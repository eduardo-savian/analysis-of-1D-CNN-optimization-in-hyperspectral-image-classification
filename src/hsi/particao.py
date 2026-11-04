"""Partição da cena: N pedaços treinados ou inferidos ao mesmo tempo.

A segmentação roda uma vez na cena inteira, antes de partir; segmentar cada pedaço
daria superpixels diferentes nas fronteiras. Os pedaços são faixas horizontais com
número parecido de pixels rotulados, então o teto do speedup é perto de N.

Dois modos:
- "independente": cada pedaço treina e testa o próprio modelo;
- "replicado": um modelo treinado com a cena inteira, e só a inferência é dividida.

Executores: "sequencial" (a referência do speedup), "thread" e "process". As
partições sempre rodam na CPU, para que os três se comparem no mesmo dispositivo.
"""

import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn

from hsi.ambiente import configure_determinism, seed_everything, select_device
from hsi.dados import build_loaders, prepare_data
from hsi.modelo import build_model
from hsi.treino import fit, metrics, predict

CPU = torch.device("cpu")


def partition_scene(coords: np.ndarray, n: int, modo: str = "faixas", seed: int = 42) -> np.ndarray:
    """Índice da partição de cada amostra.

    "faixas" ordena as amostras pela linha da cena e corta em N partes do mesmo
    tamanho; uma fronteira pode cair no meio de uma linha. "aleatoria" sorteia, e serve
    de controle: mesmo custo, sem contiguidade espacial.
    """
    if modo == "faixas":
        ordem = np.argsort(coords[:, 0], kind="stable")
    else:
        ordem = np.random.default_rng(seed).permutation(len(coords))
    particao = np.empty(len(coords), dtype=np.int64)
    for k, bloco in enumerate(np.array_split(ordem, n)):
        particao[bloco] = k
    return particao


def slice_data(d: dict, mascara: np.ndarray) -> dict:
    """Recorta os dados para as amostras de uma partição, renumerando os índices."""
    indices = np.flatnonzero(mascara)
    nova_posicao = np.full(len(mascara), -1)
    nova_posicao[indices] = np.arange(len(indices))

    def renumerar(parte):
        posicoes = nova_posicao[parte]
        return posicoes[posicoes >= 0]

    return {**d, "espectros": d["espectros"][indices], "rotulos": d["rotulos"][indices],
            "segmentos": d["segmentos"][indices], "coords": d["coords"][indices],
            "treino": renumerar(d["treino"]), "val": renumerar(d["val"]),
            "teste": renumerar(d["teste"])}


def train_and_predict(tarefa):
    """Trabalho de uma partição: treina (se não vier modelo) e prevê o teste dela.

    Fica no nível do módulo porque o executor de processos a envia por pickle, que
    só serializa funções pelo nome.
    """
    k, pedaco, config, modelo = tarefa
    inicio = time.perf_counter()
    loaders = build_loaders(pedaco, config, CPU)
    if modelo is None:
        seed_everything(config.seed)
        modelo, perda, otimizador = build_model(config, pedaco, CPU)
        fit(modelo, loaders, perda, otimizador, config, CPU)
    previsto = predict(modelo, loaders["teste"], CPU)
    return k, previsto, pedaco["rotulos"][pedaco["teste"]], time.perf_counter() - inicio


def init_worker():
    """Roda uma vez em cada thread ou processo do pool.

    O número de threads do PyTorch vale por thread, e um processo novo sobe com o
    padrão da máquina. Sem isto, N workers usariam N x 32 threads nos mesmos núcleos.
    """
    torch.set_num_threads(1)
    cv.setNumThreads(1)


def new_pool(executor: str, n: int):
    """Pool de `n` workers, ou None para o executor sequencial."""
    if executor == "sequencial":
        return None
    if executor == "thread":
        return ThreadPoolExecutor(max_workers=n, initializer=init_worker)
    # spawn, e não fork: o PyTorch recusa fork depois que o autograd rodou em threads.
    # O preço é que cada processo sobe um interpretador novo e reimporta o PyTorch.
    return ProcessPoolExecutor(max_workers=n, mp_context=multiprocessing.get_context("spawn"),
                               initializer=init_worker)


def _aquecer(_):
    # Força o worker a terminar de subir e a inicializar o autograd antes da medição.
    # O sleep segura o worker ocupado, para o pool criar todos em vez de reusar um.
    x = torch.zeros(8, 1, 64, requires_grad=True)
    nn.Conv1d(1, 16, 9)(x).sum().backward()
    time.sleep(0.25)


def warm_up(pool, n: int) -> float:
    """Sobe e aquece os `n` workers de um pool; devolve quanto isso levou."""
    inicio = time.perf_counter()
    if pool is not None:
        list(pool.map(_aquecer, range(n)))
    return time.perf_counter() - inicio


def run_experiment_particionado(config, n: int, modo: str = "independente",
                                particao: str = "faixas", executor: str = "thread",
                                pool=None) -> dict:
    """Divide a cena em `n` partições e as processa com o executor pedido.

    Com `pool`, usa um pool já criado e aquecido (pool quente); sem ele, o pool é
    criado e fechado dentro da medição (pool frio). As métricas são calculadas sobre o
    teste inteiro, remontado a partir das previsões de cada partição.
    """
    if modo == "replicado" and executor == "process":
        raise ValueError("o modo replicado com processos exigiria enviar o modelo a cada um")
    configure_determinism()
    inicio = time.perf_counter()
    dados, tempos_preparo = prepare_data(config, "thread")
    tempo_preparo = time.perf_counter() - inicio
    grupo = partition_scene(dados["coords"], n, particao, config.seed)
    pedacos = [slice_data(dados, grupo == k) for k in range(n)]

    tempo_treino_unico, modelo = 0.0, None
    if modo == "replicado":
        # Treina uma vez com tudo, no dispositivo padrão, e desce o modelo para a CPU,
        # onde todas as partições inferem.
        inicio = time.perf_counter()
        device = select_device(verbose=False)
        loaders = build_loaders(dados, config, device)
        seed_everything(config.seed)
        modelo, perda, otimizador = build_model(config, dados, device)
        fit(modelo, loaders, perda, otimizador, config, device)
        modelo = modelo.to(CPU)
        tempo_treino_unico = time.perf_counter() - inicio

    tarefas = [(k, pedacos[k], config, modelo) for k in range(n)]
    pool_frio = pool is None and executor != "sequencial"
    inicio = time.perf_counter()
    if pool_frio:
        pool = new_pool(executor, n)
    if executor == "sequencial":
        saidas = [train_and_predict(t) for t in tarefas]
    else:
        saidas = list(pool.map(train_and_predict, tarefas))
    if pool_frio:
        pool.shutdown()
    tempo_paralelo = time.perf_counter() - inicio

    previsto = np.concatenate([s[1] for s in saidas])
    verdade = np.concatenate([s[2] for s in saidas])
    tempos = [s[3] for s in saidas]
    return {
        **metrics(verdade, previsto),
        "tempo_preparo": tempo_preparo, "tempo_segmentacao": tempos_preparo["espacial"],
        "tempo_treino_unico": tempo_treino_unico,
        # O bloco concorrente: treino e inferência no modo independente, só a
        # inferência no replicado.
        "tempo_paralelo": tempo_paralelo,
        "tempos_por_particao": tempos,
        # Com executor concorrente cada tempo já foi medido sob disputa; o speedup
        # honesto é contra a parede do executor sequencial, não soma ÷ parede.
        "soma_partes": sum(tempos),
        "teto": sum(tempos) / max(tempos),
        "overhead": tempo_paralelo - max(tempos),
        "amostras_por_particao": [len(p["rotulos"]) for p in pedacos],
    }
