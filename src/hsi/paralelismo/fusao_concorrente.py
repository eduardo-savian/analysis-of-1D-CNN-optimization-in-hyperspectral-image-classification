import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from hsi.dados import spatial_branch


def new_thread_pool(n: int) -> ThreadPoolExecutor:
    # torch.set_num_threads vale por thread; sem o initializer, cada thread do pool usaria o
    # padrão da máquina e as convoluções de blocos pequenos ficariam muito mais lentas.
    return ThreadPoolExecutor(max_workers=n, thread_name_prefix="T",
                              initializer=torch.set_num_threads,
                              initargs=(torch.get_num_threads(),))


def warm_pool(modelo, entrada: dict, n: int) -> ThreadPoolExecutor:
    """Cria o pool e aquece cada thread com uma passada do modelo, fora da medição.

    A barreira força a criação das `n` threads; sem ela, uma thread rápida pegaria várias passadas.
    """
    pool = new_thread_pool(n)
    barreira = threading.Barrier(n)
    x = torch.from_numpy(entrada["espectros"][:8]).float().unsqueeze(1)

    def passada():
        with torch.inference_mode():
            modelo(x, torch.zeros(len(x), 1))
        barreira.wait()

    for futuro in [pool.submit(passada) for _ in range(n)]:
        futuro.result()
    return pool


def _blocos(n: int, bloco: int) -> list[np.ndarray]:
    return [np.arange(i, min(i + bloco, n)) for i in range(0, n, bloco)]


def _segmentar(entrada: dict) -> np.ndarray:
    return spatial_branch(entrada["pixels"], entrada["forma"], entrada["segmentador"],
                          entrada["params"], entrada["seed"])


def _publicar(mapa: dict, rotulos: np.ndarray) -> None:
    # Chamar com a trava em mãos.
    mapa["rotulos"][...] = rotulos
    mapa["publicado"] = True
    mapa["trava"].notify_all()


def _t0(entrada, mapa, secao, origem, relogio):
    relogio["t0_inicio"] = time.perf_counter() - origem
    if secao == "desenho":
        with mapa["trava"]:
            _publicar(mapa, _segmentar(entrada))
    else:
        rotulos = _segmentar(entrada)
        with mapa["trava"]:
            _publicar(mapa, rotulos)
    relogio["t0_fim"] = time.perf_counter() - origem


def _convolucoes(modelo, entrada, posicoes, origem):
    inicio = time.perf_counter() - origem
    # O modo sem gradiente vale por thread: precisa ser ligado dentro da tarefa.
    with torch.inference_mode():
        x = torch.from_numpy(entrada["espectros"][posicoes]).float().unsqueeze(1)
        caracteristicas = modelo.convolucoes(x)
    return posicoes, caracteristicas, inicio, time.perf_counter() - origem


def _fundir(modelo, entrada, mapa, convolucoes, secao, origem):
    """Espera o mapa, lê o id de cada posição, concatena e passa pelas densas."""
    posicoes, caracteristicas, inicio, fim_convolucoes = convolucoes
    linhas, colunas = entrada["coords"][posicoes, 0], entrada["coords"][posicoes, 1]
    with torch.inference_mode():
        with mapa["trava"]:
            mapa["trava"].wait_for(lambda: mapa["publicado"])
            pegou_mapa = time.perf_counter() - origem
            ids = mapa["rotulos"][linhas, colunas].copy()
            if secao == "desenho":
                fundido = modelo.fundir(caracteristicas, torch.from_numpy(ids).float().reshape(-1, 1))
        if secao == "minima":
            fundido = modelo.fundir(caracteristicas, torch.from_numpy(ids).float().reshape(-1, 1))
        previsto = modelo.densas(fundido).argmax(dim=1).numpy()
    tarefa = {"thread": threading.current_thread().name, "n_posicoes": len(posicoes),
              "inicio": inicio, "fim_convolucoes": fim_convolucoes,
              "pegou_mapa": pegou_mapa, "fim": time.perf_counter() - origem}
    return posicoes, ids, previsto, tarefa


def _tarefa(modelo, entrada, mapa, posicoes, secao, origem):
    # Convoluções e fusão na mesma tarefa: a thread que termina antes de T0 publicar fica
    # parada esperando o mapa, sem pegar outra tarefa.
    return _fundir(modelo, entrada, mapa, _convolucoes(modelo, entrada, posicoes, origem),
                   secao, origem)


def _tarefa_fusao(modelo, entrada, mapa, futuro_convolucoes, secao, origem):
    # Variante desacoplada. Esperar outro futuro dentro do pool não trava: as convoluções
    # entram na fila antes das fusões, e a fila é atendida em ordem.
    return _fundir(modelo, entrada, mapa, futuro_convolucoes.result(), secao, origem)


def executar_concorrente(modelo, entrada: dict, n: int, bloco: int = 1, secao: str = "minima",
                         desacoplado: bool = False, pool=None) -> dict:
    """T0 segmenta e `n` threads processam as posições em blocos de `bloco` pixels.

    `entrada` traz a cena para T0 (`pixels`, `forma`, `segmentador`, `params`, `seed`) e,
    para as threads, `espectros` e `coords` de cada pixel rotulado.
    """
    modelo.eval()
    mapa = {"rotulos": np.full(entrada["forma"], -1, dtype=np.int32),  # -1: nenhum segmentador produz
            "publicado": False, "trava": threading.Condition()}
    relogio = {}
    pool_proprio = pool is None
    pool = pool or new_thread_pool(n)
    origem = time.perf_counter()
    t0 = threading.Thread(target=_t0, args=(entrada, mapa, secao, origem, relogio), name="T0")
    t0.start()
    blocos = _blocos(len(entrada["espectros"]), bloco)
    if desacoplado:
        convolucoes = [pool.submit(_convolucoes, modelo, entrada, p, origem) for p in blocos]
        futuros = [pool.submit(_tarefa_fusao, modelo, entrada, mapa, c, secao, origem)
                   for c in convolucoes]
    else:
        futuros = [pool.submit(_tarefa, modelo, entrada, mapa, p, secao, origem) for p in blocos]
    saidas = [f.result() for f in futuros]
    t0.join()
    parede = time.perf_counter() - origem
    if pool_proprio:
        pool.shutdown()

    previsto = np.empty(len(entrada["espectros"]), dtype=np.int64)
    ids = np.empty(len(entrada["espectros"]), dtype=np.int32)
    for posicoes, ids_bloco, previsto_bloco, _ in saidas:
        previsto[posicoes] = previsto_bloco
        ids[posicoes] = ids_bloco
    return {"arranjo": "desacoplado" if desacoplado else "concorrente", "n": n,
            "bloco": bloco, "secao": secao, "previsto": previsto, "ids": ids,
            "parede": parede, "t0_inicio": relogio["t0_inicio"], "t0_fim": relogio["t0_fim"],
            "tarefas": [s[3] for s in saidas]}


def executar_sequencial(modelo, entrada: dict, bloco: int | None = None) -> dict:
    """Referência numa thread só: segmenta e depois infere.

    Sem `bloco`, infere tudo num lote; com `bloco`, na granularidade das tarefas
    concorrentes, o denominador justo do speedup.
    """
    modelo.eval()
    origem = time.perf_counter()
    rotulos = _segmentar(entrada)
    t0_fim = time.perf_counter() - origem
    n = len(entrada["espectros"])
    ids = rotulos[entrada["coords"][:, 0], entrada["coords"][:, 1]].astype(np.int32)
    previsto = np.empty(n, dtype=np.int64)
    with torch.inference_mode():
        for p in _blocos(n, bloco or n):
            x = torch.from_numpy(entrada["espectros"][p]).float().unsqueeze(1)
            seg = torch.from_numpy(ids[p]).float().reshape(-1, 1)
            previsto[p] = modelo(x, seg).argmax(dim=1).numpy()
    return {"arranjo": "lote" if bloco is None else "sequencial", "n": 1, "bloco": bloco or n,
            "secao": "-", "previsto": previsto, "ids": ids,
            "parede": time.perf_counter() - origem, "t0_inicio": 0.0, "t0_fim": t0_fim,
            "tarefas": []}
