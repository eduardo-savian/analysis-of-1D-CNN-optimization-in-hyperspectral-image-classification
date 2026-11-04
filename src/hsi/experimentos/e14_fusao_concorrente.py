"""Experimento 14: a fusão concorrente do orientador contra a inferência em lote.

Com o mesmo modelo treinado (a configuração herdada, com SEEDS) e todos os pixels
rotulados da cena, mede quatro arranjos:
- "lote": segmenta e infere tudo de uma vez, com 1 e com 16 threads do PyTorch;
- "sequencial": segmenta e percorre as posições em blocos, numa thread só;
- "concorrente": o esquema, com N threads, blocos de 1, 64 e 1.024 posições e as duas
  extensões da região crítica;
- "desacoplado": o mesmo, com todas as convoluções enfileiradas antes das fusões.

As threads de trabalho usam uma thread do PyTorch cada, e o pool é aquecido fora da
medição. Cada execução confere se os ids lidos do mapa são os do treino e se as
classes previstas são as do lote.

`espera_total` soma, nas tarefas, o tempo entre o fim das convoluções e a leitura do
mapa. Na variante desacoplada inclui o tempo em fila esperando a tarefa de fusão, e
não mede bloqueio; ali a leitura certa é `parede` e `conv_antes_do_mapa`.
"""

import numpy as np
import pandas as pd
import torch

from hsi.config import ExperimentConfig
from hsi.dados import load_scene, pixel_table, segmenter_params
from hsi.experimentos.infra import caminho_resultado, carregar_feito, salvar
from hsi.fusao_concorrente import executar_concorrente, executar_sequencial, warm_pool
from hsi.pipeline import run_experiment

THREADS = (1, 2, 4, 8, 16)
BLOCOS = (1, 64, 1024)
SECOES = ("desenho", "minima")
REPETICOES = 5
# Com bloco 1 é uma tarefa por pixel: só Indian Pines e 3 repetições. Em Pavia e
# Salinas seriam 43 e 54 mil tarefas por execução.
REPETICOES_BLOCO_1 = 3
CENAS_BLOCO_1 = ("IP",)
# (n, bloco, seção) da execução cuja linha do tempo é gravada inteira, uma por arranjo.
LINHA_DO_TEMPO = (4, 64, "minima")


def _preparar(cena):
    """Treina o modelo uma vez e monta a entrada da inferência concorrente."""
    config = ExperimentConfig(dataset=cena)
    r = run_experiment(config)
    modelo = r["modelo"].to(torch.device("cpu")).eval()
    dados = r["dados"]
    cubo, gt = load_scene(cena)
    entrada = {"pixels": pixel_table(cubo, gt)[0], "forma": gt.shape,
               "segmentador": config.segmenter, "params": segmenter_params(config),
               "seed": config.seed, "coords": dados["coords"],
               "espectros": np.ascontiguousarray(dados["espectros"], dtype=np.float32)}
    return modelo, entrada, dados


def _linha(cena, r, rep, torch_threads, dados, referencia) -> dict:
    tarefas = r["tarefas"]
    esperas = [t["pegou_mapa"] - t["fim_convolucoes"] for t in tarefas]
    # Fração das posições cujas convoluções acabaram antes de T0 publicar o mapa.
    antes = sum(t["n_posicoes"] for t in tarefas if t["fim_convolucoes"] <= r["t0_fim"])
    teste = dados["teste"]
    return {
        "cena": cena, "arranjo": r["arranjo"], "n": r["n"], "bloco": r["bloco"],
        "secao": r["secao"], "torch_threads": torch_threads, "rep": rep,
        "parede": r["parede"], "segmentacao": r["t0_fim"] - r["t0_inicio"],
        "t0_fim": r["t0_fim"], "espera_total": sum(esperas),
        "espera_maxima": max(esperas, default=0.0),
        "conv_antes_do_mapa": antes / sum(t["n_posicoes"] for t in tarefas) if tarefas else 0.0,
        "tarefas": len(tarefas),
        "OA_teste": float(np.mean(r["previsto"][teste] == dados["rotulos"][teste])),
        "ids_iguais_ao_treino": bool(np.array_equal(r["ids"], dados["segmentos"])),
        "classes_iguais_ao_lote": bool(np.array_equal(r["previsto"], referencia)),
    }


def experimento_14(*cenas: str) -> pd.DataFrame:
    """Fusão concorrente: T0 segmenta enquanto T1..TN extraem e fundem."""
    cenas = cenas or ("IP",)
    destino = caminho_resultado("14_fusao_concorrente.csv")
    chaves = ("cena", "arranjo", "n", "bloco", "secao", "torch_threads", "rep")
    linhas, feitas = carregar_feito(destino, chaves)
    for cena in cenas:
        torch.set_num_threads(16)
        modelo, entrada, dados = _preparar(cena)
        torch.set_num_threads(1)
        # As classes de referência são as do lote com uma thread.
        referencia = executar_sequencial(modelo, entrada)["previsto"]

        def medir(arranjo, n, bloco, secao, torch_threads, executar):
            reps = REPETICOES_BLOCO_1 if bloco == 1 else REPETICOES
            for rep in range(reps):
                if (cena, arranjo, n, bloco, secao, torch_threads, rep) in feitas:
                    continue
                r = executar()
                linha = _linha(cena, r, rep, torch_threads, dados, referencia)
                if not (linha["ids_iguais_ao_treino"] and linha["classes_iguais_ao_lote"]):
                    print(f"DIVERGÊNCIA: {arranjo} N={n} bloco={bloco} {secao}")
                if rep == 0 and r["tarefas"] and (n, bloco, secao) == LINHA_DO_TEMPO:
                    tempos = pd.DataFrame(r["tarefas"]).assign(t0_inicio=r["t0_inicio"],
                                                               t0_fim=r["t0_fim"])
                    tempos.to_csv(caminho_resultado(f"14_linha_do_tempo_{cena}_{arranjo}.csv"),
                                  index=False)
                linhas.append(linha)
            salvar(linhas, destino)

        n_posicoes = len(entrada["espectros"])
        blocos = [b for b in BLOCOS if b != 1 or cena in CENAS_BLOCO_1]
        for torch_threads in (1, 16):
            torch.set_num_threads(torch_threads)
            medir("lote", 1, n_posicoes, "-", torch_threads,
                  lambda: executar_sequencial(modelo, entrada))
        torch.set_num_threads(1)
        for bloco in blocos:
            medir("sequencial", 1, bloco, "-", 1,
                  lambda bloco=bloco: executar_sequencial(modelo, entrada, bloco))
        for n in THREADS:
            pool = warm_pool(modelo, entrada, n)
            for arranjo in ("concorrente", "desacoplado"):
                for bloco in blocos:
                    for secao in SECOES:
                        medir(arranjo, n, bloco, secao, 1,
                              lambda n=n, bloco=bloco, secao=secao, arranjo=arranjo:
                              executar_concorrente(modelo, entrada, n, bloco, secao,
                                                   desacoplado=arranjo == "desacoplado", pool=pool))
            pool.shutdown()
        print(f"{cena}: {len([l for l in linhas if l['cena'] == cena])} execuções", flush=True)

    quadro = pd.DataFrame(linhas)
    ok = quadro["ids_iguais_ao_treino"].all() and quadro["classes_iguais_ao_lote"].all()
    print(f"conferência: {'todas' if ok else 'NEM TODAS'} as execuções repetem os ids do "
          f"treino e as classes do lote")
    print(f"CSV: {destino}")
    return quadro
