"""Experimento 07: pool frio contra pool quente na partição da cena.

Frio: o pool é criado e fechado dentro da medição, e cada execução paga a partida dos
workers (com processos, um interpretador novo que reimporta o PyTorch). Quente: o pool
é criado e aquecido uma vez, fora da medição, e atende as repetições.

O speedup é contra o executor sequencial nas mesmas partições, no mesmo dispositivo.
A OA também é conferida: sequencial e processos repetem o valor; threads no modo
independente não, porque as partições disputam o gerador aleatório global do PyTorch.

Indian Pines, 25 épocas, lote 64, semente 42, uma thread do PyTorch por worker.
"""

import numpy as np
import pandas as pd
import torch

from hsi.config import CONFIG, ExperimentConfig
from hsi.experimentos.infra import caminho_resultado, carregar_feito, salvar
from hsi.paralelismo.particao import new_pool, run_experiment_particionado, warm_up

C = CONFIG["experimentos"]["07"]
PARTICOES_POOL = tuple(C["particoes"])
REPETICOES_POOL = C["repeticoes"]
# O sequencial vem primeiro porque é o denominador do speedup.
ARRANJOS_POOL = (("independente", "sequencial"), ("independente", "process"),
            ("replicado", "sequencial"), ("replicado", "thread"),
            ("independente", "thread"))
# Um pool que não cabe com 25% de folga é recusado (ver config.toml).
MB_POR_PROCESSO = C["mb_por_processo"]


def _memoria_livre_mb() -> float:
    with open("/proc/meminfo") as f:
        for linha in f:
            if linha.startswith("MemAvailable:"):
                return int(linha.split()[1]) / 1024
    return float("inf")


def _medir(cena, modo, executor, n, estado) -> list[dict]:
    if executor == "process" and 1.25 * n * MB_POR_PROCESSO > _memoria_livre_mb():
        raise MemoryError(f"{n} processos não cabem na memória livre")
    config = ExperimentConfig(dataset=cena, epochs=C["epochs"], batch_size=C["batch_size"], seed=42)
    pool, aquecimento = None, 0.0
    if estado == "quente":
        pool = new_pool(executor, n)
        aquecimento = warm_up(pool, n)
    linhas = []
    try:
        for rep in range(REPETICOES_POOL[estado]):
            r = run_experiment_particionado(config, n, modo, executor, pool=pool)
            linhas.append({"cena": cena, "modo": modo, "executor": executor, "n": n,
                           "estado": estado, "rep": rep, "bloco_paralelo": r["tempo_paralelo"],
                           "soma_partes": r["soma_partes"], "overhead": r["overhead"],
                           "teto": r["teto"], "tempo_aquecimento": aquecimento, "OA": r["OA"]})
    finally:
        if pool is not None:
            pool.shutdown()
    return linhas


def experimento_07(*cenas: str) -> pd.DataFrame:
    """Pool frio contra pool quente, por executor e número de partições."""
    cenas = cenas or tuple(C["cenas"])
    torch.set_num_threads(1)
    destino = caminho_resultado("07_pool_quente.csv")
    linhas, feitas = carregar_feito(destino, ("cena", "modo", "executor", "n", "estado", "rep"))
    for cena in cenas:
        for modo, executor in ARRANJOS_POOL:
            for n in PARTICOES_POOL:
                for estado in ("frio", "quente"):
                    if all((cena, modo, executor, n, estado, rep) in feitas
                           for rep in range(REPETICOES_POOL[estado])):
                        continue
                    try:
                        novas = _medir(cena, modo, executor, n, estado)
                    except MemoryError as erro:
                        print(f"{modo} {executor} N={n} {estado}: {erro}")
                        continue
                    linhas += novas
                    salvar(linhas, destino)
                    print(f"{cena} {modo:<13}{executor:<11}N={n:<3}{estado:<7}"
                          f"bloco {np.mean([l['bloco_paralelo'] for l in novas]):6.2f}s", flush=True)

    quadro = pd.DataFrame(linhas)
    print("\nspeedup sobre o sequencial (média do bloco paralelo):")
    medias = quadro.groupby(["cena", "modo", "estado", "n", "executor"])["bloco_paralelo"].mean()
    tabela = medias.unstack("executor")
    print((1 / tabela.div(tabela["sequencial"], axis=0)).round(2))
    print(f"\nCSV: {destino}")
    return quadro
