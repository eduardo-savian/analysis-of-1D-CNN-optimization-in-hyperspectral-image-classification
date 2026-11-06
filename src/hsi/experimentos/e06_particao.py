"""Experimento 06: a cena dividida em N = 2, 4, 8 e 16 faixas, nos modos independente
e replicado.

No modo independente cada partição treina num processo: com threads elas dividem o
gerador aleatório global do PyTorch e a OA muda entre execuções iguais. O replicado só
infere, sem sorteio, e usa threads. 25 épocas, lote 64, uma thread do PyTorch, 3 sementes. O
monolítico treina na GPU e as partições na CPU, então o speedup contra ele mistura
dispositivo com paralelismo; o speedup entre executores está no experimento 07.
"""

import time

import numpy as np
import pandas as pd
import torch

from hsi.config import CONFIG, ExperimentConfig
from hsi.experimentos.infra import caminho_resultado, carregar_feito, salvar
from hsi.paralelismo.particao import run_experiment_particionado
from hsi.pipeline import run_experiment

C = CONFIG["experimentos"]["06"]
PARTICOES = tuple(C["particoes"])
SEMENTES_PARTICAO = tuple(C["sementes"])
# Independente: cada faixa treina e avalia o próprio modelo. Replicado: um modelo treinado
# com a cena inteira infere em cada faixa.
MODOS = ("independente", "replicado")


def _config_particao(cena, semente):
    return ExperimentConfig(dataset=cena, epochs=C["epochs"], batch_size=C["batch_size"], seed=semente)


def experimento_06(*cenas: str) -> pd.DataFrame:
    """Escala da partição da cena e o custo em acurácia."""
    cenas = cenas or tuple(C["cenas"])
    torch.set_num_threads(1)
    destino = caminho_resultado("06_paralelismo_particao.csv")
    linhas, feitas = carregar_feito(destino, ("cena", "arranjo", "n", "semente"))
    for cena in cenas:
        for semente in SEMENTES_PARTICAO:
            if (cena, "monolítico", 1, semente) in feitas:
                continue
            inicio = time.perf_counter()
            r = run_experiment(_config_particao(cena, semente))
            linhas.append({"cena": cena, "arranjo": "monolítico", "n": 1, "semente": semente,
                           "tempo_total": time.perf_counter() - inicio,
                           "OA": r["OA"], "AA": r["AA"], "Kappa": r["Kappa"]})
            salvar(linhas, destino)

        for modo in MODOS:
            for n in PARTICOES:
                for semente in SEMENTES_PARTICAO:
                    if (cena, modo, n, semente) in feitas:
                        continue
                    inicio = time.perf_counter()
                    executor = "process" if modo == "independente" else "thread"
                    r = run_experiment_particionado(_config_particao(cena, semente), n, modo,
                                                    executor=executor)
                    linhas.append({
                        "cena": cena, "arranjo": modo, "n": n, "semente": semente,
                        "tempo_total": time.perf_counter() - inicio,
                        "bloco_paralelo": r["tempo_paralelo"], "soma_partes": r["soma_partes"],
                        "speedup_bloco": r["soma_partes"] / r["tempo_paralelo"],
                        "teto": r["teto"], "OA": r["OA"], "AA": r["AA"], "Kappa": r["Kappa"],
                    })
                    salvar(linhas, destino)
                oa = [l["OA"] for l in linhas
                      if l["cena"] == cena and l["arranjo"] == modo and l["n"] == n]
                print(f"{cena} {modo:<13} N={n:<3} OA {np.mean(oa):.4f}", flush=True)
    print(f"\nCSV: {destino}")
    return pd.DataFrame(linhas)
