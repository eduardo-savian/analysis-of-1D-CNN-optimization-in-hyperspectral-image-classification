import ast

import numpy as np
import pandas as pd

from hsi.config import CONFIG, SEGMENTERS, ExperimentConfig
from hsi.experimentos.infra import caminho_resultado, carregar_feito, salvar
from hsi.hpo import ORIGINAL
from hsi.pipeline import run_experiment

C = CONFIG["experimentos"]["05"]
SEMENTES = tuple(C["sementes"])


def _config_impacto(vencedores, cena, segmentador, configuracao, semente):
    comum = dict(dataset=cena, segmenter=segmentador, seed=semente)
    if configuracao == "herdada":
        return ExperimentConfig(**comum, **ORIGINAL)
    v = vencedores[(vencedores["cena"] == cena) & (vencedores["condição"] == segmentador)].iloc[0]
    return ExperimentConfig(
        **comum, learning_rate=float(v["learning_rate"]), dropout=float(v["dropout"]),
        weight_decay=float(v["weight_decay"]), batch_size=int(v["batch_size"]),
        pca_components=int(v["pca_components"]),
        # literal_eval lê o dicionário do CSV sem executar código.
        segmenter_overrides=ast.literal_eval(v["overrides"]),
    )


def experimento_05(*cenas: str) -> pd.DataFrame:
    """Configuração herdada contra configuração otimizada, por segmentador."""
    cenas = cenas or tuple(C["cenas"])
    vencedores = pd.read_csv(caminho_resultado("03_resumo.csv"))
    destino = caminho_resultado("05_impacto_hpo.csv")
    linhas, feitas = carregar_feito(destino, ("cena", "condição", "configuração", "semente"))
    for cena in cenas:
        print(f"\n### {cena}")
        for segmentador in SEGMENTERS:
            for configuracao in ("herdada", "otimizada"):
                for semente in SEMENTES:
                    if (cena, segmentador, configuracao, semente) in feitas:
                        continue
                    r = run_experiment(_config_impacto(vencedores, cena, segmentador, configuracao, semente))
                    linhas.append({"cena": cena, "condição": segmentador, "configuração": configuracao,
                                   "semente": semente, "OA": r["OA"], "AA": r["AA"],
                                   "Kappa": r["Kappa"], "épocas": r["epocas"]})
                    salvar(linhas, destino)
            oa = {configuracao: np.array([l["OA"] for l in linhas if l["cena"] == cena
                                     and l["condição"] == segmentador and l["configuração"] == configuracao])
                  for configuracao in ("herdada", "otimizada")}
            ganho = 100 * (oa["otimizada"].mean() - oa["herdada"].mean())
            # Diferença afirmável quando passa de duas vezes o desvio combinado.
            desvio = 100 * np.hypot(oa["otimizada"].std(), oa["herdada"].std())
            print(f"{segmentador:<10} herdada {oa['herdada'].mean():.4f}  "
                  f"otimizada {oa['otimizada'].mean():.4f}  ganho {ganho:+.2f} pp "
                  f"({'afirmável' if abs(ganho) > 2 * desvio else 'dentro do ruído'})", flush=True)
    print(f"\nCSV: {destino}")
    return pd.DataFrame(linhas)
