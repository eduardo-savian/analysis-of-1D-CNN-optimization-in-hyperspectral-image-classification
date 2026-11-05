"""Experimento 03: cada segmentador com a CNN ajustada para ele.

Para cada cena e segmentador: otimização bayesiana de 16 trials restrita àquele
segmentador, escolha do vencedor pela validação e retreino do vencedor com 5 sementes.
O SEEDS, segmentador do trabalho original, passa pelo mesmo tratamento e é a referência
de `ganho_OA_pp`. Treino com os 85% do original, até 200 épocas e parada após 10 sem
melhora, como na linha de base, nos trials e nos retreinos.

Grava `03_resumo.csv` (uma linha por cena e segmentador, lida pelo experimento 05) e
`03_por_seed_<cenas>.csv`.
"""

import time
from dataclasses import replace

import numpy as np
import pandas as pd

from hsi.config import CONFIG, SEGMENTERS, ExperimentConfig
from hsi.experimentos.infra import caminho_resultado, carregar_feito, salvar
from hsi.hpo import automated_hpo
from hsi.pipeline import run_experiment

C = CONFIG["experimentos"]["03"]
TRIALS_SEGMENTADOR = C["trials"]
SEMENTES = tuple(C["sementes"])


def experimento_03(*cenas: str) -> list:
    """Segmentadores sob otimização conjunta."""
    cenas = cenas or tuple(C["cenas"])
    destino = caminho_resultado("03_resumo.csv")
    resumo, feitas = carregar_feito(destino, ("cena", "condição"))
    por_semente = []
    for cena in cenas:
        print(f"\n### {cena}")
        # SEEDS primeiro: é a referência do ganho.
        for segmentador in SEGMENTERS:
            if (cena, segmentador) in feitas:
                continue
            inicio = time.perf_counter()
            base = ExperimentConfig(dataset=cena, segmenter=segmentador, seed=42)
            trials = automated_hpo(base, TRIALS_SEGMENTADOR, search_epochs=base.epochs, strategy="bayes",
                                   segmenters=(segmentador,))
            vencedor = trials.loc[trials["val_accuracy (objetivo)"].idxmax()]
            config = vencedor["config"]

            # Guarda só as métricas de cada retreino; modelo e dados são liberados.
            medidas = []
            for semente in SEMENTES:
                r = run_experiment(replace(config, seed=semente))
                medidas.append({"cena": cena, "condição": segmentador, "seed": semente,
                                "OA": r["OA"], "AA": r["AA"], "Kappa": r["Kappa"],
                                "treino_s": r["tempo_treino"], "épocas": r["epocas"]})
            por_semente += [{k: m[k] for k in ("cena", "condição", "seed", "OA", "AA", "Kappa")}
                            for m in medidas]
            oa, aa, kappa = (np.array([m[k] for m in medidas]) for k in ("OA", "AA", "Kappa"))
            resumo.append({
                "cena": cena, "condição": segmentador,
                "val_busca": float(vencedor["val_accuracy (objetivo)"]),
                "OA": oa.mean(), "OA_std": oa.std(), "AA": aa.mean(), "AA_std": aa.std(),
                "Kappa": kappa.mean(), "Kappa_std": kappa.std(),
                "ganho_OA_pp": float("nan"),
                "learning_rate": config.learning_rate, "dropout": config.dropout,
                "weight_decay": config.weight_decay, "batch_size": config.batch_size,
                "pca_components": config.pca_components,
                "overrides": str(config.segmenter_overrides),
                "treino_s": np.mean([m["treino_s"] for m in medidas]),
                "épocas": np.mean([m["épocas"] for m in medidas]),
                "minutos_totais": (time.perf_counter() - inicio) / 60,
            })
            salvar(resumo, destino)
            print(f"{segmentador:<10} val {vencedor['val_accuracy (objetivo)']:.4f}  "
                  f"OA {oa.mean():.4f} ± {oa.std():.4f}", flush=True)

        # Ganho de cada segmentador sobre o SEEDS, os dois otimizados.
        da_cena = [r for r in resumo if r["cena"] == cena]
        referencia = next((r["OA"] for r in da_cena if r["condição"] == "seeds"), None)
        if referencia is not None:
            for r in da_cena:
                r["ganho_OA_pp"] = 100 * (r["OA"] - referencia)
            salvar(resumo, destino)

    if por_semente:
        pd.DataFrame(por_semente).to_csv(
            caminho_resultado(f"03_por_seed_{'_'.join(cenas)}.csv"), index=False)
    print(f"\nCSV: {destino}")
    return resumo
