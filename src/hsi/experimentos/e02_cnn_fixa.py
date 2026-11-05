"""Experimento 02: os três segmentadores com a mesma configuração da CNN.

É o desenho previsto no projeto, e ficou como contraexemplo: com a CNN fixa, a
comparação mede o quanto essa configuração combina com cada segmentador, e não a qualidade
do segmentador. O experimento 03 refaz a comparação com a CNN ajustada para cada um.

Treino com os 85% do original, até 200 épocas e parada após 10 sem melhora,
como na linha de base; 5 sementes.
"""

import numpy as np
import pandas as pd

from hsi.config import CONFIG, SEGMENTERS, ExperimentConfig
from hsi.dados import apply_pca, load_scene, pixel_table
from hsi.experimentos.infra import caminho_resultado
from hsi.pipeline import run_experiment
from hsi.segmentadores import SEGMENTER_PARAMS, rgb_composite, segment

C = CONFIG["experimentos"]["02"]
SEMENTES = tuple(C["sementes"])


def qualidade_segmentacao(cena: str, segmentador: str) -> dict:
    """Número de regiões e pureza: a fração da classe dominante em cada região,
    ponderada pelos pixels rotulados."""
    cubo, gt = load_scene(cena)
    pixels, _ = pixel_table(cubo, gt)
    rgb = rgb_composite(apply_pca(pixels, 3, 42), gt.shape)
    regioes = segment(rgb, segmentador, SEGMENTER_PARAMS[segmentador][cena]).ravel()
    rotulado = gt.ravel() > 0
    ids, classes = regioes[rotulado], gt.ravel()[rotulado]
    dominantes = sum(np.bincount(classes[ids == r]).max() for r in np.unique(ids))
    return {"regioes": len(np.unique(regioes)), "pureza": dominantes / len(classes)}


def experimento_02(*cenas: str) -> pd.DataFrame:
    """Segmentadores com a CNN fixa (contraexemplo)."""
    cenas = cenas or tuple(C["cenas"])
    linhas = []
    for cena in cenas:
        print(f"\n### {cena}")
        for segmentador in SEGMENTERS:
            for semente in SEMENTES:
                config = ExperimentConfig(dataset=cena, segmenter=segmentador, seed=semente)
                r = run_experiment(config)
                linhas.append({"cena": cena, "condição": segmentador, "seed": semente,
                               "OA": r["OA"], "AA": r["AA"], "Kappa": r["Kappa"],
                               "treino_s": r["tempo_treino"], "épocas": r["epocas"]})
            oa = np.array([l["OA"] for l in linhas[-len(SEMENTES):]])
            q = qualidade_segmentacao(cena, segmentador)
            print(f"{segmentador:<10} OA {oa.mean():.4f} ± {oa.std():.4f}  "
                  f"regiões {q['regioes']:>5}  pureza {q['pureza']:.4f}", flush=True)

    quadro = pd.DataFrame(linhas)
    destino = caminho_resultado("02_impacto_segmentadores.csv")
    quadro.to_csv(destino, index=False)
    print(f"\nCSV: {destino}")
    return quadro
