"""Impacto de cada segmentador na classificação, com hiperparâmetros fixos.

**Este desenho é inválido e está preservado de propósito.** Ele mantém os
hiperparâmetros da CNN fixos para os quatro segmentadores, achando que isso isola a
variável. Isola a variável errada: mede o quanto aquele conjunto fixo por acaso
serve para cada segmentador, não o que cada segmentador consegue entregar.

O corte do orientador em 17/09: *"não faz sentido neste trabalho analisar ele
separado"*. Neste trabalho o segmentador é um eixo da otimização conjunta, e nunca
vai ser usado com hiperparâmetros que não foram escolhidos para ele. A pergunta
certa está no experimento 06.

Fica porque é a evidência de que a otimização conjunta não é capricho metodológico:
com CNN fixa, o SLIC em Pavia dá Kappa 0,0346 — praticamente aleatório — contra
0,9360 do SEEDS, e a conclusão seria "o SLIC não serve para Pavia". Sob otimização
conjunta (experimento 06) ele empata com o SEEDS ali: OA 0,9534 contra 0,9515.

A referência é o SEEDS, o segmentador do trabalho herdado. O log guardado ainda traz
uma linha "sem fusão", condição que saiu do desenho: o trabalho otimiza a
arquitetura herdada, que já tem fusão, e não a compara com uma sem.

Protocolo: 10% de treino (estratificado), teto de 200 épocas com patience 10. Os 200
são teto, não alvo — com poucas amostras o modelo não bate a meta de 0,95 e o early
stopping governa.

Ao contrário dos outros experimentos, este não é retomável: grava o CSV só no fim.
"""

import gc

import numpy as np
import pandas as pd

from hsi.config import DATASETS, ExperimentConfig
from hsi.dados.cenas import load_scene, to_pixel_table
from hsi.dados.preprocessamento import apply_pca, rgb_composite
from hsi.dados.segmentacao import (
    LSC_PARAMS,
    SEEDS_PARAMS,
    SLIC_PARAMS,
    WATERSHED_PARAMS,
    segment_superpixels,
)
from hsi.experimentos.infra import caminho_resultado
from hsi.pipeline import release_result, run_experiment
from hsi.visualizacao.diagnosticos import segmentation_quality


def _e05_estrutura(cena: str, segmentador: str) -> dict:
    """Métricas estruturais da segmentação — calculadas uma vez, não por seed."""
    tabelas = {"seeds": SEEDS_PARAMS, "slic": SLIC_PARAMS,
               "lsc": LSC_PARAMS, "watershed": WATERSHED_PARAMS}
    cenario = load_scene(cena, download=False)
    pixels, _ = to_pixel_table(cenario)
    componentes, _ = apply_pca(pixels, n_components=3, random_state=42)
    rgb = rgb_composite(componentes, cenario.shape)
    resultado = segment_superpixels(rgb, segmentador, tabelas[segmentador][cena])
    return segmentation_quality(resultado.labels, cenario.labels)


def experimento_05(*cenas: str) -> "pd.DataFrame":
    """Contraexemplo: os quatro segmentadores sob hiperparâmetros fixos."""
    cenas = cenas or ("IP", "PU", "SA")
    sementes = (11, 29, 42, 73, 101)
    condicoes = ("seeds", "slic", "lsc", "watershed")

    print("IMPACTO DO SEGMENTADOR NA CLASSIFICAÇÃO — CNN1D, hiperparâmetros fixos")
    print("DESENHO INVÁLIDO, preservado como contraexemplo (ver docstring).")
    print(f"10% de treino | teto de 200 épocas | patience 10 | {len(sementes)} seeds",
          flush=True)
    linhas = []
    for cena in cenas:
        print(f"\n### {cena} — {DATASETS[cena]}", flush=True)
        print(f"{'condição':<12}{'OA':>18}{'AA':>18}{'Kappa':>18}"
              f"{'regiões':>9}{'pureza':>9}{'treino':>9}{'épocas':>8}", flush=True)
        for condicao in condicoes:
            metricas, tempos, epocas = [], [], []
            for semente in sementes:
                config = ExperimentConfig(
                    dataset=cena, model_type="cnn1d",
                    two_input=True, segmenter=condicao,
                    fusion="scalar", train_fraction=0.10,
                    epochs=200, patience=10, early_stopping_target=0.95, seed=semente,
                )
                resultado = run_experiment(config, backend="thread", verbose=False)
                metricas.append((resultado.metrics.overall_accuracy,
                                 resultado.metrics.average_accuracy,
                                 resultado.metrics.kappa))
                tempos.append(resultado.training.total_time)
                epocas.append(len(resultado.training.history))
                linhas.append({
                    "cena": cena, "condição": condicao, "seed": semente,
                    "OA": metricas[-1][0], "AA": metricas[-1][1], "Kappa": metricas[-1][2],
                    "treino_s": tempos[-1], "épocas": epocas[-1],
                })
                release_result(resultado)
                gc.collect()
            valores = np.array(metricas)
            qualidade = _e05_estrutura(cena, condicao)
            regioes = qualidade["superpixels"]
            pureza = f"{qualidade['pureza rotulada']:.4f}"
            print(f"{condicao:<12}"
                  f"{valores[:,0].mean():>11.4f}±{valores[:,0].std():.4f}"
                  f"{valores[:,1].mean():>11.4f}±{valores[:,1].std():.4f}"
                  f"{valores[:,2].mean():>11.4f}±{valores[:,2].std():.4f}"
                  f"{regioes:>9}{pureza:>9}"
                  f"{np.mean(tempos):>8.1f}s{np.mean(epocas):>8.0f}", flush=True)

    quadro = pd.DataFrame(linhas)
    destino = caminho_resultado("05_impacto_segmentadores.csv")
    quadro.to_csv(destino, index=False)
    print("\nDiferença para o SEEDS, o segmentador herdado (OA média):", flush=True)
    for cena in cenas:
        sub = quadro[quadro["cena"] == cena].groupby("condição")["OA"].mean()
        base = sub["seeds"]
        ganhos = ", ".join(f"{k}={100*(v-base):+.2f}pp"
                           for k, v in sub.items() if k != "seeds")
        print(f"  {cena}: {ganhos}", flush=True)
    print(f"\nCSV: {destino}", flush=True)
    return quadro
