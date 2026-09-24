"""Qual segmentador é melhor QUANDO CADA UM RECEBE SEUS PRÓPRIOS HIPERPARÂMETROS.

CRITÉRIO — e este é o ponto do experimento.
Comparar segmentadores com a CNN fixa não mede o segmentador: mede o quanto aquele
conjunto fixo de hiperparâmetros por acaso serve para ele. O contraexemplo está em
`05_cnn_fixa_contraexemplo.log`: com hiperparâmetros fixos, o SLIC em Pavia deu
Kappa de 0,0346 — praticamente aleatório. Não porque o SLIC seja ruim, mas porque
ele produz 1.428 regiões, e um id escalar indo até 1.428 contra um espectro de
desvio 1,0 exige um learning rate e um dropout completamente diferentes.

Neste trabalho o segmentador é um eixo da otimização conjunta, não uma variável
isolada. Então a pergunta certa é: **qual é o melhor resultado que cada segmentador
alcança quando a CNN é ajustada para ele?**

PROTOCOLO, em duas etapas por condição:
  1. HPO restrito àquele segmentador (busca learning rate, dropout, weight decay,
     batch, componentes de PCA e os parâmetros internos do próprio segmentador).
     Seleção só por validação.
  2. O vencedor é retreinado com várias sementes, para ter barra de erro no número
     que vai ser reportado.

A condição "sem fusão" recebe o mesmo tratamento — HPO sobre os eixos da CNN — para
ser um piso justo, e não um piso com hiperparâmetros arbitrários.

A estratégia de busca é fixada em TPE nas cinco condições: assim a comparação isola
o segmentador, e não mistura diferenças entre estratégias de busca.
"""
from __future__ import annotations

import gc
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from _bootstrap import carregar, caminho_resultado
from _retomada import carregar_feito, salvar

ns = carregar()
EC = ns["ExperimentConfig"]

TRIALS = 16               # orçamento de busca por condição
SEEDS_FINAIS = (11, 29, 42, 73, 101)
EPOCAS = 200              # teto; com 10% de treino o early stopping governa (~65)
FRACAO = 0.10
ESTRATEGIA = "tpe"
# Cenas pela linha de comando: `python 06_....py PU SA` roda só essas duas.
# Indian Pines foi medida antes da correção do piso de componentes de PCA e
# continua válida — a receita 30 nunca gerou opção abaixo de 15. Ver `06_IP.log`.
CENAS = tuple(a for a in sys.argv[1:] if not a.startswith("-")) or ("IP", "PU", "SA")
CONDICOES = ("sem fusão", "seeds", "slic", "lsc", "watershed")


def hpo_da_condicao(cena: str, condicao: str):
    """Etapa 1: busca os melhores hiperparâmetros para esta condição."""
    com_fusao = condicao != "sem fusão"
    base = EC(
        dataset=cena, model_type="cnn1d",
        two_input=com_fusao,
        segmenter=("seeds" if not com_fusao else condicao),
        fusion="scalar", train_fraction=FRACAO,
        epochs=EPOCAS, patience=10, early_stopping_target=0.95, seed=42,
    )
    tabela, _ = ns["automated_hpo"](
        base, trials_per_cnn=TRIALS, search_epochs=EPOCAS, backend="thread",
        model_types=("cnn1d",),
        segmenters=(base.segmenter,),          # busca restrita a este segmentador
        strategy=ESTRATEGIA, retain="winner",
    )
    vencedor = tabela.loc[tabela["val_accuracy (objetivo)"].idxmax()]
    return vencedor["config"], float(vencedor["val_accuracy (objetivo)"]), tabela


def retreinar(config, sementes):
    """Etapa 2: repete a receita vencedora com várias sementes."""
    metricas, tempos, epocas = [], [], []
    for semente in sementes:
        resultado = ns["run_experiment"](
            replace(config, seed=int(semente)), backend="thread", verbose=False
        )
        metricas.append((resultado.metrics.overall_accuracy,
                         resultado.metrics.average_accuracy,
                         resultado.metrics.kappa))
        tempos.append(resultado.training.total_time)
        epocas.append(len(resultado.training.history))
        ns["release_result"](resultado)
        gc.collect()
    return np.array(metricas), float(np.mean(tempos)), float(np.mean(epocas))


def main() -> None:
    print("SEGMENTADORES SOB OTIMIZAÇÃO CONJUNTA — cada um com seus hiperparâmetros")
    print(f"busca: {ESTRATEGIA}, {TRIALS} trials por condição | "
          f"final: {len(SEEDS_FINAIS)} sementes | {int(FRACAO*100)}% de treino | "
          f"teto de {EPOCAS} épocas", flush=True)

    destino = caminho_resultado("06_resumo.csv")
    resumo, feitas = carregar_feito(destino, ("cena", "condição"))
    if feitas:
        print(f"retomando: {len(feitas)} células já medidas em {destino.name}", flush=True)
    linhas = []
    for cena in CENAS:
        print(f"\n{'='*104}\n### {cena} — {ns['DATASETS'][cena]}\n{'='*104}", flush=True)
        print(f"{'condição':<12}{'val (busca)':>12}{'OA':>18}{'AA':>18}{'Kappa':>18}"
              f"{'lr':>8}{'drop':>6}{'batch':>7}{'PCA':>5}", flush=True)
        piso_oa = next((r["OA"] for r in resumo
                        if r["cena"] == cena and r["condição"] == "sem fusão"), None)
        for condicao in CONDICOES:
            if (cena, condicao) in feitas:
                pronto = next(r for r in resumo
                              if r["cena"] == cena and r["condição"] == condicao)
                print(f"{condicao:<12}{'(já medido)':>12}{pronto['OA']:>11.4f}"
                      f"{'':>7}{pronto['AA']:>11.4f}{'':>7}{pronto['Kappa']:>11.4f}", flush=True)
                continue
            inicio = time.perf_counter()
            config, melhor_val, _ = hpo_da_condicao(cena, condicao)
            valores, tempo_medio, epocas_medias = retreinar(config, SEEDS_FINAIS)
            if piso_oa is None:
                piso_oa = valores[:, 0].mean()

            print(f"{condicao:<12}{melhor_val:>12.4f}"
                  f"{valores[:,0].mean():>11.4f}±{valores[:,0].std():.4f}"
                  f"{valores[:,1].mean():>11.4f}±{valores[:,1].std():.4f}"
                  f"{valores[:,2].mean():>11.4f}±{valores[:,2].std():.4f}"
                  f"{config.learning_rate:>8.4f}{config.dropout:>6.2f}"
                  f"{config.batch_size:>7}{config.pca_components:>5}", flush=True)

            for indice, semente in enumerate(SEEDS_FINAIS):
                linhas.append({
                    "cena": cena, "condição": condicao, "seed": semente,
                    "OA": valores[indice, 0], "AA": valores[indice, 1],
                    "Kappa": valores[indice, 2],
                })
            resumo.append({
                "cena": cena, "condição": condicao,
                "val_busca": melhor_val,
                "OA": valores[:, 0].mean(), "OA_std": valores[:, 0].std(),
                "AA": valores[:, 1].mean(), "AA_std": valores[:, 1].std(),
                "Kappa": valores[:, 2].mean(), "Kappa_std": valores[:, 2].std(),
                "ganho_OA_pp": 100 * (valores[:, 0].mean() - piso_oa),
                "learning_rate": config.learning_rate, "dropout": config.dropout,
                "weight_decay": config.weight_decay, "batch_size": config.batch_size,
                "pca_components": config.pca_components,
                "overrides": str(config.segmenter_overrides),
                "treino_s": tempo_medio, "épocas": epocas_medias,
                "minutos_totais": (time.perf_counter() - inicio) / 60,
            })
            # Grava a cada célula concluída: uma interrupção perde no máximo uma.
            salvar(resumo, destino)

        sub = [r for r in resumo if r["cena"] == cena]
        ganhos = ", ".join(f"{r['condição']}={r['ganho_OA_pp']:+.2f}pp"
                           for r in sub if r["condição"] != "sem fusão")
        print(f"\n  ganho de OA sobre 'sem fusão' (ambos otimizados): {ganhos}", flush=True)

        if linhas:
            pd.DataFrame(linhas).to_csv(
                caminho_resultado(f"06_por_seed_{'_'.join(CENAS)}.csv"), index=False)

    print(f"\nCSVs em {caminho_resultado('').parent}", flush=True)


if __name__ == "__main__":
    main()
