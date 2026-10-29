"""Qual segmentador é melhor quando cada um recebe seus próprios hiperparâmetros.

CRITÉRIO — e este é o ponto do experimento. Comparar segmentadores com a CNN fixa
não mede o segmentador: mede o quanto aquele conjunto fixo por acaso serve para ele.
O contraexemplo está no experimento 05: com hiperparâmetros fixos, o SLIC em Pavia
deu Kappa de 0,0346. Não porque o SLIC seja ruim, mas porque ele produz 1.428
regiões, e um id escalar indo até 1.428 contra um espectro de desvio 1,0 exige um
learning rate e um dropout completamente diferentes.

Neste trabalho o segmentador é um eixo da otimização conjunta, não uma variável
isolada. A pergunta certa é: **qual é o melhor resultado que cada segmentador
alcança quando a CNN é ajustada para ele?**

PROTOCOLO, em duas etapas por condição:
  1. HPO restrito àquele segmentador (busca learning rate, dropout, weight decay,
     batch, componentes de PCA e os parâmetros internos do próprio segmentador).
     Seleção só por validação.
  2. O vencedor é retreinado com várias sementes, para ter barra de erro no número
     que vai ser reportado.

A REFERÊNCIA é o SEEDS, o segmentador do trabalho herdado, com o mesmo tratamento:
HPO e retreino. `ganho_OA_pp` é quanto trocar o segmentador acrescenta sobre a
escolha herdada, os dois otimizados. Uma condição "sem fusão" não entra: o trabalho
otimiza a arquitetura herdada, que já tem fusão, e não a compara com uma sem.

A estratégia de busca é fixada em TPE nas quatro condições: assim a comparação isola
o segmentador, e não mistura diferenças entre estratégias de busca.
"""

import gc
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from hsi.config import DATASETS, ExperimentConfig
from hsi.experimentos.infra import caminho_resultado, carregar_feito, salvar
from hsi.hpo.busca import automated_hpo
from hsi.pipeline import release_result, run_experiment

#: Orçamento de busca por condição, sementes do retreino, teto de épocas e fração.
#: "seeds" vem primeiro de propósito: é a referência de `ganho_OA_pp`, e os ganhos
#: só são calculados depois que ela está medida.
E06 = dict(trials=16, sementes=(11, 29, 42, 73, 101), epocas=200,
           fracao=0.10, estrategia="tpe",
           condicoes=("seeds", "slic", "lsc", "watershed"))


def _e06_hpo_da_condicao(cena: str, condicao: str):
    """Etapa 1: busca os melhores hiperparâmetros para esta condição."""
    base = ExperimentConfig(
        dataset=cena, model_type="cnn1d", two_input=True, segmenter=condicao,
        fusion="scalar", train_fraction=E06["fracao"],
        epochs=E06["epocas"], patience=10, early_stopping_target=0.95, seed=42,
    )
    tabela, _ = automated_hpo(
        base, trials_per_cnn=E06["trials"], search_epochs=E06["epocas"],
        backend="thread", model_types=("cnn1d",),
        segmenters=(base.segmenter,),          # busca restrita a este segmentador
        strategy=E06["estrategia"], retain="winner",
    )
    vencedor = tabela.loc[tabela["val_accuracy (objetivo)"].idxmax()]
    return vencedor["config"], float(vencedor["val_accuracy (objetivo)"]), tabela


def _e06_retreinar(config, sementes):
    """Etapa 2: repete a receita vencedora com várias sementes."""
    metricas, tempos, epocas = [], [], []
    for semente in sementes:
        resultado = run_experiment(
            replace(config, seed=int(semente)), backend="thread", verbose=False)
        metricas.append((resultado.metrics.overall_accuracy,
                         resultado.metrics.average_accuracy,
                         resultado.metrics.kappa))
        tempos.append(resultado.training.total_time)
        epocas.append(len(resultado.training.history))
        release_result(resultado)
        gc.collect()
    return np.array(metricas), float(np.mean(tempos)), float(np.mean(epocas))


def experimento_06(*cenas: str) -> list:
    """Cada segmentador com os hiperparâmetros buscados para ele."""
    cenas = cenas or ("IP", "PU", "SA")
    print("SEGMENTADORES SOB OTIMIZAÇÃO CONJUNTA — cada um com seus hiperparâmetros")
    print(f"busca: {E06['estrategia']}, {E06['trials']} trials por condição | "
          f"final: {len(E06['sementes'])} sementes | {int(E06['fracao']*100)}% de "
          f"treino | teto de {E06['epocas']} épocas", flush=True)

    destino = caminho_resultado("06_resumo.csv")
    resumo, feitas = carregar_feito(destino, ("cena", "condição"))
    if feitas:
        print(f"retomando: {len(feitas)} células já medidas em {destino.name}", flush=True)
    linhas = []
    for cena in cenas:
        print(f"\n{'='*104}\n### {cena} — {DATASETS[cena]}\n{'='*104}", flush=True)
        print(f"{'condição':<12}{'val (busca)':>12}{'OA':>18}{'AA':>18}{'Kappa':>18}"
              f"{'lr':>8}{'drop':>6}{'batch':>7}{'PCA':>5}", flush=True)
        for condicao in E06["condicoes"]:
            if (cena, condicao) in feitas:
                pronto = next(r for r in resumo
                              if r["cena"] == cena and r["condição"] == condicao)
                print(f"{condicao:<12}{'(já medido)':>12}{pronto['OA']:>11.4f}"
                      f"{'':>7}{pronto['AA']:>11.4f}{'':>7}{pronto['Kappa']:>11.4f}",
                      flush=True)
                continue
            inicio = time.perf_counter()
            config, melhor_val, _ = _e06_hpo_da_condicao(cena, condicao)
            valores, tempo_medio, epocas_medias = _e06_retreinar(config, E06["sementes"])

            print(f"{condicao:<12}{melhor_val:>12.4f}"
                  f"{valores[:,0].mean():>11.4f}±{valores[:,0].std():.4f}"
                  f"{valores[:,1].mean():>11.4f}±{valores[:,1].std():.4f}"
                  f"{valores[:,2].mean():>11.4f}±{valores[:,2].std():.4f}"
                  f"{config.learning_rate:>8.4f}{config.dropout:>6.2f}"
                  f"{config.batch_size:>7}{config.pca_components:>5}", flush=True)

            for indice, semente in enumerate(E06["sementes"]):
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
                "ganho_OA_pp": float("nan"),  # preenchido no fim da cena, contra o SEEDS
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
        referencia = next((r["OA"] for r in sub if r["condição"] == "seeds"), None)
        if referencia is not None:
            for r in sub:
                r["ganho_OA_pp"] = 100 * (r["OA"] - referencia)
            salvar(resumo, destino)
            ganhos = ", ".join(f"{r['condição']}={r['ganho_OA_pp']:+.2f}pp"
                               for r in sub if r["condição"] != "seeds")
            print(f"\n  ganho de OA sobre o SEEDS herdado (ambos otimizados): {ganhos}",
                  flush=True)

        if linhas:
            pd.DataFrame(linhas).to_csv(
                caminho_resultado(f"06_por_seed_{'_'.join(cenas)}.csv"), index=False)

    print(f"\nCSV: {destino}", flush=True)
    return resumo
