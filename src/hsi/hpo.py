"""Busca de hiperparâmetros: o espaço, as três estratégias e o laço que treina cada trial.

Um candidato é uma tupla de índices, um por gene (segmentador, configuração do
segmentador, taxa de aprendizado, dropout, weight decay, lote e componentes do PCA).
As três estratégias trabalham sobre essa mesma representação e começam com a mesma
cobertura: os primeiros trials usam segmentadores distintos. O trial vale pela maior
acurácia de validação do seu histórico; o teste só é relatado.
"""

import json
from dataclasses import replace

import numpy as np
import optuna
import pandas as pd

from hsi.config import PCA_COMPONENTS, SEGMENTERS
from hsi.dados import SEGMENTER_PARAMS
from hsi.modelo import MIN_COMPONENTS
from hsi.pipeline import run_experiment

# Cada eixo inclui o valor da configuração herdada (taxa 1e-3, dropout 0,5, wd 0, lote 32).
HPO_SPACE = {
    "learning_rate": (3e-4, 1e-3, 3e-3),
    "dropout": (0.2, 0.35, 0.5),
    "weight_decay": (0.0, 1e-5, 1e-4),
    "batch_size": (32, 64, 128),
}
STRATEGIES = ("random", "tpe", "genetic")


def pca_options(dataset: str) -> list[int]:
    # 0,5x, 1x e 1,5x o valor da cena, sem descer do mínimo que a CNN aceita. Em
    # University of Pavia e Salinas Valley o 0,5x vira 15 e sobram duas opções.
    base = PCA_COMPONENTS[dataset]
    return sorted({max(MIN_COMPONENTS, int(round(base * f))) for f in (0.5, 1.0, 1.5)})


def segmenter_options(segmenter: str, dataset: str) -> list[dict]:
    """Três configurações por segmentador: a padrão da cena no meio e uma de cada lado."""
    p = SEGMENTER_PARAMS[segmenter][dataset]
    if segmenter == "seeds":
        return [{"num_superpixels": max(40, int(p["num_superpixels"] * f)),
                 "num_iterations": it, "prior": prior}
                for f, it, prior in ((.70, 120, 1), (1.0, p["num_iterations"], p["prior"]),
                                     (1.35, 250, 3))]
    if segmenter == "slic":
        return [{"region_size": max(3, p["region_size"] - 3), "ruler": 5.0},
                {"region_size": p["region_size"], "ruler": p["ruler"]},
                {"region_size": p["region_size"] + 3, "ruler": 20.0}]
    if segmenter == "lsc":
        return [{"region_size": max(3, p["region_size"] - 3), "ratio": .04},
                {"region_size": p["region_size"], "ratio": p["ratio"]},
                {"region_size": p["region_size"] + 3, "ratio": .12}]
    return [{"markers": max(20, int(p["markers"] * f))} for f in (.70, 1.0, 1.35)]


def genes(dataset: str, segmenters=SEGMENTERS) -> list[tuple[str, tuple]]:
    # O segmentador precisa ser o gene 0: é a posição que a cobertura inicial fixa.
    return ([("segmenter", tuple(segmenters)), ("override_index", (0, 1, 2))]
            + list(HPO_SPACE.items()) + [("pca_components", tuple(pca_options(dataset)))])


def decode(lista_genes, candidato, dataset) -> dict:
    """Índices -> campos de ExperimentConfig."""
    valores = {nome: opcoes[i % len(opcoes)] for (nome, opcoes), i in zip(lista_genes, candidato)}
    segmentador = valores.pop("segmenter")
    opcoes = segmenter_options(segmentador, dataset)
    escolhida = opcoes[valores.pop("override_index") % len(opcoes)]
    return {"segmenter": segmentador, "segmenter_overrides": dict(escolhida), **valores}


# ------------------------------------------------------------------ estratégias
# Cada estratégia chama `avaliar(candidato)`, que treina e devolve a validação, até
# gastar `n_trials`.

def random_search(lista_genes, n_trials, seed, avaliar):
    """Sorteio uniforme sem repetir candidatos; não usa os resultados anteriores."""
    rng = np.random.default_rng(seed)

    def sortear():
        return tuple(int(rng.integers(len(op))) for _, op in lista_genes)

    cobertura = list(range(len(lista_genes[0][1])))
    vistos = set()
    for _ in range(n_trials):
        forcado = cobertura.pop(0) if cobertura else None
        sorteado = sortear()
        # Até 32 novos sorteios para fugir de um candidato já visto.
        for _ in range(32):
            candidato = sorteado if forcado is None else (forcado,) + sorteado[1:]
            if candidato not in vistos:
                break
            sorteado = sortear()
        vistos.add(candidato)
        avaliar(candidato)


def tpe_search(lista_genes, n_trials, seed, avaliar):
    """TPE do Optuna. Segmentador e configuração do segmentador são categóricos; os demais eixos são
    inteiros ordenados, porque neles o índice seguinte é de fato um valor maior."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cobertura = range(len(lista_genes[0][1]))
    # Sem isso o Optuna sortearia 10 trials antes de começar a modelar.
    estudo = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(
        seed=int(seed), n_startup_trials=max(4, len(cobertura))))
    for i in cobertura:
        estudo.enqueue_trial({"segmenter": int(i)}, skip_if_exists=False)
    for _ in range(n_trials):
        trial = estudo.ask()
        candidato = []
        for nome, opcoes in lista_genes:
            if nome in ("segmenter", "override_index"):
                candidato.append(int(trial.suggest_categorical(nome, list(range(len(opcoes))))))
            else:
                candidato.append(int(trial.suggest_int(nome, 0, len(opcoes) - 1)))
        estudo.tell(trial, float(avaliar(tuple(candidato))))


def genetic_search(lista_genes, n_trials, seed, avaliar, population=8, elite=2,
                   mutation=0.25, tournament=3):
    """Algoritmo genético: torneio, cruzamento uniforme, mutação por gene e elitismo.

    A primeira geração é a cobertura completada por sorteios. Os `elite` melhores
    passam à geração seguinte sem treinar de novo, então cada geração nova custa
    `population - elite` trials.
    """
    rng = np.random.default_rng(seed)

    def sortear():
        return tuple(int(rng.integers(len(op))) for _, op in lista_genes)

    fila = [(i,) + sortear()[1:] for i in range(len(lista_genes[0][1]))]
    while len(fila) < population:
        fila.append(sortear())
    geracao = []          # (validação, candidato) da geração atual
    for _ in range(n_trials):
        if not fila:
            # sorted é estável: no empate, fica na frente quem foi avaliado antes.
            anterior = sorted(geracao, key=lambda par: par[0], reverse=True)
            geracao = anterior[:elite]
            while len(fila) < population - elite:
                pais = [max((anterior[int(rng.integers(len(anterior)))]
                             for _ in range(tournament)), key=lambda par: par[0])[1]
                        for _ in range(2)]
                filho = tuple(pais[0][g] if rng.random() < .5 else pais[1][g]
                              for g in range(len(lista_genes)))
                filho = tuple(int(rng.integers(len(op))) if rng.random() < mutation else gene
                              for gene, (_, op) in zip(filho, lista_genes))
                fila.append(filho)
        candidato = fila.pop(0)
        geracao.append((float(avaliar(candidato)), candidato))


SEARCHERS = {"random": random_search, "tpe": tpe_search, "genetic": genetic_search}


# ------------------------------------------------------------------------ busca

def automated_hpo(base, n_trials, search_epochs, strategy="random", segmenters=SEGMENTERS,
                  population=8):
    """Executa a busca e devolve a tabela de trials, uma linha por trial.

    A coluna `config` guarda a configuração de cada trial; o vencedor é a linha de
    maior `val_accuracy (objetivo)` (no empate, o trial mais antigo).
    """
    segmenters = tuple(segmenters)
    lista_genes = genes(base.dataset, segmenters)
    linhas = []

    def avaliar(candidato):
        config = replace(base, **decode(lista_genes, candidato, base.dataset))
        # Com poucas épocas por trial a paciência 10 nunca dispararia: ela cai para um
        # terço do orçamento (3 com 10 épocas; com 40 ou 200 continua 10).
        config = replace(config, epochs=search_epochs,
                         patience=min(config.patience, max(2, search_epochs // 3)))
        r = run_experiment(config)
        validacao = float(max(r["historico"]["val_accuracy"]))
        melhor = max([validacao] + [l["val_accuracy (objetivo)"] for l in linhas])
        linhas.append({
            "estratégia": strategy, "trial": len(linhas) + 1,
            "segmentador": config.segmenter,
            "parâmetros segmentador": json.dumps(config.segmenter_overrides, sort_keys=True),
            "learning rate": config.learning_rate, "dropout": config.dropout,
            "weight decay": config.weight_decay, "batch": config.batch_size,
            "PCA": config.pca_components,
            "val_accuracy (objetivo)": validacao,
            "OA teste (relato)": r["OA"], "AA teste (relato)": r["AA"],
            "Kappa teste (relato)": r["Kappa"],
            "tempo treino (s)": r["tempo_treino"], "épocas executadas": r["epocas"],
            "melhor até agora": melhor, "config": config,
        })
        return validacao

    n_trials = max(n_trials, len(segmenters))
    if strategy == "genetic":
        genetic_search(lista_genes, n_trials, base.seed, avaliar, population=population)
    else:
        SEARCHERS[strategy](lista_genes, n_trials, base.seed, avaliar)
    return pd.DataFrame(linhas)
