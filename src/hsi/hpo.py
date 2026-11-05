"""Busca de hiperparâmetros: o espaço, as duas estratégias do projeto (busca em grade e
otimização bayesiana) e o laço que treina cada trial.

Um candidato é uma tupla de índices, um por eixo (segmentador, configuração do
segmentador, taxa de aprendizado, dropout, weight decay, lote e componentes do PCA).
O trial vale pela maior acurácia de validação do seu histórico; o teste só é relatado.
"""

import itertools
import json
import warnings
from dataclasses import replace

import optuna
import pandas as pd

from hsi.config import CONFIG, PCA_COMPONENTS, SEGMENTERS
from hsi.modelo import MIN_COMPONENTS
from hsi.pipeline import run_experiment
from hsi.segmentadores import SEGMENTER_PARAMS

# O espaço, a grade e as variações vêm de [hpo] em config.toml.
_H = CONFIG["hpo"]
HPO_SPACE = {eixo: tuple(valores) for eixo, valores in _H["espaco"].items()}
# Os valores da configuração original em cada eixo, os padrões de [execucao].
ORIGINAL = {eixo: CONFIG["execucao"][eixo] for eixo in HPO_SPACE}
STRATEGIES = ("grid", "bayes")
GRID_AXES = tuple(_H["eixos_da_grade"])


def pca_options(dataset: str, cnn: str = "projeto") -> list[int]:
    # Os fatores de [hpo] sobre o valor da cena, sem descer do mínimo que a CNN aceita.
    base = PCA_COMPONENTS[dataset]
    return sorted({max(MIN_COMPONENTS[cnn], int(round(base * f))) for f in _H["fatores_pca"]})


def segmenter_options(segmenter: str, dataset: str) -> list[dict]:
    """Três configurações por segmentador: a padrão da cena no meio e uma de cada lado.

    LSC e Watershed usam as três listas de [segmentadores]; o SEEDS, as variações de
    [hpo.seeds] em torno do padrão da cena."""
    if segmenter == "seeds":
        p, v = SEGMENTER_PARAMS["seeds"][dataset], _H["seeds"]
        menor, maior = [{"num_superpixels": max(v["minimo_superpixels"], int(p["num_superpixels"] * f)),
                         "num_iterations": it, "prior": prior}
                        for f, it, prior in zip(v["fator_superpixels"], v["num_iterations"], v["prior"])]
        padrao = {k: p[k] for k in ("num_superpixels", "num_iterations", "prior")}
        return [menor, padrao, maior]
    s = CONFIG["segmentadores"]
    if segmenter == "lsc":
        return [{"region_size": lado, "ratio": razao}
                for lado, razao in zip(s["lsc"]["region_size"][dataset], s["lsc"]["ratio"])]
    return [{"markers": m} for m in s["watershed"]["markers"][dataset]]


def genes(dataset: str, segmenters=SEGMENTERS, cnn: str = "projeto") -> list[tuple[str, tuple]]:
    # O segmentador precisa ser o eixo 0: é a posição que a cobertura inicial fixa.
    return ([("segmenter", tuple(segmenters)), ("override_index", (0, 1, 2))]
            + list(HPO_SPACE.items()) + [("pca_components", tuple(pca_options(dataset, cnn)))])


def decode(lista_genes, candidato, dataset) -> dict:
    """Índices -> campos de ExperimentConfig."""
    valores = {nome: opcoes[i % len(opcoes)] for (nome, opcoes), i in zip(lista_genes, candidato)}
    segmentador = valores.pop("segmenter")
    opcoes = segmenter_options(segmentador, dataset)
    escolhida = opcoes[valores.pop("override_index") % len(opcoes)]
    return {"segmenter": segmentador, "segmenter_overrides": dict(escolhida), **valores}


# ------------------------------------------------------------------ estratégias
# Cada estratégia chama `avaliar(candidato)`, que treina e devolve a validação.

def _indice_original(lista_genes, dataset) -> list[int]:
    """O candidato da configuração original: configuração padrão do segmentador, taxa
    1e-3, dropout 0,5, wd 0, lote 32 e o PCA padrão da cena."""
    original = {**ORIGINAL, "override_index": 1, "pca_components": PCA_COMPONENTS[dataset]}
    return [0 if nome == "segmenter" else opcoes.index(original[nome])
            for nome, opcoes in lista_genes]


def grid_search(lista_genes, dataset, seed, avaliar):
    """Busca em grade sobre o segmentador e os eixos de GRID_AXES, com os demais eixos
    no valor da configuração original. Não usa a semente: a grade é sempre a mesma."""
    posicao = {nome: g for g, (nome, _) in enumerate(lista_genes)}
    eixos = ["segmenter", *GRID_AXES]
    base = _indice_original(lista_genes, dataset)
    for indices in itertools.product(*(range(len(lista_genes[posicao[e]][1])) for e in eixos)):
        candidato = list(base)
        for eixo, i in zip(eixos, indices):
            candidato[posicao[eixo]] = i
        avaliar(tuple(candidato))


def bayesian_search(lista_genes, n_trials, seed, avaliar):
    """Otimização bayesiana com processo gaussiano (GPSampler do Optuna). Segmentador e
    configuração do segmentador são categóricos; os demais eixos são inteiros ordenados,
    porque neles o índice seguinte é de fato um valor maior."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # O GPSampler ainda é marcado como experimental no Optuna 4.
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    cobertura = range(len(lista_genes[0][1]))
    # Os primeiros trials são sorteados (um por segmentador, depois aleatórios) e só
    # então o processo gaussiano passa a escolher; o padrão do Optuna seria 10.
    estudo = optuna.create_study(direction="maximize", sampler=optuna.samplers.GPSampler(
        seed=int(seed), n_startup_trials=max(_H["bayes_trials_iniciais"], len(cobertura))))
    for i in cobertura:
        estudo.enqueue_trial({"segmenter": int(i)}, skip_if_exists=False)
    # O processo gaussiano às vezes propõe um candidato já treinado. Ele recebe o valor
    # medido e não conta no orçamento; o limite de propostas evita um laço sem fim.
    medidos = {}
    for _ in range(8 * n_trials):
        if len(medidos) == n_trials:
            break
        trial = estudo.ask()
        candidato = []
        for nome, opcoes in lista_genes:
            if nome in ("segmenter", "override_index"):
                candidato.append(int(trial.suggest_categorical(nome, list(range(len(opcoes))))))
            else:
                candidato.append(int(trial.suggest_int(nome, 0, len(opcoes) - 1)))
        candidato = tuple(candidato)
        if candidato not in medidos:
            medidos[candidato] = float(avaliar(candidato))
        estudo.tell(trial, medidos[candidato])


# ------------------------------------------------------------------------ busca

def automated_hpo(base, n_trials, search_epochs, strategy="bayes", segmenters=SEGMENTERS):
    """Executa a busca e devolve a tabela de trials, uma linha por trial.

    Na grade, `n_trials` não vale: o número de trials é o tamanho da grade.

    A coluna `config` guarda a configuração de cada trial; o vencedor é a linha de
    maior `val_accuracy (objetivo)` (no empate, o trial mais antigo).
    """
    segmenters = tuple(segmenters)
    lista_genes = genes(base.dataset, segmenters, base.cnn)
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

    if strategy == "grid":
        grid_search(lista_genes, base.dataset, base.seed, avaliar)
    else:
        bayesian_search(lista_genes, max(n_trials, len(segmenters)), base.seed, avaliar)
    return pd.DataFrame(linhas)
