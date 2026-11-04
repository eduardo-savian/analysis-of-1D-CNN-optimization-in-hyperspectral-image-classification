"""Experimento 08: busca aleatória, TPE e algoritmo genético no mesmo espaço.

Cada estratégia roda 20 trials com tetos de 10, 40 e 200 épocas por trial, e cada
busca é repetida com 3 sementes. Todos os trials são gravados, não só os vencedores:
é deles que sai a validação média por valor de cada hiperparâmetro. 10% do treino.
"""

import pandas as pd

from hsi.config import ExperimentConfig
from hsi.experimentos.infra import caminho_resultado, carregar_feito, salvar
from hsi.hpo import STRATEGIES, automated_hpo

TRIALS = 20
ORCAMENTOS = (10, 40, 200)
SEMENTES = (11, 29, 42)
EIXOS = ("learning rate", "dropout", "weight decay", "batch", "PCA")


def experimento_08(*cenas: str) -> pd.DataFrame:
    """Estratégias de busca, orçamento de épocas e peso de cada eixo."""
    cenas = cenas or ("IP",)
    destino = caminho_resultado("08_trials.csv")
    todos, feitas = carregar_feito(destino, ("cena", "orçamento", "estratégia", "semente"))
    for cena in cenas:
        for orcamento in ORCAMENTOS:
            for estrategia in STRATEGIES:
                for semente in SEMENTES:
                    if (cena, orcamento, estrategia, semente) in feitas:
                        continue
                    base = ExperimentConfig(dataset=cena, train_fraction=0.10, epochs=orcamento,
                                            target=0.95, seed=semente)
                    # A semente governa a busca e o treino. Com população 6, os 20 trials
                    # do genético dão quatro gerações.
                    trials = automated_hpo(base, TRIALS, search_epochs=orcamento,
                                           strategy=estrategia, population=6)
                    trials = trials.drop(columns="config").assign(
                        cena=cena, orçamento=orcamento, semente=semente)
                    todos += trials.to_dict(orient="records")
                    salvar(todos, destino)
                    print(f"{cena} {orcamento:>3} épocas  {estrategia:<8} s={semente:<3} "
                          f"melhor val {trials['val_accuracy (objetivo)'].max():.4f}", flush=True)

    quadro = pd.DataFrame(todos)
    melhor = quadro.groupby(["cena", "orçamento", "estratégia", "semente"])[
        "val_accuracy (objetivo)"].max()
    print("\nmelhor validação por busca, média e desvio entre sementes:")
    print(melhor.groupby(["cena", "orçamento", "estratégia"]).agg(["mean", "std"]).round(4))
    print("\namplitude da validação média entre os valores de cada eixo (pp):")
    for eixo in EIXOS:
        medias = quadro.groupby(eixo)["val_accuracy (objetivo)"].mean()
        print(f"  {eixo:<14} {100 * (medias.max() - medias.min()):6.2f}")
    print(f"\nCSV: {destino}")
    return quadro
