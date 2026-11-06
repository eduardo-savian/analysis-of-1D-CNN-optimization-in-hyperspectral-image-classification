import pandas as pd

from hsi.config import CONFIG, ExperimentConfig
from hsi.experimentos.infra import caminho_resultado, carregar_feito, salvar
from hsi.hpo import STRATEGIES, automated_hpo

C = CONFIG["experimentos"]["04"]
TRIALS_BUSCA = C["trials"]
ORCAMENTOS = tuple(C["orcamentos_de_epocas"])
SEMENTES_BUSCA = tuple(C["sementes"])
EIXOS = ("learning rate", "dropout", "weight decay", "batch", "PCA")


def experimento_04(*cenas: str) -> pd.DataFrame:
    """Estratégias de busca, orçamento de épocas e peso de cada eixo."""
    cenas = cenas or tuple(C["cenas"])
    destino = caminho_resultado("04_trials.csv")
    todos, feitas = carregar_feito(destino, ("cena", "orçamento", "estratégia", "semente"))
    for cena in cenas:
        for orcamento in ORCAMENTOS:
            for estrategia in STRATEGIES:
                for semente in SEMENTES_BUSCA:
                    if (cena, orcamento, estrategia, semente) in feitas:
                        continue
                    base = ExperimentConfig(dataset=cena, train_fraction=C["train_fraction"],
                                            epochs=orcamento, target=C["target"], seed=semente)
                    # A semente governa a busca e o treino.
                    trials = automated_hpo(base, TRIALS_BUSCA, search_epochs=orcamento,
                                           strategy=estrategia)
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
