"""Gráficos do HPO: convergência das estratégias, espaço buscado e vencedores."""

import matplotlib.pyplot as plt

from hsi.config import SEGMENTERS
from hsi.hpo.estrategias import HPO_STRATEGIES


def plot_hpo_strategies(comparison):
    """Convergência, validação por trial e onde cada estratégia gastou o orçamento."""
    import seaborn as sns

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2), constrained_layout=True)
    colors = dict(zip(HPO_STRATEGIES, ("#4C78A8", "#F58518", "#54A24B")))
    for (strategy, cnn), group in comparison.groupby(["estratégia", "CNN"]):
        ordered = group.sort_values("trial")
        axes[0].plot(ordered["trial"], ordered["melhor até agora"], marker="o", ms=4,
                     color=colors.get(strategy), label=f"{strategy} · {cnn}")
        axes[1].scatter(ordered["trial"], ordered["val_accuracy (objetivo)"],
                        color=colors.get(strategy), s=45, alpha=.75, label=f"{strategy} · {cnn}")
    axes[0].set(title="Convergência: melhor validação até o trial n", xlabel="Trial",
                ylabel="melhor val_accuracy")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=.25)
    axes[1].set(title="Validação de cada trial individual", xlabel="Trial", ylabel="val_accuracy")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=.25)
    usage = comparison.groupby(["estratégia", "segmentador"]).size().reset_index(name="trials")
    sns.barplot(data=usage, x="estratégia", y="trials", hue="segmentador", ax=axes[2])
    axes[2].set(title="Onde cada estratégia gastou o orçamento", ylabel="trials")
    axes[2].grid(axis="y", alpha=.25)
    fig.suptitle("Random search × TPE (bayesiano) × algoritmo genético — mesmo espaço, mesmo orçamento",
                 fontsize=13)
    return fig


#: Colunas da tabela mostradas por `plot_hpo_space`, uma por painel. O segundo campo
#: (escala do eixo) não é usado: o stripplot trata cada valor como categoria.
HPO_PLOT_AXES = (
    ("learning rate", "log"), ("dropout", "linear"), ("weight decay", "symlog"),
    ("batch", "linear"), ("PCA", "linear"),
)


def plot_hpo_space(hpo_table):
    """Mostra a validação obtida em cada valor de cada eixo do espaço de busca.

    É o gráfico que responde "o que a busca de fato variou e o que importou":
    um eixo cujo valor não muda a validação é candidato a sair do espaço na
    próxima campanha, liberando orçamento para os eixos que importam.
    """
    import seaborn as sns

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
    # Cinco eixos em seis painéis: o que sobra fica vazio.
    for axis in axes.ravel()[len(HPO_PLOT_AXES):]:
        axis.set_axis_off()
    for axis, (column, scale) in zip(axes.ravel(), HPO_PLOT_AXES):
        if column not in hpo_table or hpo_table[column].nunique() < 1:
            axis.set_axis_off()
            continue
        sns.stripplot(data=hpo_table, x=column, y="val_accuracy (objetivo)", hue="CNN",
                      dodge=True, jitter=.18, s=8, ax=axis)
        axis.set(title=f"Validação × {column}", ylim=(0, 1), ylabel="melhor val_accuracy")
        axis.grid(axis="y", alpha=.25)
        axis.legend_.remove() if axis.legend_ else None
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="CNN", loc="outside right upper")
    strategies = hpo_table["estratégia"].unique() if "estratégia" in hpo_table else ["random"]
    fig.suptitle(f"Espaço de busca ({', '.join(strategies)}) — um painel por eixo", fontsize=13)
    return fig


def plot_hpo_by_cnn(hpo_table, model_type):
    """Mostra candidatos, segmentadores e score de validação de uma única CNN."""
    subset = hpo_table[hpo_table["CNN"] == model_type.upper()].copy().sort_values("trial")
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5), constrained_layout=True)
    colors = {name: color for name, color in zip(SEGMENTERS, plt.cm.tab10.colors)}
    axes[0].bar(subset["trial"], subset["val_accuracy (objetivo)"], color=[colors[name] for name in subset["segmentador"]])
    for _, row in subset.iterrows():
        marker = "★" if row["vencedor"] else ""
        axes[0].text(row["trial"], row["val_accuracy (objetivo)"], marker, ha="center", va="bottom", fontsize=14)
    axes[0].set(title=f"HPO — {model_type.upper()} (seleção por validação)", xlabel="Trial", ylabel="Melhor val_accuracy", ylim=(0, 1))
    axes[0].grid(axis="y", alpha=.25)
    for segmenter, group in subset.groupby("segmentador"):
        axes[1].scatter(group["tempo treino (s)"], group["val_accuracy (objetivo)"],
                        label=segmenter, s=100, color=colors[segmenter], edgecolor="black")
    axes[1].set(title="Qualidade de validação × custo", xlabel="Tempo de treino (s)", ylabel="Melhor val_accuracy", ylim=(0, 1))
    axes[1].legend(title="Segmentador"); axes[1].grid(alpha=.25)
    return fig, subset.drop(columns="resultado", errors="ignore")


def plot_hpo_compilation(hpo_table):
    """Compila busca, segmentadores e vencedores do HPO de cada CNN numa única figura."""
    summary = hpo_table.groupby("CNN", as_index=False).apply(
        lambda group: group.loc[group["val_accuracy (objetivo)"].idxmax()], include_groups=False
    ).reset_index(drop=True)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), constrained_layout=True)
    for cnn, group in hpo_table.groupby("CNN"):
        axes[0].scatter(group["trial"], group["val_accuracy (objetivo)"], label=cnn, s=85)
    axes[0].set(title="Busca por arquitetura", xlabel="Trial", ylabel="val_accuracy", ylim=(0, 1))
    axes[0].legend(); axes[0].grid(alpha=.25)
    pivot = hpo_table.pivot_table(index="CNN", columns="segmentador", values="val_accuracy (objetivo)", aggfunc="max")
    pivot.plot.bar(ax=axes[1], ylim=(0, 1), rot=0, title="Melhor validação por segmentador")
    axes[1].set_ylabel("val_accuracy"); axes[1].grid(axis="y", alpha=.25)
    # Barras lado a lado: sobrepostas, a mais baixa some dentro da mais alta.
    positions = range(len(summary))
    axes[2].bar([p - .2 for p in positions], summary["OA teste (relato)"], width=.4, label="OA teste")
    axes[2].bar([p + .2 for p in positions], summary["AA teste (relato)"], width=.4, label="AA teste")
    axes[2].set_xticks(list(positions), summary["CNN"])
    axes[2].set(title="Vencedores HPO — relatório de teste", ylabel="Pontuação", ylim=(0, 1))
    axes[2].legend(); axes[2].grid(axis="y", alpha=.25)
    return fig, summary
