"""Gráficos: curvas de treino, matriz de confusão e etapas da segmentação.

Todas as funções recebem os dados prontos e devolvem a figura — nada de ler
variável global nem de chamar `plt.show()` escondido dentro do cálculo, como no
notebook original.

Reproduzem as figuras do original, inclusive os títulos em inglês. Nenhuma célula
de execução as chama: os painéis em uso são os dos diagnósticos visuais (seção
3.3) e os dos dashboards.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _grid(images: dict[str, np.ndarray], cmaps=None, figsize=(20, 10), columns: int = 4):
    cmaps = cmaps or {}
    fig = plt.figure(figsize=figsize)
    for i, (title, image) in enumerate(images.items(), start=1):
        axis = fig.add_subplot(2, columns, i)
        axis.imshow(image, cmap=cmaps.get(title))
        axis.axis("off")
        axis.set_title(title)
    return fig


def plot_history(history, metric: str, save_to: Path | None = None):
    """Curva de treino x validação para 'Loss' ou 'Accuracy'."""
    keys = {"Loss": ("loss", "val_loss"), "Accuracy": ("accuracy", "val_accuracy")}
    if metric not in keys:
        raise ValueError(f"metric deve ser um de {sorted(keys)}, recebido {metric!r}")
    train_key, val_key = keys[metric]
    values = history.to_dict()

    fig = plt.figure(figsize=(10, 7))
    plt.plot(history.epochs, values[train_key], "g", label="Training")
    plt.plot(history.epochs, values[val_key], "b", label="Validation")
    plt.title(f"Training and Validation {metric}")
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(loc="upper center")
    plt.grid()
    if save_to is not None:
        plt.savefig(save_to)
    return fig


def plot_confusion(confusion: np.ndarray, normalize: bool = False, save_to: Path | None = None):
    """Matriz de confusão; `normalize=True` mostra o percentual por linha."""
    import pandas as pd
    import seaborn as sns

    matrix, fmt = confusion, "d"
    if normalize:
        matrix = 100 * confusion.astype(float) / confusion.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"

    fig = plt.figure(figsize=(10, 7))
    sns.heatmap(pd.DataFrame(matrix), annot=True, fmt=fmt, cmap="Blues")
    plt.xlabel("Previsão")
    plt.ylabel("Ground Truth")
    if save_to is not None:
        plt.savefig(save_to)
    return fig


def plot_pca_components(components: np.ndarray, shape: tuple[int, int], count: int = 3):
    """As primeiras componentes principais como imagens."""
    height, width = shape
    return _grid({
        f"Band - {i}": components[:, i].reshape(height, width) for i in range(count)
    }, cmaps={f"Band - {i}": "nipy_spectral" for i in range(count)})


def plot_rgb_channels(rgb: np.ndarray):
    """Os três canais da composição RGB, cada um em seu próprio degradê."""
    def ramp(color):
        return plt.cm.colors.ListedColormap(np.linspace([0, 0, 0], color, 256))

    channels = {"Red": (rgb[:, :, 0], [1, 0, 0]), "Green": (rgb[:, :, 1], [0, 1, 0]), "Blue": (rgb[:, :, 2], [0, 0, 1])}
    return _grid(
        {name: image for name, (image, _) in channels.items()},
        cmaps={name: ramp(color) for name, (_, color) in channels.items()},
    )


def plot_segmentation_steps(steps: dict[str, np.ndarray]):
    """Máscara de contorno, fundo, frente e composição final da segmentação.

    Recebe as imagens de `contour_overlay`, que servem a qualquer segmentador; no
    original só havia o SEEDS.
    """
    return _grid(steps)
