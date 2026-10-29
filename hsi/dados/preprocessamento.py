"""Redução de dimensionalidade (PCA) e composição RGB para a segmentação."""

import numpy as np
from sklearn.decomposition import PCA


def apply_pca(x: np.ndarray, n_components: int, random_state: int = 42, whiten: bool = True):
    """Reduz o número de bandas mantendo as componentes de maior variância.

    `whiten=True` e `random_state` fixo reproduzem o notebook original. Hoje o
    `random_state` não muda nada: com o sklearn fixado e estes shapes (muito mais
    pixels que bandas), `svd_solver="auto"` escolhe `covariance_eigh`, que é exato —
    medido nas três cenas, sementes diferentes dão a mesma projeção. Ele só voltaria
    a importar se o solver randomizado entrasse em jogo.

    Retorna `(X_transformado, modelo_ajustado)`.
    """
    model = PCA(n_components=n_components, whiten=whiten, random_state=random_state)
    model.fit(x)
    return model.transform(x), model


def mat2gray(a: np.ndarray) -> np.ndarray:
    """Normaliza para [0, 1] (equivalente do `mat2gray` do MATLAB)."""
    lo, hi = np.min(a), np.max(a)
    return (a - lo) / (hi - lo)


def rgb_composite(components: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Três primeiras componentes principais -> imagem RGB uint8 (H, W, 3).

    É essa imagem que alimenta os quatro segmentadores: eles trabalham sobre uma
    imagem de 3 canais, não sobre o cubo inteiro.
    """
    if components.shape[1] < 3:
        raise ValueError(f"São necessárias 3 componentes, recebidas {components.shape[1]}.")
    height, width = shape
    channels = [(255 * mat2gray(components[:, i])).astype(np.uint8).reshape(height, width) for i in range(3)]
    return np.dstack(channels)
