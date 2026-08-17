"""Extração vetorizada de patches espaciais para as CNNs 2D e 3D.

CNN1D classifica apenas a assinatura espectral do pixel. As variantes 2D e 3D
precisam de uma vizinhança centrada no pixel rotulado. Este módulo extrai todas as
vizinhanças em ordem *row-major*, a mesma ordem de ``to_pixel_table`` e de
``remove_background``; assim não há risco de desalinhar amostra e rótulo.
"""

from __future__ import annotations

import numpy as np

__all__ = ["extract_patches", "spatial_patches", "spectral_spatial_patches"]


def extract_patches(cube: np.ndarray, labels: np.ndarray, patch_size: int) -> np.ndarray:
    """Extrai um patch por pixel não-background de um cubo ``(H, W, C)``.

    Args:
        cube: Imagem/cubo de entrada com canais na última dimensão.
        labels: Ground truth ``(H, W)`` ainda contendo background igual a zero.
        patch_size: Lado ímpar da janela quadrada centrada no pixel.

    Returns:
        Array ``(N, C, P, P)`` float32, em que ``N`` é o número de pixels
        rotulados e ``P`` é ``patch_size``.

    O padding ``reflect`` preserva contexto nas bordas sem introduzir faixas
    artificiais de zero. ``sliding_window_view`` cria uma visão antes da seleção;
    a cópia final é necessária para manter memória contígua para PyTorch.
    """
    if cube.ndim != 3:
        raise ValueError(f"cube deve ter shape (H, W, C), recebido {cube.shape}.")
    if labels.shape != cube.shape[:2]:
        raise ValueError("labels deve ter as mesmas dimensões espaciais de cube.")
    if patch_size < 3 or not patch_size % 2:
        raise ValueError("patch_size deve ser ímpar e maior ou igual a 3.")

    radius = patch_size // 2
    padded = np.pad(cube, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
    # (H, W, C, P, P); o filtro booleano preserva a mesma ordem dos labels achatados.
    windows = np.lib.stride_tricks.sliding_window_view(padded, (patch_size, patch_size), axis=(0, 1))
    selected = windows[labels > 0]
    return np.ascontiguousarray(selected, dtype=np.float32)


def spatial_patches(components: np.ndarray, scene_shape: tuple[int, int], labels: np.ndarray, patch_size: int) -> np.ndarray:
    """Converte PCA espacial ``(H*W, 3)`` em patches RGB-latentes para CNN2D."""
    height, width = scene_shape
    cube = components.reshape(height, width, components.shape[1])
    return extract_patches(cube, labels.reshape(height, width), patch_size)


def spectral_spatial_patches(image: np.ndarray, labels: np.ndarray, patch_size: int) -> np.ndarray:
    """Normaliza o cubo cru e extrai ``(N, 1, B, P, P)`` para CNN3D.

    A normalização global para [0, 1] evita que a escala inteira original dos
    sensores domine a otimização. O eixo de bandas vira a profundidade do
    ``Conv3d``; o canal de entrada é 1, como em uma sequência espectral única.
    """
    image = image.astype(np.float32, copy=False)
    maximum = float(image.max())
    normalized = image / maximum if maximum else image
    patches = extract_patches(normalized, labels, patch_size)
    return patches[:, np.newaxis, :, :, :]
