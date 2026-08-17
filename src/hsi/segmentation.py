"""Segmentação em superpixels com SEEDS, SLIC e LSC (OpenCV contrib).

O id do superpixel de cada pixel vira a *segunda entrada* da rede: além do
espectro, o modelo recebe a que região homogênea aquele pixel pertence.

Requer `opencv-contrib-python[-headless]` — `cv.ximgproc` não existe no pacote
`opencv-python` comum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "SeedsParams", "SlicParams", "LscParams", "SEEDS_PARAMS", "SLIC_PARAMS", "LSC_PARAMS",
    "SEGMENTERS", "SegmentationResult", "SegmentationArtifacts", "segment_superpixels",
    "superpixel_labels", "contour_overlay", "colorize_segments",
]

SEGMENTERS = ("seeds", "slic", "lsc")


@dataclass(frozen=True)
class SeedsParams:
    """Parâmetros do `cv.ximgproc.createSuperpixelSEEDS`.

    num_superpixels é um alvo, não uma garantia: o algoritmo pode devolver menos.
    """

    num_iterations: int = 150
    prior: int = 1  # suavização de forma 3x3 quando > 0; válido em [0, 5]
    double_step: bool = True  # itera cada nível de bloco duas vezes (mais preciso)
    num_superpixels: int = 150
    num_levels: int = 8
    num_histogram_bins: int = 15


# Indian Pines tem 145x145 e classes agrícolas em faixas estreitas: precisa de mais
# superpixels e mais iterações. PaviaU e Salinas usam o mesmo conjunto.
SEEDS_PARAMS = {
    "IP": SeedsParams(
        num_iterations=200, prior=2, double_step=True,
        num_superpixels=350, num_levels=10, num_histogram_bins=5,
    ),
    "PU": SeedsParams(),
    "SA": SeedsParams(),
}


@dataclass(frozen=True)
class SlicParams:
    """Parâmetros do SLICO do OpenCV.

    ``region_size`` é o lado aproximado de cada superpixel. SLICO adapta a
    compacidade automaticamente e, por isso, é uma boa comparação com SEEDS sem
    introduzir mais um hiperparâmetro de cor/forma.
    """

    region_size: int = 12
    ruler: float = 10.0
    num_iterations: int = 10


@dataclass(frozen=True)
class LscParams:
    """Parâmetros do Linear Spectral Clustering (LSC) do OpenCV."""

    region_size: int = 12
    ratio: float = 0.075
    num_iterations: int = 10


SLIC_PARAMS = {"IP": SlicParams(region_size=8), "PU": SlicParams(), "SA": SlicParams()}
LSC_PARAMS = {"IP": LscParams(region_size=8), "PU": LscParams(), "SA": LscParams()}


@dataclass(frozen=True)
class SegmentationResult:
    """Saída da segmentação, em arrays puros.

    Guardar a máscara de contorno aqui — em vez do objeto SEEDS do OpenCV, que
    não é serializável — é o que permite rodar a segmentação num processo
    separado e trazer o resultado de volta (ver `hsi.parallel`).
    """

    labels: np.ndarray  # (H, W) int32 — id do superpixel de cada pixel
    contour_mask: np.ndarray  # (H, W) uint8 — bordas entre superpixels

    @property
    def n_superpixels(self) -> int:
        return len(np.unique(self.labels))


@dataclass(frozen=True)
class SegmentationArtifacts:
    """Tudo o que o ramo espacial produz, incluindo os intermediários visuais.

    Só os gráficos precisam de `components` e `rgb`; o pipeline em si usa apenas os
    ids de superpixel. Todos são arrays puros, então a estrutura atravessa a
    fronteira de um processo sem problema.
    """

    components: np.ndarray  # (H*W, 3) — as 3 primeiras componentes principais
    rgb: np.ndarray  # (H, W, 3) uint8 — entrada do SEEDS
    result: "SegmentationResult"

    @property
    def labels(self) -> np.ndarray:
        return self.result.labels

    @property
    def contour_mask(self) -> np.ndarray:
        return self.result.contour_mask

    @property
    def n_superpixels(self) -> int:
        return self.result.n_superpixels


def segment_superpixels(
    rgb: np.ndarray,
    algorithm: str,
    params: SeedsParams | SlicParams | LscParams,
) -> SegmentationResult:
    """Segmenta RGB com o algoritmo escolhido e devolve labels/contornos puros.

    Args:
        rgb: Composição uint8 ``(H, W, 3)`` construída das componentes de PCA.
        algorithm: ``seeds``, ``slic`` (SLICO) ou ``lsc``.
        params: Dataclass correspondente ao algoritmo e à cena.

    Todas as variantes são superpixels do ``cv.ximgproc``. A saída uniforme
    permite comparar algoritmos sem alterar DataLoader, modelo ou métricas.
    """
    import cv2 as cv

    if not hasattr(cv, "ximgproc"):
        raise ImportError(
            "cv.ximgproc ausente: instale opencv-contrib-python-headless "
            "(o pacote opencv-python comum não traz o SEEDS)."
        )
    if algorithm == "seeds":
        if not isinstance(params, SeedsParams):
            raise TypeError("SEEDS requer SeedsParams.")
        height, width, channels = rgb.shape
        segmenter = cv.ximgproc.createSuperpixelSEEDS(
            width, height, channels,
            params.num_superpixels, params.num_levels, params.prior,
            params.num_histogram_bins, params.double_step,
        )
    elif algorithm == "slic":
        if not isinstance(params, SlicParams):
            raise TypeError("SLIC requer SlicParams.")
        segmenter = cv.ximgproc.createSuperpixelSLIC(
            rgb, algorithm=cv.ximgproc.SLICO, region_size=params.region_size, ruler=params.ruler,
        )
    elif algorithm == "lsc":
        if not isinstance(params, LscParams):
            raise TypeError("LSC requer LscParams.")
        segmenter = cv.ximgproc.createSuperpixelLSC(rgb, region_size=params.region_size, ratio=params.ratio)
    else:
        raise ValueError(f"segmentador inválido: {algorithm!r}. Use um de {SEGMENTERS}.")

    # A API do SEEDS recebe a imagem em cada ``iterate``; SLIC/LSC recebem a
    # imagem no construtor e aceitam somente o número de iterações.
    if algorithm == "seeds":
        segmenter.iterate(rgb, params.num_iterations)
    else:
        segmenter.iterate(params.num_iterations)
    return SegmentationResult(
        labels=segmenter.getLabels(), contour_mask=segmenter.getLabelContourMask(False),
    )


def superpixel_labels(rgb: np.ndarray, params: SeedsParams) -> SegmentationResult:
    """Compatibilidade com a API anterior: executa especificamente SEEDS."""
    return segment_superpixels(rgb, "seeds", params)


def contour_overlay(rgb: np.ndarray, segmentation) -> dict[str, np.ndarray]:
    """Imagens intermediárias da segmentação, para inspeção visual.

    Aceita tanto um `SegmentationResult` quanto um `SegmentationArtifacts`.
    """
    import cv2 as cv

    mask = segmentation.contour_mask
    mask_inv = cv.bitwise_not(mask)
    background = cv.bitwise_and(rgb, rgb, mask=mask_inv)
    foreground = cv.bitwise_and(rgb, rgb, mask=mask)
    return {
        "mask_inv": mask_inv,
        "result_bg": background,
        "result_fg": foreground,
        "result": cv.add(background, foreground),
    }


def colorize_segments(labels: np.ndarray, seed: int = 42) -> np.ndarray:
    """Pinta cada superpixel com uma cor aleatória (só para visualização).

    Usa um gerador próprio para não mexer no RNG global do NumPy — no notebook
    original esta visualização consumia o mesmo `np.random` do resto do pipeline.
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(labels)
    palette = rng.integers(0, 256, size=(unique.max() + 1, 3), dtype=np.uint8)
    return palette[labels]
