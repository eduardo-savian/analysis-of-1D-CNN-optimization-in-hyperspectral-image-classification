"""Segmentação espacial com SEEDS, SLIC, LSC (OpenCV contrib) e Watershed.

O id da região de cada pixel vira a *segunda entrada* da rede: além do espectro,
o modelo recebe a que região homogênea aquele pixel pertence.

Os quatro algoritmos devolvem o mesmo contrato — `SegmentationResult` com labels
e máscara de contorno — e por isso DataLoader, CNN e métricas não mudam quando o
segmentador muda. SEEDS/SLIC/LSC são superpixels do `cv.ximgproc` e exigem
`opencv-contrib-python[-headless]`; Watershed usa bacias de gradiente do OpenCV
base, iniciado por uma malha regular de marcadores, e é a alternativa
conceitualmente distinta do conjunto.

A tupla canônica `SEGMENTERS` é definida em `hsi/config.py` (contrato de configuração),
porque `ExperimentConfig` a valida; aqui ficam apenas os parâmetros e a execução.
"""

from dataclasses import dataclass

import numpy as np

from hsi.config import SEGMENTERS


@dataclass(frozen=True)
class SeedsParams:
    """Parâmetros do `cv.ximgproc.createSuperpixelSEEDS`.

    num_superpixels é um alvo, não uma garantia: o algoritmo pode devolver bem menos
    (medido: 350 viram 81 regiões em Indian Pines; 150 viram 45 em PaviaU).
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
    #: Sem efeito no SLICO, que adapta a compacidade sozinho: os rótulos saem
    #: idênticos de 5 a 80 (medido em Indian Pines). Só o SLIC clássico o usa.
    ruler: float = 10.0
    num_iterations: int = 10


@dataclass(frozen=True)
class LscParams:
    """Parâmetros do Linear Spectral Clustering (LSC) do OpenCV."""

    region_size: int = 12
    ratio: float = 0.075
    num_iterations: int = 10


@dataclass(frozen=True)
class WatershedParams:
    """Sementes do Watershed do OpenCV, numa malha regular sobre a cena.

    `markers` é um alvo, como o `num_superpixels` do SEEDS: as sementes da borda se
    perdem (ver `_watershed_segments`), e 350 rendem 273 regiões em Indian Pines,
    contando a região 0 dos contornos.
    """

    markers: int = 350


# Pela mesma razão do SEEDS, Indian Pines usa regiões menores: 324 no SLIC e no LSC.
SLIC_PARAMS = {"IP": SlicParams(region_size=8), "PU": SlicParams(), "SA": SlicParams()}


LSC_PARAMS = {"IP": LscParams(region_size=8), "PU": LscParams(), "SA": LscParams()}


WATERSHED_PARAMS = {
    "IP": WatershedParams(markers=350),
    "PU": WatershedParams(markers=500),
    "SA": WatershedParams(markers=500),
}


@dataclass(frozen=True)
class SegmentationResult:
    """Saída da segmentação, em arrays puros.

    Guardar a máscara de contorno aqui — em vez do objeto do segmentador do OpenCV,
    que não é serializável — é o que permite rodar a segmentação num processo
    separado e trazer o resultado de volta (backend `process` de
    `prepare_data_parallel`, em `hsi/paralelo/ramos.py`).
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
    rgb: np.ndarray  # (H, W, 3) uint8 — entrada do segmentador
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


def _boundary_mask(labels: np.ndarray) -> np.ndarray:
    """Converte transições entre regiões em máscara de contorno uint8.

    O Watershed do OpenCV não expõe ``getLabelContourMask``; derivar a máscara das
    diferenças entre vizinhos mantém o mesmo contrato visual dos superpixels.
    """
    mask = np.zeros(labels.shape, dtype=np.uint8)
    mask[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    mask[1:, :] |= labels[1:, :] != labels[:-1, :]
    return (255 * mask).astype(np.uint8)


def _watershed_segments(rgb: np.ndarray, params: WatershedParams) -> SegmentationResult:
    """Executa Watershed com uma malha de sementes regulares e reindexa as regiões."""
    import cv2 as cv

    height, width = rgb.shape[:2]
    target = max(1, params.markers)
    # Malha com a proporção da cena: rows/cols ≈ height/width e rows*cols ≈ target.
    rows = max(1, int(round(np.sqrt(target * height / width))))
    cols = max(1, int(round(target / rows)))
    markers = np.zeros((height, width), dtype=np.int32)
    marker_id = 1
    # A primeira e a última linha e coluna da malha caem na moldura da imagem, que o
    # `cv.watershed` sobrescreve com -1: essas sementes se perdem (70 de 342 em IP).
    for y in np.linspace(0, height - 1, rows, dtype=int):
        for x in np.linspace(0, width - 1, cols, dtype=int):
            markers[y, x] = marker_id
            marker_id += 1
    raw = cv.watershed(cv.cvtColor(rgb, cv.COLOR_RGB2BGR), markers)
    # -1 marca as linhas entre bacias e a moldura inteira. Todos esses pixels viram
    # um único id, 0, espalhado pela cena (21% dos pixels de Indian Pines).
    raw[raw < 0] = 0
    _, labels = np.unique(raw, return_inverse=True)
    labels = labels.reshape(raw.shape).astype(np.int32)
    return SegmentationResult(labels=labels, contour_mask=_boundary_mask(labels))


def segment_superpixels(
    rgb: np.ndarray,
    algorithm: str,
    params: SeedsParams | SlicParams | LscParams | WatershedParams,
) -> SegmentationResult:
    """Segmenta RGB com o algoritmo escolhido e devolve labels/contornos puros.

    Args:
        rgb: Composição uint8 ``(H, W, 3)`` construída das componentes de PCA.
        algorithm: ``seeds``, ``slic`` (SLICO), ``lsc`` ou ``watershed``.
        params: Dataclass correspondente ao algoritmo e à cena.

    A saída uniforme permite comparar algoritmos sem alterar DataLoader, modelo ou
    métricas. Watershed é resolvido antes da checagem do ``ximgproc`` porque é o
    único que não depende do pacote contrib.
    """
    import cv2 as cv

    if algorithm == "watershed":
        if not isinstance(params, WatershedParams):
            raise TypeError("Watershed requer WatershedParams.")
        return _watershed_segments(rgb, params)
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

    Aceita tanto um `SegmentationResult` quanto um `SegmentationArtifacts`. Como no
    original, `result` (fundo + frente) reconstitui o próprio RGB, porque máscara e
    inversa particionam os pixels: os contornos só aparecem, em preto, em `result_bg`.
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
