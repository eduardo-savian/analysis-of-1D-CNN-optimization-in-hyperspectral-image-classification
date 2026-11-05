"""Os segmentadores do projeto: SEEDS, LSC e Watershed, sobre a composição RGB da cena.

Os parâmetros por cena vêm de [segmentadores] em config.toml: os do SEEDS são os do
notebook original, e LSC e Watershed foram calibrados para gerar o mesmo número de
regiões que ele. Em LSC e Watershed cada parâmetro tem três valores (variação menor,
padrão e maior); o padrão é o do meio, e a busca de hiperparâmetros usa os três.
"""

import cv2 as cv
import numpy as np

from hsi.config import CONFIG, DATASETS

_S = CONFIG["segmentadores"]
SEGMENTER_PARAMS = {
    "seeds": _S["seeds"],
    "lsc": {c: {"region_size": _S["lsc"]["region_size"][c][1], "ratio": _S["lsc"]["ratio"][1],
                "num_iterations": _S["lsc"]["num_iterations"]} for c in DATASETS},
    "watershed": {c: {"markers": _S["watershed"]["markers"][c][1]} for c in DATASETS},
}


def rgb_composite(componentes: np.ndarray, forma: tuple[int, int]) -> np.ndarray:
    """Três componentes principais -> imagem RGB uint8, a entrada dos segmentadores."""
    canais = []
    for i in range(3):
        c = componentes[:, i]
        normalizado = (c - c.min()) / (c.max() - c.min())
        canais.append((255 * normalizado).astype(np.uint8).reshape(forma))
    return np.dstack(canais)


def segmenter_params(config) -> dict:
    return {**SEGMENTER_PARAMS[config.segmenter][config.dataset], **config.segmenter_overrides}


def segment(rgb: np.ndarray, algoritmo: str, p: dict) -> np.ndarray:
    """Mapa (H, W) com o id da região de cada pixel."""
    if algoritmo == "watershed":
        return _watershed(rgb, p["markers"])
    if algoritmo == "seeds":
        h, w, canais = rgb.shape
        s = cv.ximgproc.createSuperpixelSEEDS(w, h, canais, p["num_superpixels"], p["num_levels"],
                                              p["prior"], p["num_histogram_bins"], p["double_step"])
        s.iterate(rgb, p["num_iterations"])
    elif algoritmo == "lsc":
        s = cv.ximgproc.createSuperpixelLSC(rgb, region_size=p["region_size"], ratio=p["ratio"])
        s.iterate(p["num_iterations"])
    else:
        raise ValueError(f"segmentador inválido: {algoritmo!r}")
    return s.getLabels()


def _watershed(rgb: np.ndarray, n_marcadores: int) -> np.ndarray:
    # Marcadores numa grade com a proporção da cena. Os da moldura se perdem, porque o
    # cv.watershed escreve -1 na borda da imagem.
    h, w = rgb.shape[:2]
    linhas = max(1, int(round(np.sqrt(n_marcadores * h / w))))
    colunas = max(1, int(round(n_marcadores / linhas)))
    marcadores = np.zeros((h, w), dtype=np.int32)
    k = 1
    for y in np.linspace(0, h - 1, linhas, dtype=int):
        for x in np.linspace(0, w - 1, colunas, dtype=int):
            marcadores[y, x] = k
            k += 1
    bruto = cv.watershed(cv.cvtColor(rgb, cv.COLOR_RGB2BGR), marcadores)
    # As linhas entre bacias (-1) viram uma região própria, a de id 0.
    bruto[bruto < 0] = 0
    _, rotulos = np.unique(bruto, return_inverse=True)
    return rotulos.reshape(bruto.shape).astype(np.int32)
