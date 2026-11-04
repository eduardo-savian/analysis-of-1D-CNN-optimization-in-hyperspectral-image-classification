"""Da cena em disco aos DataLoaders, na ordem em que uma execução percorre.

carga da cena -> ramo espectral (PCA) e ramo espacial (PCA de 3 componentes, RGB e
segmentador) -> divisão em treino, validação e teste -> DataLoaders.

Os dois ramos não dependem um do outro; `prepare_data` pode executá-los em sequência,
em duas threads ou com o ramo espacial num processo filho.
"""

import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import cv2 as cv
import numpy as np
import scipy.io as sio
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from hsi.config import DADOS

# Arquivo .mat e chave do array dentro dele, para o cubo e para a verdade terrestre.
SCENE_FILES = {
    "IP": (("Indian_pines_corrected.mat", "indian_pines_corrected"),
           ("Indian_pines_gt.mat", "indian_pines_gt")),
    "PU": (("PaviaU.mat", "paviaU"), ("PaviaU_gt.mat", "paviaU_gt")),
    "SA": (("Salinas_corrected.mat", "salinas_corrected"), ("Salinas_gt.mat", "salinas_gt")),
}


def load_scene(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    """Cubo (H, W, B) e verdade terrestre (H, W), em que 0 é pixel sem rótulo."""
    (arquivo_cubo, chave_cubo), (arquivo_gt, chave_gt) = SCENE_FILES[dataset]
    cubo = sio.loadmat(DADOS / arquivo_cubo)[chave_cubo]
    gt = sio.loadmat(DADOS / arquivo_gt)[chave_gt]
    return cubo, gt


def pixel_table(cubo: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Achata a cena em uma linha por pixel: (H*W, B) e (H*W,)."""
    h, w, bandas = cubo.shape
    return cubo.reshape(h * w, bandas), gt.reshape(h * w)


def apply_pca(x: np.ndarray, n_components: int, seed: int) -> np.ndarray:
    # Branqueamento e semente fixa como no original. Fica fit + transform, e não
    # fit_transform, porque os dois caminhos diferem nos últimos dígitos.
    pca = PCA(n_components=n_components, whiten=True, random_state=seed)
    pca.fit(x)
    return pca.transform(x)


def rgb_composite(componentes: np.ndarray, forma: tuple[int, int]) -> np.ndarray:
    """Três componentes principais -> imagem RGB uint8, a entrada dos segmentadores."""
    canais = []
    for i in range(3):
        c = componentes[:, i]
        normalizado = (c - c.min()) / (c.max() - c.min())
        canais.append((255 * normalizado).astype(np.uint8).reshape(forma))
    return np.dstack(canais)


# ----------------------------------------------------------------- segmentadores

# Parâmetros padrão por cena. Indian Pines (145 x 145 px, talhões estreitos) pede
# regiões menores. Os do SEEDS são os do notebook original.
SEGMENTER_PARAMS = {
    "seeds": {
        "IP": dict(num_superpixels=350, num_iterations=200, prior=2,
                   num_levels=10, num_histogram_bins=5, double_step=True),
        "PU": dict(num_superpixels=150, num_iterations=150, prior=1,
                   num_levels=8, num_histogram_bins=15, double_step=True),
        "SA": dict(num_superpixels=150, num_iterations=150, prior=1,
                   num_levels=8, num_histogram_bins=15, double_step=True),
    },
    # O SLIC roda no modo SLICO, que ajusta a compacidade sozinho: `ruler` não muda
    # os rótulos, mas fica registrado porque faz parte do espaço de busca.
    "slic": {c: dict(region_size=8 if c == "IP" else 12, ruler=10.0, num_iterations=10)
             for c in ("IP", "PU", "SA")},
    "lsc": {c: dict(region_size=8 if c == "IP" else 12, ratio=0.075, num_iterations=10)
            for c in ("IP", "PU", "SA")},
    "watershed": {"IP": dict(markers=350), "PU": dict(markers=500), "SA": dict(markers=500)},
}


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
    elif algoritmo == "slic":
        s = cv.ximgproc.createSuperpixelSLIC(rgb, algorithm=cv.ximgproc.SLICO,
                                             region_size=p["region_size"], ruler=p["ruler"])
        s.iterate(p["num_iterations"])
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


# ------------------------------------------------------------ os dois ramos

def spectral_branch(espectros: np.ndarray, n_components: int, seed: int) -> np.ndarray:
    return apply_pca(espectros, n_components, seed)


def spatial_branch(pixels, forma, algoritmo, params, seed) -> np.ndarray:
    # Segmenta a cena inteira, pixels sem rótulo incluídos: os superpixels precisam da
    # imagem completa.
    rgb = rgb_composite(apply_pca(pixels, 3, seed), forma)
    return segment(rgb, algoritmo, params)


def _cronometrar(funcao, *args):
    inicio = time.perf_counter()
    return funcao(*args), time.perf_counter() - inicio


# ---------------------------------------------------------- divisão dos dados

def split_indices(rotulos, test_ratio, valid_ratio, seed):
    """Sorteio estratificado em duas etapas, como no original."""
    indices = np.arange(len(rotulos))
    treino, resto = train_test_split(indices, test_size=test_ratio, random_state=seed,
                                     stratify=rotulos)
    # No original, o primeiro retorno desta segunda chamada é o teste.
    teste, val = train_test_split(resto, test_size=valid_ratio, random_state=seed,
                                  stratify=rotulos[resto])
    return treino, val, teste


def subsample_train(treino, rotulos, fracao, seed):
    """Fica com `fracao` do treino, sorteada por classe para nenhuma classe sumir."""
    if fracao >= 1:
        return treino
    rng = np.random.default_rng(seed)
    escolhidos = []
    for classe in np.unique(rotulos[treino]):
        da_classe = treino[rotulos[treino] == classe]
        n = max(1, int(round(len(da_classe) * fracao)))
        escolhidos.append(rng.choice(da_classe, size=n, replace=False))
    return np.sort(np.concatenate(escolhidos))


# ------------------------------------------------------------- preparação

def prepare_data(config, backend: str = "sequential", cena=None):
    """Executa os dois ramos e divide as amostras. Devolve `(dados, tempos)`.

    `dados` é um dicionário com os espectros reduzidos, as classes, o id do superpixel
    de cada pixel rotulado, os índices de treino, validação e teste e a posição
    (linha, coluna) de cada amostra. `tempos` traz a duração de cada ramo e a do
    bloco inteiro.
    """
    cubo, gt = cena if cena is not None else load_scene(config.dataset)
    pixels, rotulos_cena = pixel_table(cubo, gt)
    rotulados = np.flatnonzero(rotulos_cena > 0)
    espectros = pixels[rotulados]
    rotulos = (rotulos_cena[rotulados] - 1).astype(np.int64)

    args_espectral = (espectros, config.pca_components, config.seed)
    args_espacial = (pixels, gt.shape, config.segmenter, segmenter_params(config), config.seed)
    inicio = time.perf_counter()
    if backend == "sequential":
        reduzidos, t_espectral = _cronometrar(spectral_branch, *args_espectral)
        mapa, t_espacial = _cronometrar(spatial_branch, *args_espacial)
    elif backend == "thread":
        # PCA e OpenCV liberam o GIL no trabalho pesado, então as duas threads rodam
        # de fato ao mesmo tempo, compartilhando os arrays sem cópia.
        with ThreadPoolExecutor(max_workers=2) as pool:
            espacial = pool.submit(_cronometrar, spatial_branch, *args_espacial)
            reduzidos, t_espectral = pool.submit(_cronometrar, spectral_branch,
                                                 *args_espectral).result()
            mapa, t_espacial = espacial.result()
    elif backend == "process":
        # Fork: o filho herda o OpenCV com uma thread, que o LSC precisa.
        with ProcessPoolExecutor(max_workers=1,
                                 mp_context=multiprocessing.get_context("fork")) as pool:
            espacial = pool.submit(_cronometrar, spatial_branch, *args_espacial)
            reduzidos, t_espectral = _cronometrar(spectral_branch, *args_espectral)
            mapa, t_espacial = espacial.result()
    else:
        raise ValueError(f"backend inválido: {backend!r}")
    tempos = {"espectral": t_espectral, "espacial": t_espacial,
              "parede": time.perf_counter() - inicio}

    treino, val, teste = split_indices(rotulos, config.test_ratio, config.valid_ratio, config.seed)
    treino = subsample_train(treino, rotulos, config.train_fraction, config.seed)
    dados = {
        "espectros": reduzidos,
        "rotulos": rotulos,
        "segmentos": mapa.reshape(-1)[rotulados],
        "treino": treino, "val": val, "teste": teste,
        "coords": np.stack(np.divmod(rotulados, gt.shape[1]), axis=1),
        "forma": gt.shape,
        "n_classes": int(rotulos.max()) + 1,
    }
    check_data(dados)
    return dados, tempos


def check_data(d: dict) -> None:
    """Conferências baratas antes de gastar tempo de treino."""
    n = len(d["rotulos"])
    if not np.isfinite(d["espectros"]).all():
        raise ValueError("espectros com NaN ou infinito")
    if not np.array_equal(np.unique(d["rotulos"]), np.arange(d["n_classes"])):
        raise ValueError("as classes precisam ir de 0 a C-1 sem buracos")
    if len(d["espectros"]) != n or len(d["segmentos"]) != n:
        raise ValueError("espectros, segmentos e rótulos precisam de uma linha por amostra")
    todos = np.concatenate([d["treino"], d["val"], d["teste"]])
    if len(np.unique(todos)) != len(todos):
        raise ValueError("uma amostra aparece em mais de uma partição")


def build_loaders(d: dict, config, device) -> dict[str, DataLoader]:
    """Um DataLoader por partição. Cada lote traz (espectro, id do superpixel, classe)."""
    def tensores(indices):
        x = torch.from_numpy(np.ascontiguousarray(d["espectros"][indices])).float().unsqueeze(1)
        seg = torch.from_numpy(np.ascontiguousarray(d["segmentos"][indices])).float().reshape(-1, 1)
        y = torch.from_numpy(np.ascontiguousarray(d["rotulos"][indices]))
        return TensorDataset(x, seg, y)

    pin = device.type == "cuda"
    return {
        # Gerador próprio: a ordem dos lotes depende só da semente da configuração.
        "treino": DataLoader(tensores(d["treino"]), batch_size=config.batch_size, shuffle=True,
                             generator=torch.Generator().manual_seed(config.seed), pin_memory=pin),
        "val": DataLoader(tensores(d["val"]), batch_size=config.batch_size, pin_memory=pin),
        "teste": DataLoader(tensores(d["teste"]), batch_size=config.batch_size, pin_memory=pin),
    }
