from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.io as sio
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.extmath import randomized_svd
from torch.utils.data import DataLoader, TensorDataset

from hsi.config import DADOS, SEMENTE_DADOS
from hsi.segmentadores import rgb_composite, segment, segmenter_params

# Arquivo .mat e chave do array, para o cubo e para a verdade terrestre.
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
    """PCA com branqueamento igual ao PCA(whiten=True) do sklearn < 1.5, usado no original.

    O sklearn atual decide o sinal das componentes por V e inverte algumas; no PCA de 3
    componentes isso inverte um canal do RGB e muda a segmentação.
    """
    centrado = x.astype(np.float64) - x.mean(axis=0)
    _, s, vt = randomized_svd(centrado, n_components, n_oversamples=10, n_iter="auto",
                              power_iteration_normalizer="auto", flip_sign=True,
                              random_state=seed)
    return (centrado @ vt.T) / np.sqrt(s ** 2 / (len(x) - 1))


def spectral_branch(espectros: np.ndarray, n_components: int, seed: int) -> np.ndarray:
    return apply_pca(espectros, n_components, seed)


def spatial_branch(pixels, forma, algoritmo, params, seed) -> np.ndarray:
    # Cena inteira, com os pixels sem rótulo: os superpixels precisam da imagem completa.
    rgb = rgb_composite(apply_pca(pixels, 3, seed), forma)
    return segment(rgb, algoritmo, params)


def split_indices(rotulos, test_ratio, valid_ratio, seed):
    """Divisão estratificada em duas etapas, como no original. Devolve (treino, val, teste)."""
    indices = np.arange(len(rotulos))
    treino, resto = train_test_split(indices, test_size=test_ratio, random_state=seed,
                                     stratify=rotulos)
    # Como no original: o primeiro retorno é o teste, e estratificar pelo one-hot muda o
    # sorteio em relação aos rótulos inteiros.
    one_hot = np.eye(rotulos.max() + 1, dtype=np.float32)[rotulos[resto]]
    teste, val = train_test_split(resto, test_size=valid_ratio, random_state=seed,
                                  stratify=one_hot)
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


def prepare_data(config, backend: str = "sequential") -> dict:
    """Executa os dois ramos e divide as amostras.

    Devolve espectros reduzidos, classes, id do superpixel e (linha, coluna) de cada pixel
    rotulado, mais os índices de treino, val e teste. `backend="thread"` roda os ramos juntos.
    """
    cubo, gt = load_scene(config.dataset)
    pixels, rotulos_cena = pixel_table(cubo, gt)
    rotulados = np.flatnonzero(rotulos_cena > 0)
    espectros = pixels[rotulados]
    rotulos = (rotulos_cena[rotulados] - 1).astype(np.int64)

    args_espectral = (espectros, config.pca_components, SEMENTE_DADOS)
    args_espacial = (pixels, gt.shape, config.segmenter, segmenter_params(config), SEMENTE_DADOS)
    if backend == "thread":
        # PCA e OpenCV liberam o GIL no trabalho pesado; as threads rodam de fato juntas.
        with ThreadPoolExecutor(max_workers=2) as pool:
            espacial = pool.submit(spatial_branch, *args_espacial)
            reduzidos = spectral_branch(*args_espectral)
            mapa = espacial.result()
    else:
        reduzidos = spectral_branch(*args_espectral)
        mapa = spatial_branch(*args_espacial)

    treino, val, teste = split_indices(rotulos, config.test_ratio, config.valid_ratio, SEMENTE_DADOS)
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
    return dados


def check_data(d: dict) -> None:
    """Valida o dicionário de `prepare_data` antes do treino."""
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
