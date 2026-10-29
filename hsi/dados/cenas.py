"""Carga das cenas hiperespectrais e transformação para tabela de pixels."""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio

from hsi.config import default_data_dir, repo_root

# Arquivo e chave do dicionário `.mat` de cada cena. O notebook original repetia
# isso num if/elif de três ramos com o caminho montado na mão.
SCENE_FILES = {
    "IP": {
        "image": ("Indian_pines_corrected.mat", "indian_pines_corrected"),
        "labels": ("Indian_pines_gt.mat", "indian_pines_gt"),
    },
    "PU": {
        "image": ("PaviaU.mat", "paviaU"),
        "labels": ("PaviaU_gt.mat", "paviaU_gt"),
    },
    "SA": {
        "image": ("Salinas_corrected.mat", "salinas_corrected"),
        "labels": ("Salinas_gt.mat", "salinas_gt"),
    },
}


@dataclass(frozen=True)
class Scene:
    """Cena hiperespectral crua: cubo (H, W, B) e ground truth (H, W)."""

    code: str
    image: np.ndarray
    labels: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape[0], self.image.shape[1]

    @property
    def n_bands(self) -> int:
        return self.image.shape[2]

    @property
    def n_classes(self) -> int:
        """Classes reais, sem contar o background (rótulo 0)."""
        return int(self.labels.max())

    def __str__(self) -> str:
        return (
            f"{self.code}: cubo {self.image.shape} {self.image.dtype} | "
            f"gt {self.labels.shape} {self.labels.dtype} | {self.n_classes} classes"
        )


def ensure_dataset(dataset: str, directory: Path | None = None) -> Path:
    """Confere se os `.mat` da cena estão em disco e devolve o diretório deles.

    Se faltar algum, delegaria o download a `scripts/download_data.py`, que não
    existe mais no repositório: na prática, arquivo ausente vira `FileNotFoundError`.
    Coloque os `.mat` em `data/dataHSI/` ou aponte `HSI_DATA_DIR` para eles.
    """
    directory = Path(directory) if directory else default_data_dir()
    required = [name for name, _ in SCENE_FILES[dataset].values()]
    missing = [f for f in required if not (directory / f).exists()]
    if not missing:
        return directory

    script = repo_root() / "scripts" / "download_data.py"
    if not script.exists():
        raise FileNotFoundError(
            f"Arquivos ausentes em {directory}: {missing}. "
            f"{script} não encontrado — baixe manualmente (ver SETUP.md) ou defina HSI_DATA_DIR."
        )
    directory.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(script), "--dataset", dataset, "--out", str(directory)],
        check=True,
    )
    return directory


def load_scene(dataset: str, directory: Path | None = None, download: bool = True) -> Scene:
    """Carrega cubo e ground truth de uma cena."""
    if dataset not in SCENE_FILES:
        raise ValueError(f"Dataset inválido: {dataset!r}. Use IP, PU ou SA.")
    directory = ensure_dataset(dataset, directory) if download else (Path(directory) if directory else default_data_dir())

    spec = SCENE_FILES[dataset]
    (img_file, img_key), (gt_file, gt_key) = spec["image"], spec["labels"]
    image = sio.loadmat(directory / img_file)[img_key]
    labels = sio.loadmat(directory / gt_file)[gt_key]
    return Scene(code=dataset, image=image, labels=labels)


def to_pixel_table(scene: Scene) -> tuple[np.ndarray, np.ndarray]:
    """Cubo (H, W, B) -> tabela (H*W, B) para classificação por pixel.

    Retorna também os rótulos achatados (H*W,), ainda com o background.
    """
    h, w, bands = scene.image.shape
    return scene.image.reshape(h * w, bands), scene.labels.reshape(h * w)


def remove_background(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Descarta as amostras de background (classe 0) e reindexa as classes para 0..C-1.

    Funciona tanto para a tabela espectral (N, B) quanto para uma coluna auxiliar
    (N, 1) — o filtro é sempre pelo vetor de rótulos `y`.
    """
    keep = y > 0
    x_out, y_out = x[keep], y[keep]
    return x_out, y_out - 1
