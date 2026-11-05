"""Versões das bibliotecas, dispositivo, sementes e modo determinístico."""

import importlib.metadata
import os
import sys

import cv2 as cv
import numpy as np
import torch

# As versões de requirements.txt, com que os resultados foram medidos.
EXPECTED_VERSIONS = {
    "torch": "2.13.0",
    "numpy": "2.5.1",
    "scipy": "1.18.0",
    "scikit-learn": "1.9.0",
    "pandas": "3.0.5",
    "matplotlib": "3.11.1",
    "opencv-contrib-python-headless": "5.0.0.93",
    "optuna": "4.9.0",
}


def check_versions() -> list[str]:
    """Imprime as versões instaladas e devolve as que diferem das esperadas."""
    divergentes = []
    print(f"python {sys.version.split()[0]}")
    for pacote, esperada in EXPECTED_VERSIONS.items():
        try:
            instalada = importlib.metadata.version(pacote)
        except importlib.metadata.PackageNotFoundError:
            instalada = "ausente"
        # O torch da GPU vem como 2.13.0+cu130; o sufixo não conta.
        ok = instalada.split("+")[0] == esperada
        if not ok:
            divergentes.append(pacote)
        print(f"{pacote:32s} {instalada:16s} {'ok' if ok else f'esperada {esperada}'}")
    # SEEDS e LSC estão só no pacote opencv-contrib.
    if not hasattr(cv, "ximgproc"):
        print("cv2.ximgproc ausente: instale opencv-contrib-python-headless")
    return divergentes


def select_device(verbose: bool = True) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        nome = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
        print(f"dispositivo: {device} ({nome})")
    return device


def configure_determinism() -> None:
    """Deixa as execuções repetíveis: mesmas sementes, mesmo resultado.

    O OpenCV fica com uma thread porque o LSC paralelo devolvia rótulos diferentes a
    cada chamada com a mesma imagem. Com uma thread, o LSC fica de duas a quatro vezes
    mais lento (LSC_THREADS_MS em experimentos/resultados/medicoes_avulsas.md).
    """
    cv.setNumThreads(1)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_everything(seed: int) -> None:
    """Semeia NumPy e PyTorch. Chamar logo antes de criar o modelo, que é quem
    consome o gerador do PyTorch nos pesos iniciais."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
