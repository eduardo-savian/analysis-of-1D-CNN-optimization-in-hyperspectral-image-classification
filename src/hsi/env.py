"""Ambiente: conferência de versões, seleção de GPU, determinismo e sementes.

Duas armadilhas que estas funções detectam em vez de deixar estourar no meio do
treino:

- **RTX 50xx (Blackwell, `sm_120`)** só roda em build do PyTorch com CUDA 12.8+.
  Um wheel cu121/cu124 importa sem erro e falha no primeiro kernel com
  *"no kernel image is available for execution on the device"*.
- **`opencv-python` em vez de `opencv-contrib`**: `cv.ximgproc.createSuperpixelSEEDS`
  só existe no pacote contrib, e a falha só apareceria na célula de segmentação.
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

__all__ = ["EXPECTED_VERSIONS", "check_versions", "select_device", "configure_determinism", "seed_everything"]

#: Versões com que o projeto foi validado (as mesmas de `requirements.txt`).
EXPECTED_VERSIONS = {
    "python": "3.12.13",
    "torch": "2.13.0",
    "numpy": "2.5.1",
    "scipy": "1.18.0",
    "scikit-learn": "1.9.0",
    "pandas": "3.0.5",
    "matplotlib": "3.11.1",
    "seaborn": "0.13.2",
    "opencv-contrib-python-headless": "5.0.0.93",
}


def _installed_version(package: str) -> str | None:
    if package == "python":
        return sys.version.split()[0]
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        # A variante com GUI serve igual; só não é a recomendada para servidor.
        if package.endswith("-headless"):
            try:
                return importlib_metadata.version(package[: -len("-headless")]) + " (com GUI)"
            except importlib_metadata.PackageNotFoundError:
                pass
        return None


def check_versions(verbose: bool = True) -> list[str]:
    """Compara o ambiente com `EXPECTED_VERSIONS`. Só avisa — não interrompe.

    Retorna a lista de pacotes divergentes ou ausentes.
    """
    import cv2 as cv

    divergent = []
    if verbose:
        print(f"{'pacote':32s} {'esperado':14s} {'instalado':22s} status")
        print("-" * 80)

    for package, expected in EXPECTED_VERSIONS.items():
        found = _installed_version(package)
        if found is None:
            status = "AUSENTE"
        else:
            base = found.split("+")[0].split(" ")[0]  # ignora '+cu130' e sufixos
            status = "ok" if base == expected else "DIVERGENTE"
        if status != "ok":
            divergent.append(package)
        if verbose:
            print(f"{package:32s} {expected:14s} {str(found):22s} {status}")

    if verbose:
        # O build CUDA vem do índice usado na instalação, não do requirements.txt.
        print("-" * 80)
        cuda_build = torch.version.cuda
        if cuda_build is None:
            cuda_status = "CPU-only (sem GPU)"
        elif tuple(map(int, cuda_build.split("."))) >= (12, 8):
            cuda_status = "ok (suporta sm_120)"
        else:
            cuda_status = "INCOMPATÍVEL com sm_120 (RTX 50xx)"
        print(f"{'torch CUDA build':32s} {'>= 12.8':14s} {str(cuda_build):22s} {cuda_status}")
        print(
            f"{'cv2 / ximgproc (SEEDS)':32s} {'-':14s} {cv.__version__:22s} "
            f"{'ok' if hasattr(cv, 'ximgproc') else 'AUSENTE: instale opencv-contrib'}"
        )
        if divergent:
            print("\nDivergências:", ", ".join(divergent))
            print("Para reproduzir exatamente: uv pip install -r requirements.txt   (ver SETUP.md)")
        else:
            print("\nAmbiente idêntico ao de validação.")

    return divergent


def _arch_number(arch: str, prefix: str) -> int | None:
    value = arch[len(prefix):]
    return int(value) if arch.startswith(prefix) and value.isdigit() else None


def _build_supports_gpu(capability: tuple[int, int], arch_list: list[str]) -> tuple[str, str]:
    """O build do PyTorch atende esta GPU? Retorna (nível, explicação).

    Três formas de atender, em ordem de preferência:
      1. cubin exato para sm_XY;
      2. cubin sm_XZ do mesmo major com Z <= Y — a compatibilidade binária do CUDA
         vale entre minors do mesmo major (sm_86 numa L4 sm_89), nunca entre majors;
      3. PTX (compute_*) de versão <= a da GPU, compilado por JIT no primeiro uso.
    """
    major, minor = capability
    sm = f"sm_{major}{minor}"

    if sm in arch_list:
        return "ok", f"Build tem kernels nativos para {sm}."

    numbers = [_arch_number(a, "sm_") for a in arch_list]
    if any(n is not None and n // 10 == major and n % 10 <= minor for n in numbers):
        return "ok", (
            f"Sem cubin {sm} exato, mas há um do mesmo major reutilizável "
            "(compatibilidade binária CUDA)."
        )

    ptx = [_arch_number(a, "compute_") for a in arch_list]
    if any(n is not None and n <= major * 10 + minor for n in ptx):
        return "aviso", f"{sm} não está no build; deve funcionar via JIT de PTX (primeira execução mais lenta)."

    return "erro", f"Este build do PyTorch não suporta {sm}."


#: Prefixo impresso para cada nível de suporte devolvido por `_build_supports_gpu`.
_SUPPORT_TAG = {"ok": "ok", "aviso": "!", "erro": "ERRO"}


def select_device(verbose: bool = True) -> torch.device:
    """Escolhe GPU se houver, reportando (e testando) se ela realmente executa kernels."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print("Python: ", sys.version.split()[0])
        print("PyTorch:", torch.__version__)
        print("Device: ", device)

    if device.type != "cuda":
        if verbose:
            print("GPU: None — o treino vai rodar em CPU (mais lento, porém funcional).")
        return device

    props = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    arch_list = torch.cuda.get_arch_list()
    level, message = _build_supports_gpu(capability, arch_list)

    if verbose:
        print("GPU:            ", props.name)
        print("Compute cap.:   ", f"{capability[0]}.{capability[1]}", f"(sm_{capability[0]}{capability[1]})")
        print("SMs / memória:  ", props.multi_processor_count, "SMs |", f"{props.total_memory / 1024**3:.2f} GB")
        print("CUDA runtime:   ", torch.version.cuda, "| cuDNN:", torch.backends.cudnn.version())
        print("Build compilado:", " ".join(arch_list))
        print(f"\n[{_SUPPORT_TAG[level]}] {message}")
        if level == "erro" and capability[0] >= 12:
            print("       RTX 50xx (Blackwell) exige wheel CUDA 12.8+:")
            print("       uv pip install torch --index-url https://download.pytorch.org/whl/cu130")

    # Prova real: um kernel de convolução igual ao usado no modelo.
    try:
        probe = nn.Conv1d(1, 16, 9).to(device)
        out = probe(torch.zeros(2, 1, 32, device=device))
        torch.cuda.synchronize()
        if verbose:
            print("[ok] Convolução de teste executada na GPU:", tuple(out.shape))
        del probe, out
        torch.cuda.empty_cache()
    except RuntimeError as error:
        print("[ERRO] A GPU não executou o kernel de teste:", error)

    return device


def configure_determinism(deterministic: bool = True, verbose: bool = True) -> None:
    """Troca explícita entre reprodutibilidade e velocidade.

    `cudnn.benchmark=True` mede kernels e escolhe o mais rápido por shape, mas a
    escolha pode mudar entre execuções e com ela os últimos dígitos do resultado.
    TF32 acelera as densas (Ampere+) truncando a mantissa: é determinístico, porém
    dá números diferentes de FP32 puro.
    """
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # cuBLAS determinístico
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        if verbose:
            print("\nModo determinístico: cudnn.deterministic=True, benchmark=False, TF32 off")
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if verbose:
            print("\nModo desempenho: cudnn.benchmark=True, TF32 on (resultado pode variar)")


def seed_everything(seed: int = 42) -> None:
    """Semeia NumPy e PyTorch (CPU e todas as GPUs).

    Deve ser chamada imediatamente antes de construir o modelo: é a construção
    que consome o RNG do PyTorch e define os pesos iniciais.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
