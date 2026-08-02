"""Resolução de caminhos — funciona igual no Jupyter, no CLI e numa instância AWS.

O notebook original montava o Google Drive e usava `os.getcwd()`, o que quebrava
conforme o diretório em que o Jupyter tinha sido iniciado. Aqui a raiz do
repositório é descoberta subindo a árvore de diretórios.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["repo_root", "data_dir", "output_dir"]


def repo_root(start: Path | None = None) -> Path:
    """Raiz do repositório, independente de onde o processo foi iniciado."""
    here = (start or Path.cwd()).resolve()
    for base in (here, *here.parents):
        if (base / ".git").is_dir() or (base / "scripts" / "download_data.py").exists():
            return base
    return here


def data_dir(root: Path | None = None) -> Path:
    """Diretório dos `.mat`.

    `HSI_DATA_DIR` tem precedência (útil para apontar um EBS/EFS na AWS); sem ela,
    `data/dataHSI/` dentro do repositório. O caminho do ambiente pode estar fora do
    repositório, por isso a raiz é descoberta à parte e não derivada daqui.
    """
    from_env = os.environ.get("HSI_DATA_DIR")
    if from_env:
        return Path(from_env).expanduser().resolve()
    return (root or repo_root()) / "data" / "dataHSI"


def output_dir(root: Path | None = None, create: bool = True) -> Path:
    """Diretório de saídas (curvas, checkpoints, relatórios)."""
    out = (root or repo_root()) / "outputs"
    if create:
        out.mkdir(parents=True, exist_ok=True)
    return out
