"""Salvar e recarregar o modelo treinado.

O padrão do PyTorch é salvar apenas o `state_dict` (os tensores de peso) e
reconstruir a arquitetura antes de carregá-lo. Por isso o checkpoint guarda
também os parâmetros de que `load_checkpoint` precisa para remontar a rede.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .config import ExperimentConfig
from .model import CNN1D

__all__ = ["save_checkpoint", "load_checkpoint"]


def save_checkpoint(model: CNN1D, config: ExperimentConfig, seq_len: int, classes: int, path: Path) -> Path:
    """Grava pesos + tudo o que `load_checkpoint` precisa para reconstruir o modelo."""
    torch.save(
        {
            "state_dict": model.state_dict(),
            "seq_len": seq_len,
            "classes": classes,
            "two_input": config.two_input,
            "dropout": config.dropout,
            "dataset": config.dataset,
            "config": config.to_dict(),
        },
        path,
    )
    return path


def load_checkpoint(path: Path, device=None) -> tuple[CNN1D, dict]:
    """Reconstrói a arquitetura, carrega os pesos e devolve o modelo em `eval()`."""
    device = device or torch.device("cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = CNN1D(
        seq_len=checkpoint["seq_len"],
        classes=checkpoint["classes"],
        two_input=checkpoint["two_input"],
        dropout=checkpoint.get("dropout", 0.5),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
