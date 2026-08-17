"""Salvar e recarregar CNN1D, CNN2D ou CNN3D treinadas.

O padrão do PyTorch é salvar apenas o `state_dict` (os tensores de peso) e
reconstruir a arquitetura antes de carregá-lo. Por isso o checkpoint guarda
também `model_type` e o layout da entrada que `load_checkpoint` precisa para
remontar a rede correta.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .config import ExperimentConfig
from .model import CNN1D, CNN2D, CNN3D

__all__ = ["save_checkpoint", "load_checkpoint"]


def save_checkpoint(model, config: ExperimentConfig, input_shape: int | tuple[int, ...], classes: int, path: Path) -> Path:
    """Grava pesos e a arquitetura necessária para reconstruí-los.

    ``input_shape`` preserva o layout específico: ``(1, Kb)`` para 1D,
    ``(3, P, P)`` para 2D e ``(1, B, P, P)`` para 3D.
    """
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_shape": list(input_shape) if not isinstance(input_shape, int) else input_shape,
            "classes": classes,
            "model_type": config.model_type,
            "two_input": config.two_input,
            "dropout": config.dropout,
            "dataset": config.dataset,
            "config": config.to_dict(),
        },
        path,
    )
    return path


def load_checkpoint(path: Path, device=None) -> tuple[object, dict]:
    """Reconstrói a arquitetura, carrega os pesos e devolve o modelo em `eval()`."""
    device = device or torch.device("cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_type = checkpoint.get("model_type", "cnn1d")
    input_shape = checkpoint.get("input_shape", checkpoint.get("seq_len"))
    if model_type == "cnn1d":
        seq_len = input_shape if isinstance(input_shape, int) else input_shape[-1]
        model = CNN1D(seq_len, checkpoint["classes"], two_input=checkpoint["two_input"],
                      dropout=checkpoint.get("dropout", 0.5))
    elif model_type == "cnn2d":
        model = CNN2D(input_shape[0], checkpoint["classes"], two_input=checkpoint["two_input"],
                      dropout=checkpoint.get("dropout", 0.5))
    elif model_type == "cnn3d":
        model = CNN3D(checkpoint["classes"], two_input=checkpoint["two_input"],
                      dropout=checkpoint.get("dropout", 0.5))
    else:
        raise ValueError(f"Checkpoint com model_type desconhecido: {model_type!r}.")
    model = model.to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
