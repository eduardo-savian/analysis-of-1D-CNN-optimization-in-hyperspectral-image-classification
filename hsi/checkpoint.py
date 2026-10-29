"""Salvar e recarregar a CNN1D treinada.

O padrão do PyTorch é salvar apenas o `state_dict` (os tensores de peso) e
reconstruir a arquitetura antes de carregá-lo. Por isso o checkpoint guarda
também o que define os shapes da rede — `model_type`, layout da entrada, número
de classes, fusão e tamanho da tabela de embeddings —, que `load_checkpoint` usa
para remontá-la.
"""

from pathlib import Path

import torch

from hsi.config import ExperimentConfig
from hsi.modelo import CNN1D


def save_checkpoint(model, config: ExperimentConfig, input_shape: int | tuple[int, ...], classes: int, path: Path) -> Path:
    """Grava pesos e a arquitetura necessária para reconstruí-los.

    ``input_shape`` é o layout de uma amostra, ``(1, Kb)``, vindo de
    ``PreparedData.input_shape()``; ``load_checkpoint`` usa só o último eixo.
    """
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_shape": list(input_shape) if not isinstance(input_shape, int) else input_shape,
            "classes": classes,
            "model_type": config.model_type,
            "two_input": config.two_input,
            "dropout": config.dropout,
            "fusion": config.fusion,
            # Não está na config: vem dos dados (maior id + 1). A tabela de
            # embeddings do próprio modelo garante o mesmo shape ao recarregar.
            "n_segments": getattr(model, "segment_embedding", None).num_embeddings
            if getattr(model, "segment_embedding", None) is not None else 0,
            "embedding_dim": config.segment_embedding_dim,
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
    # Checkpoints antigos gravavam só `seq_len` (o inteiro Kb).
    input_shape = checkpoint.get("input_shape", checkpoint.get("seq_len"))
    # Checkpoints anteriores à fusão por embedding não têm estas chaves; os
    # defaults reproduzem exatamente o comportamento escalar antigo.
    fusion_kwargs = {
        "fusion": checkpoint.get("fusion", "scalar"),
        "n_segments": checkpoint.get("n_segments", 0),
        "embedding_dim": checkpoint.get("embedding_dim", 8),
    }
    if model_type != "cnn1d":
        raise ValueError(f"Checkpoint de arquitetura fora do escopo: {model_type!r}.")
    seq_len = input_shape if isinstance(input_shape, int) else input_shape[-1]
    model = CNN1D(seq_len, checkpoint["classes"], two_input=checkpoint["two_input"],
                  dropout=checkpoint.get("dropout", 0.5), **fusion_kwargs)
    model = model.to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
