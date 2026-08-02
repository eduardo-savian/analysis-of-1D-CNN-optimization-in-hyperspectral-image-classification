"""Laço de treino: épocas, histórico e parada antecipada.

O histórico guarda quatro séries por época (`loss`, `accuracy`, `val_loss`,
`val_accuracy`) e o treino para sozinho quando a perda de validação deixa de
melhorar por `patience` épocas seguidas.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

__all__ = ["EarlyStopping", "History", "TrainResult", "unpack_batch", "run_epoch", "fit"]


def unpack_batch(batch, device):
    """Batch -> `(x, seg, y)` já no device; `seg` é None no modo de uma entrada.

    Centraliza o desempacotamento para que os laços de treino e de predição não
    precisem saber quantas entradas o modelo tem.
    """
    if len(batch) == 3:
        x, seg, y = batch
        seg = seg.to(device, non_blocking=True)
    else:
        (x, y), seg = batch, None
    return x.to(device, non_blocking=True), seg, y.to(device, non_blocking=True)


class EarlyStopping:
    """Interrompe o treino quando `val_loss` para de melhorar.

    Com `restore_best_weights=False` (o padrão, e o comportamento da implementação
    de referência) os pesos finais são os da **última** época, não os da melhor —
    o critério de parada olha a melhor, mas não volta atrás. Ative a opção para
    também restaurar os pesos da melhor época.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0, restore_best_weights: bool = False) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best = np.inf
        self.wait = 0
        self.stopped_epoch = 0
        self._best_state = None

    def step(self, value: float, model: nn.Module, epoch: int) -> bool:
        """Registra a métrica da época. Retorna True quando é hora de parar."""
        if value < self.best - self.min_delta:
            self.best = value
            self.wait = 0
            if self.restore_best_weights:
                self._best_state = copy.deepcopy(model.state_dict())
            return False

        self.wait += 1
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            if self.restore_best_weights and self._best_state is not None:
                model.load_state_dict(self._best_state)
            return True
        return False


@dataclass
class History:
    """Séries por época: perda e acurácia, em treino e validação."""

    loss: list[float] = field(default_factory=list)
    accuracy: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)

    def append(self, loss: float, accuracy: float, val_loss: float, val_accuracy: float) -> None:
        self.loss.append(loss)
        self.accuracy.append(accuracy)
        self.val_loss.append(val_loss)
        self.val_accuracy.append(val_accuracy)

    def __len__(self) -> int:
        return len(self.loss)

    @property
    def epochs(self) -> range:
        return range(1, len(self) + 1)

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "val_loss": self.val_loss,
            "val_accuracy": self.val_accuracy,
        }


@dataclass
class TrainResult:
    history: History
    epoch_times: list[float]
    total_time: float
    stopped_early: bool
    best_val_loss: float

    def summary(self) -> str:
        linhas = []
        if self.stopped_early:
            linhas.append(
                f"Early stopping na época {len(self.history)} "
                f"(melhor val_loss: {self.best_val_loss:.4f})."
            )
        linhas.append(
            f"Tempo total de treino: {self.total_time:.2f}s | "
            f"média por época: {np.mean(self.epoch_times):.2f}s ({len(self.epoch_times)} épocas)"
        )
        return "\n".join(linhas)


def run_epoch(model, loader: DataLoader, criterion, device, optimizer=None) -> tuple[float, float]:
    """Uma passagem completa sobre o loader.

    `optimizer=None` faz a passagem ser só de avaliação: sem backward, sem passo
    do otimizador e com o modelo em modo `eval` (dropout desligado).
    """
    training = optimizer is not None
    model.train(training)

    total_loss, total_correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            x, seg, y = unpack_batch(batch, device)

            logits = model(x, seg)
            loss = criterion(logits, y)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * y.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)

    return total_loss / total, total_correct / total


def fit(model, loaders, criterion, optimizer, config, device, verbose: bool = True) -> TrainResult:
    """Treina até `config.epochs` ou até a parada antecipada em `val_loss`."""
    # Repõe a semente do gerador do loader de treino: criar um iterador sobre ele
    # (inspecionar um batch, por exemplo) sorteia uma permutação e adianta o estado.
    # Sem isso, olhar os dados antes de treinar mudaria a ordem dos batches — e o
    # resultado — de forma silenciosa.
    generator = getattr(loaders["train"], "generator", None)
    if generator is not None:
        generator.manual_seed(config.seed)

    stopper = EarlyStopping(patience=config.patience)
    history = History()
    epoch_times: list[float] = []
    stopped_early = False

    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_acc = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, loaders["val"], criterion, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_times.append(time.perf_counter() - epoch_start)

        history.append(train_loss, train_acc, val_loss, val_acc)

        if verbose:
            print(
                f"Epoch {epoch}/{config.epochs} - {epoch_times[-1]:.2f}s"
                f" - loss: {train_loss:.4f} - accuracy: {train_acc:.4f}"
                f" - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}"
            )

        if stopper.step(val_loss, model, epoch):
            stopped_early = True
            break

    return TrainResult(
        history=history,
        epoch_times=epoch_times,
        total_time=time.perf_counter() - started,
        stopped_early=stopped_early,
        best_val_loss=float(stopper.best),
    )
