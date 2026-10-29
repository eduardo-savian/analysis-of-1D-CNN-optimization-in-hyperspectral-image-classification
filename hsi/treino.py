"""Laço de treino: épocas, histórico e parada antecipada.

O histórico guarda quatro séries por época, com os nomes do `history` do Keras no
original: `loss`, `accuracy`, `val_loss` e `val_accuracy`. A parada antecipada
monitora uma métrica de validação configurável e encerra o treino após uma janela
(`patience`) sem ganho mínimo significativo, ou antes, se a meta opcional
(`early_stopping_target`) for atingida.
"""

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader


def unpack_batch(batch, device):
    """Batch -> `(x, seg, y)` já no device; `seg` é None no modo de uma entrada.

    Centraliza o desempacotamento para que os laços de treino e de predição não
    precisem saber quantas entradas o modelo tem.

    `non_blocking=True` só sobrepõe a cópia CPU→GPU ao cálculo quando o tensor
    está em memória fixada (`pin_memory`, que `build_loaders` liga em CUDA); fora
    disso, a cópia é síncrona como sempre.
    """
    if len(batch) == 3:
        x, seg, y = batch
        seg = seg.to(device, non_blocking=True)
    else:
        (x, y), seg = batch, None
    return x.to(device, non_blocking=True), seg, y.to(device, non_blocking=True)


class EarlyStopping:
    """Encerra após uma janela de épocas sem melhoria relevante na validação.

    Para ``val_accuracy``, uma época só reinicia a janela se superar a melhor
    acurácia por mais de ``min_delta`` (comparação estrita, como no Keras). Para
    ``val_loss``, a redução também precisa passar de ``min_delta``. Ao parar, por
    meta ou por patience, restaura opcionalmente os pesos da melhor época (só
    nesse caso; ver ``fit``).
    """

    def __init__(self, patience, min_delta, monitor, target, restore_best_weights):
        if monitor not in {"val_accuracy", "val_loss"}:
            raise ValueError("monitor deve ser 'val_accuracy' ou 'val_loss'.")
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = "max" if monitor == "val_accuracy" else "min"
        self.target = target
        self.target_reached = False
        self.stop_reason = None
        self.restore_best_weights = restore_best_weights
        self.best = -np.inf if self.mode == "max" else np.inf
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0
        self._best_state = None

    def _improved(self, value):
        return value > self.best + self.min_delta if self.mode == "max" else value < self.best - self.min_delta

    def _target_reached(self, value):
        if self.target is None:
            return False
        return value >= self.target if self.mode == "max" else value <= self.target

    def _restore(self, model):
        if self.restore_best_weights and self._best_state is not None:
            model.load_state_dict(self._best_state)

    def step(self, value, model, epoch):
        """Registra uma época e para ao atingir meta ou esgotar patience."""
        if self._improved(value):
            self.best, self.best_epoch, self.wait = value, epoch, 0
            if self.restore_best_weights:
                # `state_dict()` devolve referências aos tensores vivos; sem a
                # cópia, o "melhor" estado mudaria junto com o treino.
                self._best_state = copy.deepcopy(model.state_dict())
        else:
            self.wait += 1
        if self._target_reached(value):
            self.target_reached, self.stopped_epoch = True, epoch
            self.stop_reason = f"meta {self.monitor}={self.target:g} atingida"
            self._restore(model)
            return True
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.stop_reason = f"{self.patience} épocas sem melhoria de pelo menos {self.min_delta:g} em {self.monitor}"
            self._restore(model)
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
    """Histórico e decisão de early stopping de uma execução."""

    history: History
    epoch_times: list[float]
    total_time: float
    stopped_early: bool
    best_val_loss: float
    best_epoch: int
    best_monitor_value: float
    early_stopping_monitor: str
    early_stopping_min_delta: float
    patience: int
    restore_best_weights: bool
    early_stopping_target: float | None
    target_reached: bool
    stop_reason: str

    def summary(self):
        status = (f"Early stopping na época {len(self.history)}: {self.stop_reason}."
                  if self.stopped_early else "Treino encerrou ao atingir o limite de épocas.")
        return (f"Melhor {self.early_stopping_monitor}: {self.best_monitor_value:.4f} "
                f"na época {self.best_epoch} (Δ mínimo: {self.early_stopping_min_delta:g}).\n"
                f"{status} Pesos restaurados: {self.restore_best_weights}.\n"
                f"Tempo total: {self.total_time:.2f}s | {len(self.epoch_times)} épocas")


def run_epoch(
    model,
    loader: DataLoader,
    criterion,
    device,
    optimizer=None,
    gradient_accumulation_steps: int = 1,
) -> tuple[float, float]:
    """Uma passagem completa sobre o loader.

    `optimizer=None` faz a passagem ser só de avaliação: sem backward, sem passo
    do otimizador e com o modelo em modo `eval` (dropout desligado).

    Quando ``gradient_accumulation_steps > 1``, cada batch contribui com sua soma
    de gradientes e a janela é normalizada pelo número **real** de amostras antes
    de ``optimizer.step``. Assim a última janela, que pode ser menor, continua
    equivalente a um batch concatenado. Com valor 1 o caminho numérico original
    é preservado, mantendo a reprodutibilidade do baseline.
    """
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps deve ser ao menos 1.")

    training = optimizer is not None
    # Reposto a cada passagem: `fit` alterna treino e validação na mesma época, e a
    # validação deixa o modelo em `eval` (dropout desligado).
    model.train(training)

    total_loss, total_correct, total = 0.0, 0, 0
    samples_in_window = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
        for batch_index, batch in enumerate(loader, start=1):
            x, seg, y = unpack_batch(batch, device)

            logits = model(x, seg)
            loss = criterion(logits, y)

            if training:
                if gradient_accumulation_steps == 1:
                    # Mantém exatamente o caminho numérico do baseline.
                    loss.backward()
                else:
                    # CrossEntropyLoss devolve a média do batch. Acumulamos a
                    # soma e normalizamos uma vez pelo total real da janela; isso
                    # também trata corretamente a última janela incompleta.
                    (loss * y.size(0)).backward()
                    samples_in_window += y.size(0)
                if batch_index % gradient_accumulation_steps == 0 or batch_index == len(loader):
                    if gradient_accumulation_steps > 1:
                        for parameter in model.parameters():
                            if parameter.grad is not None:
                                parameter.grad.div_(samples_in_window)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    samples_in_window = 0

            total_loss += loss.item() * y.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)

    return total_loss / total, total_correct / total


def fit(model, loaders, criterion, optimizer, config, device, verbose: bool = True) -> TrainResult:
    """Treina, valida, mede tempo e aplica parada antecipada.

    Cada época faz uma passagem de treino, outra de validação e sincroniza CUDA
    antes de medir duração. A acumulação pertence somente ao treino; validação
    sempre mede os logits do modelo sem gradientes. O retorno preserva todo o
    histórico necessário para curvas e para ``metrics.json``.

    Os pesos da melhor época só voltam quando a parada antecipada dispara. Se o
    laço chega a ``config.epochs`` sem parar, o modelo fica com os pesos da última
    época, ainda que ``restore_best_weights`` seja True.
    """
    # Repõe a semente do gerador do loader de treino: criar um iterador sobre ele
    # (inspecionar um batch, por exemplo) sorteia uma permutação e adianta o estado.
    # Sem isso, olhar os dados antes de treinar mudaria a ordem dos batches — e o
    # resultado — de forma silenciosa.
    generator = getattr(loaders["train"], "generator", None)
    if generator is not None:
        generator.manual_seed(config.seed)

    stopper = EarlyStopping(
        patience=config.patience,
        min_delta=config.early_stopping_min_delta,
        monitor=config.early_stopping_monitor,
        target=config.early_stopping_target,
        restore_best_weights=config.restore_best_weights,
    )
    history = History()
    epoch_times: list[float] = []
    stopped_early = False

    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_acc = run_epoch(
            model, loaders["train"], criterion, device, optimizer,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )
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

        monitored_value = val_acc if config.early_stopping_monitor == "val_accuracy" else val_loss
        if stopper.step(monitored_value, model, epoch):
            stopped_early = True
            break

    return TrainResult(
        history=history,
        epoch_times=epoch_times,
        total_time=time.perf_counter() - started,
        stopped_early=stopped_early,
        best_val_loss=float(min(history.val_loss)),
        best_epoch=stopper.best_epoch,
        best_monitor_value=float(stopper.best),
        early_stopping_monitor=config.early_stopping_monitor,
        early_stopping_min_delta=config.early_stopping_min_delta,
        patience=config.patience,
        restore_best_weights=config.restore_best_weights,
        early_stopping_target=config.early_stopping_target,
        target_reached=stopper.target_reached,
        stop_reason=stopper.stop_reason if stopped_early else "limite de épocas atingido",
    )
