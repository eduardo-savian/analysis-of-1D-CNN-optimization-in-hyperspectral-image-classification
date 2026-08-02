"""Divisão treino/validação/teste e construção dos DataLoaders.

O notebook original empilhava espectro e segmentação num tensor `(N, Kb, 1, 1, 2)`
só para conseguir passar os dois pelo mesmo `train_test_split`, e depois desfazia
o empilhamento com fatiamento posicional. Aqui divide-se o **índice** uma vez e
cada array é indexado por ele — mesma partição, sem o malabarismo de shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

__all__ = ["Splits", "PreparedData", "split_indices", "build_loaders"]


@dataclass(frozen=True)
class Splits:
    """Índices das três partições, referentes às linhas de `PreparedData`."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def sizes(self) -> dict[str, int]:
        return {"treino": len(self.train), "validação": len(self.val), "teste": len(self.test)}


@dataclass(frozen=True)
class PreparedData:
    """Dados prontos para o modelo, já sem background e com PCA aplicado.

    spectra:  (N, Kb) float64 — componentes principais do espectro
    labels:   (N,)    int64   — classes reindexadas para 0..C-1
    segments: (N,)    ou None — id do superpixel de cada pixel (segunda entrada)
    """

    spectra: np.ndarray
    labels: np.ndarray
    segments: np.ndarray | None
    splits: Splits
    scene_shape: tuple[int, int]

    @property
    def n_features(self) -> int:
        return self.spectra.shape[1]

    @property
    def n_classes(self) -> int:
        return int(self.labels.max()) + 1

    @property
    def two_input(self) -> bool:
        return self.segments is not None


def split_indices(labels: np.ndarray, test_ratio: float, valid_ratio: float, seed: int = 42) -> Splits:
    """Split estratificado em duas etapas, como no original.

    Primeiro separa treino de (validação + teste); depois reparte esse resto.
    `stratify` mantém a proporção de cada classe nas três partições — necessário
    porque as cenas são bastante desbalanceadas.
    """
    index = np.arange(len(labels))
    train, rest = train_test_split(
        index, test_size=test_ratio, random_state=seed, stratify=labels
    )
    # Atenção à ordem: no original o primeiro retorno desta segunda chamada é o
    # conjunto de *teste* e o segundo é o de validação.
    test, val = train_test_split(
        rest, test_size=valid_ratio, random_state=seed, stratify=labels[rest]
    )
    return Splits(train=train, val=val, test=test)


def _tensors(data: PreparedData, index: np.ndarray) -> tuple[torch.Tensor, ...]:
    """Fatia uma partição e converte para os tensores que o modelo consome.

    O espectro vira `(N, 1, Kb)` — o layout *channels-first* que o `nn.Conv1d`
    espera: uma sequência de comprimento Kb com um canal só. A segmentação é um
    escalar por amostra, concatenado depois do achatamento das convoluções.
    """
    spectra = torch.from_numpy(np.ascontiguousarray(data.spectra[index])).float().unsqueeze(1)
    labels = torch.from_numpy(np.ascontiguousarray(data.labels[index]))
    if data.segments is None:
        return spectra, labels
    segments = torch.from_numpy(np.ascontiguousarray(data.segments[index])).float().reshape(-1, 1)
    return spectra, segments, labels


def build_loaders(data: PreparedData, config, device) -> dict[str, DataLoader]:
    """DataLoaders de treino, validação e teste.

    O loader de treino recebe um `generator` semeado explicitamente: a ordem dos
    batches passa a depender só de `config.seed`, e não do estado global do RNG.
    Criar um iterador sobre esse loader (para inspecionar um batch, por exemplo)
    adianta o gerador — por isso `fit` repõe a semente antes da primeira época.
    """
    pin = device.type == "cuda"
    datasets = {
        name: TensorDataset(*_tensors(data, index))
        for name, index in (
            ("train", data.splits.train),
            ("val", data.splits.val),
            ("test", data.splits.test),
        )
    }
    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=config.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(config.seed),
            pin_memory=pin, num_workers=config.num_workers, drop_last=False,
        )
    }
    for name in ("val", "test"):
        loaders[name] = DataLoader(
            datasets[name], batch_size=config.batch_size, shuffle=False,
            pin_memory=pin, num_workers=config.num_workers,
        )
    return loaders
