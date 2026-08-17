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

__all__ = ["Splits", "PreparedData", "split_indices", "validate_prepared_data", "build_loaders"]


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
    #: (N, 3, P, P) para CNN2D; None quando não foi solicitada.
    patches_2d: np.ndarray | None = None
    #: (N, 1, B, P, P) para CNN3D; None quando não foi solicitada.
    patches_3d: np.ndarray | None = None

    @property
    def n_features(self) -> int:
        return self.spectra.shape[1]

    @property
    def n_classes(self) -> int:
        return int(self.labels.max()) + 1

    @property
    def two_input(self) -> bool:
        return self.segments is not None

    def input_shape(self, model_type: str) -> tuple[int, ...]:
        """Shape de uma amostra já no layout channels-first da arquitetura.

        A validação acontece aqui, perto dos dados, para que pedir CNN2D/3D sem
        os patches correspondentes produza erro claro em vez de falhar dentro de
        uma convolução PyTorch.
        """
        if model_type == "cnn1d":
            return (1, self.n_features)
        patches = self.patches_2d if model_type == "cnn2d" else self.patches_3d
        if patches is None:
            raise ValueError(f"Dados para {model_type} não foram preparados.")
        return tuple(patches.shape[1:])


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


def validate_prepared_data(data: PreparedData) -> dict[str, int | bool]:
    """Aplica o contrato de dados antes de alocar GPU ou iniciar o treino.

    A validação concentra falhas comuns de experimentos: arrays desalinhados,
    NaN/infinito após PCA, classes fora de ``0..C-1`` e splits com repetição ou
    amostras perdidas. Retorna um resumo serializável para o tracker; levanta
    ``ValueError`` com uma causa objetiva quando o contrato não é atendido.
    """
    n_samples = len(data.labels)
    if data.spectra.ndim != 2 or len(data.spectra) != n_samples:
        raise ValueError("spectra deve ser uma matriz com uma linha por rótulo.")
    if data.labels.ndim != 1 or n_samples == 0:
        raise ValueError("labels deve ser um vetor não vazio.")
    if not np.isfinite(data.spectra).all():
        raise ValueError("spectra contém NaN ou infinito.")
    expected_classes = np.arange(data.n_classes)
    if not np.array_equal(np.unique(data.labels), expected_classes):
        raise ValueError("labels deve conter classes contíguas de 0 até C-1.")
    if data.segments is not None and (len(data.segments) != n_samples or not np.isfinite(data.segments).all()):
        raise ValueError("segments deve ser finito e ter uma entrada por rótulo.")
    for name, patches, dimensions in (("patches_2d", data.patches_2d, 4), ("patches_3d", data.patches_3d, 5)):
        if patches is not None and (patches.ndim != dimensions or len(patches) != n_samples or not np.isfinite(patches).all()):
            raise ValueError(f"{name} tem shape inválido, NaN/infinito ou quantidade desalinhada.")

    all_indices = np.concatenate([data.splits.train, data.splits.val, data.splits.test])
    if len(all_indices) != n_samples or len(np.unique(all_indices)) != n_samples:
        raise ValueError("splits deve particionar cada amostra exatamente uma vez.")
    if all_indices.min(initial=0) < 0 or all_indices.max(initial=-1) >= n_samples:
        raise ValueError("splits contém índice fora dos dados preparados.")
    return {
        "valid": True,
        "samples": n_samples,
        "classes": data.n_classes,
        "has_segments": data.segments is not None,
        "has_patches_2d": data.patches_2d is not None,
        "has_patches_3d": data.patches_3d is not None,
    }


def _tensors(data: PreparedData, index: np.ndarray, model_type: str) -> tuple[torch.Tensor, ...]:
    """Fatia uma partição e converte para os tensores que o modelo consome.

    CNN1D recebe espectro ``(N, 1, Kb)``, CNN2D recebe ``(N, 3, P, P)`` e CNN3D
    recebe ``(N, 1, B, P, P)``. A segmentação continua um escalar por amostra,
    concatenado depois do achatamento das convoluções em qualquer arquitetura.
    """
    if model_type == "cnn1d":
        features = torch.from_numpy(np.ascontiguousarray(data.spectra[index])).float().unsqueeze(1)
    elif model_type == "cnn2d" and data.patches_2d is not None:
        features = torch.from_numpy(np.ascontiguousarray(data.patches_2d[index])).float()
    elif model_type == "cnn3d" and data.patches_3d is not None:
        features = torch.from_numpy(np.ascontiguousarray(data.patches_3d[index])).float()
    else:
        raise ValueError(f"Dados para {model_type} não foram preparados.")
    labels = torch.from_numpy(np.ascontiguousarray(data.labels[index]))
    if data.segments is None:
        return features, labels
    segments = torch.from_numpy(np.ascontiguousarray(data.segments[index])).float().reshape(-1, 1)
    return features, segments, labels


def build_loaders(data: PreparedData, config, device) -> dict[str, DataLoader]:
    """DataLoaders de treino, validação e teste.

    O loader de treino recebe um `generator` semeado explicitamente: a ordem dos
    batches passa a depender só de `config.seed`, e não do estado global do RNG.
    Criar um iterador sobre esse loader (para inspecionar um batch, por exemplo)
    adianta o gerador — por isso `fit` repõe a semente antes da primeira época.

    Em CUDA, ``pin_memory=True`` torna a transferência CPU→GPU não bloqueante em
    ``unpack_batch``. ``prefetch_factor`` e ``persistent_workers`` só são
    passados quando há workers, pois o PyTorch os rejeita/ignora com zero workers.
    Validação e teste podem usar batch maior porque não armazenam gradientes.
    """
    pin = device.type == "cuda"
    datasets = {
        name: TensorDataset(*_tensors(data, index, config.model_type))
        for name, index in (
            ("train", data.splits.train),
            ("val", data.splits.val),
            ("test", data.splits.test),
        )
    }
    worker_options = {}
    if config.num_workers:
        worker_options = {
            "prefetch_factor": config.prefetch_factor,
            "persistent_workers": config.persistent_workers,
        }

    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=config.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(config.seed),
            pin_memory=pin, num_workers=config.num_workers, drop_last=False,
            **worker_options,
        )
    }
    for name in ("val", "test"):
        loaders[name] = DataLoader(
            datasets[name], batch_size=config.resolved_eval_batch_size, shuffle=False,
            pin_memory=pin, num_workers=config.num_workers,
            **worker_options,
        )
    return loaders
