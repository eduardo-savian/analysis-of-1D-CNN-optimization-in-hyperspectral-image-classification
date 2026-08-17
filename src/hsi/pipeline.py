"""Pipeline completo: cena crua -> dados prontos -> modelo treinado -> métricas.

Este módulo é a única peça que conhece a ordem das etapas. O notebook e o CLI
`scripts/run_experiment.py` são só dois frontends para as mesmas funções, o que
garante que os dois produzam exatamente o mesmo resultado.

A preparação dos dados vive em `hsi.parallel`, que a executa com os dois ramos
(espectral e espacial) em sequência ou concorrentemente, conforme o backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .data import Scene
from .dataset import PreparedData, build_loaders, validate_prepared_data
from .env import configure_determinism, seed_everything, select_device
from .evaluate import Metrics, evaluate
from .model import build_model
from .parallel import StageTimings, prepare_data_parallel
from .segmentation import SegmentationArtifacts
from .train import TrainResult, fit

__all__ = ["prepare_data", "run_experiment", "ExperimentResult"]


def prepare_data(
    config: ExperimentConfig,
    data_dir: Path | None = None,
    scene: Scene | None = None,
    backend: str = "sequential",
) -> tuple[PreparedData, SegmentationArtifacts | None]:
    """Da cena em disco até os arrays prontos para virar tensores.

    Etapas: carga -> tabela de pixels -> remoção do background -> PCA ->
    (opcional) segmentação SEEDS -> split estratificado.

    `backend` seleciona como os dois ramos independentes são executados; veja
    `hsi.parallel`. O resultado é o mesmo em qualquer um deles — use
    `prepare_data_parallel` diretamente quando quiser também os tempos.
    """
    data, artifacts, _ = prepare_data_parallel(config, data_dir, scene, backend=backend)
    return data, artifacts


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    data: PreparedData
    model: object
    training: TrainResult
    metrics: Metrics
    predicted: np.ndarray
    device: object
    prep_timings: StageTimings
    #: Resultado do contrato de dados validado antes de construir DataLoaders.
    data_contract: dict


def run_experiment(
    config: ExperimentConfig | None = None,
    data_dir: Path | None = None,
    backend: str = "sequential",
    verbose: bool = True,
) -> ExperimentResult:
    """Execução completa de ponta a ponta — o que o CLI chama."""
    config = config or ExperimentConfig()

    device = select_device(verbose=verbose)
    configure_determinism(config.deterministic, verbose=verbose)

    data, _, prep_timings = prepare_data_parallel(config, data_dir, backend=backend)
    data_contract = validate_prepared_data(data)
    if verbose:
        print("\n" + config.summary())
        print("Amostras:", data.splits.sizes(), "| classes:", data.n_classes)
        print("\n" + prep_timings.summary())

    loaders = build_loaders(data, config, device)

    # A semente é reposta imediatamente antes de construir o modelo: são os pesos
    # iniciais que dependem do RNG global do PyTorch.
    seed_everything(config.seed)
    # Cada arquitetura declara seu próprio layout: 1D=(1,Kb), 2D=(3,P,P) e
    # 3D=(1,B,P,P). O modelo nunca precisa inferir ou rearranjar dados sozinho.
    model, criterion, optimizer = build_model(
        config, data.input_shape(config.model_type), data.n_classes, device,
    )
    if verbose:
        print("\nParâmetros treináveis:", model.n_parameters)

    training = fit(model, loaders, criterion, optimizer, config, device, verbose=verbose)
    if verbose:
        print("\n" + training.summary())

    metrics, predicted = evaluate(
        model, loaders["test"], criterion, device, data.labels[data.splits.test]
    )
    if verbose:
        print("\n" + metrics.report)
        print(metrics.summary())

    return ExperimentResult(
        config=config,
        data=data,
        model=model,
        training=training,
        metrics=metrics,
        predicted=predicted,
        device=device,
        prep_timings=prep_timings,
        data_contract=data_contract,
    )
