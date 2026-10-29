"""Pipeline completo: cena crua -> dados prontos -> modelo treinado -> métricas.

Este módulo é a única peça que conhece a ordem das etapas. O notebook, o HPO e os
experimentos — no Jupyter ou pelo lançador `experimentos/executar.py` — chamam estas
funções, e não uma cópia delas.

A preparação dos dados vive em `prepare_data_parallel` (`hsi/paralelo/ramos.py`), que a executa com
os dois ramos (espectral e espacial) em sequência ou concorrentemente, conforme o
backend.
"""

import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from hsi.ambiente import configure_determinism, seed_everything, select_device
from hsi.avaliacao import Metrics, evaluate
from hsi.config import ExperimentConfig
from hsi.dados.cenas import Scene
from hsi.dados.particoes import PreparedData, build_loaders, validate_prepared_data
from hsi.dados.segmentacao import SegmentationArtifacts
from hsi.modelo import build_model
from hsi.paralelo.ramos import StageTimings, prepare_data_parallel
from hsi.treino import TrainResult, fit


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
    `BACKENDS`. O resultado é o mesmo em qualquer um deles — use
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


#: Campos de `PreparedData` que dominam a memória e não são necessários para
#: relatar um trial já avaliado. `labels` e `splits` ficam: são pequenos e vários
#: relatórios os consultam.
_HEAVY_DATA_FIELDS = ("spectra", "segments", "coords")


def result_footprint(result: "ExperimentResult") -> dict[str, float]:
    """Quantos MB de arrays um resultado está segurando, por campo.

    Serve para justificar com número — e não com intuição — quando vale a pena
    liberar um resultado em vez de retê-lo.
    """
    data = getattr(result, "data", None)
    sizes = {}
    for field in _HEAVY_DATA_FIELDS:
        array = getattr(data, field, None) if data is not None else None
        if array is not None:
            sizes[f"data.{field}"] = array.nbytes / 1e6
    model = getattr(result, "model", None)
    if model is not None:
        sizes["modelo"] = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    if getattr(result, "predicted", None) is not None:
        sizes["predicted"] = result.predicted.nbytes / 1e6
    sizes["total"] = sum(sizes.values())
    return sizes


def release_result(result: "ExperimentResult | None", keep_model: bool = False) -> None:
    """Solta os arrays pesados de um resultado que não será mais analisado.

    Por que isso existe: uma campanha de HPO instancia dezenas de
    `ExperimentResult`, e cada um segura sua própria cópia dos dados preparados.
    Em Indian Pines são 2,5 MB por trial; em Salinas, 6,5 MB. Reter dezenas de
    trials acumula sem necessidade, e o custo cresce com o tamanho da cena.

    A prática correta é a que o orientador apontou: instanciar, treinar, extrair
    as métricas (que são kilobytes), persistir o checkpoint em disco se o modelo
    ainda for necessário, e liberar o resto antes do próximo trial.

    O que sobrevive: config, métricas, matriz de confusão, histórico de treino,
    tempos e splits — tudo o que os relatórios e os PDFs consomem.
    """
    if result is None:
        return
    data = getattr(result, "data", None)
    if data is not None:
        for field in _HEAVY_DATA_FIELDS:
            # PreparedData é frozen: a atribuição precisa contornar o __setattr__.
            object.__setattr__(data, field, None)
    if not keep_model:
        result.model = None
    result.predicted = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def run_experiment(
    config: ExperimentConfig | None = None,
    data_dir: Path | None = None,
    backend: str = "sequential",
    verbose: bool = True,
) -> ExperimentResult:
    """Execução de ponta a ponta — a que a seção 8, o HPO e os experimentos chamam."""
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
    # O layout vem de `PreparedData.input_shape`: (1, Kb), channels-first, na
    # CNN 1D. O modelo nunca precisa inferir ou rearranjar dados sozinho.
    model, criterion, optimizer = build_model(
        config, data.input_shape(config.model_type), data.n_classes, device,
        n_segments=data.n_segments,
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
