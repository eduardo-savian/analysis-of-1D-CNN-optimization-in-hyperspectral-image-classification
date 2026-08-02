"""Classificação de imagens hiperespectrais com CNN 1D.

Implementação em PyTorch do TCC de Sydney Matheus de Souza, reorganizada como
biblioteca: o notebook e o CLI `scripts/run_experiment.py` são frontends das
mesmas funções, e por isso produzem resultados idênticos.

Uso típico:

    from hsi import ExperimentConfig, run_experiment

    result = run_experiment(ExperimentConfig(dataset="PU", two_input=True))
    print(result.metrics.summary())
"""

from __future__ import annotations

from . import plots
from .config import DATASETS, ExperimentConfig
from .data import Scene, load_scene, remove_background, to_pixel_table
from .dataset import PreparedData, Splits, build_loaders, split_indices
from .env import check_versions, configure_determinism, seed_everything, select_device
from .evaluate import Metrics, binary_readout, evaluate, predict_proba
from .checkpoint import load_checkpoint, save_checkpoint
from .model import CNN1D, build_model
from .parallel import BACKENDS, StageTimings, prepare_data_parallel
from .paths import data_dir, output_dir, repo_root
from .pipeline import ExperimentResult, prepare_data, run_experiment
from .preprocess import apply_pca, mat2gray, rgb_composite
from .segmentation import (
    SEEDS_PARAMS,
    SegmentationArtifacts,
    SegmentationResult,
    SeedsParams,
    colorize_segments,
    contour_overlay,
    superpixel_labels,
)
from .train import EarlyStopping, History, TrainResult, fit, run_epoch
from .variants import BASELINE, OPTIMIZED, VARIANTS, Variant, VariantRun, run_variant

__version__ = "1.0.0"

__all__ = [
    "plots",
    "DATASETS", "ExperimentConfig",
    "Scene", "load_scene", "remove_background", "to_pixel_table",
    "PreparedData", "Splits", "build_loaders", "split_indices",
    "check_versions", "configure_determinism", "seed_everything", "select_device",
    "Metrics", "binary_readout", "evaluate", "predict_proba",
    "load_checkpoint", "save_checkpoint",
    "CNN1D", "build_model",
    "BACKENDS", "StageTimings", "prepare_data_parallel",
    "data_dir", "output_dir", "repo_root",
    "ExperimentResult", "prepare_data", "run_experiment",
    "apply_pca", "mat2gray", "rgb_composite",
    "SEEDS_PARAMS", "SeedsParams", "SegmentationArtifacts", "SegmentationResult",
    "colorize_segments", "contour_overlay", "superpixel_labels",
    "EarlyStopping", "History", "TrainResult", "fit", "run_epoch",
    "BASELINE", "OPTIMIZED", "VARIANTS", "Variant", "VariantRun", "run_variant",
]
