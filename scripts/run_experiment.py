#!/usr/bin/env python3
"""Executa um experimento completo sem notebook.

Útil para rodar em instância AWS via SSH, para varrer configurações e para
verificar reprodutibilidade — é exatamente o mesmo pipeline que o notebook usa.

Uso:
    python scripts/run_experiment.py                       # PaviaU, duas entradas
    python scripts/run_experiment.py -d IP --no-two-input  # Indian Pines, só espectro
    python scripts/run_experiment.py --epochs 5 --quiet
    python scripts/run_experiment.py --json outputs/pu.json
    python scripts/run_experiment.py --backend thread        # preparação paralela
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hsi import BACKENDS, DATASETS, ExperimentConfig, output_dir, run_experiment  # noqa: E402
from hsi.checkpoint import save_checkpoint  # noqa: E402
from hsi.plots import plot_history  # noqa: E402


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        dataset=args.dataset,
        two_input=not args.no_two_input,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        deterministic=not args.fast,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", "-d", default="PU", choices=sorted(DATASETS),
                        help="cena a usar (padrão: PU)")
    parser.add_argument("--no-two-input", action="store_true",
                        help="treinar só com o espectro, sem a segmentação SEEDS")
    parser.add_argument("--epochs", type=int, default=60, help="máximo de épocas (padrão: 60)")
    parser.add_argument("--batch-size", type=int, default=32, help="tamanho do batch (padrão: 32)")
    parser.add_argument("--seed", type=int, default=42, help="semente (padrão: 42)")
    parser.add_argument("--fast", action="store_true",
                        help="desliga o modo determinístico (mais rápido, resultado pode variar)")
    parser.add_argument("--out", type=Path, default=None, help="diretório de saída (padrão: outputs/)")
    parser.add_argument("--json", type=Path, default=None, help="grava as métricas em JSON neste caminho")
    parser.add_argument("--no-save", action="store_true", help="não gravar checkpoint nem curvas")
    parser.add_argument("--backend", "-b", default="sequential", choices=BACKENDS,
                        help="execução dos ramos independentes da preparação (padrão: sequential)")
    parser.add_argument("--quiet", "-q", action="store_true", help="só imprime o resultado final")
    args = parser.parse_args()

    config = build_config(args)
    result = run_experiment(config, backend=args.backend, verbose=not args.quiet)

    if args.quiet:
        print(config.summary())
        print(result.prep_timings.summary())
        print(result.metrics.summary())

    out = args.out or output_dir()
    if not args.no_save:
        out.mkdir(parents=True, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        for metric in ("Loss", "Accuracy"):
            plot_history(result.training.history, metric,
                         save_to=out / f"{config.model_name}_{metric}_curve_torch.pdf")
        path = save_checkpoint(result.model, config, result.data.n_features, result.data.n_classes,
                               out / f"{config.model_name}_torch.pt")
        print("\nModelo salvo em", path)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "config": config.to_dict(),
            "metrics": {
                "overall_accuracy": result.metrics.overall_accuracy,
                "average_accuracy": result.metrics.average_accuracy,
                "kappa": result.metrics.kappa,
                "test_loss": result.metrics.test_loss,
                "test_accuracy": result.metrics.test_accuracy,
                "per_class_accuracy": result.metrics.per_class_accuracy.tolist(),
            },
            "history": result.training.history.to_dict(),
            "epochs_run": len(result.training.history),
            "prep_timings": {
                "backend": result.prep_timings.backend,
                "wall": result.prep_timings.wall,
                "stages": result.prep_timings.stages,
            },
            "train_time": result.training.total_time,
            "stopped_early": result.training.stopped_early,
            "best_val_loss": result.training.best_val_loss,
        }, indent=2))
        print("Métricas em", args.json)

    return 0


if __name__ == "__main__":
    sys.exit(main())
