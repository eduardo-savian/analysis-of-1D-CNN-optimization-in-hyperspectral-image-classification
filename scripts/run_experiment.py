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

from hsi import (  # noqa: E402
    BACKENDS, DATASETS, ExperimentConfig, MODEL_TYPES, SEGMENTERS, create_run, fail_run, output_dir, record_run,
    result_metrics, run_experiment,
)
from hsi.checkpoint import save_checkpoint  # noqa: E402
from hsi.plots import plot_history  # noqa: E402
from hsi.registry import promote_candidate  # noqa: E402


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    """Converte argumentos da CLI na única receita aceita pelo pipeline.

    Concentrar o mapeamento aqui impede que notebook, CLI e funções internas
    tenham interpretações diferentes para batch, determinismo ou DataLoader.
    ``ExperimentConfig`` valida os valores antes de qualquer treinamento.
    """
    return ExperimentConfig(
        dataset=args.dataset,
        two_input=not args.no_two_input,
        model_type=args.model,
        segmenter=args.segmenter,
        patch_size=args.patch_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.accumulate_steps,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        seed=args.seed,
        deterministic=not args.fast,
    )


def main() -> int:
    """Orquestra a execução, persistência de artefatos e fechamento MLOps.

    A ordem importa: ``create_run`` registra a receita antes de gastar GPU;
    qualquer exceção de ``run_experiment`` passa por ``fail_run``; somente após
    checkpoint e curvas existirem ``record_run`` marca o manifesto como sucesso.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", "-d", default="PU", choices=sorted(DATASETS),
                        help="cena a usar (padrão: PU)")
    parser.add_argument("--no-two-input", action="store_true",
                        help="desligar a fusão por id de superpixel")
    parser.add_argument("--model", default="cnn1d", choices=MODEL_TYPES,
                        help="arquitetura: cnn1d (baseline), cnn2d ou cnn3d")
    parser.add_argument("--segmenter", default="seeds", choices=SEGMENTERS,
                        help="superpixel para fusão: seeds, slic ou lsc")
    parser.add_argument("--patch-size", type=int, default=5,
                        help="lado ímpar do patch espacial usado por cnn2d/cnn3d (padrão: 5)")
    parser.add_argument("--epochs", type=int, default=60, help="máximo de épocas (padrão: 60)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="batch físico na GPU (padrão: 32; meça com benchmark_batches.py)")
    parser.add_argument("--eval-batch-size", type=int, default=0,
                        help="batch de validação/teste; 0 reutiliza --batch-size")
    parser.add_argument("--accumulate-steps", type=int, default=1,
                        help="batches a acumular por update (batch efetivo = batch-size × este valor)")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="workers do DataLoader; mantenha 0 até uma medição justificar outro valor")
    parser.add_argument("--prefetch-factor", type=int, default=2,
                        help="batches pré-carregados por worker quando num-workers > 0")
    parser.add_argument("--persistent-workers", action="store_true",
                        help="mantém workers do DataLoader entre épocas (só com num-workers > 0)")
    parser.add_argument("--seed", type=int, default=42, help="semente (padrão: 42)")
    parser.add_argument("--fast", action="store_true",
                        help="desliga o modo determinístico (mais rápido, resultado pode variar)")
    parser.add_argument("--out", type=Path, default=None,
                        help="diretório de artefatos; padrão: outputs/runs/<run-id>/")
    parser.add_argument("--json", type=Path, default=None, help="grava as métricas em JSON neste caminho")
    parser.add_argument("--no-save", action="store_true", help="não gravar checkpoint nem curvas")
    parser.add_argument("--promote", action="store_true",
                        help="promover para o registry local se superar o campeão comparável")
    parser.add_argument("--no-track", action="store_true",
                        help="não criar outputs/runs/<run-id> com a trilha MLOps")
    parser.add_argument("--backend", "-b", default="sequential", choices=BACKENDS,
                        help="execução dos ramos independentes da preparação (padrão: sequential)")
    parser.add_argument("--quiet", "-q", action="store_true", help="só imprime o resultado final")
    args = parser.parse_args()

    config = build_config(args)
    if args.promote and (args.no_track or args.no_save):
        parser.error("--promote requer rastreamento e checkpoint (não use --no-track/--no-save).")
    run = None if args.no_track else create_run(config)
    try:
        result = run_experiment(config, backend=args.backend, verbose=not args.quiet)
    except Exception as error:
        if run:
            fail_run(run, error)
        raise

    if args.quiet:
        print(config.summary())
        print(result.prep_timings.summary())
        print(result.metrics.summary())

    # No modo rastreado cada execução possui diretório próprio: elimina a
    # sobrescrita silenciosa de checkpoint/curvas entre experimentos.
    out = args.out or (run.directory if run else output_dir())
    artifacts = {}
    if not args.no_save:
        out.mkdir(parents=True, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        for metric in ("Loss", "Accuracy"):
            plot_history(result.training.history, metric,
                         save_to=out / f"{config.model_name}_{metric}_curve_torch.pdf")
        path = save_checkpoint(result.model, config, result.data.input_shape(config.model_type), result.data.n_classes,
                               out / f"{config.model_name}_torch.pt")
        artifacts["checkpoint"] = path
        artifacts["loss_curve"] = out / f"{config.model_name}_Loss_curve_torch.pdf"
        artifacts["accuracy_curve"] = out / f"{config.model_name}_Accuracy_curve_torch.pdf"
        print("\nModelo salvo em", path)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"config": config.to_dict(), **result_metrics(result)}, indent=2))
        print("Métricas em", args.json)

    if run:
        record_run(run, result, artifacts)
        print("Execução rastreada em", run.directory)
        if args.promote:
            decision = promote_candidate(run, result, artifacts["checkpoint"])
            status = "PROMOVIDO" if decision.promoted else "não promovido"
            print(f"Registry: {status} — {decision.reason} ({decision.registry_path})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
