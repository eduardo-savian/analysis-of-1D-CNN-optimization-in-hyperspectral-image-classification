#!/usr/bin/env python3
"""Mede a paralelização inter-etapas da preparação dos dados.

Compara os backends `sequential`, `thread` e `process` (ver `hsi.parallel`) em cada
cena, com repetições, e reporta o speedup. Com `--with-training`, mede também o
treino, para pôr o ganho em perspectiva: a lei de Amdahl limita o efeito de acelerar
a preparação ao peso dela no tempo total.

Uso:
    python scripts/benchmark_parallel.py                      # as três cenas, 5 repetições
    python scripts/benchmark_parallel.py -d PU -n 10
    python scripts/benchmark_parallel.py --with-training      # inclui o treino no total
    python scripts/benchmark_parallel.py --json outputs/paralelismo.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from hsi import BACKENDS, DATASETS, ExperimentConfig, load_scene, prepare_data_parallel  # noqa: E402


def measure(config: ExperimentConfig, scene, backend: str, repeats: int, warmup: int) -> dict:
    """Roda um backend `repeats` vezes e devolve as estatísticas dos tempos."""
    for _ in range(warmup):
        prepare_data_parallel(config, scene=scene, backend=backend)

    walls, spectral, spatial, results = [], [], [], []
    for _ in range(repeats):
        data, artifacts, timings = prepare_data_parallel(config, scene=scene, backend=backend)
        walls.append(timings.wall)
        spectral.append(timings.stages.get("espectral", 0.0))
        spatial.append(timings.stages.get("espacial", 0.0))
        results.append((data.spectra, data.segments))

    # Todas as repetições têm de produzir exatamente o mesmo resultado.
    first = results[0]
    estavel = all(np.array_equal(a, b) for r in results[1:] for a, b in zip(first, r))

    return {
        "backend": backend,
        "wall_median": statistics.median(walls),
        "wall_min": min(walls),
        "wall_mean": statistics.mean(walls),
        "wall_stdev": statistics.stdev(walls) if len(walls) > 1 else 0.0,
        "espectral_median": statistics.median(spectral),
        "espacial_median": statistics.median(spatial),
        "walls": walls,
        "estavel": estavel,
        "n_superpixels": artifacts.n_superpixels if artifacts else None,
    }


def train_once(config: ExperimentConfig, backend: str) -> dict:
    """Execução completa, para saber quanto a preparação pesa no total."""
    from hsi import run_experiment

    started = time.perf_counter()
    result = run_experiment(config, backend=backend, verbose=False)
    total = time.perf_counter() - started
    return {
        "total": total,
        "prep": result.prep_timings.wall,
        "treino": result.training.total_time,
        "epocas": len(result.training.history),
        "oa": result.metrics.overall_accuracy,
        "kappa": result.metrics.kappa,
    }


def report_dataset(dataset: str, rows: list[dict], repeats: int) -> None:
    baseline = next(r for r in rows if r["backend"] == "sequential")
    print(f"\n{dataset} — {DATASETS[dataset]}  ({repeats} repetições, mediana)")
    print(f"  {'backend':11s} {'bloco':>9s} {'espectral':>10s} {'espacial':>9s} "
          f"{'soma ramos':>11s} {'speedup':>8s} {'overhead':>9s}")
    print("  " + "-" * 72)
    for row in rows:
        soma = row["espectral_median"] + row["espacial_median"]
        speedup = baseline["wall_median"] / row["wall_median"]
        overhead = row["wall_median"] - max(row["espectral_median"], row["espacial_median"])
        marca = "" if row["estavel"] else "  [!] resultado instável entre repetições"
        print(f"  {row['backend']:11s} {row['wall_median']:8.3f}s {row['espectral_median']:9.3f}s "
              f"{row['espacial_median']:8.3f}s {soma:10.3f}s {speedup:7.2f}x {overhead:+8.3f}s{marca}")

    ideal = max(baseline["espectral_median"], baseline["espacial_median"])
    print(f"  teto teórico (max dos ramos, overhead zero): {ideal:.3f}s "
          f"-> {baseline['wall_median'] / ideal:.2f}x")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", "-d", action="append", choices=sorted(DATASETS),
                        help="cena a medir (repetível); padrão: todas")
    parser.add_argument("--repeats", "-n", type=int, default=5, help="repetições por backend (padrão: 5)")
    parser.add_argument("--warmup", type=int, default=1, help="execuções descartadas antes de medir (padrão: 1)")
    parser.add_argument("--backend", "-b", action="append", choices=BACKENDS,
                        help="backend a medir (repetível); padrão: todos")
    parser.add_argument("--with-training", action="store_true",
                        help="mede também um treino completo, para o peso da preparação no total")
    parser.add_argument("--json", type=Path, default=None, help="grava os resultados em JSON")
    args = parser.parse_args()

    datasets = args.dataset or sorted(DATASETS)
    backends = args.backend or list(BACKENDS)

    print(f"Máquina: {platform.processor() or platform.machine()} | Python {sys.version.split()[0]}")
    print(f"Backends: {', '.join(backends)} | repetições: {args.repeats} (+{args.warmup} de aquecimento)")

    saida = {"datasets": {}, "with_training": args.with_training}
    for dataset in datasets:
        config = ExperimentConfig(dataset=dataset)
        scene = load_scene(dataset)
        rows = [measure(config, scene, backend, args.repeats, args.warmup) for backend in backends]
        report_dataset(dataset, rows, args.repeats)
        saida["datasets"][dataset] = {"prep": rows}

        if args.with_training:
            treino = train_once(config, backend="thread")
            peso = 100 * treino["prep"] / treino["total"]
            baseline = next(r for r in rows if r["backend"] == "sequential")
            melhor = min(rows, key=lambda r: r["wall_median"])
            ganho = baseline["wall_median"] - melhor["wall_median"]
            print(f"  execução completa: {treino['total']:.1f}s "
                  f"(preparação {treino['prep']:.2f}s = {peso:.1f}%, "
                  f"treino {treino['treino']:.1f}s em {treino['epocas']} épocas)")
            print(f"  ganho do melhor backend no total: {ganho:.3f}s "
                  f"= {100 * ganho / treino['total']:.2f}% do tempo de ponta a ponta")
            saida["datasets"][dataset]["treino"] = treino

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(saida, indent=2))
        print("\nResultados em", args.json)

    return 0


if __name__ == "__main__":
    sys.exit(main())
