#!/usr/bin/env python3
"""Escolhe batch físico para a GPU a partir de uma medição do pipeline real.

O modelo é pequeno: aumentar o batch costuma reduzir lançamentos de kernel por
amostra. Este script mede uma época de treino por candidato, descarta OOM e salva
a decisão com o ambiente. Ele não mede acurácia nem substitui o experimento final.

Uso:
    python scripts/benchmark_batches.py -d PU --model cnn1d
    python scripts/benchmark_batches.py -d IP --model cnn3d --patch-size 5 --batches 16 32 64
    python scripts/benchmark_batches.py --batches 32 64 128 256 512 --json outputs/batches-pu.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

from hsi import (  # noqa: E402
    DATASETS, MODEL_TYPES, SEGMENTERS, ExperimentConfig, build_loaders, build_model, configure_determinism,
    prepare_data, run_epoch, seed_everything, select_device,
)


def synchronize(device: torch.device) -> None:
    """Espera a fila CUDA antes de iniciar/parar um cronômetro.

    Chamadas CUDA são assíncronas: sem essa barreira, ``perf_counter`` mediria só
    o lançamento de kernels, e não a época realmente executada pela GPU. Em CPU
    não há fila assíncrona e a função é propositalmente um no-op.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()


def measure(config: ExperimentConfig, data, device: torch.device, repeats: int, warmup: int) -> dict:
    """Mede throughput de um batch físico no pipeline de treino real.

    Uma época de aquecimento força a criação de contexto CUDA, autotuning e
    alocações iniciais; ela não entra na estatística. Cada repetição mede a época
    inteira e repõe o gerador do loader para não comparar ordens de batch
    diferentes. Falta de memória é um resultado esperado de exploração, por isso
    volta como ``status=oom`` em vez de abortar todos os candidatos.
    """
    try:
        loaders = build_loaders(data, config, device)
        seed_everything(config.seed)
        model, criterion, optimizer = build_model(
            config, data.input_shape(config.model_type), data.n_classes, device,
        )

        for _ in range(warmup):
            loaders["train"].generator.manual_seed(config.seed)
            run_epoch(model, loaders["train"], criterion, device, optimizer)
        synchronize(device)

        elapsed = []
        for _ in range(repeats):
            loaders["train"].generator.manual_seed(config.seed)
            synchronize(device)
            started = time.perf_counter()
            run_epoch(model, loaders["train"], criterion, device, optimizer)
            synchronize(device)
            elapsed.append(time.perf_counter() - started)

        median = statistics.median(elapsed)
        return {
            "batch_size": config.batch_size,
            "status": "ok",
            "epoch_seconds_median": median,
            "samples_per_second": len(data.splits.train) / median,
            "trials_seconds": elapsed,
        }
    except torch.cuda.OutOfMemoryError:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {"batch_size": config.batch_size, "status": "oom"}


def main() -> int:
    """Lê candidatos, prepara a cena uma vez e imprime/salva o vencedor.

    A preparação não entra no cronômetro porque é idêntica para todos os batches;
    o que se quer comparar é a ocupação da GPU durante treino. O vencedor é o
    maior ``samples_per_second`` entre candidatos que terminaram sem OOM.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", "-d", default="PU", choices=sorted(DATASETS))
    parser.add_argument("--model", default="cnn1d", choices=MODEL_TYPES,
                        help="arquitetura a medir; compare batches apenas dentro do mesmo modelo")
    parser.add_argument("--segmenter", default="seeds", choices=SEGMENTERS,
                        help="segmentador usado quando a fusão está ativa")
    parser.add_argument("--patch-size", type=int, default=5,
                        help="lado do patch de CNN2D/CNN3D")
    parser.add_argument("--batches", type=int, nargs="+", default=[32, 64, 128, 256, 512, 1024])
    parser.add_argument("--repeats", "-n", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--backend", default="thread", choices=("sequential", "thread", "process"),
                        help="backend da preparação; não muda a medição de batch")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if any(value < 1 for value in args.batches):
        parser.error("todos os valores de --batches devem ser positivos")
    if args.repeats < 1 or args.warmup < 0:
        parser.error("--repeats deve ser positivo e --warmup não pode ser negativo")

    device = select_device(verbose=True)
    configure_determinism(False, verbose=True)
    base = ExperimentConfig(
        dataset=args.dataset, model_type=args.model, segmenter=args.segmenter,
        patch_size=args.patch_size, deterministic=False,
    )
    data, _ = prepare_data(base, backend=args.backend)
    rows = [measure(replace(base, batch_size=size), data, device, args.repeats, args.warmup) for size in args.batches]
    successful = [row for row in rows if row["status"] == "ok"]
    best = max(successful, key=lambda row: row["samples_per_second"]) if successful else None

    print("\nBatch físico | época (mediana) | amostras/s | status")
    print("-" * 57)
    for row in rows:
        if row["status"] == "ok":
            marker = "  <- recomendado" if best is row else ""
            print(f"{row['batch_size']:13d} | {row['epoch_seconds_median']:15.3f}s | "
                  f"{row['samples_per_second']:10.1f} | ok{marker}")
        else:
            print(f"{row['batch_size']:13d} | {'-':>16s} | {'-':>10s} | OOM")

    report = {
        "dataset": args.dataset,
        "model_type": args.model,
        "segmenter": args.segmenter,
        "patch_size": args.patch_size,
        "backend_preparation": args.backend,
        "device": str(device),
        "recommended_batch_size": None if best is None else best["batch_size"],
        "results": rows,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print("\nBenchmark em", args.json)
    return 0 if best else 1


if __name__ == "__main__":
    raise SystemExit(main())
