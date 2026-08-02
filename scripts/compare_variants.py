#!/usr/bin/env python3
"""Compara o baseline (Souza, 2023) com a nossa implementação.

Roda as duas variantes com a mesma cena, a mesma semente e os mesmos hiperparâmetros,
e reporta tempo por etapa, speedup e as métricas de classificação. As métricas têm de
sair **idênticas**: a otimização muda quando cada etapa roda, não o que ela calcula.
Se divergirem, o script sinaliza — é bug, não ganho.

Uso:
    python scripts/compare_variants.py                    # PaviaU
    python scripts/compare_variants.py -d IP -d SA
    python scripts/compare_variants.py --repeats 3        # média de 3 execuções
    python scripts/compare_variants.py --json outputs/comparacao.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hsi import DATASETS  # noqa: E402
from hsi.variants import BASELINE, OPTIMIZED, run_variant  # noqa: E402


def run_pair(dataset: str, epochs: int, repeats: int, verbose: bool):
    """Executa as duas variantes `repeats` vezes cada, alternando a ordem.

    Alternar evita que efeitos de aquecimento (cache de disco, clock da GPU) sejam
    creditados sistematicamente a uma das variantes.
    """
    runs = {"baseline": [], "optimized": []}
    for i in range(repeats):
        ordem = (BASELINE, OPTIMIZED) if i % 2 == 0 else (OPTIMIZED, BASELINE)
        for variant in ordem:
            runs[variant.key].append(
                run_variant(variant, dataset=dataset, epochs=epochs, verbose=verbose)
            )
    return runs


def median_of(runs, attr: str) -> float:
    return statistics.median(getattr(r, attr) for r in runs)


def report(dataset: str, runs: dict, repeats: int) -> dict:
    base, ours = runs["baseline"], runs["optimized"]

    print(f"\n{'=' * 78}")
    print(f"{dataset} — {DATASETS[dataset]}   ({repeats} execução(ões) de cada variante, mediana)")
    print("=" * 78)

    linhas = [
        ("preparação", "prep_time"),
        ("treino", "train_time"),
        ("total", "total_time"),
    ]
    print(f"  {'etapa':14s} {'baseline':>12s} {'nossa impl.':>13s} {'diferença':>12s} {'speedup':>9s}")
    print("  " + "-" * 64)
    tempos = {}
    for rotulo, attr in linhas:
        b, o = median_of(base, attr), median_of(ours, attr)
        print(f"  {rotulo:14s} {b:11.3f}s {o:12.3f}s {b - o:+11.3f}s {b / o if o else float('nan'):8.2f}x")
        tempos[rotulo] = {"baseline": b, "optimized": o}

    m_base, m_ours = base[0].metrics, ours[0].metrics
    identico = base[0].matches(ours[0])
    print(f"\n  {'métrica':14s} {'baseline':>12s} {'nossa impl.':>13s}")
    print("  " + "-" * 42)
    for rotulo, attr in (("OA", "overall_accuracy"), ("AA", "average_accuracy"), ("kappa", "kappa")):
        print(f"  {rotulo:14s} {getattr(m_base, attr):12.6f} {getattr(m_ours, attr):13.6f}")

    print(f"\n  Classificação idêntica entre as variantes: {'sim' if identico else 'NÃO'}")
    if not identico:
        print("  [!] As métricas divergiram. A otimização alterou o que o pipeline calcula —")
        print("      investigue antes de reportar qualquer ganho de tempo.")

    prep_share = 100 * tempos["preparação"]["baseline"] / tempos["total"]["baseline"]
    ganho_total = tempos["total"]["baseline"] - tempos["total"]["optimized"]
    print(f"\n  A preparação é {prep_share:.1f}% do tempo de ponta a ponta do baseline.")
    print(f"  Ganho absoluto no total: {ganho_total:+.3f}s "
          f"({100 * ganho_total / tempos['total']['baseline']:+.2f}%)")
    print("  Pela lei de Amdahl, acelerar só a preparação não pode render mais que "
          f"{100 / (100 - prep_share):.2f}x no total.")

    return {
        "dataset": dataset,
        "tempos": tempos,
        "metricas_identicas": identico,
        "metricas": {
            "baseline": {"oa": m_base.overall_accuracy, "aa": m_base.average_accuracy, "kappa": m_base.kappa},
            "optimized": {"oa": m_ours.overall_accuracy, "aa": m_ours.average_accuracy, "kappa": m_ours.kappa},
        },
        "prep_share_pct": prep_share,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", "-d", action="append", choices=sorted(DATASETS),
                        help="cena a comparar (repetível); padrão: PU")
    parser.add_argument("--epochs", type=int, default=60, help="máximo de épocas (padrão: 60)")
    parser.add_argument("--repeats", "-n", type=int, default=1,
                        help="execuções de cada variante (padrão: 1)")
    parser.add_argument("--json", type=Path, default=None, help="grava a comparação em JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="mostra a saída de cada execução")
    args = parser.parse_args()

    print(BASELINE.name + ":", BASELINE.description)
    print(OPTIMIZED.name + ":", OPTIMIZED.description)

    saida = []
    for dataset in (args.dataset or ["PU"]):
        runs = run_pair(dataset, args.epochs, args.repeats, args.verbose)
        saida.append(report(dataset, runs, args.repeats))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(saida, indent=2))
        print("\nComparação em", args.json)

    return 0 if all(r["metricas_identicas"] for r in saida) else 1


if __name__ == "__main__":
    sys.exit(main())
