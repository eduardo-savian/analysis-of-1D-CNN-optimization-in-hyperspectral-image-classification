"""Leitura e sumarização dos artefatos de MLOps já produzidos pelos runs.

Não há serviço externo: esta camada transforma os JSONs imutáveis de
``outputs/runs`` em registros comparáveis, adequados para terminal, Markdown e
integração posterior com dashboards/MLflow.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hsi.config import output_dir


@dataclass(frozen=True)
class RunObservation:
    """Visão plana dos campos que importam para observar uma execução."""

    run_id: str
    directory: Path
    status: str
    config: dict[str, Any]
    metrics: dict[str, Any] | None
    train_time: float | None
    prep_time: float | None
    error: str | None = None

    def row(self) -> dict[str, Any]:
        """Converte o run para uma linha JSON/tabular sem objetos Path ou NumPy."""
        metrics = self.metrics or {}
        return {
            "run_id": self.run_id,
            "status": self.status,
            "dataset": self.config.get("dataset"),
            "model_type": self.config.get("model_type", "cnn1d"),
            "segmenter": self.config.get("segmenter", "seeds"),
            "two_input": self.config.get("two_input"),
            "epochs": self.config.get("epochs"),
            "oa": metrics.get("overall_accuracy"),
            "aa": metrics.get("average_accuracy"),
            "kappa": metrics.get("kappa"),
            "train_seconds": self.train_time,
            "prep_seconds": self.prep_time,
            "error": self.error,
            "directory": str(self.directory),
        }


def _read(path: Path) -> dict[str, Any]:
    """Lê um JSON de run; arquivos malformados são erro visível ao operador."""
    return json.loads(path.read_text(encoding="utf-8"))


def discover_runs(runs_dir: Path | None = None) -> list[RunObservation]:
    """Descobre runs concluídos, falhos ou em andamento sem modificar artefatos.

    Diretórios sem manifesto são ignorados: podem ser cópias parciais durante um
    upload. O resultado é ordenado do mais recente para o mais antigo, pois o
    identificador começa por data UTC.
    """
    base = runs_dir or output_dir(create=False) / "runs"
    if not base.exists():
        return []
    observations = []
    for directory in sorted((path for path in base.iterdir() if path.is_dir()), reverse=True):
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = _read(manifest_path)
        config = _read(directory / "config.json") if (directory / "config.json").exists() else {}
        metrics_file = directory / "metrics.json"
        metrics_record = _read(metrics_file) if metrics_file.exists() else {}
        failure_file = directory / "failure.json"
        failure = _read(failure_file) if failure_file.exists() else {}
        observations.append(RunObservation(
            run_id=manifest.get("run_id", directory.name), directory=directory,
            status=manifest.get("status", "unknown"), config=config,
            metrics=metrics_record.get("metrics"), train_time=metrics_record.get("train_time"),
            prep_time=metrics_record.get("prep_timings", {}).get("wall"),
            error=failure.get("message"),
        ))
    return observations


def filter_runs(
    runs: list[RunObservation], dataset: str | None = None, model_type: str | None = None,
    segmenter: str | None = None, include_incomplete: bool = False,
) -> list[RunObservation]:
    """Filtra runs sem misturar candidatos falhos nas métricas comparáveis."""
    return [
        run for run in runs
        if (include_incomplete or run.status == "completed")
        and (dataset is None or run.config.get("dataset") == dataset)
        and (model_type is None or run.config.get("model_type", "cnn1d") == model_type)
        and (segmenter is None or run.config.get("segmenter", "seeds") == segmenter)
    ]


def runs_summary(runs: list[RunObservation]) -> dict[str, Any]:
    """Calcula contagens e melhor run por AA, kappa e OA para observabilidade."""
    completed = [run for run in runs if run.status == "completed" and run.metrics]
    best = max(
        completed,
        key=lambda run: (run.metrics["average_accuracy"], run.metrics["kappa"], run.metrics["overall_accuracy"]),
        default=None,
    )
    return {
        "total": len(runs),
        "completed": len(completed),
        "failed": sum(run.status == "failed" for run in runs),
        "best_by_aa": None if best is None else best.row(),
    }


def markdown_report(runs: list[RunObservation]) -> str:
    """Renderiza um resumo textual compacto pronto para relatório, issue ou TCC."""
    summary = runs_summary(runs)
    lines = [
        "# Observabilidade de experimentos", "",
        f"Runs: {summary['total']} | concluídos: {summary['completed']} | falhos: {summary['failed']}", "",
        "| run | status | cena | modelo | segmentador | OA | AA | kappa | treino (s) |",
        "|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for run in runs:
        row = run.row()
        def metric(value):
            return "-" if value is None else f"{value:.4f}"
        lines.append(
            f"| {row['run_id']} | {row['status']} | {row['dataset'] or '-'} | "
            f"{row['model_type']} | {row['segmenter']} | {metric(row['oa'])} | "
            f"{metric(row['aa'])} | {metric(row['kappa'])} | {metric(row['train_seconds'])} |"
        )
    if summary["best_by_aa"]:
        best = summary["best_by_aa"]
        lines.extend(["", f"Melhor por AA: `{best['run_id']}` ({best['aa']:.4f})."])
    return "\n".join(lines) + "\n"
