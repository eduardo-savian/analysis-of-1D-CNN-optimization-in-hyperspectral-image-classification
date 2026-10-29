"""Escada de batch físico da GPU, medida sobre os vencedores do HPO.

O HPO (`hsi/hpo/busca.py`) já varia `batch_size` como eixo de **qualidade**, misturado aos
demais. Este sweep é complementar e responde outra pergunta: fixada a receita
vencedora, qual batch entrega mais **throughput** sem perder validação. Por isso ele
varre uma escada uniforme de 50 em 50 em vez de sortear, continua mesmo se um valor
falhar por falta de memória, e escolhe pelo maior throughput dentro da tolerância.
"""

import json
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import matplotlib.pyplot as plt
import pandas as pd

from hsi.config import ExperimentConfig, output_dir
from hsi.hpo.selecao import select_hpo_winners
from hsi.mlops.campanha import (
    _campaign_json,
    _serializable_records,
    run_audited_experiment,
)
from hsi.mlops.paineis import generate_all_runs_pdf
from hsi.mlops.relatorios import generate_pdf_report


def run_audited_batch_sweep(config, batch_start=50, batch_max=300, step=50, epochs=10, quality_tolerance=.005, backend="thread", campaign_root=None):
    """Compara uma escada uniforme e escolhe velocidade sem sacrificar validação."""
    if any(not isinstance(value, int) or value <= 0 for value in (batch_start, batch_max, step)):
        raise ValueError("batch_start, batch_max e step devem ser inteiros positivos.")
    if batch_start > batch_max or (batch_max - batch_start) % step:
        raise ValueError("A escada deve terminar exatamente em batch_max.")
    if not 0 <= quality_tolerance < 1:
        raise ValueError("quality_tolerance deve estar no intervalo [0, 1).")
    batch_sizes = list(range(batch_start, batch_max + 1, step))
    started = datetime.now(timezone.utc)
    campaign_id = f"batch_sweep_{started.strftime('%Y%m%dT%H%M%SZ')}_{config.model_type}_{uuid4().hex[:8]}"
    directory = Path(campaign_root or output_dir() / "campaigns") / campaign_id
    directory.mkdir(parents=True, exist_ok=False)
    plan = [
        {"trial_key": f"{config.model_type}_batch_{batch}", "batch_size": batch,
         "config": replace(config, batch_size=batch, epochs=epochs, patience=min(config.patience, max(2, epochs // 3))).to_dict()}
        for batch in batch_sizes
    ]
    manifest = {
        "schema_version": 1, "campaign_id": campaign_id, "status": "running", "kind": "batch_sweep",
        "started_at_utc": started.isoformat(), "selection_metric": "validation_best then throughput",
        "plan_sha256": sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(), "batch_sizes": batch_sizes,
        "batch_ladder": {"start": batch_start, "step": step, "max": batch_max, "rationale": "incrementos fixos medem ganho marginal e limite de memória"},
        "quality_tolerance": quality_tolerance,
    }
    _campaign_json(directory / "campaign.json", manifest)
    _campaign_json(directory / "plan.json", plan)
    records = []
    for recipe in plan:
        candidate = ExperimentConfig(**recipe["config"])
        record = run_audited_experiment(candidate, directory, campaign_id, recipe["trial_key"], backend)
        record["batch_size"] = candidate.batch_size
        record["effective_batch"] = candidate.effective_batch_size
        if record["status"] == "completed":
            result = record["result"]
            record["validation_best"] = float(max(result.training.history.val_accuracy))
            # Amostras de treino por segundo, contando só as épocas executadas: o
            # early stopping encerra cada degrau num ponto diferente, e o tempo total
            # sozinho premiaria o batch que parou antes.
            record["throughput_samples_s"] = len(result.data.splits.train) * len(result.training.history) / max(result.training.total_time, 1e-12)
        records.append(record)
        _campaign_json(directory / "results.json", _serializable_records(records))
    frame = pd.DataFrame(_serializable_records(records))
    completed = frame.query("status == 'completed'").copy()
    if not completed.empty:
        best_validation = completed["validation_best"].max()
        viable = completed[completed["validation_best"] >= best_validation - quality_tolerance]
        winner = viable.sort_values(["throughput_samples_s", "batch_size"], ascending=[False, True]).iloc[0].to_dict()
    else:
        winner = None
    frame.to_csv(directory / "summary.csv", index=False)
    analysis = {"selection": f"maior throughput com validação até {quality_tolerance:.3f} do melhor valor", "batch_ladder_rationale": "incrementos fixos isolam o efeito de +50 amostras e revelam saturação/OOM", "quality_tolerance": quality_tolerance, "winner": winner}
    _campaign_json(directory / "analysis.json", analysis)
    manifest.update({"status": "completed", "finished_at_utc": datetime.now(timezone.utc).isoformat(), "winner": winner, "artifacts": ["plan.json", "results.json", "summary.csv", "analysis.json", "report.pdf"]})
    _campaign_json(directory / "campaign.json", manifest)
    pdf_report = generate_pdf_report(directory, f"Sweep de batches {campaign_id}")
    global_pdf, _, _ = generate_all_runs_pdf()
    return {"directory": directory, "records": frame, "winner": winner, "batch_sizes": batch_sizes, "quality_tolerance": quality_tolerance, "pdf_report": pdf_report, "global_pdf_report": global_pdf, "objects": {r["run_id"]: r["result"] for r in records if r["result"] is not None}}


def plot_batch_sweep(sweep):
    """Visualiza qualidade, custo e throughput em função do batch físico da GPU."""
    frame = sweep["records"].query("status == 'completed'").sort_values("batch_size")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    axes[0, 0].plot(frame["batch_size"], frame["validation_best"], marker="o", label="validação")
    axes[0, 0].plot(frame["batch_size"], frame["oa_test"], marker="o", label="OA teste (relato)")
    axes[0, 0].set(title="Qualidade × batch", xlabel="Batch físico", ylabel="Pontuação", ylim=(0, 1)); axes[0, 0].legend(); axes[0, 0].grid(alpha=.25)
    axes[0, 1].plot(frame["batch_size"], frame["train_seconds"], marker="o", color="#F58518")
    axes[0, 1].set(title="Tempo de treino × batch", xlabel="Batch físico", ylabel="Segundos"); axes[0, 1].grid(alpha=.25)
    axes[1, 0].plot(frame["batch_size"], frame["throughput_samples_s"], marker="o", color="#54A24B")
    axes[1, 0].set(title="Throughput × batch", xlabel="Batch físico", ylabel="Amostras/s"); axes[1, 0].grid(alpha=.25)
    # Barras lado a lado: sobrepostas, a mais baixa some dentro da mais alta.
    positions = range(len(frame))
    axes[1, 1].bar([p - .2 for p in positions], frame["aa_test"], width=.4, label="AA")
    axes[1, 1].bar([p + .2 for p in positions], frame["kappa_test"], width=.4, label="Kappa")
    axes[1, 1].set_xticks(list(positions), frame["batch_size"].astype(str))
    axes[1, 1].set(title="Robustez por batch", xlabel="Batch físico", ylabel="Pontuação", ylim=(0, 1)); axes[1, 1].legend(); axes[1, 1].grid(axis="y", alpha=.25)
    return fig


def run_batch_sweeps_for_hpo_winners(hpo_table, batch_start=50, batch_max=300, step=50, epochs=10, quality_tolerance=.005, backend="thread"):
    """Executa a mesma escada justificável para o melhor HPO de cada CNN."""
    winners = select_hpo_winners(hpo_table)
    return {
        name: run_audited_batch_sweep(row["config"], batch_start=batch_start, batch_max=batch_max, step=step, epochs=epochs, quality_tolerance=quality_tolerance, backend=backend)
        for name, row in winners.items()
    }
