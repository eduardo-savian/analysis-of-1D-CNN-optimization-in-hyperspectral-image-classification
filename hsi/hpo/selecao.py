"""Ponte entre screening de HPO e campanha final de resultados científicos.

O HPO é uma fase de triagem: avalia candidatos com orçamento reduzido e seleciona
por validação. A campanha final não mistura candidatos inferiores no comparativo;
ela repete exclusivamente o vencedor de cada CNN, com várias seeds e orçamento de
épocas completo. A decisão fica congelada em `selection.json`, e todos os candidatos,
em `hpo_screening.json`.
"""

import json
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import pandas as pd

from hsi.config import ExperimentConfig, output_dir
from hsi.mlops.campanha import (
    _campaign_json,
    _serializable_records,
    analyze_campaign,
    campaign_report,
    run_audited_experiment,
)
from hsi.mlops.paineis import generate_all_runs_pdf
from hsi.mlops.relatorios import generate_pdf_report


def select_hpo_winners(hpo_table):
    """Seleciona o melhor trial por CNN usando apenas `val_accuracy (objetivo)`.

    Critérios de desempate, nesta ordem: maior validação, menor tempo de treino e
    menor número do trial. OA/AA/Kappa de teste não participam da escolha.
    """
    winners = {}
    for cnn, group in hpo_table.groupby("CNN", sort=True):
        ranked = group.sort_values(
            ["val_accuracy (objetivo)", "tempo treino (s)", "trial"],
            ascending=[False, True, True],
        )
        winners[cnn.lower()] = ranked.iloc[0]
    return winners


def run_best_hpo_campaign(hpo_table, final_epochs, seeds=(11, 29, 42), backend="thread", campaign_root=None):
    """Repete e audita os melhores casos HPO com múltiplas seeds.

    O diretório da campanha contém o snapshot de todos os candidatos HPO, a regra
    de seleção, um hash do plano final e os links para cada run final. Assim é
    possível responder exatamente *por que* uma receita foi promovida ao estudo
    comparativo e quais execuções formaram suas médias/intervalos.

    Só `epochs` muda para `final_epochs`: a config do vencedor é a do trial, então
    a `patience` encurtada da busca (3, com 10 épocas de busca) segue no retreino.
    """
    winners = select_hpo_winners(hpo_table)
    started = datetime.now(timezone.utc)
    campaign_id = f"best_hpo_{started.strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
    directory = Path(campaign_root or output_dir() / "campaigns") / campaign_id
    directory.mkdir(parents=True, exist_ok=False)
    screening = hpo_table.drop(columns=["resultado", "config"], errors="ignore").to_dict(orient="records")
    selection = []
    final_plan = []
    for cnn, row in winners.items():
        winner_config = replace(row["config"], epochs=final_epochs)
        selection.append({
            "CNN": cnn.upper(), "selected_trial": int(row["trial"]),
            "selection_metric": "val_accuracy (objetivo)", "selection_value": float(row["val_accuracy (objetivo)"]),
            "config": winner_config.to_dict(),
        })
        for repeat, seed in enumerate(seeds, start=1):
            final_config = replace(winner_config, seed=int(seed))
            final_plan.append({
                "trial_key": f"final_{cnn}_s{seed}", "phase": "final_best_hpo",
                "repeat": repeat, "config": final_config.to_dict(),
            })
    final_hash = sha256(json.dumps(final_plan, sort_keys=True).encode("utf-8")).hexdigest()
    campaign = {
        "schema_version": 1, "campaign_id": campaign_id, "status": "running",
        "started_at_utc": started.isoformat(), "backend": backend,
        "selection_policy": "best validation HPO per CNN; test report-only",
        "selection_metric": "validation_best", "test_policy": "report_only",
        "screening_trials": len(screening), "planned_final_trials": len(final_plan),
        "final_plan_sha256": final_hash, "seeds": list(seeds),
    }
    _campaign_json(directory / "campaign.json", campaign)
    _campaign_json(directory / "hpo_screening.json", screening)
    _campaign_json(directory / "selection.json", selection)
    _campaign_json(directory / "final_plan.json", final_plan)
    records = []
    for recipe in final_plan:
        config = ExperimentConfig(**recipe["config"])
        record = run_audited_experiment(config, directory, campaign_id, recipe["trial_key"], backend)
        record["phase"] = recipe["phase"]
        records.append(record)
        _campaign_json(directory / "results.json", _serializable_records(records))
    frame = pd.DataFrame(_serializable_records(records))
    analysis = analyze_campaign(frame)
    frame.to_csv(directory / "summary.csv", index=False)
    _campaign_json(directory / "analysis.json", analysis["json"])
    report_context = {**campaign, "plan_sha256": final_hash}
    (directory / "report.md").write_text(campaign_report(report_context, frame, analysis), encoding="utf-8")
    campaign.update({
        "status": "completed", "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_trials": int((frame["status"] == "completed").sum()),
        "failed_trials": int((frame["status"] == "failed").sum()),
        "artifacts": ["hpo_screening.json", "selection.json", "final_plan.json", "results.json", "summary.csv", "analysis.json", "report.md", "report.pdf"],
    })
    _campaign_json(directory / "campaign.json", campaign)
    pdf_report = generate_pdf_report(directory, f"Campanha HPO {campaign_id}")
    global_pdf, _, _ = generate_all_runs_pdf()
    objects = {record["run_id"]: record["result"] for record in records if record["result"] is not None}
    return {"campaign": campaign, "directory": directory, "records": frame, "objects": objects, "analysis": analysis, "selection": pd.DataFrame(selection), "pdf_report": pdf_report, "global_pdf_report": global_pdf}


def best_hpo_results_by_cnn(campaign):
    """Retorna um run representativo por CNN para os dashboards finais.

    Com a mesma receita vencedora repetida por seed, escolhe-se o run de maior
    validação apenas para visualização; o ranking científico continua sendo a média
    e o intervalo da campanha, nunca esse run individual.
    """
    results = {}
    completed = campaign["records"].query("status == 'completed'")
    for model_type, group in completed.groupby("model_type"):
        selected = group.sort_values("validation_best", ascending=False).iloc[0]
        results[model_type] = campaign["objects"][selected["run_id"]]
    return results
