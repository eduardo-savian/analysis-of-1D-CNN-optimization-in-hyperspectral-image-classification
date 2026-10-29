"""Campanhas reprodutíveis para comparar modelos em vez de olhar uma única execução.

Cada tentativa é um run normal do tracker e grava configuração, ambiente,
fingerprint dos dados, métricas, histórico, checkpoint e status. A campanha agrega
esses runs por uma receita explícita, preserva o plano original, atualiza resultados
parciais após cada tentativa e calcula estatísticas somente a partir dos artefatos.
"""

import json
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hsi.checkpoint import save_checkpoint
from hsi.config import MODEL_TYPES, SEGMENTERS, ExperimentConfig, output_dir
from hsi.hpo.busca import hpo_candidates
from hsi.mlops.paineis import generate_all_runs_pdf
from hsi.mlops.rastreio import create_run, fail_run, record_run
from hsi.mlops.relatorios import generate_pdf_report
from hsi.pipeline import run_experiment


def _campaign_json(path: Path, value):
    """Grava JSON ordenado e legível, usando escrita atômica por substituição."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_campaign_plan(base_config, trials_per_cnn=4, seeds=(11, 29, 42),
                        search_epochs=10, model_types=MODEL_TYPES,
                        segmenters=None):
    """Cria a matriz auditável: candidatos por CNN × repetições por seed.

    Os candidatos vêm de `hpo_candidates` — sempre os do random search, a única
    estratégia que fixa o plano inteiro antes de treinar — e cobrem cada
    segmentador. São sorteados uma vez, com `base_config.seed`; as `seeds` só os
    repetem. A seed é parte da receita e, portanto, aparece tanto no plano quanto
    em `config.json` de todo run. A ordem é determinística: a mesma config e as
    mesmas seeds geram o mesmo plano e o mesmo `plan_sha256`, o que permite
    comparar campanhas.
    """
    segmenters = tuple(segmenters or SEGMENTERS)
    trials_per_cnn = max(trials_per_cnn, len(segmenters))
    recipes = []
    for model_type in model_types:
        candidates = hpo_candidates(base_config, model_type, segmenters, trials_per_cnn, base_config.seed)
        for candidate_index, candidate in enumerate(candidates, start=1):
            for repeat, seed in enumerate(seeds, start=1):
                config = replace(
                    candidate, seed=int(seed), epochs=search_epochs,
                    patience=min(candidate.patience, max(2, search_epochs // 3)),
                )
                recipes.append({
                    "trial_key": f"{model_type}_c{candidate_index:02d}_s{seed}",
                    "model_type": model_type,
                    "candidate_index": candidate_index,
                    "repeat": repeat,
                    "config": config.to_dict(),
                })
    return recipes


def run_audited_experiment(config, campaign_dir: Path, campaign_id: str, trial_key: str, backend="thread"):
    """Executa uma receita e torna sucesso ou falha permanentemente auditável.

    Mesmo uma exceção gera `failure.json`, é vinculada ao `trial_key` e permanece
    no resultado parcial da campanha. Assim, remover uma falha para melhorar um
    resultado não é possível sem deixar alteração explícita no plano/manifesto.

    O `except` pega `BaseException`, então uma interrupção do kernel também vira
    falha registrada — e o laço da campanha segue para a receita seguinte em vez
    de parar.
    """
    context = create_run(config)
    run_link = {
        "campaign_id": campaign_id,
        "campaign_directory": str(campaign_dir.resolve()),
        "trial_key": trial_key,
    }
    _campaign_json(context.directory / "campaign.json", run_link)
    try:
        result = run_experiment(config, backend=backend, verbose=False)
        checkpoint = context.directory / "model.pt"
        save_checkpoint(result.model, config, result.data.input_shape(config.model_type), result.data.n_classes, checkpoint)
        record_run(context, result, artifacts={"checkpoint": checkpoint, "campaign": context.directory / "campaign.json"})
        pdf_report = generate_pdf_report(context.directory, f"Run {context.run_id}", result=result)
        manifest_path = context.directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["artifacts"]["pdf_report"] = pdf_report.name
        _campaign_json(manifest_path, manifest)
        return {
            "trial_key": trial_key, "run_id": context.run_id, "status": "completed",
            "run_directory": str(context.directory.resolve()),
            "model_type": config.model_type, "segmenter": config.segmenter,
            "learning_rate": config.learning_rate, "dropout": config.dropout, "seed": config.seed,
            # Critério de seleção: validação. Métricas de teste são exclusivamente relatadas.
            "validation_best": float(max(result.training.history.val_accuracy)),
            "oa_test": result.metrics.overall_accuracy, "aa_test": result.metrics.average_accuracy,
            "kappa_test": result.metrics.kappa, "test_loss": result.metrics.test_loss,
            "train_seconds": result.training.total_time, "prep_seconds": result.prep_timings.wall,
            "checkpoint": str(checkpoint.resolve()), "pdf_report": str(pdf_report.resolve()), "result": result,
        }
    except BaseException as error:
        fail_run(context, error)
        return {
            "trial_key": trial_key, "run_id": context.run_id, "status": "failed",
            "run_directory": str(context.directory.resolve()),
            "model_type": config.model_type, "segmenter": config.segmenter,
            "learning_rate": config.learning_rate, "dropout": config.dropout, "seed": config.seed,
            "error_type": type(error).__name__, "error": str(error), "result": None,
        }


def _serializable_records(records):
    """Remove objetos PyTorch do relatório de campanha sem perder links aos artefatos."""
    return [{key: value for key, value in record.items() if key != "result"} for record in records]


def run_audited_campaign(base_config, trials_per_cnn=4, seeds=(11, 29, 42),
                         search_epochs=10, backend="thread", campaign_root=None):
    """Executa uma campanha completa e persiste o estado após *cada* tentativa.

    Arquivos produzidos em `outputs/campaigns/<id>/`:
    `campaign.json` (intenção e status), `plan.json` (matriz imutável),
    `results.json` (resultado parcial/final), `summary.csv`, `analysis.json`,
    `report.md` e `report.pdf`. Cada linha aponta para um run individual em
    `outputs/runs/`; no fim, o PDF global de todos os runs é regenerado.

    Não há retomada: cada chamada abre um diretório novo e roda o plano inteiro.
    A seção 8.5 não usa esta função — ela roda `run_best_hpo_campaign`, que repete
    só os vencedores do HPO.
    """
    started = datetime.now(timezone.utc)
    campaign_id = f"campaign_{started.strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
    directory = Path(campaign_root or output_dir() / "campaigns") / campaign_id
    directory.mkdir(parents=True, exist_ok=False)
    plan = build_campaign_plan(base_config, trials_per_cnn, seeds, search_epochs)
    plan_hash = sha256(json.dumps(plan, sort_keys=True).encode("utf-8")).hexdigest()
    campaign = {
        "schema_version": 1, "campaign_id": campaign_id, "status": "running",
        "started_at_utc": started.isoformat(), "backend": backend,
        "selection_metric": "validation_best", "test_policy": "report_only",
        "base_config": base_config.to_dict(), "plan_sha256": plan_hash,
        "planned_trials": len(plan), "seeds": list(seeds),
    }
    _campaign_json(directory / "campaign.json", campaign)
    _campaign_json(directory / "plan.json", plan)
    records = []
    for recipe in plan:
        config = ExperimentConfig(**recipe["config"])
        record = run_audited_experiment(config, directory, campaign_id, recipe["trial_key"], backend)
        records.append(record)
        _campaign_json(directory / "results.json", _serializable_records(records))
    frame = pd.DataFrame(_serializable_records(records))
    analysis = analyze_campaign(frame)
    frame.to_csv(directory / "summary.csv", index=False)
    _campaign_json(directory / "analysis.json", analysis["json"])
    (directory / "report.md").write_text(campaign_report(campaign, frame, analysis), encoding="utf-8")
    campaign.update({
        "status": "completed", "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_trials": int((frame["status"] == "completed").sum()),
        "failed_trials": int((frame["status"] == "failed").sum()),
        "artifacts": ["plan.json", "results.json", "summary.csv", "analysis.json", "report.md", "report.pdf"],
    })
    _campaign_json(directory / "campaign.json", campaign)
    pdf_report = generate_pdf_report(directory, f"Campanha {campaign_id}")
    global_pdf, _, _ = generate_all_runs_pdf()
    return {"campaign": campaign, "directory": directory, "records": frame, "objects": {r["run_id"]: r["result"] for r in records if r["result"] is not None}, "analysis": analysis, "pdf_report": pdf_report, "global_pdf_report": global_pdf}


def _bootstrap_interval(values, seed=42, samples=2000):
    """Intervalo bootstrap de 95% da média, explícito mesmo com poucas seeds.

    Semente fixa: os mesmos valores dão sempre o mesmo intervalo. As 2000
    reamostras estabilizam os quantis de 2,5% e 97,5%, mas não compensam um n
    pequeno. Com as 3 seeds padrão, a reamostra que repete três vezes o menor valor
    tem probabilidade 1/27 ≈ 3,7%, acima do corte de 2,5% (idem para o maior): o
    "IC95%" coincide com [mínimo, máximo] das seeds. Leia-o como amplitude, não
    como inferência.
    """
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return tuple(map(float, np.quantile(means, (.025, .975))))


def analyze_campaign(records: pd.DataFrame):
    """Agrega repetições por receita e produz ranking apenas pela validação.

    Uma "receita" aqui é (CNN, segmentador, learning rate, dropout) — os únicos
    eixos que o registro de cada run carrega. Candidatos que diferem só em weight
    decay, batch, PCA ou na receita interna do segmentador caem no mesmo grupo, e
    as repetições deles são agregadas juntas.

    O dataframe por receita contém média, desvio e IC95% de validação, além das
    métricas de teste somente para contextualização. `run_ids` preserva exatamente
    quais execuções contribuíram para cada número agregado.
    """
    completed = records[records["status"] == "completed"].copy()
    if completed.empty:
        return {"recipes": pd.DataFrame(), "models": pd.DataFrame(), "json": {"completed": 0}}
    groups = ["model_type", "segmenter", "learning_rate", "dropout"]
    rows = []
    for key, group in completed.groupby(groups, sort=True):
        ci_low, ci_high = _bootstrap_interval(group["validation_best"], seed=42)
        rows.append({
            **dict(zip(groups, key)), "n_execuções": len(group),
            "validação média": group["validation_best"].mean(), "validação desvio": group["validation_best"].std(ddof=0),
            "validação ic95 baixo": ci_low, "validação ic95 alto": ci_high,
            "OA teste média": group["oa_test"].mean(), "AA teste média": group["aa_test"].mean(),
            "Kappa teste média": group["kappa_test"].mean(), "treino médio (s)": group["train_seconds"].mean(),
            "run_ids": list(group["run_id"]),
        })
    recipes = pd.DataFrame(rows).sort_values(["validação média", "validação desvio"], ascending=[False, True]).reset_index(drop=True)
    recipes.index += 1
    recipes.index.name = "ranking por validação"
    model_rows = []
    for model_type, group in completed.groupby("model_type", sort=True):
        low, high = _bootstrap_interval(group["validation_best"], seed=42)
        model_rows.append({
            "CNN": model_type.upper(), "n_execuções": len(group),
            "validação média": group["validation_best"].mean(), "validação desvio": group["validation_best"].std(ddof=0),
            "validação ic95 baixo": low, "validação ic95 alto": high,
            "OA teste média": group["oa_test"].mean(), "AA teste média": group["aa_test"].mean(),
            "Kappa teste média": group["kappa_test"].mean(), "treino médio (s)": group["train_seconds"].mean(),
        })
    models = pd.DataFrame(model_rows).sort_values("validação média", ascending=False).reset_index(drop=True)
    return {
        "recipes": recipes, "models": models,
        "json": {
            "completed": int(len(completed)), "failed": int((records["status"] == "failed").sum()),
            # Repete os padrões de `_bootstrap_interval`: mudar um sem o outro faz o
            # registro descrever um intervalo diferente do calculado.
            "selection_metric": "validation_best", "bootstrap": {"seed": 42, "samples": 2000, "confidence": .95},
            "ranking": json.loads(recipes.to_json(orient="records")),
            "models": json.loads(models.to_json(orient="records")),
        },
    }


def plot_campaign_analysis(records: pd.DataFrame, analysis):
    """Mostra repetição por seed, intervalos, segmentadores e custo da campanha."""
    completed = records[records["status"] == "completed"].copy()
    if completed.empty:
        print("Campanha sem runs concluídos para analisar.")
        return None
    recipes, models = analysis["recipes"], analysis["models"]
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
    for model_type, group in completed.groupby("model_type"):
        axes[0, 0].scatter(group["seed"], group["validation_best"], s=75, label=model_type.upper())
    axes[0, 0].set(title="Repetições: validação por seed", xlabel="Seed", ylabel="Melhor val_accuracy", ylim=(0, 1))
    axes[0, 0].legend(); axes[0, 0].grid(alpha=.25)
    positions = np.arange(len(models))
    means = models["validação média"].to_numpy()
    errors = np.vstack((means - models["validação ic95 baixo"], models["validação ic95 alto"] - means))
    axes[0, 1].bar(positions, means, yerr=errors, capsize=5, color="#4C78A8")
    axes[0, 1].set(title="Média por CNN com IC95% bootstrap", xticks=positions, xticklabels=models["CNN"], ylabel="val_accuracy", ylim=(0, 1))
    axes[0, 1].grid(axis="y", alpha=.25)
    pivot = completed.pivot_table(index="model_type", columns="segmenter", values="validation_best", aggfunc="mean")
    image = axes[1, 0].imshow(pivot, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    axes[1, 0].set(title="Validação média: CNN × segmentador", yticks=range(len(pivot.index)), yticklabels=[item.upper() for item in pivot.index], xticks=range(len(pivot.columns)), xticklabels=pivot.columns)
    fig.colorbar(image, ax=axes[1, 0], label="val_accuracy")
    scatter = axes[1, 1].scatter(recipes["treino médio (s)"], recipes["validação média"], c=recipes["OA teste média"], cmap="viridis", s=100, edgecolor="black")
    for rank, row in recipes.head(min(8, len(recipes))).iterrows():
        axes[1, 1].annotate(f"#{rank} {row['model_type'].upper()}/{row['segmenter']}", (row["treino médio (s)"], row["validação média"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[1, 1].set(title="Receitas: qualidade de validação × custo", xlabel="Treino médio (s)", ylabel="val_accuracy", ylim=(0, 1))
    axes[1, 1].grid(alpha=.25); fig.colorbar(scatter, ax=axes[1, 1], label="OA teste (relato)")
    return fig


def campaign_report(campaign, records: pd.DataFrame, analysis) -> str:
    """Gera relatório Markdown que aponta para os run_ids e documenta a política."""
    lines = [
        "# Relatório de campanha MLOps", "",
        f"- Campanha: `{campaign['campaign_id']}`", f"- Início: {campaign['started_at_utc']}",
        f"- Plano SHA-256: `{campaign['plan_sha256']}`", "- Seleção: maior `validation_best`; teste é somente relatório.",
        f"- Runs concluídos: {(records['status'] == 'completed').sum()} | falhos: {(records['status'] == 'failed').sum()}", "",
        "## Ranking de receitas", "",
    ]
    recipes = analysis["recipes"]
    if recipes.empty:
        return "\n".join(lines + ["Nenhum run concluído.", ""])
    lines.extend(["| rank | CNN | segmentador | lr | dropout | n | val média | IC95% | OA teste | runs |", "|---:|---|---|---:|---:|---:|---:|---|---:|---|"])
    for rank, row in recipes.iterrows():
        lines.append(
            f"| {rank} | {row['model_type'].upper()} | {row['segmenter']} | {row['learning_rate']:.0e} | {row['dropout']:.2f} | {row['n_execuções']} | "
            f"{row['validação média']:.4f} | [{row['validação ic95 baixo']:.4f}, {row['validação ic95 alto']:.4f}] | {row['OA teste média']:.4f} | {', '.join(row['run_ids'])} |"
        )
    return "\n".join(lines) + "\n"
