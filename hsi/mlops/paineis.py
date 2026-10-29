"""Painéis e PDF consolidados derivados somente dos artefatos persistidos.

Não há recomputação de métricas nem leitura dos tensores do modelo: cada ponto dos
gráficos vem de ``config.json``, ``metrics.json`` e ``manifest.json`` de um run.
Isso mantém o relatório global reproduzível e adequado para auditoria MLOps. A
única exceção é a seção de segmentadores do PDF, que segmenta a cena de novo (ver
`segmentation_report_figures`).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from matplotlib.backends.backend_pdf import PdfPages

from hsi.config import ExperimentConfig, output_dir
from hsi.dados.cenas import load_scene
from hsi.mlops.relatorios import _pdf_text
from hsi.visualizacao.diagnosticos import (
    plot_segmentation_overlay_suite,
    plot_segmentation_quality_dashboard,
)

#: Título e regra de leitura de cada figura, pela mesma chave de `all_run_figures`.
#: `generate_all_runs_pdf` indexa este dicionário com essas chaves: uma figura nova
#: sem nota aqui derruba o PDF com `KeyError`.
ALL_RUN_PLOT_NOTES = {
    "quality": ("Qualidade por arquitetura", "OA, AA e Kappa de todos os metrics.json. Compare a distribuição: caixas altas e estreitas indicam qualidade alta e estável; AA é mais segura em classes desbalanceadas."),
    "segmenter": ("AA por CNN e segmentador", "Média de AA agrupada por model_type e segmenter. A cor representa desempenho médio, não causalidade; grupos com poucos runs exigem cautela."),
    "pareto": ("Fronteira qualidade × custo", "AA e tempo de treino de cada run. Pontos no alto à esquerda são preferíveis: maior qualidade com menor custo. Cor identifica a CNN e tamanho representa batch físico."),
    "times": ("Distribuição de custos", "Tempos de treino e pré-processamento gravados em metrics.json. Use para identificar variabilidade entre seeds e custo relativo das arquiteturas."),
    "hyperparameters": ("Efeito de hiperparâmetros", "Melhor val_accuracy do histórico versus learning rate; tamanho do ponto é dropout. A métrica de validação é usada porque teste não participa de seleção."),
    "early": ("Convergência e early stopping", "Épocas realmente executadas versus melhor validação. Cor informa se houve parada antecipada; a linha/meta vem da configuração de cada run."),
    "batch": ("Batch × qualidade", "Batch físico de config.json versus validação e AA. A dispersão revela se batches maiores alteraram a qualidade; throughput deve ser lido no sweep específico."),
    "timeline": ("Evolução temporal", "Runs ordenados pelo timestamp do ID. Serve para detectar regressões após mudanças de receita; não substitui comparação controlada."),
    "classes": ("Acurácia por classe", "Média de per_class_accuracy por CNN, apenas para runs com a mesma quantidade de classes. Tons altos indicam classes mais bem reconhecidas."),
    "gaps": ("Consistência entre métricas", "OA e AA do mesmo run. A diagonal representa igualdade; OA muito maior que AA pode indicar classes frequentes mascarando classes raras."),
    "coverage": ("Cobertura experimental", "Cada barra é a quantidade de runs de uma combinação CNN/segmentador; repetição melhora a evidência."),
    "training": ("Evolução de todos os treinos", "Linhas claras são runs individuais; linha forte é média por CNN e faixa é desvio padrão. Validação deve subir enquanto a loss cai."),
    "validation_test": ("Validação × teste", "Cada ponto compara a melhor validação da receita com AA do teste. A diagonal é referência, não objetivo de seleção: o teste continua reservado para relato."),
    "stopping": ("Meta e early stopping", "Barras mostram a proporção de runs que atingiu a meta de validação e que encerrou antecipadamente. Elas revelam a eficiência do orçamento de épocas."),
    "seeds": ("Repetibilidade por seed", "Pontos mostram AA por seed; caixas mostram dispersão por arquitetura. Menor dispersão sugere resultado mais estável, desde que haja repetição suficiente."),
}


def load_all_audited_runs(runs_root=None):
    """Lê todos os runs concluídos diretamente de seus documentos auditáveis."""
    rows = []
    root = Path(runs_root or output_dir() / "runs")
    for directory in sorted(root.glob("*")):
        if not directory.is_dir():
            continue
        config_path, metrics_path, manifest_path = directory / "config.json", directory / "metrics.json", directory / "manifest.json"
        if not config_path.exists() or not metrics_path.exists():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        except (json.JSONDecodeError, OSError):
            continue
        metrics, history = payload.get("metrics", {}), payload.get("history", {})
        validation = history.get("val_accuracy", [])
        stopping = payload.get("early_stopping", {})
        rows.append({
            "run_id": directory.name, "status": manifest.get("status", "unknown"),
            "dataset": config.get("dataset"), "CNN": str(config.get("model_type", "unknown")).upper(),
            "segmentador": config.get("segmenter", "nenhum"), "batch": config.get("batch_size"),
            "batch efetivo": config.get("batch_size", 0) * config.get("gradient_accumulation_steps", 1),
            "learning rate": config.get("learning_rate"), "dropout": config.get("dropout"), "seed": config.get("seed"),
            "OA": metrics.get("overall_accuracy"), "AA": metrics.get("average_accuracy"), "Kappa": metrics.get("kappa"),
            "melhor validação": max(validation) if validation else float("nan"),
            "épocas executadas": payload.get("epochs_run", len(validation)),
            "parou cedo": payload.get("stopped_early", False),
            "meta validação": stopping.get("target", config.get("early_stopping_target")),
            "meta atingida": stopping.get("target_reached", False),
            "treino (s)": payload.get("train_time"), "pré-processamento (s)": payload.get("prep_timings", {}).get("wall"),
            "acurácia por classe": metrics.get("per_class_accuracy", []), "diretório": str(directory.resolve()),
        })
    return pd.DataFrame(rows)


def show_all_run_note(key):
    """Mostra como o gráfico foi obtido e a regra correta de leitura."""
    title, note = ALL_RUN_PLOT_NOTES[key]
    display(Markdown(f"> **Como ler — {title}:** {note}"))


# ----------------------------------------------------------------------------
# FIGURAS E PDF GLOBAIS — definição única de all_run_figures e do relatório
# ----------------------------------------------------------------------------
# Curvas de evolução, cobertura, repetibilidade e o PDF consolidado de todos os runs.
#
# Usa `ALL_RUN_PLOT_NOTES` e `load_all_audited_runs`, acima, e `_pdf_text`, de
# `hsi/mlops/relatorios.py`. É o único lugar que define `_all_run_core_figures`,
# `all_run_figures` e `generate_all_runs_pdf`.


def load_all_training_histories(runs_root=None):
    """Curvas persistidas de loss e acurácia de todos os runs auditados."""
    traces=[]
    for directory in Path(runs_root or output_dir()/"runs").glob("*"):
        mp,cp=directory/"metrics.json",directory/"config.json"
        if not mp.exists() or not cp.exists(): continue
        payload=json.loads(mp.read_text(encoding="utf-8")); history=payload.get("history",{})
        if history.get("val_accuracy"):
            config=json.loads(cp.read_text(encoding="utf-8")); traces.append({"CNN":str(config.get("model_type","unknown")).upper(),"history":history})
    return traces


def plot_training_evolution_all_runs():
    """Linhas individuais, média e desvio por CNN para quatro sinais de treino."""
    traces=load_all_training_histories()
    if not traces: return None
    figure,axes=plt.subplots(2,2,figsize=(16,10),constrained_layout=True)
    # A CNN1D tem a mesma cor fixa nos quatro painéis, em vez de depender da ordem
    # da paleta automática; outra arquitetura (só em runs anteriores à restrição à
    # CNN1D) cai na cor de reserva. Linhas translúcidas são runs individuais; a
    # linha espessa e a faixa são média ± desvio.
    palette = {"CNN1D": "#0072B2"}
    for axis,(key,title) in zip(axes.ravel(),[("accuracy","Acurácia de treino"),("val_accuracy","Acurácia de validação"),("loss","Loss de treino"),("val_loss","Loss de validação")]):
        for cnn in sorted({x["CNN"] for x in traces}):
            color = palette.get(cnn, "#CC79A7")
            series=[np.asarray(x["history"].get(key,[]),float) for x in traces if x["CNN"]==cnn and x["history"].get(key)]
            if not series: continue
            for values in series:
                axis.plot(np.arange(1,len(values)+1), values, color=color, alpha=.10, lw=.9)
            length=max(map(len,series)); stack=np.full((len(series),length),np.nan)
            for row,values in enumerate(series): stack[row,:len(values)]=values
            mean,std=np.nanmean(stack,axis=0),np.nanstd(stack,axis=0); epochs=np.arange(1,length+1)
            axis.plot(epochs, mean, color=color, lw=2.8, label=f"{cnn} — média")
            axis.fill_between(epochs, mean-std, mean+std, color=color, alpha=.18, label=f"{cnn} — ±1 desvio")
        axis.set(title=title,xlabel="Época",ylabel="pontuação" if "Acurácia" in title else "cross-entropy")
        axis.legend(fontsize=8, ncol=2); axis.grid(alpha=.25)
    return figure


SEGMENTATION_REPORT_NOTES = {
    "segmenter_overlays": (
        "Saída visual dos segmentadores",
        "Cada painel usa a mesma composição PCA/RGB; a linha colorida é o contorno de um superpixel. Compare se os contornos acompanham limites visuais, sem confundir superpixel com classe prevista.",
    ),
    "segmenter_geometry": (
        "Escala e homogeneidade dos segmentadores",
        "O painel compara número e tamanho das regiões, densidade de borda, pureza rotulada e distribuições de área. Pureza mede homogeneidade da região e não é acurácia da CNN.",
    ),
}


def segmentation_report_figures(frame):
    """Gera a evidência visual dos quatro segmentadores para o PDF consolidado.

    A cena é a do run mais antigo entre os `config.json` auditados; com runs de
    várias cenas, só ela aparece. O cálculo usa as receitas padrão de visualização,
    não altera qualquer run e falha de maneira informativa caso os arquivos da cena
    não estejam disponíveis localmente.
    """
    datasets = [] if frame.empty else frame["dataset"].dropna().unique().tolist()
    dataset = datasets[0] if datasets else ExperimentConfig().dataset
    try:
        scene = load_scene(dataset, download=False)
        return {
            "segmenter_overlays": plot_segmentation_overlay_suite(scene),
            "segmenter_geometry": plot_segmentation_quality_dashboard(scene),
        }, None
    except Exception as error:
        return {}, f"Não foi possível gerar a saída dos segmentadores para {dataset}: {type(error).__name__}: {error}"


def _all_run_core_figures(frame):
    """Gráficos-base do conjunto de runs; `all_run_figures` os complementa."""
    import seaborn as sns
    palette = dict(zip(sorted(frame["CNN"].dropna().unique()), sns.color_palette("tab10")))
    figures = {}
    melted = frame.melt(id_vars=["CNN"], value_vars=["OA", "AA", "Kappa"], var_name="métrica", value_name="valor").dropna()
    figure, axis = plt.subplots(figsize=(13, 6)); sns.boxplot(data=melted, x="métrica", y="valor", hue="CNN", ax=axis, palette=palette); sns.stripplot(data=melted, x="métrica", y="valor", hue="CNN", dodge=True, ax=axis, palette=palette, alpha=.45, legend=False); axis.set(title="Qualidade de todos os runs", ylim=(0, 1)); axis.grid(axis="y", alpha=.25); figures["quality"] = figure
    heat = frame.pivot_table(index="CNN", columns="segmentador", values="AA", aggfunc="mean")
    if not heat.empty:
        figure, axis = plt.subplots(figsize=(10, 5)); sns.heatmap(heat, annot=True, fmt=".3f", vmin=0, vmax=1, cmap="YlGnBu", ax=axis); axis.set_title("AA média por CNN e segmentador"); figures["segmenter"] = figure
    figure, axis = plt.subplots(figsize=(11, 6)); sns.scatterplot(data=frame, x="treino (s)", y="AA", hue="CNN", size="batch", sizes=(35, 240), palette=palette, ax=axis); axis.set(title="Pareto: qualidade × custo", ylim=(0, 1)); axis.grid(alpha=.25); figures["pareto"] = figure
    times = frame.melt(id_vars=["CNN"], value_vars=["treino (s)", "pré-processamento (s)"], var_name="etapa", value_name="segundos").dropna()
    figure, axis = plt.subplots(figsize=(12, 6)); sns.boxplot(data=times, x="etapa", y="segundos", hue="CNN", palette=palette, ax=axis); axis.set(title="Distribuição de tempo por etapa"); axis.grid(axis="y", alpha=.25); figures["times"] = figure
    figure, axis = plt.subplots(figsize=(11, 6)); sns.scatterplot(data=frame, x="learning rate", y="melhor validação", hue="CNN", size="dropout", sizes=(40, 220), palette=palette, ax=axis); axis.set_xscale("log"); axis.set(title="Hiperparâmetros: learning rate × validação", ylim=(0, 1)); axis.grid(alpha=.25); figures["hyperparameters"] = figure
    figure, axis = plt.subplots(figsize=(11, 6)); sns.scatterplot(data=frame, x="épocas executadas", y="melhor validação", hue="parou cedo", style="CNN", palette={True: "#E45756", False: "#54A24B"}, ax=axis); axis.set(title="Convergência e parada antecipada", ylim=(0, 1)); axis.grid(alpha=.25); figures["early"] = figure
    figure, axis = plt.subplots(figsize=(11, 6)); sns.scatterplot(data=frame, x="batch", y="melhor validação", hue="CNN", style="segmentador", palette=palette, ax=axis); axis.set(title="Batch físico × melhor validação", ylim=(0, 1)); axis.grid(alpha=.25); figures["batch"] = figure
    ordered = frame.sort_values("run_id").reset_index(drop=True)
    figure, axis = plt.subplots(figsize=(14, 6)); axis.plot(ordered.index + 1, ordered["AA"], marker="o", label="AA"); axis.plot(ordered.index + 1, ordered["melhor validação"], marker=".", label="melhor validação"); axis.set(title="Linha do tempo dos runs", xlabel="Ordem temporal do run", ylabel="Pontuação", ylim=(0, 1)); axis.legend(); axis.grid(alpha=.25); figures["timeline"] = figure
    compatible = frame[frame["acurácia por classe"].map(len) == frame["acurácia por classe"].map(len).max()]
    if not compatible.empty and compatible["acurácia por classe"].map(len).iloc[0] > 0:
        values = compatible.groupby("CNN")["acurácia por classe"].apply(lambda series: np.mean(np.vstack(series), axis=0))
        matrix = np.vstack(values.to_list())
        figure, axis = plt.subplots(figsize=(14, max(3, len(values) * .8))); sns.heatmap(matrix, vmin=0, vmax=1, cmap="viridis", annot=True, fmt=".2f", yticklabels=values.index, ax=axis); axis.set(title="Acurácia média por classe", xlabel="Classe", ylabel="CNN"); figures["classes"] = figure
    return figures


def all_run_figures(frame):
    """Todos os gráficos inferíveis de config.json, metrics.json e manifest.json.

    Definição única: os painéis-base vêm de `_all_run_core_figures` e os painéis de
    evolução, cobertura e repetibilidade são acrescentados aqui. Não há redefinição
    posterior, então chamar duas vezes produz exatamente o mesmo objeto.
    """
    if frame.empty:
        return {}
    figures = _all_run_core_figures(frame)
    import seaborn as sns
    figure,axis=plt.subplots(figsize=(8,7)); sns.scatterplot(data=frame,x="OA",y="AA",hue="CNN",style="segmentador",s=85,ax=axis); axis.plot([0,1],[0,1],"--",color="black",lw=1); axis.set(title="OA × AA: efeito do desbalanceamento",xlim=(0,1),ylim=(0,1)); axis.grid(alpha=.25); figures["gaps"]=figure
    coverage=frame.groupby(["CNN","segmentador"]).size().reset_index(name="runs"); figure,axis=plt.subplots(figsize=(12,5)); sns.barplot(data=coverage,x="CNN",y="runs",hue="segmentador",ax=axis); axis.set(title="Cobertura de runs por CNN e segmentador",ylabel="quantidade de runs"); axis.grid(axis="y",alpha=.25); figures["coverage"]=figure
    training=plot_training_evolution_all_runs()
    if training is not None: figures["training"]=training

    figure, axis = plt.subplots(figsize=(9, 7))
    sns.scatterplot(data=frame, x="melhor validação", y="AA", hue="CNN", style="segmentador", s=90, ax=axis)
    axis.plot([0, 1], [0, 1], "--", color="black", lw=1, label="referência 1:1")
    axis.set(title="Melhor validação × AA de teste", xlim=(0, 1), ylim=(0, 1), xlabel="melhor validação", ylabel="AA de teste")
    axis.grid(alpha=.25); axis.legend()
    figures["validation_test"] = figure
    stopping = frame.groupby("CNN")[["meta atingida", "parou cedo"]].mean().reset_index().melt(id_vars="CNN", var_name="evento", value_name="proporção")
    figure, axis = plt.subplots(figsize=(10, 5))
    sns.barplot(data=stopping, x="CNN", y="proporção", hue="evento", ax=axis)
    axis.set(title="Meta de validação e early stopping", ylabel="proporção de runs", ylim=(0, 1))
    axis.grid(axis="y", alpha=.25)
    figures["stopping"] = figure
    if "seed" in frame and frame["seed"].notna().any():
        figure, axis = plt.subplots(figsize=(11, 6))
        sns.boxplot(data=frame, x="CNN", y="AA", hue="CNN", legend=False, ax=axis)
        sns.stripplot(data=frame, x="CNN", y="AA", hue="seed", dodge=False, palette="tab10", ax=axis)
        axis.set(title="AA por seed: repetibilidade", ylabel="AA de teste", ylim=(0, 1))
        axis.grid(axis="y", alpha=.25); axis.legend(title="seed", bbox_to_anchor=(1.02, 1), loc="upper left")
        figures["seeds"] = figure
    return figures


def generate_all_runs_pdf(runs_root=None, reports_root=None):
    """PDF visual de todos os runs, mais o CSV bruto para conferência externa.

    O PDF é inteiramente visual (sem páginas de tabela); quem precisar dos números
    exatos usa o CSV, que carrega as mesmas linhas lidas dos artefatos auditados.
    """
    frame = load_all_audited_runs(runs_root)
    reports = Path(reports_root or output_dir() / "reports")
    reports.mkdir(parents=True, exist_ok=True)
    csv_path, pdf_path = reports / "all_runs.csv", reports / "all_runs_report.pdf"
    frame.drop(columns=["acurácia por classe"], errors="ignore").to_csv(csv_path, index=False)
    with PdfPages(pdf_path) as pdf:
        _pdf_text(pdf,"Relatório visual consolidado",f"Runs incluídos: {len(frame)}\nDatasets: {sorted(frame['dataset'].dropna().unique().tolist()) if not frame.empty else []}\nOrigem: config.json, metrics.json e manifest.json de cada run.")
        for key,figure in all_run_figures(frame).items():
            pdf.savefig(figure,bbox_inches="tight"); _pdf_text(pdf,f"Como ler: {ALL_RUN_PLOT_NOTES[key][0]}",ALL_RUN_PLOT_NOTES[key][1]); plt.close(figure)
        segment_figures, segment_error = segmentation_report_figures(frame)
        if segment_error:
            _pdf_text(pdf, "Saídas dos segmentadores", segment_error)
        for key, figure in segment_figures.items():
            pdf.savefig(figure, bbox_inches="tight")
            _pdf_text(pdf, f"Como ler: {SEGMENTATION_REPORT_NOTES[key][0]}", SEGMENTATION_REPORT_NOTES[key][1])
            plt.close(figure)
    return pdf_path, csv_path, frame
