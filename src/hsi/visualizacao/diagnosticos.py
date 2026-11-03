"""Visualizações explicáveis para classificação hiperespectral.

As figuras são diagnósticos, não apenas ilustrações: elas ajudam a verificar se a
cena foi lida corretamente, se o PCA preserva estruturas espaciais, se os
superpixels são coerentes e se a qualidade do classificador é uniforme entre as
classes. Nenhuma altera os dados que recebe. As de segmentação refazem PCA e
superpixels a partir da cena, com os parâmetros padrão de cada segmentador; as de
avaliação usam o resultado de um run, e `profile_inference` repete a inferência de
teste só para cronometrá-la.
"""

from __future__ import annotations

from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    jaccard_score,
    precision_recall_fscore_support,
)

from hsi.avaliacao import predict_proba
from hsi.config import ExperimentConfig
from hsi.dados.cenas import Scene, to_pixel_table
from hsi.dados.particoes import build_loaders
from hsi.dados.preprocessamento import apply_pca, rgb_composite
from hsi.dados.segmentacao import colorize_segments, segment_superpixels
from hsi.paralelo.ramos import StageTimings, _segmenter_params
from hsi.pipeline import ExperimentResult


def _robust_rgb(cube: np.ndarray) -> np.ndarray:
    """Cria uma composição RGB de falsa cor com três bandas bem espaçadas.

    O percentil 2–98 reduz a influência de pixels extremos; ele melhora a leitura
    visual, mas nunca é usado como entrada do modelo ou da segmentação.
    """
    bands = np.linspace(0, cube.shape[2] - 1, 3, dtype=int)
    image = cube[:, :, bands].astype(np.float64)
    low, high = np.percentile(image, (2, 98), axis=(0, 1), keepdims=True)
    return np.clip((image - low) / np.maximum(high - low, 1e-12), 0, 1)


def _label_cmap(n_classes: int):
    """Mapa discreto com uma cor por classe (tab20 reamostrado).

    O fundo não é tratado aqui: quem chama marca os pixels sem rótulo como
    inválidos (máscara ou NaN), e o matplotlib não pinta valores inválidos.
    """
    return plt.colormaps.get_cmap("tab20").resampled(max(n_classes, 1))


def plot_scene_overview(scene: Scene):
    """Mostra a cena, ground truth, desbalanceamento e assinaturas espectrais.

    A distribuição de classes explica por que OA sozinha pode enganar: uma classe
    muito frequente pode dominar a média. As curvas espectrais mostram o quanto as
    classes são separáveis antes da CNN.
    """
    labels = scene.labels
    classes, counts = np.unique(labels[labels > 0], return_counts=True)
    pixels, flat_labels = to_pixel_table(scene)
    cmap = _label_cmap(scene.n_classes)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    axes[0, 0].imshow(_robust_rgb(scene.image))
    axes[0, 0].set_title("Composição de falsa cor (bandas espaçadas)")
    axes[0, 0].axis("off")

    # A classe k usa a cor k-1 do mapa, a mesma da sua barra no gráfico de
    # contagens; o fundo mascarado fica sem cor.
    ground_truth = np.ma.masked_where(labels == 0, labels - 1)
    im = axes[0, 1].imshow(ground_truth, cmap=cmap, vmin=0, vmax=max(scene.n_classes - 1, 0))
    axes[0, 1].set_title("Ground truth — apenas pixels rotulados")
    axes[0, 1].axis("off")
    fig.colorbar(im, ax=axes[0, 1], shrink=.78, label="Classe (índice)")

    axes[1, 0].bar(classes, counts, color=cmap(np.arange(len(classes))))
    axes[1, 0].set(title="Amostras rotuladas por classe", xlabel="Classe", ylabel="Pixels")
    axes[1, 0].grid(axis="y", alpha=.25)

    for class_id in classes:
        spectra = pixels[flat_labels == class_id]
        axes[1, 1].plot(np.mean(spectra, axis=0), lw=1.2, label=str(class_id))
    axes[1, 1].set(title="Assinatura espectral média por classe", xlabel="Banda", ylabel="Intensidade média")
    axes[1, 1].legend(title="Classe", ncol=2, fontsize=8)
    axes[1, 1].grid(alpha=.25)
    return fig


def pca_diagnostics(scene: Scene, n_components: int = 12):
    """Calcula PCA sem whitening para interpretar variância e mapas de características.

    O pipeline usa `whiten=True`, que só reescala cada componente; sem ele, a escala
    de cada mapa acompanha a variância da componente. O ajuste usa a cena inteira,
    como o ramo espacial (o ramo espectral ajusta o seu só nos pixels rotulados).
    """
    pixels, _ = to_pixel_table(scene)
    n_components = min(n_components, pixels.shape[1], pixels.shape[0])
    model = PCA(n_components=n_components, random_state=42)
    components = model.fit_transform(pixels)
    return components, model


def plot_pca_diagnostics(scene: Scene, n_components: int = 12, maps: int = 6):
    """Exibe a variância explicada e os mapas das primeiras componentes principais.

    As três primeiras PCs são, a menos de escala, os canais do RGB que o
    segmentador recebe no ramo espacial: regiões com padrões semelhantes devem ter
    valores semelhantes nelas.
    """
    components, model = pca_diagnostics(scene, n_components)
    maps = min(maps, components.shape[1])
    ncols = 3
    nrows = int(np.ceil(maps / ncols))
    fig = plt.figure(figsize=(15, 4 + 3.5 * nrows), constrained_layout=True)
    grid = fig.add_gridspec(nrows + 1, ncols)
    axis = fig.add_subplot(grid[0, :])
    explained = model.explained_variance_ratio_
    axis.bar(np.arange(1, len(explained) + 1), explained * 100, alpha=.55, label="por componente")
    axis.plot(np.arange(1, len(explained) + 1), np.cumsum(explained) * 100, marker="o", color="crimson", label="acumulada")
    axis.set(title="PCA: variância explicada", xlabel="Componente", ylabel="Variância (%)")
    axis.legend()
    axis.grid(axis="y", alpha=.25)
    for index in range(maps):
        ax = fig.add_subplot(grid[1 + index // ncols, index % ncols])
        image = components[:, index].reshape(scene.shape)
        im = ax.imshow(image, cmap="viridis")
        ax.set_title(f"Mapa de característica: PC {index + 1}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=.75)
    return fig, components, model


def segmentation_quality(labels: np.ndarray, ground_truth: np.ndarray) -> dict:
    """Mede tamanho, bordas e pureza sem confundir superpixel com classe.

    *Pureza* é a fração da classe dominante dentro de cada superpixel, ponderada
    pelos pixels rotulados. É diagnóstico espacial; não é uma métrica de acurácia
    do classificador e não deve ser usada para escolher modelo isoladamente.
    """
    areas = np.bincount(labels.ravel())
    valid = ground_truth.ravel() > 0
    segment_ids = labels.ravel()[valid]
    true_labels = ground_truth.ravel()[valid]
    weighted_hits = 0
    for segment_id in np.unique(segment_ids):
        segment_truth = true_labels[segment_ids == segment_id]
        weighted_hits += np.bincount(segment_truth).max()
    # Densidade de borda: pares de vizinhos (horizontais e verticais) em regiões
    # diferentes, por pixel da cena. Cresce com a fragmentação.
    horizontal = np.count_nonzero(labels[:, 1:] != labels[:, :-1])
    vertical = np.count_nonzero(labels[1:, :] != labels[:-1, :])
    return {
        "superpixels": int(len(areas)),
        "área média": float(areas.mean()),
        "área mediana": float(np.median(areas)),
        "densidade de borda": float((horizontal + vertical) / labels.size),
        "pureza rotulada": float(weighted_hits / max(len(true_labels), 1)),
    }


def plot_segmentation_comparison(scene: Scene, algorithms=("seeds", "slic", "lsc", "watershed")):
    """Compara contornos, regiões e estatísticas dos segmentadores disponíveis.

    Todos recebem a mesma composição RGB criada pelas três primeiras PCs; assim a
    comparação isola o algoritmo. Os parâmetros são os padrão da cena
    (`SEEDS_PARAMS` e afins), sem os `segmenter_overrides` de `config` ou do HPO.
    """
    pixels, _ = to_pixel_table(scene)
    components, _ = apply_pca(pixels, n_components=3, random_state=42)
    rgb = rgb_composite(components, scene.shape)
    fig, axes = plt.subplots(2, len(algorithms), figsize=(6 * len(algorithms), 9), constrained_layout=True)
    axes = np.asarray(axes).reshape(2, len(algorithms))
    rows = []
    for index, algorithm in enumerate(algorithms):
        local = ExperimentConfig(dataset=scene.code, segmenter=algorithm)
        result = segment_superpixels(rgb, algorithm, _segmenter_params(local))
        outlined = rgb.copy()
        outlined[result.contour_mask > 0] = (255, 255, 255)
        axes[0, index].imshow(outlined)
        axes[0, index].set_title(f"{algorithm.upper()}: contornos")
        axes[1, index].imshow(colorize_segments(result.labels, seed=42))
        axes[1, index].set_title(f"{algorithm.upper()}: regiões")
        for axis in axes[:, index]:
            axis.axis("off")
        rows.append({"segmentador": algorithm.upper(), **segmentation_quality(result.labels, scene.labels)})
    return fig, pd.DataFrame(rows).set_index("segmentador")


def classification_metrics_table(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Calcula métricas por classe e os resumos macro para classes desbalanceadas.

    OA, AA (recall macro) e Kappa são o trio consagrado em HSI. Precisão, F1 e IoU
    expõem erros assimétricos por classe; elas complementam, mas não substituem, o
    trio para comparação com trabalhos da literatura.

    OA e Kappa não são recalculados aqui: vêm de `Metrics`. A coluna `classe` volta
    à numeração 1..C do ground truth; os rótulos de entrada são 0..C-1.
    """
    labels = np.arange(max(int(y_true.max()), int(y_pred.max())) + 1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0,
    )
    iou = jaccard_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class = pd.DataFrame({
        "classe": labels + 1,
        "suporte": support,
        "precisão": precision,
        "revocação / AA por classe": recall,
        "F1": f1,
        "IoU": iou,
    })
    summary = {
        "AA / balanced accuracy": balanced_accuracy_score(y_true, y_pred),
        "F1 macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "IoU macro": jaccard_score(y_true, y_pred, average="macro", zero_division=0),
    }
    return per_class, summary


def plot_evaluation_dashboard(result: ExperimentResult, scene: Scene):
    """Exibe convergência, métricas, matriz de confusão e mapa de erros do teste."""
    metrics = result.metrics
    history = result.training.history.to_dict()
    y_true = result.data.labels[result.data.splits.test]
    y_pred = result.predicted
    per_class, extra = classification_metrics_table(y_true, y_pred)
    fig, axes = plt.subplots(2, 3, figsize=(20, 11), constrained_layout=True)

    epochs = result.training.history.epochs
    axes[0, 0].plot(epochs, history["loss"], label="treino")
    axes[0, 0].plot(epochs, history["val_loss"], label="validação")
    axes[0, 0].set(title="Convergência: loss", xlabel="Época", ylabel="Cross-entropy")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=.25)
    axes[0, 1].plot(epochs, history["accuracy"], label="treino")
    axes[0, 1].plot(epochs, history["val_accuracy"], label="validação")
    training = result.training
    axes[0, 1].axvline(training.best_epoch, color="#E45756", ls="--", lw=1.5,
                        label=f"melhor época ({training.best_epoch})")
    if training.stopped_early:
        axes[0, 1].axvline(len(training.history), color="#4C78A8", ls=":", lw=2,
                            label=f"parada antecipada ({len(training.history)})")
    target = getattr(training, "early_stopping_target", None)
    if target is not None:
        axes[0, 1].axhline(target, color="#54A24B", ls="-.", lw=1.6,
                            label=f"meta de validação ({target:.3f})")
    axes[0, 1].set(title=(f"Convergência: acurácia | {training.early_stopping_monitor}, "
                           f"patience={training.patience}, Δ≥{training.early_stopping_min_delta:g}"),
                   xlabel="Época", ylabel="Acurácia")
    axes[0, 1].legend(); axes[0, 1].grid(alpha=.25)

    values = [metrics.overall_accuracy, metrics.average_accuracy, metrics.kappa, extra["F1 macro"], extra["IoU macro"]]
    names = ["OA", "AA", "Kappa", "F1 macro", "IoU macro"]
    axes[0, 2].bar(names, values, color=["#4C78A8", "#72B7B2", "#F58518", "#54A24B", "#E45756"])
    axes[0, 2].set(title="Métricas globais", ylabel="Pontuação", ylim=(0, 1))
    axes[0, 2].grid(axis="y", alpha=.25)

    normalized = metrics.confusion / np.maximum(metrics.confusion.sum(axis=1, keepdims=True), 1)
    sns.heatmap(normalized, cmap="Blues", vmin=0, vmax=1, ax=axes[1, 0], cbar_kws={"label": "fração"})
    axes[1, 0].set(title="Matriz de confusão normalizada", xlabel="Predição", ylabel="Classe real")

    # Barras lado a lado: sobrepostas, a mais baixa some dentro da mais alta.
    axes[1, 1].bar(per_class["classe"] - .2, per_class["revocação / AA por classe"], width=.4, label="recall / AA")
    axes[1, 1].bar(per_class["classe"] + .2, per_class["F1"], width=.4, label="F1")
    axes[1, 1].set(title="Qualidade por classe", xlabel="Classe", ylabel="Pontuação", ylim=(0, 1))
    axes[1, 1].legend(); axes[1, 1].grid(axis="y", alpha=.25)

    # As linhas de `PreparedData` seguem a ordem row-major dos pixels rotulados, e
    # `predicted` segue `splits.test` (o loader de teste não embaralha); por isso
    # `flatnonzero` devolve cada amostra à sua posição. NaN fora do teste não é pintado.
    positions = np.flatnonzero(scene.labels.ravel() > 0)[result.data.splits.test]
    errors = np.full(scene.labels.size, np.nan)
    errors[positions] = (y_pred != y_true).astype(float)
    axes[1, 2].imshow(errors.reshape(scene.shape), cmap="Reds", vmin=0, vmax=1)
    axes[1, 2].set_title("Erros no conjunto de teste (vermelho)")
    axes[1, 2].axis("off")
    return fig, per_class, extra


def profile_inference(result: ExperimentResult) -> pd.DataFrame:
    """Mede inferência real e memória de GPU sem misturar esse tempo ao treino.

    O pico parte do que já estava alocado ao zerar a estatística — os pesos deste
    modelo e de qualquer outro ainda vivo na GPU —, não só da inferência.
    """
    loaders = build_loaders(result.data, result.config, result.device)
    # CUDA é assíncrona: sem sincronizar antes e depois, o cronômetro incluiria
    # kernels pendentes de antes e pararia sem esperar os desta inferência.
    if result.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(result.device)
        torch.cuda.synchronize(result.device)
    started = perf_counter()
    probabilities = predict_proba(result.model, loaders["test"], result.device)
    if result.device.type == "cuda":
        torch.cuda.synchronize(result.device)
        peak_memory = torch.cuda.max_memory_allocated(result.device) / 2**20
    else:
        peak_memory = np.nan
    elapsed = perf_counter() - started
    return pd.DataFrame([{
        "amostras": len(probabilities),
        "inferência (s)": elapsed,
        "ms / pixel": 1000 * elapsed / max(len(probabilities), 1),
        "pico GPU (MiB)": peak_memory,
        "parâmetros": result.model.n_parameters,
    }])


def plot_parallel_profile(timings: StageTimings):
    """Visualiza os ramos independentes e o ganho potencial do paralelismo pré-fusão.

    Em backend paralelo, cada ramo é cronometrado enquanto disputa CPU com o outro,
    então a soma tende a superestimar o tempo sequencial real. A comparação honesta
    é o `wall` de uma execução `sequential` (`StageTimings.speedup_over`).
    """
    durations = dict(timings.stages)
    fig, axis = plt.subplots(figsize=(10, 3.5), constrained_layout=True)
    colors = {"espectral": "#4C78A8", "espacial": "#F58518"}
    for row, (name, duration) in enumerate(durations.items()):
        axis.barh(row, duration, color=colors.get(name, "#72B7B2"), label=name)
        axis.text(duration, row, f" {duration:.3f}s", va="center")
    sequential = sum(durations.values())
    axis.axvline(timings.wall, color="black", ls="--", label=f"wall: {timings.wall:.3f}s")
    axis.axvline(sequential, color="gray", ls=":", label=f"sequencial estimado: {sequential:.3f}s")
    axis.set(yticks=range(len(durations)), yticklabels=list(durations), xlabel="Tempo (s)", title=f"Pré-processamento — backend: {timings.backend}")
    axis.legend(loc="lower right")
    return fig


def plot_runs_dashboard(runs):
    """Compara qualidade e custo dos runs rastreados; o ideal fica no alto à esquerda."""
    rows = [run.row() for run in runs if run.metrics]
    if not rows:
        print("Não há runs concluídos com métricas para visualizar.")
        return None
    # O `run_id` começa pelo carimbo UTC, então ordená-lo é ordem cronológica; o
    # rótulo usa só o sufixo aleatório final, curto e único.
    frame = pd.DataFrame(rows).sort_values("run_id")
    labels = [run_id.split("_")[-1] for run_id in frame["run_id"]]
    fig, axes = plt.subplots(1, 2, figsize=(17, 5), constrained_layout=True)
    axes[0].plot(labels, frame["oa"], marker="o", label="OA")
    axes[0].plot(labels, frame["aa"], marker="o", label="AA")
    axes[0].plot(labels, frame["kappa"], marker="o", label="Kappa")
    axes[0].set(title="Métricas por run", xlabel="Sufixo do run", ylabel="Pontuação", ylim=(0, 1))
    axes[0].tick_params(axis="x", rotation=45); axes[0].legend(); axes[0].grid(alpha=.25)
    scatter = axes[1].scatter(frame["train_seconds"], frame["oa"], s=90, c=frame["aa"], cmap="viridis", edgecolor="black")
    for _, row in frame.iterrows():
        axes[1].annotate(row["model_type"], (row["train_seconds"], row["oa"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[1].set(title="Trade-off: qualidade × tempo de treino", xlabel="Tempo de treino (s)", ylabel="OA")
    axes[1].grid(alpha=.25)
    fig.colorbar(scatter, ax=axes[1], label="AA")
    return fig


def plot_per_class_metrics(per_class):
    """Todas as métricas de classificação por classe em um único gráfico."""
    figure, axis = plt.subplots(figsize=(14, 5))
    classes = per_class["classe"].to_numpy(); width=.25
    axis.bar(classes-width, per_class["precisão"], width, label="precisão")
    axis.bar(classes, per_class["revocação / AA por classe"], width, label="recall / AA")
    axis.bar(classes+width, per_class["F1"], width, label="F1")
    axis.plot(classes, per_class["IoU"], "o-", color="#E45756", label="IoU")
    axis.set(title="Classificação por classe", xlabel="Classe", ylabel="pontuação", ylim=(0,1))
    axis.legend(ncol=4); axis.grid(axis="y", alpha=.25)
    return figure


def plot_inference_profile_chart(profile):
    """Tempo, latência e pico de memória em gráficos."""
    row=profile.iloc[0]; figure, axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for axis,label,value,color,ylabel,title in zip(axes,["inferência","pixel","GPU"],[row["inferência (s)"],row["ms / pixel"],row["pico GPU (MiB)"]],["#4C78A8","#F58518","#54A24B"],["segundos","ms","MiB"],["Tempo de inferência","Latência por pixel","Pico de memória"]):
        axis.bar([label],[value],color=color); axis.set(ylabel=ylabel,title=title); axis.grid(axis="y",alpha=.25)
    return figure


def plot_segment_metrics_visual(segments):
    """Acurácia, confiança, entropia e F1 dos superpixels."""
    frame=segments.copy().sort_values("acurácia",ascending=False)
    figure,axes=plt.subplots(1,2,figsize=(16,5),constrained_layout=True)
    scatter=axes[0].scatter(frame["confiança"],frame["acurácia"],c=frame["entropia"],s=55,cmap="magma_r",edgecolor="black",linewidth=.25)
    axes[0].set(title="Segmentos: confiança × acurácia",xlabel="confiança",ylabel="acurácia",xlim=(0,1),ylim=(0,1)); axes[0].grid(alpha=.25); figure.colorbar(scatter,ax=axes[0],label="entropia")
    order=np.arange(1,len(frame)+1); axes[1].plot(order,frame["F1 da classe"],label="F1 da classe"); axes[1].plot(order,frame["entropia"],label="entropia")
    axes[1].set(title="Segmentos ordenados por acurácia",xlabel="posição do segmento",ylabel="pontuação",ylim=(0,1)); axes[1].legend(); axes[1].grid(alpha=.25)
    return figure


def plot_cnn_summary_visual(summary):
    """Métricas, custo e memória do dashboard individual."""
    figure,axes=plt.subplots(1,3,figsize=(17,4.5),constrained_layout=True)
    metrics=["OA","AA","Kappa","precisão macro","F1 macro","IoU macro"]; axes[0].bar(metrics,[summary.get(x,np.nan) for x in metrics],color=plt.cm.tab10.colors[:len(metrics)]); axes[0].set(title="Qualidade global",ylabel="pontuação",ylim=(0,1)); axes[0].tick_params(axis="x",rotation=35)
    costs=["pré-processamento (s)","treino (s)","inferência (s)"]; axes[1].bar(["pré-processamento","treino","inferência"],[summary.get(x,np.nan) for x in costs],color=["#72B7B2","#F58518","#4C78A8"]); axes[1].set(title="Custo temporal",ylabel="segundos")
    axes[2].bar(["parâmetros","pico GPU (MiB)"],[summary.get("parâmetros",np.nan),summary.get("pico GPU (MiB)",np.nan)],color=["#54A24B","#B279A2"]); axes[2].set(title="Capacidade e memória")
    for axis in axes: axis.grid(axis="y",alpha=.25)
    return figure


def plot_layer_statistics_visual(layers):
    """Média, desvio e esparsidade de todas as camadas monitoradas."""
    figure,axes=plt.subplots(1,2,figsize=(16,5),constrained_layout=True)
    names=layers["camada"].astype(str) if "camada" in layers else np.arange(len(layers)).astype(str)
    axes[0].bar(names,layers["média"],yerr=layers["desvio"],capsize=3,color="#4C78A8"); axes[0].set(title="Ativação média ± desvio",xlabel="Camada",ylabel="ativação"); axes[0].tick_params(axis="x",rotation=35)
    axes[1].bar(names,layers["% zero"],color="#E45756"); axes[1].set(title="Esparsidade por camada",xlabel="Camada",ylabel="% zero",ylim=(0,100)); axes[1].tick_params(axis="x",rotation=35)
    for axis in axes: axis.grid(axis="y",alpha=.25)
    return figure


def plot_segmentation_quality_dashboard(scene, algorithms=("seeds", "slic", "lsc", "watershed")):
    """Converte cada propriedade dos superpixels em gráficos comparáveis.

    O painel mede somente a geometria e a homogeneidade dos segmentos. Pureza não
    é acurácia do classificador: ela responde se uma região tende a conter uma
    única classe rotulada antes da etapa de classificação.
    """
    pixels, _ = to_pixel_table(scene)
    components, _ = apply_pca(pixels, n_components=3, random_state=42)
    rgb = rgb_composite(components, scene.shape)
    rows, area_samples = [], {}
    for algorithm in algorithms:
        local = ExperimentConfig(dataset=scene.code, segmenter=algorithm)
        segmentation = segment_superpixels(rgb, algorithm, _segmenter_params(local))
        rows.append({"segmentador": algorithm.upper(), **segmentation_quality(segmentation.labels, scene.labels)})
        area_samples[algorithm.upper()] = np.bincount(segmentation.labels.ravel())
    frame = pd.DataFrame(rows).set_index("segmentador")
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    colors = list(plt.cm.tab10.colors[:len(frame)])
    frame["superpixels"].plot.bar(ax=axes[0, 0], color=colors, legend=False)
    axes[0, 0].set(title="Quantidade de superpixels", xlabel="Segmentador", ylabel="Regiões")
    frame[["área média", "área mediana"]].plot.bar(ax=axes[0, 1], color=["#4C78A8", "#F58518"])
    axes[0, 1].set(title="Tamanho das regiões", xlabel="Segmentador", ylabel="Pixels por região")
    frame["densidade de borda"].plot.bar(ax=axes[0, 2], color=colors, legend=False)
    axes[0, 2].set(title="Densidade de borda", xlabel="Segmentador", ylabel="Fração de vizinhos em borda")
    frame["pureza rotulada"].plot.bar(ax=axes[1, 0], color=colors, legend=False, ylim=(0, 1))
    axes[1, 0].set(title="Pureza rotulada", xlabel="Segmentador", ylabel="Fração da classe dominante")
    for name, areas in area_samples.items():
        axes[1, 1].hist(areas, bins=30, alpha=.45, label=name, density=True)
        axes[1, 2].plot(np.sort(areas), np.linspace(0, 1, len(areas)), label=name)
    axes[1, 1].set(title="Distribuição do tamanho das regiões", xlabel="Pixels por região", ylabel="Densidade")
    axes[1, 2].set(title="Escala acumulada das regiões", xlabel="Pixels por região", ylabel="Fração acumulada", ylim=(0, 1))
    for axis in axes.ravel():
        axis.grid(axis="y", alpha=.25)
    axes[0, 1].legend(title="Medição")
    axes[1, 1].legend(title="Segmentador", fontsize=8)
    axes[1, 2].legend(title="Segmentador", fontsize=8)
    figure.suptitle("Geometria e homogeneidade dos segmentadores", fontsize=15)
    return figure


def plot_segmentation_overlay_suite(scene,algorithms=("seeds","slic","lsc","watershed")):
    """Contornos de todos os segmentadores sobre a mesma composição PCA/RGB."""
    pixels,_=to_pixel_table(scene); components,_=apply_pca(pixels,n_components=3,random_state=42); rgb=rgb_composite(components,scene.shape)
    figure,axes=plt.subplots(2,2,figsize=(15,12),constrained_layout=True); colors=["#00FFFF","#FF00FF","#FFFF00","#FF7F0E"]
    for axis,algorithm,color in zip(axes.ravel(),algorithms,colors):
        local=ExperimentConfig(dataset=scene.code,segmenter=algorithm); segmentation=segment_superpixels(rgb,algorithm,_segmenter_params(local))
        axis.imshow(rgb)
        axis.contour(segmentation.contour_mask > 0, levels=[.5], colors=[color], linewidths=.45)
        axis.plot([], [], color=color, label="contorno do superpixel")
        axis.legend(loc="lower right", fontsize=8, framealpha=.8)
        axis.set(title=f"Sobreposição {algorithm.upper()}")
        axis.axis("off")
    return figure
