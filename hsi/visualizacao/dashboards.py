"""Dashboard da CNN1D.

Todo resultado passa exatamente pelo mesmo protocolo de avaliação: métricas
globais, métricas por classe, custo, incerteza, Grad-CAM, saliência e estatísticas
de todas as camadas-folha. É o que torna os dashboards comparáveis no compilado
final. A visualização de cada camada segue a geometria do tensor: filtros ×
posição nas convoluções, vetor nas camadas densas.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from hsi.visualizacao.diagnosticos import (
    classification_metrics_table,
    plot_cnn_summary_visual,
    plot_evaluation_dashboard,
    plot_layer_statistics_visual,
    plot_per_class_metrics,
    profile_inference,
)
from hsi.visualizacao.explicabilidade import (
    explanation_sample,
    grad_cam,
    plot_grad_cam,
    plot_input_saliency,
    plot_maxpool_effect,
    plot_prediction_uncertainty,
)


def capture_all_layer_outputs(model, x, seg=None):
    """Observa cada camada-folha executada numa inferência da amostra escolhida.

    Na CNN1D são as convoluções e suas ReLUs, o Flatten, as camadas lineares e o
    Dropout (identidade em `eval`), mais a `nn.Embedding` quando
    `fusion="embedding"`. A fusão de segmento é uma operação funcional
    (`torch.cat`) e, por não ser um módulo PyTorch, aparece indiretamente na
    primeira Linear da cabeça classificadora.
    """
    records, handles = [], []
    for name, module in model.named_modules():
        if not name or any(module.children()):
            continue
        def hook(_module, _inputs, output, layer_name=name, layer_type=type(module).__name__):
            value = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(value):
                records.append({
                    "nome": layer_name,
                    "tipo": layer_type,
                    "ativação": value.detach()[0].float().cpu().numpy(),
                })
        handles.append(module.register_forward_hook(hook))
    previous_mode = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits = model(x, seg)
    finally:
        for handle in handles:
            handle.remove()
        model.train(previous_mode)
    return records, int(logits.argmax(dim=1).item())


def layer_statistics(records):
    """Resume shape, dispersão e esparsidade de todas as ativações capturadas."""
    rows = []
    for index, record in enumerate(records, start=1):
        values = record["ativação"]
        rows.append({
            "ordem": index,
            "layer": record["nome"],
            "tipo": record["tipo"],
            "shape": " × ".join(map(str, values.shape)),
            "média": float(values.mean()),
            "desvio": float(values.std()),
            "mínimo": float(values.min()),
            "máximo": float(values.max()),
            "% zero": float(100 * np.mean(np.isclose(values, 0))),
        })
    return pd.DataFrame(rows)


def _layer_visual(axis, values, layer_type):
    """Escolhe a representação mais legível para a geometria de cada camada."""
    if values.ndim == 1:  # vetores: Flatten, Embedding e toda a cabeça densa
        visible = min(96, len(values))
        axis.bar(np.arange(visible), values[:visible], width=1, color="#4C78A8")
        axis.set(xlabel="Neurônio", ylabel="Ativação")
    elif values.ndim == 2:  # filtros × posição: CNN1D
        image = axis.imshow(values, aspect="auto", cmap="magma")
        axis.set(xlabel="Posição espectral", ylabel="Filtro")
        plt.colorbar(image, ax=axis, fraction=.046)
        axis.set(xlabel="X", ylabel="Y")
        plt.colorbar(image, ax=axis, fraction=.046)
        axis.set(xlabel="X", ylabel="Y")
        plt.colorbar(image, ax=axis, fraction=.046)
    else:
        axis.text(.5, .5, f"Tensor {values.shape}\nsem projeção visual", ha="center", va="center")
    axis.set_title(f"{layer_type}: {values.shape}", fontsize=9)
    axis.grid(False)


def plot_all_layers(records, layers_per_page=6):
    """Cria páginas de dashboard para *todas* as camadas, inclusive ReLU e Linear.

    À esquerda está a representação da ativação; à direita, o histograma dela. As
    páginas evitam miniaturas ilegíveis em redes com muitas camadas.
    """
    pages = {}
    for page_start in range(0, len(records), layers_per_page):
        subset = records[page_start:page_start + layers_per_page]
        fig, axes = plt.subplots(len(subset), 2, figsize=(15, 4.2 * len(subset)), constrained_layout=True)
        axes = np.atleast_2d(axes)
        for row, record in enumerate(subset):
            values = record["ativação"]
            _layer_visual(axes[row, 0], values, record["tipo"])
            axes[row, 0].set_title(f"{record['nome']} — {record['tipo']} | {values.shape}", loc="left", fontsize=10, fontweight="bold")
            axes[row, 1].hist(values.ravel(), bins=40, color="#F58518", alpha=.85)
            axes[row, 1].set(title="Distribuição das ativações", xlabel="Valor", ylabel="Contagem")
            axes[row, 1].grid(alpha=.2)
        page = page_start // layers_per_page + 1
        fig.suptitle(f"Todas as camadas — página {page}", fontsize=15)
        pages[f"camadas {page}"] = fig
    return pages


def dashboard_summary(result):
    """Calcula o mesmo vetor de métricas para qualquer resultado.

    É a base comum do compilado final. Tempo de inferência e pico de memória vêm
    de `profile_inference`, que repete a inferência de teste para medi-los.
    """
    y_true = result.data.labels[result.data.splits.test]
    y_pred = result.predicted
    per_class, extra = classification_metrics_table(y_true, y_pred)
    inference = profile_inference(result).iloc[0].to_dict()
    return {
        "CNN": result.config.model_type.upper(),
        "OA": result.metrics.overall_accuracy,
        "AA": result.metrics.average_accuracy,
        "Kappa": result.metrics.kappa,
        "precisão macro": float(per_class["precisão"].mean()),
        "F1 macro": extra["F1 macro"],
        "IoU macro": extra["IoU macro"],
        "loss teste": result.metrics.test_loss,
        "pré-processamento (s)": result.prep_timings.wall,
        "treino (s)": result.training.total_time,
        "inferência (s)": inference["inferência (s)"],
        "ms / pixel": inference["ms / pixel"],
        "pico GPU (MiB)": inference["pico GPU (MiB)"],
        "parâmetros": result.model.n_parameters,
    }, per_class


def _render_cnn_dashboard_core(result, scene, max_channels=8):
    """Núcleo do dashboard explicável de uma CNN já treinada.

    Retorna os artefatos em vez de chamar `display`, permitindo que a mesma função
    seja usada no dashboard individual e no compilado final. A versão
    pública `render_cnn_dashboard`, em `hsi/visualizacao/explicacao_inversa.py`, acrescenta a explicação espacial.
    """
    x, seg, true_class = explanation_sample(result, split="test", batch_index=0)
    records, predicted_class = capture_all_layer_outputs(result.model, x, seg)
    summary, per_class = dashboard_summary(result)
    cam, cam_class, cam_layer = grad_cam(result.model, x, seg)
    figures = {
        "métricas e erros": plot_evaluation_dashboard(result, scene)[0],
        "saliência": plot_input_saliency(result.model, x, seg, result.config.model_type),
        "Grad-CAM": plot_grad_cam(cam, result.config.model_type, cam_class, cam_layer),
        "incerteza": plot_prediction_uncertainty(result, scene),
    }
    figures.update({f"{key}": value for key, value in plot_all_layers(records).items()})
    figures.update({f"MaxPool — {key}": value for key, value in plot_maxpool_effect(records).items()})
    return {
        "summary": summary,
        "per_class": per_class,
        "layers": layer_statistics(records),
        "figures": figures,
        "classe real": true_class + 1,
        "classe prevista": predicted_class + 1,
    }


def _display_cnn_dashboard_core(name, dashboard):
    """Núcleo da exibição de um dashboard individual.

    Mostra os resumos (métricas, classes, camadas) e depois cada figura do
    dashboard sob um título Markdown.
    """
    from IPython.display import Markdown, display
    display(Markdown(f"## Dashboard — {name.upper()}"))
    display(plot_cnn_summary_visual(dashboard["summary"]))
    display(plot_per_class_metrics(dashboard["per_class"]))
    display(plot_layer_statistics_visual(dashboard["layers"]))
    print(f"Pixel explicado — classe real: {dashboard['classe real']} | prevista: {dashboard['classe prevista']}")
    for title, figure in dashboard["figures"].items():
        display(Markdown(f"#### {title}"))
        display(figure)
        # Fechar depois de exibir: o backend inline mostraria de novo, no fim da
        # célula, toda figura que continuasse aberta — e ela seguiria na memória.
        plt.close(figure)


def plot_final_cnn_compilation(dashboards):
    """Compila as mesmas métricas, classes e custo de vários dashboards numa visão final.

    Cada chave de `dashboards` vira uma linha; as células de execução do notebook usam uma
    por arquitetura, o que hoje significa só `cnn1d`.
    """
    summary = pd.DataFrame([dashboard["summary"] for dashboard in dashboards.values()]).set_index("CNN")
    per_class_rows = []
    for name, dashboard in dashboards.items():
        table = dashboard["per_class"].copy()
        table["CNN"] = name.upper()
        per_class_rows.append(table)
    per_class = pd.concat(per_class_rows, ignore_index=True)
    ordered = summary.index.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(21, 12), constrained_layout=True)
    summary[["OA", "AA", "Kappa", "precisão macro", "F1 macro", "IoU macro"]].T.plot.bar(ax=axes[0, 0], rot=0)
    axes[0, 0].set(title="Métricas globais iguais para todas as CNNs", ylabel="Pontuação", ylim=(0, 1))
    summary[["pré-processamento (s)", "treino (s)", "inferência (s)"]].plot.bar(ax=axes[0, 1], rot=0)
    axes[0, 1].set(title="Custo temporal", ylabel="Segundos")
    summary[["parâmetros", "pico GPU (MiB)"]].plot.bar(ax=axes[0, 2], rot=0)
    axes[0, 2].set(title="Capacidade e memória")
    f1_matrix = per_class.pivot(index="CNN", columns="classe", values="F1").reindex(ordered)
    im = axes[1, 0].imshow(f1_matrix, aspect="auto", vmin=0, vmax=1, cmap="YlGnBu")
    axes[1, 0].set(title="F1 por classe", xlabel="Classe", ylabel="CNN", xticks=range(f1_matrix.shape[1]), xticklabels=f1_matrix.columns)
    fig.colorbar(im, ax=axes[1, 0], label="F1")
    recall_matrix = per_class.pivot(index="CNN", columns="classe", values="revocação / AA por classe").reindex(ordered)
    im = axes[1, 1].imshow(recall_matrix, aspect="auto", vmin=0, vmax=1, cmap="magma")
    axes[1, 1].set(title="Revocação / AA por classe", xlabel="Classe", ylabel="CNN", xticks=range(recall_matrix.shape[1]), xticklabels=recall_matrix.columns)
    fig.colorbar(im, ax=axes[1, 1], label="Recall")
    axes[1, 2].scatter(summary["parâmetros"], summary["OA"], s=120, c=summary["treino (s)"], cmap="viridis", edgecolor="black")
    for name, row in summary.iterrows():
        axes[1, 2].annotate(name, (row["parâmetros"], row["OA"]), xytext=(5, 5), textcoords="offset points")
    axes[1, 2].set(title="Trade-off: capacidade × OA", xlabel="Parâmetros", ylabel="OA", ylim=(0, 1))
    axes[1, 2].grid(alpha=.25)
    for axis in axes.ravel()[:3]:
        axis.grid(axis="y", alpha=.25)
    return fig, summary, per_class
