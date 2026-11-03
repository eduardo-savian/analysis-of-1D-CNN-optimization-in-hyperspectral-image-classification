"""Reconstrói decisões e métricas nas coordenadas originais da imagem HSI.

O modelo é treinado apenas nos pixels rotulados; por isso mapas de predição,
confiança e erro são explicitamente mascarados fora do teste. Essa escolha evita
inventar uma métrica onde não há ground truth. A segmentação é reconstruída de
forma determinística com a receita registrada, e cada superpixel recebe estatísticas
agregadas das previsões que o atravessam.
"""

from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from hsi.avaliacao import predict_proba
from hsi.dados.cenas import to_pixel_table
from hsi.dados.particoes import _tensors, build_loaders
from hsi.dados.segmentacao import SegmentationArtifacts, colorize_segments
from hsi.paralelo.ramos import _segmenter_params, spatial_branch
from hsi.treino import unpack_batch
from hsi.visualizacao.dashboards import (
    _display_cnn_dashboard_core,
    _render_cnn_dashboard_core,
)
from hsi.visualizacao.diagnosticos import (
    _label_cmap,
    classification_metrics_table,
    plot_segment_metrics_visual,
)
from hsi.visualizacao.explicabilidade import grad_cam

_SPATIAL_CONTEXT_CACHE = {}


def rebuild_spatial_context(result, scene):
    """Reproduz RGB, superpixels e contornos da receita do run, com cache em memória.

    A chave é (cena, forma, segmentador, seed) e não inclui `segmenter_overrides`:
    dois runs com o mesmo segmentador e a mesma seed, mas receitas internas
    diferentes (caso comum entre vencedores do HPO), recebem a segmentação de quem
    chegou primeiro.
    """
    key = (scene.code, scene.shape, result.config.segmenter, result.config.seed)
    if key not in _SPATIAL_CONTEXT_CACHE:
        pixels, raw_labels = to_pixel_table(scene)
        spatial = spatial_branch(
            pixels, raw_labels, scene.shape, _segmenter_params(result.config),
            result.config.seed, result.config.segmenter,
        )
        _, components, rgb, segmentation = spatial
        _SPATIAL_CONTEXT_CACHE[key] = SegmentationArtifacts(components=components, rgb=rgb, result=segmentation)
    return _SPATIAL_CONTEXT_CACHE[key]


def inverse_prediction_maps(result, scene):
    """Projeta probabilidades e métricas do teste de volta à matriz H×W da cena.

    Refaz a inferência, então precisa de um resultado que `release_result` não
    tenha liberado (modelo, espectros e previsões).
    """
    loaders = build_loaders(result.data, result.config, result.device)
    probabilities = predict_proba(result.model, loaders["test"], result.device)
    predicted = probabilities.argmax(axis=1)
    true = result.data.labels[result.data.splits.test]
    positions = np.flatnonzero(scene.labels.ravel() > 0)[result.data.splits.test]
    if not np.array_equal(predicted, result.predicted):
        raise RuntimeError("Ordem de probabilidades não coincide com as previsões registradas do run.")
    per_class, _ = classification_metrics_table(true, predicted)
    f1_lookup = per_class.set_index("classe")["F1"].to_dict()
    total = scene.labels.size
    def canvas(values):
        image = np.full(total, np.nan, dtype=float)
        image[positions] = values
        return image.reshape(scene.shape)
    confidence = probabilities.max(axis=1)
    entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum(axis=1) / np.log(probabilities.shape[1])
    return {
        "positions": positions, "probabilities": probabilities, "true": true, "predicted": predicted,
        "confidence": confidence, "entropy": entropy, "per_class": per_class,
        "ground_truth": canvas(true + 1), "prediction": canvas(predicted + 1),
        "correct": canvas((true == predicted).astype(float)), "confidence_map": canvas(confidence),
        "entropy_map": canvas(entropy), "f1_map": canvas(np.array([f1_lookup[int(label + 1)] for label in true])),
    }


def segment_decision_metrics(result, scene, inverse, artifacts):
    """Agrega acerto, confiança, entropia e F1 por superpixel na imagem original."""
    segment_ids = artifacts.labels.ravel()[inverse["positions"]]
    rows, maps = [], {name: np.full(scene.labels.size, np.nan) for name in ("acurácia", "confiança", "entropia", "F1 da classe", "amostras teste")}
    for segment_id in np.unique(segment_ids):
        mask = segment_ids == segment_id
        spatial_mask = artifacts.labels.ravel() == segment_id
        true = inverse["true"][mask]
        predicted = inverse["predicted"][mask]
        row = {
            "segmento": int(segment_id), "amostras teste": int(mask.sum()),
            "acurácia": float(np.mean(true == predicted)),
            "confiança": float(np.mean(inverse["confidence"][mask])),
            "entropia": float(np.mean(inverse["entropy"][mask])),
            "F1 da classe": float(np.mean(inverse["f1_map"].ravel()[inverse["positions"]][mask])),
            "classe real dominante": int(np.bincount(true).argmax() + 1),
            "classe prevista dominante": int(np.bincount(predicted).argmax() + 1),
        }
        rows.append(row)
        for name in maps:
            maps[name][spatial_mask] = row[name]
    return pd.DataFrame(rows).sort_values(["acurácia", "amostras teste"]), {name: value.reshape(scene.shape) for name, value in maps.items()}


def _contoured(axis, base, contours, title, cmap=None, vmin=None, vmax=None, discrete=False):
    """Exibe um mapa e coloca contornos de superpixels para ligar métrica e região."""
    if base.ndim == 3:
        axis.imshow(base)
    else:
        image = axis.imshow(base, cmap=cmap, vmin=vmin, vmax=vmax)
        if not discrete:
            plt.colorbar(image, ax=axis, shrink=.72)
    axis.contour(contours > 0, levels=[.5], colors="white", linewidths=.35, alpha=.8)
    axis.set_title(title); axis.axis("off")


def plot_inverse_prediction_dashboard(result, scene, inverse, artifacts):
    """Mostra exatamente onde o modelo acertou, errou, ficou incerto e teve F1 baixo."""
    fig, axes = plt.subplots(2, 3, figsize=(19, 12), constrained_layout=True)
    _contoured(axes[0, 0], artifacts.rgb, artifacts.contour_mask, f"Entrada PCA/RGB + {result.config.segmenter.upper()}")
    _contoured(axes[0, 1], inverse["ground_truth"], artifacts.contour_mask, "Ground truth — pixels de teste", cmap=_label_cmap(result.data.n_classes), vmin=1, vmax=result.data.n_classes, discrete=True)
    _contoured(axes[0, 2], inverse["prediction"], artifacts.contour_mask, "Classe prevista — pixels de teste", cmap=_label_cmap(result.data.n_classes), vmin=1, vmax=result.data.n_classes, discrete=True)
    _contoured(axes[1, 0], inverse["correct"], artifacts.contour_mask, "Acerto por pixel (verde) / erro (vermelho)", cmap=ListedColormap(["#d73027", "#1a9850"]), vmin=0, vmax=1)
    _contoured(axes[1, 1], inverse["confidence_map"], artifacts.contour_mask, "Confiança máxima da previsão", cmap="viridis", vmin=0, vmax=1)
    _contoured(axes[1, 2], inverse["f1_map"], artifacts.contour_mask, "F1 da classe real aplicado à imagem", cmap="magma", vmin=0, vmax=1)
    fig.suptitle(f"Processo inverso — {result.config.model_type.upper()}: modelo → métrica → localização", fontsize=15)
    return fig


def plot_segment_metric_dashboard(result, artifacts, segment_maps):
    """Aplica métricas agregadas a cada região produzida pelo segmentador."""
    fig, axes = plt.subplots(2, 3, figsize=(19, 12), constrained_layout=True)
    _contoured(axes[0, 0], colorize_segments(artifacts.labels), artifacts.contour_mask, "IDs dos segmentos")
    _contoured(axes[0, 1], segment_maps["amostras teste"], artifacts.contour_mask, "Cobertura: pixels de teste por segmento", cmap="Blues")
    _contoured(axes[0, 2], segment_maps["acurácia"], artifacts.contour_mask, "Acurácia por segmento", cmap="RdYlGn", vmin=0, vmax=1)
    _contoured(axes[1, 0], segment_maps["confiança"], artifacts.contour_mask, "Confiança média por segmento", cmap="viridis", vmin=0, vmax=1)
    _contoured(axes[1, 1], segment_maps["entropia"], artifacts.contour_mask, "Entropia média por segmento", cmap="magma", vmin=0, vmax=1)
    _contoured(axes[1, 2], segment_maps["F1 da classe"], artifacts.contour_mask, "F1 médio da classe real por segmento", cmap="magma", vmin=0, vmax=1)
    fig.suptitle(f"Segmentador + decisões da {result.config.model_type.upper()}", fontsize=15)
    return fig


def _single_input_from_index(result, index):
    """Reconstrói exatamente o tensor de uma amostra do teste para Grad-CAM local."""
    tensors = _tensors(result.data, np.asarray([index]), result.config.model_type)
    return unpack_batch(tensors, result.device)


def plot_representative_detections(result, inverse, limit=4):
    """Explica automaticamente casos fáceis, incertos e erros confiantes com Grad-CAM.

    Os exemplos são escolhidos por regras auditáveis: maior confiança correta,
    maior entropia, maior confiança incorreta e menor confiança. Uma amostra que
    satisfaz duas regras aparece uma vez só, e `limit` corta a lista. Dessa forma o
    painel não escolhe manualmente casos visualmente convenientes.
    """
    true, predicted = inverse["true"], inverse["predicted"]
    correct = true == predicted
    candidates = OrderedDict()
    if correct.any():
        candidates["acerto mais confiante"] = int(np.flatnonzero(correct)[np.argmax(inverse["confidence"][correct])])
    candidates["maior incerteza"] = int(np.argmax(inverse["entropy"]))
    if (~correct).any():
        candidates["erro mais confiante"] = int(np.flatnonzero(~correct)[np.argmax(inverse["confidence"][~correct])])
    candidates["menor confiança"] = int(np.argmin(inverse["confidence"]))
    unique = list(OrderedDict((index, label) for label, index in candidates.items()).items())[:limit]
    fig, axes = plt.subplots(len(unique), 2, figsize=(13, 4 * len(unique)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    test_indices = result.data.splits.test
    for row, (local_index, label) in enumerate(unique):
        x, seg, _ = _single_input_from_index(result, int(test_indices[local_index]))
        cam, predicted_class, layer = grad_cam(result.model, x, seg)
        # Entrada e Grad-CAM são perfis 1D: espectro PCA do pixel e importância por
        # posição após as convoluções.
        axes[row, 0].plot(x[0, 0].detach().cpu(), color="#4C78A8")
        axes[row, 0].set(xlabel="Componente PCA", ylabel="Valor")
        axes[row, 1].plot(cam, color="crimson")
        axes[row, 1].fill_between(np.arange(len(cam)), cam, alpha=.25, color="crimson")
        axes[row, 1].set(xlabel="Posição após convoluções", ylabel="Grad-CAM", ylim=(0, 1.05))
        axes[row, 0].set_title(f"{label}: entrada local")
        axes[row, 1].set_title(f"Grad-CAM: real {true[local_index] + 1}, prevista {predicted_class + 1}\n{layer}")
        for axis in axes[row]: axis.grid(alpha=.25)
    fig.suptitle("Detecções representativas escolhidas automaticamente", fontsize=15)
    return fig


def render_spatial_explanation(result, scene):
    """Executa a explicação inversa completa e retorna figuras e métricas por segmento."""
    artifacts = rebuild_spatial_context(result, scene)
    inverse = inverse_prediction_maps(result, scene)
    segment_table, segment_maps = segment_decision_metrics(result, scene, inverse, artifacts)
    return {
        "segmentos": segment_table,
        "figures": {
            "decisões remontadas na cena": plot_inverse_prediction_dashboard(result, scene, inverse, artifacts),
            "métricas aplicadas aos segmentos": plot_segment_metric_dashboard(result, artifacts, segment_maps),
            "detecções locais explicadas": plot_representative_detections(result, inverse),
        },
    }


# Integra a explicação espacial à rotina de dashboard da CNN1D.
def render_cnn_dashboard(result, scene, max_channels=8):
    """Dashboard completo: núcleo do dashboard por CNN + decisões remontadas na cena.

    Definição única e não reentrante — chama `_render_cnn_dashboard_core`, um nome
    fixo, em vez de guardar um alias da versão anterior de si mesma.
    """
    dashboard = _render_cnn_dashboard_core(result, scene, max_channels=max_channels)
    spatial = render_spatial_explanation(result, scene)
    dashboard["segmentos"] = spatial["segmentos"]
    dashboard["figures"].update({f"imagem — {name}": figure for name, figure in spatial["figures"].items()})
    return dashboard


def display_cnn_dashboard(name, dashboard):
    """Exibe o dashboard padrão e os indicadores de superpixels como gráfico."""
    _display_cnn_dashboard_core(name, dashboard)
    if "segmentos" in dashboard:
        from IPython.display import Markdown, display
        display(Markdown("#### Métricas agregadas por segmento"))
        display(plot_segment_metrics_visual(dashboard["segmentos"]))
