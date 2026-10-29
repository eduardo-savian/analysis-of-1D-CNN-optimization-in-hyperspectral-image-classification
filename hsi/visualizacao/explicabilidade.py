"""Ferramentas de explicabilidade visual para a CNN 1D.

Os hooks observam saídas intermediárias durante uma inferência e são removidos no
fim. Portanto, não alteram pesos, gradientes de treino nem a predição registrada.
Ativação não é uma explicação causal completa: ela indica quais padrões passaram
pelas camadas; Grad-CAM e saliência complementam mostrando a influência local na
classe prevista.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from hsi.avaliacao import predict_proba
from hsi.dados.cenas import Scene
from hsi.dados.particoes import build_loaders
from hsi.pipeline import ExperimentResult
from hsi.treino import unpack_batch


def explanation_sample(result: ExperimentResult, split: str = "test", batch_index: int = 0):
    """Obtém uma única amostra já no device, estável e sem alterar o DataLoader.

    `batch_index` escolhe a posição dentro do primeiro batch do split. Para explicar
    outro pixel, aumente-o até o tamanho desse batch menos 1 (em validação e teste,
    `config.resolved_eval_batch_size`) ou construa um DataLoader dedicado.

    Devolve `(x, seg, y)`, com `y` como inteiro 0..C-1.
    """
    loaders = build_loaders(result.data, result.config, result.device)
    x, seg, y = unpack_batch(next(iter(loaders[split])), result.device)
    if not 0 <= batch_index < len(x):
        raise IndexError(f"batch_index deve estar entre 0 e {len(x) - 1}.")
    return x[batch_index:batch_index + 1], None if seg is None else seg[batch_index:batch_index + 1], y[batch_index].item()


def capture_activations(model, x: torch.Tensor, seg: torch.Tensor | None = None) -> list[dict]:
    """Captura Conv, MaxPool e pooling adaptativo da passagem direta de uma amostra.

    Na CNN1D só há as quatro Conv1d; os poolings ficam na lista para uma arquitetura
    que os tenha. Devolve `(registros, classe_prevista)`, cada registro com nome,
    tipo e a ativação da amostra, já sem o eixo do batch.
    """
    watched = (torch.nn.Conv1d, torch.nn.MaxPool1d, torch.nn.AdaptiveAvgPool1d)
    records, handles = [], []
    for name, module in model.named_modules():
        if not isinstance(module, watched):
            continue
        # Os argumentos padrão congelam o nome e o tipo desta iteração; sem eles,
        # todo hook leria as variáveis do laço já na última camada.
        def hook(_module, _inputs, output, module_name=name, module_type=type(module).__name__):
            tensor = output[0] if isinstance(output, tuple) else output
            records.append({"nome": module_name, "tipo": module_type, "ativação": tensor.detach()[0].float().cpu().numpy()})
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
    predicted = int(logits.argmax(dim=1).item())
    return records, predicted


def _activation_image(activation: np.ndarray, channel: int) -> np.ndarray:
    """Resto das CNNs 2D/3D, que convertiam um mapa espacial em imagem.

    Na CNN1D toda ativação capturada é ``(C, L)`` e é desenhada direto pelos
    chamadores; chegar aqui é erro, por isso a função só levanta exceção.
    """
    raise ValueError("Ativação não possui dimensão espacial para imagem.")


def plot_activation_atlases(records: list[dict], max_channels: int = 12) -> dict[str, object]:
    """Gera uma figura por camada com os mapas de características dos filtros.

    Na CNN1D são os `max_channels` primeiros filtros, na ordem do índice (escolher
    os mais ativos só existe no ramo legado 2D/3D). Cada linha é um filtro; o eixo
    horizontal é a posição na sequência de componentes PCA, encurtada a cada
    convolução — não um comprimento de onda.
    """
    figures = {}
    for record in records:
        activation = record["ativação"]
        name = f"{record['nome']} ({record['tipo']})"
        if activation.ndim == 2:  # C, L — CNN1D
            channels = min(max_channels, activation.shape[0])
            fig, axis = plt.subplots(figsize=(14, max(3, channels * .35)), constrained_layout=True)
            im = axis.imshow(activation[:channels], cmap="magma", aspect="auto")
            axis.set(title=f"Ativações espectrais — {name}", xlabel="Posição espectral", ylabel="Filtro")
            fig.colorbar(im, ax=axis, label="Ativação")
        else:  # C,H,W ou C,D,H,W — legado das CNNs 2D/3D; a CNN1D não chega aqui
            channels = min(max_channels, activation.shape[0])
            columns = min(4, channels)
            rows = int(np.ceil(channels / columns))
            fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.5 * rows), constrained_layout=True)
            axes = np.atleast_1d(axes).ravel()
            scores = np.mean(np.abs(activation), axis=tuple(range(1, activation.ndim)))
            selected = np.argsort(scores)[-channels:][::-1]
            for axis, channel in zip(axes, selected):
                image = _activation_image(activation, int(channel))
                im = axis.imshow(image, cmap="magma")
                axis.set_title(f"filtro {channel}")
                axis.axis("off")
                fig.colorbar(im, ax=axis, shrink=.72)
            for axis in axes[len(selected):]:
                axis.axis("off")
            dimension = "projeção máxima nas bandas" if activation.ndim == 4 else "mapa espacial"
            fig.suptitle(f"Ativações — {name} ({dimension})", fontsize=14)
        figures[name] = fig
    return figures


def plot_maxpool_effect(records: list[dict]) -> dict[str, object]:
    """Compara entrada e saída de cada MaxPooling disponível na arquitetura.

    O painel mostra o filtro mais ativo imediatamente antes e depois do pool. O
    MaxPooling não 'entende' uma classe sozinho: ele retém a maior resposta local,
    reduzindo resolução e tornando a representação mais tolerante a pequenos
    deslocamentos. A CNN1D atual não possui MaxPooling; ela preserva a resolução
    espectral até o flatten. Nesse caso a função imprime um aviso e devolve um
    dicionário vazio.
    """
    figures = {}
    pool_indices = [i for i, item in enumerate(records) if "MaxPool" in item["tipo"]]
    if not pool_indices:
        print("Esta CNN1D não usa MaxPooling: as convoluções preservam as posições espectrais até o Flatten.")
        return figures
    for index in pool_indices:
        before, after = records[index - 1], records[index]
        input_activation, output_activation = before["ativação"], after["ativação"]
        scores = np.mean(np.abs(input_activation), axis=tuple(range(1, input_activation.ndim)))
        channel = int(np.argmax(scores))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
        if input_activation.ndim == 2:
            axes[0].plot(input_activation[channel]); axes[1].plot(output_activation[channel])
            for axis in axes:
                axis.set(xlabel="Posição espectral", ylabel="Ativação")
        else:
            left = _activation_image(input_activation, channel)
            right = _activation_image(output_activation, channel)
            axes[0].imshow(left, cmap="magma"); axes[1].imshow(right, cmap="magma")
            for axis in axes:
                axis.axis("off")
        axes[0].set_title(f"Antes: {before['nome']}\n{input_activation.shape}")
        axes[1].set_title(f"Depois: {after['nome']}\n{output_activation.shape}")
        fig.suptitle(f"Efeito do MaxPooling — filtro {channel}: mantém a resposta local máxima")
        figures[after["nome"]] = fig
    return figures


def grad_cam(model, x: torch.Tensor, seg: torch.Tensor | None = None) -> tuple[np.ndarray, int, str]:
    """Calcula Grad-CAM no último Conv, condicionado à classe prevista.

    O gradiente da pontuação da classe pesa os filtros da última convolução. O
    resultado destaca regiões/posições cuja ativação mais elevou essa previsão;
    não é uma prova de causalidade nem substitui a avaliação quantitativa.

    Na CNN1D o mapa tem ``Kb - 14`` posições, uma por saída da última convolução,
    e não é reamostrado para a entrada: a posição *i* resume as componentes *i* a
    *i + 14*, o campo receptivo dos kernels (9, 3, 3, 3).
    """
    candidates = [(name, module) for name, module in model.named_modules()
                  if isinstance(module, torch.nn.Conv1d)]
    if not candidates:
        raise ValueError("Modelo sem camada convolucional para Grad-CAM.")
    name, target = candidates[-1]
    cache = {}
    forward = target.register_forward_hook(lambda _m, _i, out: cache.__setitem__("activation", out))
    backward = target.register_full_backward_hook(lambda _m, _gi, go: cache.__setitem__("gradient", go[0]))
    previous_mode = model.training
    model.eval()
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x, seg)
        predicted = int(logits.argmax(dim=1).item())
        logits[0, predicted].backward()
        activation, gradient = cache["activation"], cache["gradient"]
        axes = tuple(range(2, activation.ndim))
        weights = gradient.mean(dim=axes, keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1))[0]
        cam = cam.detach().float().cpu().numpy()
        cam = (cam - cam.min()) / max(cam.max() - cam.min(), 1e-12)
    finally:
        forward.remove(); backward.remove(); model.train(previous_mode)
    return cam, predicted, name


def plot_grad_cam(cam: np.ndarray, model_type: str, predicted: int, layer_name: str):
    """Desenha o Grad-CAM da CNN1D como perfil sobre as posições após as convoluções.

    ``model_type`` fica na assinatura porque os chamadores o passam; com só a
    CNN1D no projeto, o mapa é sempre 1D.
    """
    fig, axis = plt.subplots(figsize=(14, 4), constrained_layout=True)
    axis.plot(cam, color="crimson", lw=2)
    axis.fill_between(np.arange(len(cam)), cam, alpha=.25, color="crimson")
    axis.set(title=f"Grad-CAM espectral — classe prevista {predicted + 1} ({layer_name})", xlabel="Posição após convoluções", ylabel="Importância normalizada")
    axis.grid(alpha=.25)
    return fig


def plot_input_saliency(model, x: torch.Tensor, seg: torch.Tensor | None = None, model_type: str = "cnn1d"):
    """Mostra sensibilidade local da classe prevista à entrada (|d logit / d entrada|).

    Um valor por componente PCA da entrada espectral. ``model_type`` fica na
    assinatura porque os chamadores o passam; só existe a CNN1D.
    """
    input_tensor = x.detach().clone().requires_grad_(True)
    previous_mode = model.training
    model.eval(); model.zero_grad(set_to_none=True)
    logits = model(input_tensor, seg)
    predicted = int(logits.argmax(dim=1).item())
    logits[0, predicted].backward()
    saliency = input_tensor.grad.detach().abs()[0, 0].float().cpu().numpy()
    model.train(previous_mode)
    fig, axis = plt.subplots(figsize=(14, 4), constrained_layout=True)
    axis.plot(saliency, color="#4C78A8", lw=2)
    axis.fill_between(np.arange(len(saliency)), saliency, alpha=.25, color="#4C78A8")
    axis.set(title=f"Saliência da entrada espectral — classe prevista {predicted + 1}", xlabel="Componente PCA", ylabel="|gradiente|")
    axis.grid(alpha=.25)
    return fig


def plot_prediction_uncertainty(result: ExperimentResult, scene: Scene):
    """Plota confiança e entropia da previsão nos pixels de teste.

    Confiança baixa ou entropia alta sinalizam regiões candidatas a inspeção ou
    aquisição de novos rótulos; não significam automaticamente que a previsão é
    errada. A normalização de entropia mantém a escala em [0, 1].
    """
    loaders = build_loaders(result.data, result.config, result.device)
    probabilities = predict_proba(result.model, loaders["test"], result.device)
    confidence = probabilities.max(axis=1)
    entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum(axis=1) / np.log(probabilities.shape[1])
    positions = np.flatnonzero(scene.labels.ravel() > 0)[result.data.splits.test]
    confidence_map = np.full(scene.labels.size, np.nan); confidence_map[positions] = confidence
    entropy_map = np.full(scene.labels.size, np.nan); entropy_map[positions] = entropy
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    for axis, values, title, cmap in (
        (axes[0], confidence_map.reshape(scene.shape), "Confiança máxima", "viridis"),
        (axes[1], entropy_map.reshape(scene.shape), "Entropia preditiva", "magma"),
    ):
        image = axis.imshow(values, cmap=cmap, vmin=0, vmax=1)
        axis.set_title(title); axis.axis("off")
        fig.colorbar(image, ax=axis, label="0 → baixo | 1 → alto")
    axes[2].hist(confidence, bins=20, alpha=.7, label="confiança")
    axes[2].hist(entropy, bins=20, alpha=.7, label="entropia")
    axes[2].set(title="Distribuição de incerteza no teste", xlabel="Pontuação", ylabel="Pixels")
    axes[2].legend(); axes[2].grid(alpha=.25)
    return fig
