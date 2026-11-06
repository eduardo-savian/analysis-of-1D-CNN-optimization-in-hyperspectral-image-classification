import copy
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix


def run_epoch(modelo, loader, perda, device, otimizador=None) -> tuple[float, float]:
    """Uma passada pelo loader; treina se houver otimizador. Devolve (perda, acurácia)."""
    treinando = otimizador is not None
    # train(False) desliga o dropout na validação.
    modelo.train(treinando)
    soma_perda, acertos, total = 0.0, 0, 0
    if treinando:
        otimizador.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(treinando):
        for x, seg, y in loader:
            x = x.to(device, non_blocking=True)
            seg = seg.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = modelo(x, seg)
            valor = perda(logits, y)
            if treinando:
                valor.backward()
                otimizador.step()
                otimizador.zero_grad(set_to_none=True)
            soma_perda += valor.item() * y.size(0)
            acertos += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return soma_perda / total, acertos / total


def fit(modelo, loaders, perda, otimizador, config, device, verbose=False) -> dict:
    """Treina até `config.epochs` épocas ou até a parada antecipada.

    Para após `patience` épocas sem melhora maior que `min_delta` na validação, ou ao
    atingir `target`, e restaura a melhor época; sem parada, ficam os pesos da última.
    """
    # Ressemeia: olhar um lote antes do treino adiantaria o gerador do loader.
    loaders["treino"].generator.manual_seed(config.seed)
    historico = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    melhor, melhor_epoca, sem_melhora, melhor_estado = -np.inf, 0, 0, None
    motivo = "limite de épocas"
    inicio = time.perf_counter()
    for epoca in range(1, config.epochs + 1):
        loss, acc = run_epoch(modelo, loaders["treino"], perda, device, otimizador)
        val_loss, val_acc = run_epoch(modelo, loaders["val"], perda, device)
        for chave, valor in zip(historico, (loss, acc, val_loss, val_acc)):
            historico[chave].append(valor)
        if verbose:
            print(f"época {epoca:3d}  loss {loss:.4f}  acc {acc:.4f}  "
                  f"val_loss {val_loss:.4f}  val_acc {val_acc:.4f}")

        if val_acc > melhor + config.min_delta:
            melhor, melhor_epoca, sem_melhora = val_acc, epoca, 0
            # state_dict() aponta para os tensores vivos; sem cópia, mudaria com o treino.
            melhor_estado = copy.deepcopy(modelo.state_dict())
        else:
            sem_melhora += 1
        if config.target is not None and val_acc >= config.target:
            motivo = f"meta {config.target} atingida"
        elif sem_melhora >= config.patience:
            motivo = f"{config.patience} épocas sem melhora"
        else:
            continue
        modelo.load_state_dict(melhor_estado)
        break
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {"historico": historico, "tempo": time.perf_counter() - inicio,
            "melhor_epoca": melhor_epoca, "motivo": motivo}


@torch.no_grad()
def predict(modelo, loader, device) -> np.ndarray:
    modelo.eval()
    classes = []
    for x, seg, _ in loader:
        # O softmax não muda o argmax, mas fica para repetir o cálculo do original.
        probabilidades = torch.softmax(modelo(x.to(device), seg.to(device)), dim=1)
        classes.append(probabilidades.argmax(dim=1).cpu())
    return torch.cat(classes).numpy()


def metrics(verdade: np.ndarray, previsto: np.ndarray) -> dict:
    """OA, AA e Kappa. A AA é a média das acurácias por classe."""
    confusao = confusion_matrix(verdade, previsto)
    suporte = confusao.sum(axis=1)
    # Classe prevista mas ausente do teste tem suporte 0 e conta como acurácia 0.
    por_classe = np.divide(np.diag(confusao), suporte, out=np.zeros(len(suporte)),
                           where=suporte > 0)
    return {"OA": float(accuracy_score(verdade, previsto)), "AA": float(por_classe.mean()),
            "Kappa": float(cohen_kappa_score(verdade, previsto)), "confusao": confusao}
