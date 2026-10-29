"""Predição e métricas de classificação (OA, AA, kappa, matriz de confusão)."""

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)

from hsi.treino import run_epoch, unpack_batch


@dataclass(frozen=True)
class Metrics:
    """Métricas padrão da literatura de classificação hiperespectral.

    OA (overall accuracy) pesa cada amostra igualmente; AA (average accuracy) pesa
    cada *classe* igualmente, e por isso cai quando as classes minoritárias vão
    mal. Kappa desconta o acerto esperado por acaso.
    """

    overall_accuracy: float
    average_accuracy: float
    kappa: float
    per_class_accuracy: np.ndarray
    confusion: np.ndarray
    test_loss: float
    test_accuracy: float
    report: str

    def summary(self) -> str:
        return (
            f"Test loss:     {self.test_loss}\n"
            f"Test accuracy: {self.test_accuracy}\n"
            f"OA:    {self.overall_accuracy}\n"
            f"AA:    {self.average_accuracy}\n"
            f"kappa: {self.kappa}"
        )


@torch.no_grad()
def predict_proba(model, loader, device) -> np.ndarray:
    """Probabilidades por classe para todas as amostras do loader.

    O softmax é explícito porque a última camada do modelo devolve logits: no
    treino quem o aplica é a `CrossEntropyLoss`.
    """
    model.eval()
    batches = []
    for batch in loader:
        x, seg, _ = unpack_batch(batch, device)
        batches.append(torch.softmax(model(x, seg), dim=1).cpu())
    return torch.cat(batches).numpy()


def _average_accuracy(confusion: np.ndarray) -> tuple[np.ndarray, float]:
    """Acurácia por classe (diagonal / suporte) e sua média.

    Uma classe sem amostras no teste, mas prevista pelo modelo, aparece na matriz
    com suporte 0 e conta como 0 — mesmo resultado do `np.nan_to_num` do original,
    mas sem provocar divisão por zero. Ausente do teste e das previsões, ela não
    entra na matriz (montada sem `labels` em `evaluate`) nem na média.
    """
    support = confusion.sum(axis=1)
    per_class = np.divide(
        np.diag(confusion), support,
        out=np.zeros(len(support), dtype=float), where=support > 0,
    )
    return per_class, float(per_class.mean())


def evaluate(model, loader, criterion, device, y_true: np.ndarray) -> tuple[Metrics, np.ndarray]:
    """Roda a predição no loader e calcula todas as métricas.

    Retorna `(Metrics, classes_previstas)`.
    """
    probabilities = predict_proba(model, loader, device)
    predicted = np.argmax(probabilities, axis=1)

    confusion = confusion_matrix(y_true, predicted)
    per_class, average_accuracy = _average_accuracy(confusion)
    # Segunda passagem pelo teste, o equivalente ao `model.evaluate` do original:
    # dá a loss de teste. A acurácia dela coincide com a OA.
    test_loss, test_accuracy = run_epoch(model, loader, criterion, device)

    metrics = Metrics(
        overall_accuracy=float(accuracy_score(y_true, predicted)),
        average_accuracy=average_accuracy,
        kappa=float(cohen_kappa_score(y_true, predicted)),
        per_class_accuracy=per_class,
        confusion=confusion,
        test_loss=test_loss,
        test_accuracy=test_accuracy,
        report=classification_report(y_true, predicted, zero_division=1),
    )
    return metrics, predicted


def binary_readout(confusion: np.ndarray) -> dict[str, float]:
    """Leitura binária das duas primeiras classes da matriz de confusão.

    Reproduz o cálculo da seção "Evaluating (Livro)" do notebook original, que
    aplica a fórmula de TP/TN/FP/FN de um problema **binário** sobre uma matriz
    multiclasse: só olha o bloco 2x2 do canto superior esquerdo e ignora todas as
    outras classes. Mantido para comparação com o trabalho original — não é uma
    métrica válida deste problema; use `Metrics` (OA/AA/kappa).
    """
    tn, fp = float(confusion[0][0]), float(confusion[0][1])
    fn, tp = float(confusion[1][0]), float(confusion[1][1])

    accuracy = (tp + tn) / (tp + tn + fn + fp)
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f_score = (2 * recall * precision) / (recall + precision) if (recall + precision) else float("nan")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f_score": f_score}
