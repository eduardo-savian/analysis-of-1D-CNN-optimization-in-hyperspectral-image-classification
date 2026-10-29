"""Registry local de modelos aprovados, baseado nos runs rastreados.

O tracker guarda *todas* as execuções rastreadas; o registry guarda apenas o
campeão de uma família comparável. A promoção é deliberadamente explícita (uma
chamada a ``promote_candidate``; nenhuma célula do notebook a faz): uma execução
não vira referência só porque terminou sem erro.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hsi.config import output_dir

if TYPE_CHECKING:     # só anotações: importar de verdade criaria ciclo com o pipeline
    from hsi.mlops.rastreio import RunContext
    from hsi.pipeline import ExperimentResult


@dataclass(frozen=True)
class PromotionDecision:
    """Resultado explicável de uma tentativa de promover um modelo."""

    promoted: bool
    reason: str
    registry_path: Path


def comparison_key(result: "ExperimentResult") -> dict[str, Any]:
    """Define quais condições devem permanecer fixas numa disputa por campeão.

    Segmentador, backend e batch ficam fora de propósito: são justamente opções
    que o experimento pode querer comparar. Cena, arquitetura, segunda entrada,
    componentes de PCA, proporções do split e seed permanecem fixos para evitar
    promover uma métrica não comparável. `split_mode`, `train_fraction` e `fusion`
    não entram na chave: runs sob protocolos diferentes (split por pixel ou por
    região, fração de treino) disputam o mesmo campeão.
    """
    config = result.config
    return {
        "dataset": config.dataset,
        "model_type": config.model_type,
        "two_input": config.two_input,
        "pca_components": config.pca_components,
        "test_ratio": config.test_ratio,
        "valid_ratio": config.valid_ratio,
        "seed": config.seed,
    }


def champion_path(result: "ExperimentResult", root: Path | None = None) -> Path:
    """Local determinístico do campeão da cena e arquitetura da execução."""
    base = root or output_dir() / "registry"
    return base / result.config.dataset / result.config.model_type / "champion.json"


def _score(candidate: dict[str, Any]) -> tuple[float, float, float]:
    """Ordena candidatos por AA, kappa e OA, nessa ordem para classes desbalanceadas."""
    metrics = candidate["metrics"]
    return metrics["average_accuracy"], metrics["kappa"], metrics["overall_accuracy"]


def _load(path: Path) -> dict[str, Any] | None:
    """Lê um campeão existente; ausência é normal no primeiro modelo da família."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def promote_candidate(
    context: "RunContext",
    result: "ExperimentResult",
    checkpoint: Path,
    root: Path | None = None,
) -> PromotionDecision:
    """Promove candidato se ele for comparável e superior ao campeão atual.

    Args:
        context: Run rastreado que produziu a execução candidata.
        result: Métricas e configuração já concluídas.
        checkpoint: Peso salvo que será apontado pelo registry.
        root: Raiz alternativa, útil em testes; padrão ``outputs/registry``.

    A regra lexicográfica é AA → kappa → OA, todas de **teste**: ao contrário das
    campanhas, que escolhem só pela validação, aqui o teste decide. Se não existir
    campeão, o primeiro candidato válido entra. Se a chave de comparação for
    diferente, não há sobrescrita: o usuário deve criar uma família/critério
    conscientemente (por exemplo, com outra ``root``).
    """
    path = champion_path(result, root)
    candidate = {
        "run_id": context.run_id,
        "run_directory": str(context.directory.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "comparison_key": comparison_key(result),
        "config": result.config.to_dict(),
        "metrics": {
            "overall_accuracy": result.metrics.overall_accuracy,
            "average_accuracy": result.metrics.average_accuracy,
            "kappa": result.metrics.kappa,
        },
        "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    champion = _load(path)
    if champion is not None and champion["comparison_key"] != candidate["comparison_key"]:
        return PromotionDecision(False, "chave de comparação diferente do campeão atual", path)
    if champion is not None and _score(candidate) <= _score(champion):
        return PromotionDecision(False, "AA/kappa/OA não superou o campeão atual", path)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    reason = "primeiro campeão da família" if champion is None else "superou campeão em AA/kappa/OA"
    return PromotionDecision(True, reason, path)
