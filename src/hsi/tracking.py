"""Rastreabilidade local de experimentos, sem depender de um servidor externo.

Cada execução ganha um diretório imutável em ``outputs/runs/`` com a configuração,
as impressões digitais dos dados, ambiente, métricas e artefatos. Isso responde às
perguntas básicas de MLOps para o TCC: qual configuração produziu um modelo, com
qual versão de código/dados e qual foi seu resultado.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np
import torch

from .paths import output_dir, repo_root

if TYPE_CHECKING:
    from .config import ExperimentConfig
    from .dataset import PreparedData
    from .pipeline import ExperimentResult

__all__ = [
    "RunContext", "array_fingerprint", "data_fingerprint", "create_run", "record_run",
    "fail_run", "result_metrics",
]


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    """Persiste um dicionário em JSON UTF-8 legível.

    Args:
        path: Arquivo de destino dentro (ou abaixo) do diretório da execução.
        value: Estrutura já serializável por ``json``.

    Returns:
        O próprio caminho, para facilitar a composição com outras funções.

    Os campos são ordenados para tornar ``diff`` entre execuções mais útil. Esta
    é uma função interna: o chamador público é ``create_run`` ou ``record_run``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def array_fingerprint(array: np.ndarray | None) -> str | None:
    """Gera a identidade SHA-256 de um array que chega ao modelo.

    O hash inclui shape, dtype e bytes em ordem contígua. Portanto, a mesma
    matriz com dtype diferente, shape diferente ou qualquer valor modificado
    produz uma identidade diferente. ``None`` é preservado para representar a
    ausência intencional da segunda entrada no modo espectral.
    """
    if array is None:
        return None
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode())
    digest.update(contiguous.dtype.str.encode())
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def data_fingerprint(data: "PreparedData") -> dict[str, Any]:
    """Cria o contrato dos dados já processados usados nesta execução.

    Não faz hash dos `.mat` crus: o objetivo é registrar exatamente o que a CNN
    recebeu depois de PCA, remoção de background e segmentação. Os hashes dos
    índices garantem também que uma mudança de seed ou split seja detectável.
    O retorno contém apenas tipos serializáveis em JSON.
    """
    return {
        "spectra": {"shape": list(data.spectra.shape), "sha256": array_fingerprint(data.spectra)},
        "labels": {"shape": list(data.labels.shape), "sha256": array_fingerprint(data.labels)},
        "segments": None if data.segments is None else {
            "shape": list(data.segments.shape), "sha256": array_fingerprint(data.segments),
        },
        # Uma execução 2D/3D não consome só ``spectra``. Registrar estes arrays
        # impede que dois patches/modelos espaciais diferentes pareçam o mesmo
        # dataset no tracker local.
        "patches_2d": None if data.patches_2d is None else {
            "shape": list(data.patches_2d.shape), "sha256": array_fingerprint(data.patches_2d),
        },
        "patches_3d": None if data.patches_3d is None else {
            "shape": list(data.patches_3d.shape), "sha256": array_fingerprint(data.patches_3d),
        },
        "splits": {
            "train": array_fingerprint(data.splits.train),
            "val": array_fingerprint(data.splits.val),
            "test": array_fingerprint(data.splits.test),
        },
        "scene_shape": list(data.scene_shape),
        "n_classes": data.n_classes,
    }


def _git_revision() -> str | None:
    """Obtém o commit completo sem falhar fora de um repositório Git."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root(), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _environment() -> dict[str, Any]:
    """Coleta somente dados de ambiente que influenciam reprodução/performance."""
    cuda_available = torch.cuda.is_available()
    environment: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": cuda_available,
        "git_revision": _git_revision(),
    }
    if cuda_available:
        environment["gpu"] = torch.cuda.get_device_name(0)
        environment["gpu_capability"] = list(torch.cuda.get_device_capability(0))
    return environment


@dataclass(frozen=True)
class RunContext:
    """Identidade e diretório de uma execução rastreada."""

    run_id: str
    directory: Path


def create_run(config: "ExperimentConfig", base_dir: Path | None = None) -> RunContext:
    """Abre uma execução rastreada antes que dados ou pesos sejam criados.

    Args:
        config: Receita imutável que será usada pelo pipeline.
        base_dir: Raiz de runs alternativa. Sem valor, usa ``outputs/runs``.

    Returns:
        ``RunContext`` com identificador único UTC e diretório já criado.

    A função cria ``config.json``, ``environment.json`` e um manifesto com
    estado ``running``. Se o pipeline falhar depois, ``fail_run`` deve fechar
    esse estado; se terminar, ``record_run`` o substitui por ``completed``.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}_{config.dataset}_{uuid4().hex[:8]}"
    directory = (base_dir or output_dir() / "runs") / run_id
    directory.mkdir(parents=True, exist_ok=False)
    context = RunContext(run_id=run_id, directory=directory)
    _write_json(directory / "config.json", config.to_dict())
    _write_json(directory / "environment.json", _environment())
    _write_json(directory / "manifest.json", {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "started_at_utc": timestamp,
        "artifacts": ["config.json", "environment.json"],
    })
    return context


def result_metrics(result: "ExperimentResult") -> dict[str, Any]:
    """Converte ``ExperimentResult`` em um payload JSON sem tensores/arrays NumPy.

    O mesmo formato é usado em ``metrics.json`` e na saída opcional ``--json``.
    Centralizá-lo evita que uma métrica apareça em um formato no tracker e em
    outro no relatório solicitado pela CLI.
    """
    return {
        "metrics": {
            "overall_accuracy": result.metrics.overall_accuracy,
            "average_accuracy": result.metrics.average_accuracy,
            "kappa": result.metrics.kappa,
            "test_loss": result.metrics.test_loss,
            "test_accuracy": result.metrics.test_accuracy,
            "per_class_accuracy": result.metrics.per_class_accuracy.tolist(),
        },
        "history": result.training.history.to_dict(),
        "epochs_run": len(result.training.history),
        "prep_timings": {
            "backend": result.prep_timings.backend,
            "wall": result.prep_timings.wall,
            "stages": result.prep_timings.stages,
        },
        "train_time": result.training.total_time,
        "stopped_early": result.training.stopped_early,
        "best_val_loss": result.training.best_val_loss,
    }


def fail_run(context: RunContext, error: BaseException) -> None:
    """Marca uma execução como falha e persiste uma causa humana legível.

    Não relança nem tenta recuperar a exceção: essa decisão pertence à CLI. A
    função é chamada no ``except`` para que um run interrompido não pareça ter
    sido concluído com sucesso ao ser analisado posteriormente.
    """
    _write_json(context.directory / "failure.json", {
        "error_type": type(error).__name__,
        "message": str(error),
    })
    _write_json(context.directory / "manifest.json", {
        "schema_version": 1,
        "run_id": context.run_id,
        "status": "failed",
        "config": "config.json",
        "environment": "environment.json",
        "failure": "failure.json",
    })


def record_run(
    context: RunContext,
    result: "ExperimentResult",
    artifacts: dict[str, Path] | None = None,
) -> None:
    """Fecha uma execução bem-sucedida e vincula todos os seus artefatos.

    Args:
        context: Identidade retornada por ``create_run``.
        result: Objetos finais de dados, treino, modelo e avaliação.
        artifacts: Mapa lógico (por exemplo, ``checkpoint``) para o arquivo
            gerado. Caminhos dentro do run ficam relativos; externos ficam
            absolutos para não perder a referência.

    ``data.json`` e ``metrics.json`` são gravados antes do manifesto final. Isso
    garante que ``status=completed`` sempre represente uma execução auditável.
    """
    artifacts = artifacts or {}
    metrics = result_metrics(result)
    data_record = data_fingerprint(result.data)
    data_record["contract"] = result.data_contract
    _write_json(context.directory / "data.json", data_record)
    _write_json(context.directory / "metrics.json", metrics)
    def artifact_path(path: Path) -> str:
        try:
            return str(path.relative_to(context.directory))
        except ValueError:
            return str(path.resolve())

    _write_json(context.directory / "manifest.json", {
        "schema_version": 1,
        "run_id": context.run_id,
        "status": "completed",
        "config": "config.json",
        "environment": "environment.json",
        "data": "data.json",
        "metrics": "metrics.json",
        "artifacts": {name: artifact_path(path) for name, path in artifacts.items()},
    })
