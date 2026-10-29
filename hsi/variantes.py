"""As duas versões do pipeline que o trabalho compara.

`BASELINE` reproduz o fluxo da implementação de referência de Sydney Matheus de
Souza: estritamente sequencial, com o modelo e os hiperparâmetros de treino do
notebook original (learning rate, batch 32, dropout, 60 épocas). É a linha de base
contra a qual todo ganho é medido. A exceção é a parada antecipada: aqui vale a do
`ExperimentConfig` (patience 10 sobre `val_accuracy`, restaurando a melhor época),
não a do original (patience 5 sobre `val_loss`). E o que se repete **bit a bit** é
este código entre execuções: mesmo histórico de perda por época, mesmos pesos
finais e mesmos OA/AA/kappa, na mesma máquina.

`OPTIMIZED` é a nossa implementação: mesmo modelo e mesmos hiperparâmetros, mas com
as etapas independentes da preparação executadas concorrentemente (ver
`prepare_data_parallel`).

Os hiperparâmetros são idênticos nas duas de propósito. É isso que torna a comparação
interpretável: qualquer diferença de tempo vem da estratégia de execução, e as
métricas de qualidade têm de sair iguais — se não saírem, a otimização mudou o que o
código calcula, e é um bug, não um ganho.

Uso:

    base = run_variant(BASELINE, dataset="PU")
    nossa = run_variant(OPTIMIZED, dataset="PU")
    print(nossa.speedup_over(base))
"""

import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from hsi.config import ExperimentConfig
from hsi.pipeline import ExperimentResult, run_experiment


@dataclass(frozen=True)
class Variant:
    """Uma estratégia de execução do pipeline.

    `overrides` são aplicados sobre a `ExperimentConfig` base. Nenhuma variante os
    usa: as duas compartilham os hiperparâmetros de propósito, e o HPO (`hsi/hpo/`)
    não passa por aqui — entrega cada trial como uma `ExperimentConfig` pronta.
    """

    key: str
    name: str
    description: str
    backend: str
    overrides: dict = field(default_factory=dict)

    def config(self, **kwargs) -> ExperimentConfig:
        """Configuração desta variante, aceitando ajustes pontuais (dataset, épocas...)."""
        return replace(ExperimentConfig(**kwargs), **self.overrides) if self.overrides \
            else ExperimentConfig(**kwargs)


BASELINE = Variant(
    key="baseline",
    name="Baseline (Souza, 2023)",
    description=(
        "Fluxo sequencial da implementação de referência: PCA espectral, depois PCA de "
        "3 componentes, composição RGB e segmentação SEEDS, e só então o treino."
    ),
    backend="sequential",
)


OPTIMIZED = Variant(
    key="optimized",
    name="Nossa implementação",
    description=(
        "Mesmo modelo e mesmos hiperparâmetros, com os ramos espectral e espacial da "
        "preparação executados concorrentemente; sincronização apenas na fusão."
    ),
    backend="thread",
)


VARIANTS = {v.key: v for v in (BASELINE, OPTIMIZED)}


@dataclass
class VariantRun:
    """Resultado de uma variante, com os tempos separados por etapa."""

    variant: Variant
    result: ExperimentResult
    total_time: float

    @property
    def prep_time(self) -> float:
        return self.result.prep_timings.wall

    @property
    def train_time(self) -> float:
        return self.result.training.total_time

    @property
    def metrics(self):
        return self.result.metrics

    def speedup_over(self, baseline: "VariantRun") -> dict[str, float]:
        """Speedup por etapa e no total, em relação a outra execução."""
        return {
            "preparação": baseline.prep_time / self.prep_time if self.prep_time else float("nan"),
            "treino": baseline.train_time / self.train_time if self.train_time else float("nan"),
            "total": baseline.total_time / self.total_time if self.total_time else float("nan"),
        }

    def matches(self, other: "VariantRun") -> bool:
        """As duas variantes chegaram exatamente ao mesmo resultado de classificação?"""
        a, b = self.metrics, other.metrics
        return (
            a.overall_accuracy == b.overall_accuracy
            and a.average_accuracy == b.average_accuracy
            and a.kappa == b.kappa
        )

    def summary(self) -> str:
        return (
            f"{self.variant.name}\n"
            f"  preparação: {self.prep_time:7.3f}s  "
            f"(backend: {self.result.prep_timings.backend})\n"
            f"  treino:     {self.train_time:7.3f}s  "
            f"({len(self.result.training.history)} épocas)\n"
            f"  total:      {self.total_time:7.3f}s\n"
            f"  OA: {self.metrics.overall_accuracy:.6f} | "
            f"AA: {self.metrics.average_accuracy:.6f} | "
            f"kappa: {self.metrics.kappa:.6f}"
        )


def run_variant(
    variant: Variant,
    data_dir: Path | None = None,
    verbose: bool = False,
    **config_kwargs,
) -> VariantRun:
    """Executa uma variante de ponta a ponta e mede o tempo total."""
    config = variant.config(**config_kwargs)
    started = time.perf_counter()
    result = run_experiment(config, data_dir=data_dir, backend=variant.backend, verbose=verbose)
    return VariantRun(variant=variant, result=result, total_time=time.perf_counter() - started)
