"""Configuração do experimento — tudo o que antes eram variáveis globais soltas.

No notebook original os hiperparâmetros ficavam espalhados por várias células
(`dataset`, `test_ratio`, `two_input`, `Kb`, `NUM_EPOCHS`, `BATCH_SIZE`, ...) e
eram lidos implicitamente por funções que dependiam do escopo global. Aqui viram
um objeto único, explícito e serializável.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

__all__ = ["ExperimentConfig", "DATASETS", "MODEL_TYPES", "SEGMENTERS"]

# Cenas suportadas: código curto -> nome legível.
DATASETS = {
    "IP": "Indian Pines",
    "PU": "Pavia University",
    "SA": "Salinas Valley",
}

# Famílias disponíveis. As strings ficam aqui (em vez de importar os módulos de
# modelo/segmentação) para que ExperimentConfig continue leve e sem dependências
# circulares.
MODEL_TYPES = ("cnn1d", "cnn2d", "cnn3d")
SEGMENTERS = ("seeds", "slic", "lsc")

# Componentes de PCA por cena. O original usava `30 if dataset == 'IP' else 15`;
# Indian Pines precisa de mais componentes porque tem 200 bandas e 16 classes
# espectralmente próximas.
PCA_COMPONENTS = {"IP": 30, "PU": 15, "SA": 15}


@dataclass(frozen=True)
class ExperimentConfig:
    """Parâmetros de uma execução completa.

    Os valores padrão reproduzem o resultado de referência documentado em
    `SETUP.md` (PaviaU, duas entradas, SEED=42, modo determinístico).
    """

    dataset: str = "PU"
    #: True -> embedding + id do superpixel; False -> somente a entrada da CNN.
    two_input: bool = True
    #: Arquitetura comparada. cnn1d preserva a referência; cnn2d usa patches das
    #: 3 componentes espaciais; cnn3d usa cubos espectral-espaciais crus.
    model_type: str = "cnn1d"
    #: Algoritmo de superpixel usado quando `two_input=True`.
    segmenter: str = "seeds"
    #: Lado ímpar dos patches de contexto espacial para CNN2D/CNN3D.
    patch_size: int = 5

    #: Fração da base que sai do treino (vai para teste+validação).
    test_ratio: float = 0.15
    #: Fração de teste+validação que vira validação.
    valid_ratio: float = 0.5

    epochs: int = 60
    #: Batch físico enviado à GPU. Mantenha 32 para reproduzir o baseline; use o
    #: benchmark de batches para escolher um valor maior para a sua GPU.
    batch_size: int = 32
    #: Batch de validação/teste. Zero reutiliza `batch_size`; como não há
    #: gradientes nessa fase, este valor normalmente pode ser maior.
    eval_batch_size: int = 0
    #: Quantos batches físicos acumulam gradiente antes de `optimizer.step()`.
    #: O batch efetivo é `batch_size * gradient_accumulation_steps`.
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-3
    #: O padrão do PyTorch é 1e-8; 1e-7 é o valor da implementação de referência
    #: e é mantido para que os resultados sejam comparáveis com os dela.
    adam_eps: float = 1e-7
    dropout: float = 0.5
    patience: int = 5

    seed: int = 42
    #: cudnn.deterministic + TF32 off. Troque para False só ao medir desempenho.
    deterministic: bool = True

    #: num_workers=0 é intencional por padrão: os tensores já estão em RAM e os
    #: batches são pequenos. Só aumente após medir em uma máquina com carga de
    #: I/O/preprocessamento adicional.
    num_workers: int = 0
    prefetch_factor: int = 2
    persistent_workers: bool = False

    #: Preenchido no __post_init__ a partir do dataset, se não for informado.
    pca_components: int = 0

    def __post_init__(self) -> None:
        if self.dataset not in DATASETS:
            raise ValueError(f"dataset inválido: {self.dataset!r}. Use um de {sorted(DATASETS)}.")
        if self.model_type not in MODEL_TYPES:
            raise ValueError(f"model_type inválido: {self.model_type!r}. Use um de {MODEL_TYPES}.")
        if self.segmenter not in SEGMENTERS:
            raise ValueError(f"segmenter inválido: {self.segmenter!r}. Use um de {SEGMENTERS}.")
        if self.patch_size < 3 or not self.patch_size % 2:
            raise ValueError("patch_size deve ser ímpar e maior ou igual a 3.")
        for name in ("epochs", "batch_size", "gradient_accumulation_steps", "prefetch_factor"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} deve ser positivo.")
        if self.num_workers < 0:
            raise ValueError("num_workers não pode ser negativo.")
        if self.persistent_workers and not self.num_workers:
            raise ValueError("persistent_workers requer num_workers maior que zero.")
        if self.eval_batch_size < 0:
            raise ValueError("eval_batch_size não pode ser negativo.")
        if not self.pca_components:
            object.__setattr__(self, "pca_components", PCA_COMPONENTS[self.dataset])

    @property
    def scene_name(self) -> str:
        return DATASETS[self.dataset]

    @property
    def model_name(self) -> str:
        """Prefixo dos arquivos de saída (era `TORCH_MODEL` no notebook)."""
        return f"{self.dataset}_CNN"

    @property
    def resolved_eval_batch_size(self) -> int:
        """Batch de inferência efetivamente usado pelos DataLoaders."""
        return self.eval_batch_size or self.batch_size

    @property
    def effective_batch_size(self) -> int:
        """Número de amostras que contribuem para cada atualização do Adam."""
        return self.batch_size * self.gradient_accumulation_steps

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        modo = f"{self.model_type} + segmentação ({self.segmenter})" if self.two_input else self.model_type
        return (
            f"{self.dataset} — {self.scene_name} | {modo}\n"
            f"PCA: {self.pca_components} componentes | "
            f"split: {1 - self.test_ratio:.0%} treino, "
            f"{self.test_ratio * (1 - self.valid_ratio):.0%} teste, "
            f"{self.test_ratio * self.valid_ratio:.0%} validação\n"
            f"épocas: {self.epochs} (patience {self.patience}) | "
            f"batch físico/efetivo: {self.batch_size}/{self.effective_batch_size} | "
            f"patch: {self.patch_size}×{self.patch_size} | "
            f"lr: {self.learning_rate} | dropout: {self.dropout}\n"
            f"seed: {self.seed} | determinístico: {self.deterministic}"
        )
