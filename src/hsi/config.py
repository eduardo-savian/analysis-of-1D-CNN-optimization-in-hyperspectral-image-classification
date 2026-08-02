"""Configuração do experimento — tudo o que antes eram variáveis globais soltas.

No notebook original os hiperparâmetros ficavam espalhados por várias células
(`dataset`, `test_ratio`, `two_input`, `Kb`, `NUM_EPOCHS`, `BATCH_SIZE`, ...) e
eram lidos implicitamente por funções que dependiam do escopo global. Aqui viram
um objeto único, explícito e serializável.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

__all__ = ["ExperimentConfig", "DATASETS"]

# Cenas suportadas: código curto -> nome legível.
DATASETS = {
    "IP": "Indian Pines",
    "PU": "Pavia University",
    "SA": "Salinas Valley",
}

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
    #: True -> espectro + id do superpixel (SEEDS); False -> só espectro.
    two_input: bool = True

    #: Fração da base que sai do treino (vai para teste+validação).
    test_ratio: float = 0.15
    #: Fração de teste+validação que vira validação.
    valid_ratio: float = 0.5

    epochs: int = 60
    batch_size: int = 32
    learning_rate: float = 1e-3
    #: O padrão do PyTorch é 1e-8; 1e-7 é o valor da implementação de referência
    #: e é mantido para que os resultados sejam comparáveis com os dela.
    adam_eps: float = 1e-7
    dropout: float = 0.5
    patience: int = 5

    seed: int = 42
    #: cudnn.deterministic + TF32 off. Troque para False só ao medir desempenho.
    deterministic: bool = True

    #: num_workers=0 é intencional: os tensores já estão em RAM e cada batch tem
    #: 32 x Kb floats. Subprocessos custariam mais em IPC do que economizam — o
    #: gargalo deste modelo é lançamento de kernel, não I/O.
    num_workers: int = 0

    #: Preenchido no __post_init__ a partir do dataset, se não for informado.
    pca_components: int = 0

    def __post_init__(self) -> None:
        if self.dataset not in DATASETS:
            raise ValueError(f"dataset inválido: {self.dataset!r}. Use um de {sorted(DATASETS)}.")
        if not self.pca_components:
            object.__setattr__(self, "pca_components", PCA_COMPONENTS[self.dataset])

    @property
    def scene_name(self) -> str:
        return DATASETS[self.dataset]

    @property
    def model_name(self) -> str:
        """Prefixo dos arquivos de saída (era `TORCH_MODEL` no notebook)."""
        return f"{self.dataset}_CNN"

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        modo = "espectro + segmentação (SEEDS)" if self.two_input else "somente espectro"
        return (
            f"{self.dataset} — {self.scene_name} | {modo}\n"
            f"PCA: {self.pca_components} componentes | "
            f"split: {1 - self.test_ratio:.0%} treino, "
            f"{self.test_ratio * (1 - self.valid_ratio):.0%} teste, "
            f"{self.test_ratio * self.valid_ratio:.0%} validação\n"
            f"épocas: {self.epochs} (patience {self.patience}) | batch: {self.batch_size} | "
            f"lr: {self.learning_rate} | dropout: {self.dropout}\n"
            f"seed: {self.seed} | determinístico: {self.deterministic}"
        )
