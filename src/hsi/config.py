"""Caminhos do projeto e a configuração de uma execução."""

from dataclasses import dataclass, field
from pathlib import Path

# src/hsi/config.py -> raiz do repositório, onde ficam data/ e experimentos/.
RAIZ = Path(__file__).resolve().parents[2]
DADOS = RAIZ / "data" / "dataHSI"
RESULTADOS = RAIZ / "experimentos" / "resultados"

DATASETS = {"IP": "Indian Pines", "PU": "University of Pavia", "SA": "Salinas Valley"}
SEGMENTERS = ("seeds", "slic", "lsc", "watershed")

# Componentes do PCA espectral por cena, como no notebook original.
PCA_COMPONENTS = {"IP": 30, "PU": 15, "SA": 15}


@dataclass(frozen=True)
class ExperimentConfig:
    """Parâmetros de uma execução.

    Os padrões são os do notebook de Souza (2023): University of Pavia, SEEDS, lote 32,
    taxa 1e-3, dropout 0,5 e até 60 épocas. A parada antecipada é a deste trabalho:
    10 épocas sem a acurácia de validação melhorar 0,001.
    """

    dataset: str = "PU"
    segmenter: str = "seeds"
    # Mudanças sobre os parâmetros padrão do segmentador; a busca de hiperparâmetros
    # preenche este campo.
    segmenter_overrides: dict = field(default_factory=dict)
    # "scalar" concatena o id do superpixel como um número, como no original;
    # "embedding" troca o id por um vetor aprendido por região.
    fusion: str = "scalar"
    embedding_dim: int = 8

    # 15% das amostras saem do treino e se dividem entre teste e validação.
    test_ratio: float = 0.15
    valid_ratio: float = 0.5
    # Fração do treino usada de fato; validação e teste não mudam.
    train_fraction: float = 1.0

    epochs: int = 60
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    dropout: float = 0.5
    # 1e-7 é o padrão do Keras; o do PyTorch é 1e-8.
    adam_eps: float = 1e-7
    patience: int = 10
    min_delta: float = 1e-3
    # Meta de acurácia de validação que encerra o treino; None desliga.
    target: float | None = None

    seed: int = 42
    # 0 usa o valor da cena em PCA_COMPONENTS.
    pca_components: int = 0

    def __post_init__(self):
        if self.dataset not in DATASETS:
            raise ValueError(f"cena inválida: {self.dataset!r}")
        if self.segmenter not in SEGMENTERS:
            raise ValueError(f"segmentador inválido: {self.segmenter!r}")
        if self.fusion not in ("scalar", "embedding"):
            raise ValueError(f"fusão inválida: {self.fusion!r}")
        if not self.pca_components:
            object.__setattr__(self, "pca_components", PCA_COMPONENTS[self.dataset])
