import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# Raiz do repositório, com config.toml, data/ e experimentos/.
RAIZ = Path(__file__).resolve().parents[2]
with open(RAIZ / "config.toml", "rb") as arquivo:
    CONFIG = tomllib.load(arquivo)

DADOS = RAIZ / CONFIG["caminhos"]["dados"]
RESULTADOS = RAIZ / CONFIG["caminhos"]["resultados"]
FIGURAS = RAIZ / CONFIG["caminhos"]["figuras"]

DATASETS = {"IP": "Indian Pines", "PU": "University of Pavia", "SA": "Salinas Valley"}
SEGMENTERS = tuple(CONFIG["segmentadores"]["lista"])
PCA_COMPONENTS = CONFIG["dados"]["pca"]
SEMENTE_DADOS = CONFIG["dados"]["semente"]
_PADRAO = CONFIG["execucao"]


@dataclass(frozen=True)
class ExperimentConfig:
    """Parâmetros de uma execução, com os padrões da seção [execucao] de config.toml."""

    dataset: str = _PADRAO["dataset"]
    segmenter: str = _PADRAO["segmenter"]
    # Sobrescreve parâmetros padrão do segmentador; a busca preenche este campo.
    segmenter_overrides: dict = field(default_factory=dict)
    cnn: str = _PADRAO["cnn"]

    test_ratio: float = _PADRAO["test_ratio"]
    valid_ratio: float = _PADRAO["valid_ratio"]
    train_fraction: float = _PADRAO["train_fraction"]

    epochs: int = _PADRAO["epochs"]
    batch_size: int = _PADRAO["batch_size"]
    learning_rate: float = _PADRAO["learning_rate"]
    weight_decay: float = _PADRAO["weight_decay"]
    dropout: float = _PADRAO["dropout"]
    adam_eps: float = _PADRAO["adam_eps"]
    patience: int = _PADRAO["patience"]
    min_delta: float = _PADRAO["min_delta"]
    # None desliga a meta.
    target: float | None = _PADRAO.get("target")

    seed: int = _PADRAO["seed"]
    # 0 usa o valor da cena em [dados.pca].
    pca_components: int = 0

    def __post_init__(self):
        if self.dataset not in DATASETS:
            raise ValueError(f"cena inválida: {self.dataset!r}")
        if self.segmenter not in SEGMENTERS:
            raise ValueError(f"segmentador inválido: {self.segmenter!r}")
        if self.cnn not in ("projeto", "original"):
            raise ValueError(f"CNN inválida: {self.cnn!r}")
        if not self.pca_components:
            object.__setattr__(self, "pca_components", PCA_COMPONENTS[self.dataset])
