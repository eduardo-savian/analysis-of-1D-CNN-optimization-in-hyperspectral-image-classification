"""Resolução de caminhos — igual no Jupyter, em `experimentos/executar.py` e na AWS.

O notebook original montava o Google Drive e usava `os.getcwd()`, o que quebrava
conforme o diretório em que o Jupyter tinha sido iniciado. Aqui a raiz do
repositório é descoberta subindo a árvore a partir do diretório atual.
"""

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

#: O pacote vive na raiz do repositório (`<repo>/hsi/`): é dali que a busca parte.
_PACOTE = Path(__file__).resolve().parent


def repo_root(start: Path | None = None) -> Path:
    """Raiz do repositório: o primeiro diretório com `.git` a partir de `start`.

    Sem `start`, a busca parte do próprio pacote, e não do diretório atual: o
    notebook, o lançador e um script rodado de qualquer pasta encontram os mesmos
    `data/` e `outputs/`. Numa cópia sem `.git`, devolve a pasta que contém `hsi/`.
    """
    here = (start or _PACOTE).resolve()
    for base in (here, *here.parents):
        if (base / ".git").is_dir():
            return base
    return _PACOTE.parent if start is None else here


def data_dir(root: Path | None = None) -> Path:
    """Diretório dos `.mat`.

    `HSI_DATA_DIR` tem precedência (útil para apontar um EBS/EFS na AWS); sem ela,
    `data/dataHSI/` dentro do repositório. O caminho do ambiente pode estar fora do
    repositório, por isso a raiz é descoberta à parte e não derivada daqui.
    """
    from_env = os.environ.get("HSI_DATA_DIR")
    if from_env:
        return Path(from_env).expanduser().resolve()
    return (root or repo_root()) / "data" / "dataHSI"


def output_dir(root: Path | None = None, create: bool = True) -> Path:
    """Diretório de saídas (curvas, checkpoints, relatórios)."""
    out = (root or repo_root()) / "outputs"
    if create:
        out.mkdir(parents=True, exist_ok=True)
    return out


# ----------------------------------------------------------------------------
# CONFIGURAÇÃO
# ----------------------------------------------------------------------------
# Configuração do experimento — tudo o que antes eram variáveis globais soltas.
#
# No notebook original os hiperparâmetros ficavam espalhados por várias células
# (`dataset`, `test_ratio`, `two_input`, `Kb`, `NUM_EPOCHS`, `BATCH_SIZE`, ...) e
# eram lidos implicitamente por funções que dependiam do escopo global. Aqui viram
# um objeto único, explícito e serializável.


# Cenas suportadas: código curto -> nome legível.
DATASETS = {
    "IP": "Indian Pines",
    "PU": "Pavia University",
    "SA": "Salinas Valley",
}


# Famílias disponíveis. As strings ficam aqui, e não em `hsi/modelo.py` e
# `hsi/dados/segmentacao.py`, porque `ExperimentConfig` as valida e este módulo não
# pode depender daqueles, que dependem dele. `MODEL_TYPES` tem um elemento só, mas segue existindo para que
# `config.json` registre a arquitetura explicitamente em cada execução.
#: Arquitetura do trabalho. O TCC é sobre CNNs 1D, e o escopo foi restrito a ela
#: em 24/09; a versão com CNN 2D e 3D está no commit `c4665d5`.
MODEL_TYPES = ("cnn1d",)


SEGMENTERS = ("seeds", "slic", "lsc", "watershed")


# Componentes de PCA por cena. O original usava `30 if dataset == 'IP' else 15`.
# A diferença não vem do número de bandas nem de classes — Salinas tem 204 bandas
# e também 16 classes —, e sim de Indian Pines ter classes espectralmente
# próximas: milho e soja aparecem em três manejos de plantio cada.
PCA_COMPONENTS = {"IP": 30, "PU": 15, "SA": 15}


@dataclass(frozen=True)
class ExperimentConfig:
    """Parâmetros de uma execução completa.

    Os padrões repetem o notebook original — PaviaU, duas entradas, 60 épocas,
    batch 32, lr 1e-3, dropout 0,5 —, com semente 42 e modo determinístico. A
    exceção é a parada antecipada: o original parava por `val_loss` com patience 5;
    aqui vale a regra do `Context.md`, §2.2 (`val_accuracy`, patience 10), com a meta de 0,95
    desligada.
    """

    dataset: str = "PU"
    #: True -> espectro + id do superpixel como segunda entrada (ver `fusion`);
    #: False -> só o espectro.
    two_input: bool = True
    #: Mantido como campo para que `config.json` continue registrando a
    #: arquitetura explicitamente, mesmo havendo só uma.
    model_type: str = "cnn1d"
    #: Segmentador usado quando `two_input=True` ou `split_mode="region"` — o split
    #: por região usa as regiões dele mesmo sem fusão.
    segmenter: str = "seeds"
    #: Ajustes específicos rastreados do segmentador (preenchidos pelo HPO).
    segmenter_overrides: dict = field(default_factory=dict)
    #: Como o id da região entra na cabeça densa quando `two_input=True`.
    #: "scalar" concatena o id cru — é o baseline reproduzível, mas trata um
    #: rótulo nominal como número contínuo e depende da magnitude bruta do id.
    #: "embedding" aprende um vetor por região: sem ordem falsa entre ids e sem
    #: depender da escala. A ablação está em `ABLACAO_FUSAO` e `MODOS_FUSAO`
    #: (`hsi/experimentos/medicoes.py`).
    fusion: str = "scalar"
    #: Dimensão do vetor aprendido por região quando `fusion="embedding"`.
    segment_embedding_dim: int = 8

    #: "pixel" = split aleatório estratificado por pixel. É o protocolo da
    #: literatura de HSI, mas deixa o mesmo superpixel presente em treino e em
    #: teste: um modelo com fusão pode memorizar "região R -> classe C".
    #: "region" = cada superpixel inteiro vai para uma partição só; elimina esse
    #: vazamento por construção e é o modo que mede a acurácia sem ele.
    #: "block" = blocos retangulares da cena; independe do segmentador, mas a
    #: disjunção é apenas aproximada (regiões cruzam a fronteira do bloco).
    split_mode: str = "pixel"
    #: Lado do bloco em pixels quando `split_mode="block"`. 0 usa 1/4 do menor
    #: lado da cena (mínimo de 4 px; 36 px em Indian Pines) — bem maior que o
    #: superpixel típico, para que poucas regiões cruzem a fronteira.
    split_block: int = 0
    #: Validação cruzada por região, só com `split_mode="region"`. 0 mantém a
    #: partição aleatória única (que deixa poucas regiões no teste e produz desvio
    #: alto). Com k ≥ 3, as regiões são divididas em k grupos: o grupo `split_fold`
    #: é o teste e o resto treina; a validação é uma amostra estratificada por pixel
    #: desse resto (ver `split_indices_region_folds`). Percorrer `split_fold` de 0 a
    #: k-1 testa **cada região exatamente uma vez** — é o que torna o número confiável.
    split_folds: int = 0
    #: Qual grupo é o conjunto de teste, quando `split_folds` está ativo.
    split_fold: int = 0

    #: Fração do conjunto de TREINO efetivamente usada, amostrada de forma
    #: estratificada por classe. 1.0 usa tudo; 0.10 usa um décimo. Validação e
    #: teste ficam intactos, então as métricas continuam comparáveis entre
    #: frações — muda o quanto o modelo vê, não como ele é avaliado.
    #: Treinar com poucas amostras rotuladas é, além de mais barato, o protocolo
    #: usual da literatura de HSI, onde rotular pixel é caro.
    train_fraction: float = 1.0

    #: Fração da base que sai do treino (vai para teste+validação).
    test_ratio: float = 0.15
    #: Fração de teste+validação que vira validação.
    valid_ratio: float = 0.5

    epochs: int = 60
    #: Batch físico enviado à GPU. 32 é o `BATCH_SIZE` do notebook original; para
    #: escolher um valor maior, use a varredura de batch (`hsi/hpo/varredura_batch.py`, seção 8.7 do
    #: notebook).
    batch_size: int = 32
    #: Batch de validação/teste. Zero reutiliza `batch_size`; como não há
    #: gradientes nessa fase, este valor normalmente pode ser maior.
    eval_batch_size: int = 0
    #: Quantos batches físicos acumulam gradiente antes de `optimizer.step()`.
    #: O batch efetivo é `batch_size * gradient_accumulation_steps`.
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-3
    #: Regularização L2 do Adam. 0.0 reproduz o baseline; o HPO varia este eixo
    #: para testar se penalizar pesos grandes ajuda em cenas com poucas amostras.
    weight_decay: float = 0.0
    #: O padrão do PyTorch é 1e-8; 1e-7 é o padrão do Keras, que o notebook original
    #: usava, e é mantido para que os resultados sejam comparáveis com os dele.
    adam_eps: float = 1e-7
    dropout: float = 0.5
    #: Janela de épocas consecutivas sem ganho significativo antes de parar.
    #: Ex.: 10 encerra após 10 épocas sem superar a melhor validação pelo Δ mínimo.
    patience: int = 10
    #: A métrica de validação que governa a parada; sua regra é salva no run.
    early_stopping_monitor: str = "val_accuracy"
    #: Ganho absoluto mínimo. 0.001 equivale a 0,1 ponto percentual de acurácia.
    early_stopping_min_delta: float = 1e-3
    #: Meta de validação. None usa somente patience; a política do `Context.md`, §2.2, usa 0.95.
    early_stopping_target: float | None = None
    #: Recupera os pesos da melhor época, e não os da última época executada.
    restore_best_weights: bool = True

    seed: int = 42
    #: cuDNN determinístico, TF32 off e OpenCV em uma thread (ver
    #: `configure_determinism`). Troque para False só ao medir desempenho.
    deterministic: bool = True

    #: num_workers=0 é intencional por padrão: os tensores já estão em RAM e os
    #: batches são pequenos. Só aumente após medir em uma máquina com carga de
    #: I/O/pré-processamento adicional.
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
        for name in ("epochs", "batch_size", "gradient_accumulation_steps", "prefetch_factor", "patience"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} deve ser positivo.")
        if not 0 < self.train_fraction <= 1:
            raise ValueError("train_fraction deve estar em (0, 1].")
        if self.weight_decay < 0:
            raise ValueError("weight_decay não pode ser negativo.")
        if self.fusion not in {"scalar", "embedding"}:
            raise ValueError("fusion deve ser 'scalar' ou 'embedding'.")
        if self.segment_embedding_dim < 1:
            raise ValueError("segment_embedding_dim deve ser positivo.")
        if self.split_mode not in {"pixel", "region", "block"}:
            raise ValueError("split_mode deve ser 'pixel', 'region' ou 'block'.")
        if self.split_block < 0:
            raise ValueError("split_block não pode ser negativo.")
        if self.split_folds and self.split_folds < 3:
            raise ValueError("split_folds deve ser 0 (partição única) ou pelo menos 3.")
        if self.split_folds and not 0 <= self.split_fold < self.split_folds:
            raise ValueError("split_fold deve estar em [0, split_folds).")
        if self.num_workers < 0:
            raise ValueError("num_workers não pode ser negativo.")
        if self.persistent_workers and not self.num_workers:
            raise ValueError("persistent_workers requer num_workers maior que zero.")
        if self.eval_batch_size < 0:
            raise ValueError("eval_batch_size não pode ser negativo.")
        if self.early_stopping_monitor not in {"val_accuracy", "val_loss"}:
            raise ValueError("early_stopping_monitor deve ser 'val_accuracy' ou 'val_loss'.")
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta não pode ser negativo.")
        if self.early_stopping_target is not None:
            if not np.isfinite(self.early_stopping_target):
                raise ValueError("early_stopping_target deve ser finito ou None.")
            if self.early_stopping_monitor == "val_accuracy" and not 0 <= self.early_stopping_target <= 1:
                raise ValueError("A meta de val_accuracy deve estar entre 0 e 1.")
        if not self.pca_components:
            object.__setattr__(self, "pca_components", PCA_COMPONENTS[self.dataset])

    @property
    def scene_name(self) -> str:
        return DATASETS[self.dataset]

    @property
    def model_name(self) -> str:
        """Prefixo dos arquivos de saída (`TF_MODEL` no notebook original)."""
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
        modo = (
            f"{self.model_type} + segmentação ({self.segmenter}, fusão {self.fusion})"
            if self.two_input else f"{self.model_type} (sem fusão)"
        )
        return (
            f"{self.dataset} — {self.scene_name} | {modo}\n"
            f"PCA: {self.pca_components} componentes | "
            f"split: {1 - self.test_ratio:.0%} treino, "
            f"{self.test_ratio * (1 - self.valid_ratio):.0%} teste, "
            f"{self.test_ratio * self.valid_ratio:.0%} validação\n"
            f"épocas: {self.epochs} | early stopping: {self.early_stopping_monitor}, "
f"patience {self.patience}, Δ≥{self.early_stopping_min_delta:g}, "
                f"meta {self.early_stopping_target if self.early_stopping_target is not None else 'desativada'} | "
            f"batch físico/efetivo: {self.batch_size}/{self.effective_batch_size} | "
            f"lr: {self.learning_rate} | dropout: {self.dropout} | "
            f"weight decay: {self.weight_decay:g}\n"
            + (f"treino: {self.train_fraction:.0%} das amostras disponíveis\n"
               if self.train_fraction < 1 else "")
            + f"split: {self.split_mode}"
            + (f" (blocos de {self.split_block or 'auto'} px)" if self.split_mode == "block" else "")
            + (f" (regiões de {self.segmenter}" if self.split_mode == "region" else "")
            + (f", fold {self.split_fold + 1}/{self.split_folds})"
               if self.split_mode == "region" and self.split_folds
               else ")" if self.split_mode == "region" else "")
            + f"\nseed: {self.seed} | determinístico: {self.deterministic}"
        )


# Nome usado por `ensure_dataset` e `load_scene` (`hsi/dados/cenas.py`).
default_data_dir = data_dir
