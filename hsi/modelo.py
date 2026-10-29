"""CNN 1D sobre a assinatura espectral, com fusão opcional do id do superpixel.

Convolui apenas ao longo do espectro de um pixel, já reduzido por PCA: a rede não
enxerga vizinhança alguma por conta própria, e é exatamente por isso que a segunda
entrada — a que diz a qual região homogênea o pixel pertence — tem efeito tão
grande aqui.

O escopo foi restrito à CNN 1D em 24/09, como no título do TCC; as CNNs 2D e 3D
estão no commit `c4665d5`, o último antes da remoção, e a decisão, no `Context.md`
(§2.1 “Implementação”).
"""

import torch
import torch.nn as nn


def apply_xavier_init(module: nn.Module) -> None:
    """Inicializa convoluções e densas com Xavier e zera seus vieses.

    É o padrão do Keras (`glorot_uniform` e viés zero), usado pela implementação
    original; o padrão do PyTorch é outro e mudaria o ponto de partida do treino.
    """
    if isinstance(module, (nn.Conv1d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class _FusionClassifier:
    """Cabeça densa com a fusão da segunda entrada.

    A fusão é **tardia**: as convoluções nunca veem a segmentação, que entra só
    depois do ``Flatten``. As duas entradas chegam separadas, como saem dos dois
    ramos de pré-processamento — independentes entre si até aqui, e por isso
    paralelizáveis.

    Dois modos, comparáveis porque tudo mais é idêntico:

    - ``scalar``: concatena o id cru como uma única coluna. Reproduz o baseline,
      mas trata um rótulo nominal como número contínuo (o id 78 vira 78,0) e a
      escala bruta acaba dando a essa coluna uma alavancagem desproporcional.
    - ``embedding``: uma ``nn.Embedding`` aprende um vetor por região. Não impõe
      ordem entre ids, não depende da magnitude e a inicialização padrão
      ``N(0, 1)`` deixa a segunda entrada na mesma escala das ativações.

    É um mixin, e não um ``nn.Module``: vem da época em que as CNNs 2D e 3D
    também o usavam, e depende do ``nn.Module`` da classe concreta para registrar
    as camadas e para ``parameters()``. Por isso ``CNN1D`` o inicializa depois de
    ``super().__init__()``.
    """

    def __init__(self, features: int, classes: int, two_input: bool, dropout: float,
                 fusion: str = "scalar", n_segments: int = 0, embedding_dim: int = 8) -> None:
        self.two_input = two_input
        self.fusion = fusion if two_input else "scalar"
        if self.two_input and self.fusion == "embedding":
            if n_segments < 1:
                raise ValueError("fusion='embedding' exige n_segments >= 1 (use PreparedData.n_segments).")
            self.segment_embedding = nn.Embedding(n_segments, embedding_dim)
            extra = embedding_dim
        else:
            self.segment_embedding = None
            extra = int(two_input)
        self.fused_features = features + extra
        self.classifier = nn.Sequential(
            nn.Linear(self.fused_features, 128), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, classes),
        )

    def classify(self, features: torch.Tensor, seg: torch.Tensor | None) -> torch.Tensor:
        """Concatena as características das convoluções à segunda entrada e classifica.

        A segunda entrada é o id cru (``scalar``) ou o vetor aprendido da região
        (``embedding``), e sua ausência é erro quando o modelo foi construído com ela.
        """
        if self.two_input:
            if seg is None:
                raise ValueError("Modelo com duas entradas: informe o tensor de segmentação.")
            if self.segment_embedding is not None:
                # O tensor chega como float (N, 1) do DataLoader; a tabela de
                # embeddings indexa por inteiro.
                seg = self.segment_embedding(seg.reshape(-1).long())
            features = torch.cat([features, seg], dim=1)
        return self.classifier(features)

    @property
    def n_parameters(self) -> int:
        """Número de pesos treináveis, útil ao comparar custo das arquiteturas."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


#: Kernels das quatro convoluções da CNN1D, na ordem em que são aplicadas.
#: Sem padding, cada uma encurta a sequência em ``kernel - 1``. `CNN1D.features`
#: repete estes valores em vez de lê-los daqui: mudar um exige mudar o outro.
CNN1D_KERNELS = (9, 3, 3, 3)


#: Menor número de componentes espectrais que a CNN1D aceita. Com 4 convoluções
#: sem padding consumindo 14 posições, uma entrada menor que isto deixa a saída
#: vazia e o PyTorch falha com "Kernel size can't be greater than actual input
#: size" — erro que aponta para a convolução e não para a causa real, que é a
#: escolha de `pca_components`.
CNN1D_MIN_COMPONENTS = sum(k - 1 for k in CNN1D_KERNELS) + 1


class CNN1D(nn.Module, _FusionClassifier):
    """Baseline: quatro Conv1d sobre a assinatura espectral ``(N, 1, Kb)``."""

    def __init__(self, seq_len: int, classes: int, in_channels: int = 1, two_input: bool = False,
                 dropout: float = 0.5, fusion: str = "scalar",
                 n_segments: int = 0, embedding_dim: int = 8) -> None:
        super().__init__()
        if seq_len < CNN1D_MIN_COMPONENTS:
            raise ValueError(
                f"CNN1D precisa de pelo menos {CNN1D_MIN_COMPONENTS} componentes "
                f"espectrais (kernels {CNN1D_KERNELS} consomem "
                f"{CNN1D_MIN_COMPONENTS - 1}); recebeu {seq_len}. "
                "Aumente `pca_components`."
            )
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=9), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3), nn.ReLU(), nn.Flatten(),
        )
        # Passagem a seco para medir o achatamento: 64 × (Kb − 14), ou seja, 1.024
        # colunas em Indian Pines (Kb = 30).
        with torch.no_grad():
            self.n_flat = self.features(torch.zeros(1, in_channels, seq_len)).shape[1]
        _FusionClassifier.__init__(self, self.n_flat, classes, two_input, dropout,
                                   fusion, n_segments, embedding_dim)
        self.apply(apply_xavier_init)

    def forward(self, x: torch.Tensor, seg: torch.Tensor | None = None) -> torch.Tensor:
        return self.classify(self.features(x), seg)


def build_model(config, input_shape: int | tuple[int, ...], classes: int, device,
                n_segments: int = 0) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
    """Constrói a CNN1D, a loss e o Adam a partir da configuração.

    ``input_shape`` vem de ``PreparedData.input_shape()``, ``(1, Kb)``, e só o
    último eixo importa. Um inteiro também é aceito e vale como ``Kb``, a forma
    antiga da chamada. ``n_segments`` vem de ``PreparedData.n_segments`` e só é
    usado com ``two_input=True`` e ``fusion == "embedding"``.
    """
    seq_len = input_shape if isinstance(input_shape, int) else input_shape[-1]
    model: nn.Module = CNN1D(
        seq_len, classes, two_input=config.two_input, dropout=config.dropout,
        fusion=config.fusion, n_segments=n_segments,
        embedding_dim=config.segment_embedding_dim,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, eps=config.adam_eps,
        weight_decay=config.weight_decay,
    )
    return model, criterion, optimizer
