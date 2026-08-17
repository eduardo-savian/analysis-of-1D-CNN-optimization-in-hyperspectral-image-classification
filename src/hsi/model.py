"""Família de CNNs para comparar contexto espectral, espacial e espectral-espacial.

``CNN1D`` é o baseline reprodutível: convolui apenas ao longo das bandas de um
pixel. ``CNN2D`` recebe patches das três componentes principais espaciais.
``CNN3D`` recebe cubos ``bandas × altura × largura`` e aprende filtros conjuntos
nos três eixos. Todas terminam na mesma cabeça MLP e podem concatenar o id do
superpixel antes da classificação, mantendo a fusão comparável.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["CNN1D", "CNN2D", "CNN3D", "apply_xavier_init", "build_model"]


def apply_xavier_init(module: nn.Module) -> None:
    """Inicializa convoluções/densas com Xavier e zera seus vieses.

    Aplicar a mesma regra nas três arquiteturas evita atribuir uma diferença de
    convergência à inicialização em vez do tipo de convolução.
    """
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class _FusionClassifier:
    """Mixin da cabeça compartilhada; a classe concreta inicializa ``nn.Module``."""

    def __init__(self, features: int, classes: int, two_input: bool, dropout: float) -> None:
        self.two_input = two_input
        self.classifier = nn.Sequential(
            nn.Linear(features + int(two_input), 128), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, classes),
        )

    def classify(self, features: torch.Tensor, seg: torch.Tensor | None) -> torch.Tensor:
        """Funde embedding e segmentação, validando a segunda entrada quando exigida."""
        if self.two_input:
            if seg is None:
                raise ValueError("Modelo com duas entradas: informe o tensor de segmentação.")
            features = torch.cat([features, seg], dim=1)
        return self.classifier(features)

    @property
    def n_parameters(self) -> int:
        """Número de pesos treináveis, útil ao comparar custo das arquiteturas."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


class CNN1D(nn.Module, _FusionClassifier):
    """Baseline: quatro Conv1d sobre a assinatura espectral ``(N, 1, Kb)``."""

    def __init__(self, seq_len: int, classes: int, in_channels: int = 1, two_input: bool = False,
                 dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=9), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3), nn.ReLU(), nn.Flatten(),
        )
        with torch.no_grad():
            self.n_flat = self.features(torch.zeros(1, in_channels, seq_len)).shape[1]
        _FusionClassifier.__init__(self, self.n_flat, classes, two_input, dropout)
        self.apply(apply_xavier_init)

    def forward(self, x: torch.Tensor, seg: torch.Tensor | None = None) -> torch.Tensor:
        return self.classify(self.features(x), seg)


class CNN2D(nn.Module, _FusionClassifier):
    """CNN espacial para patches PCA ``(N, 3, P, P)``.

    Duas convoluções e ``AdaptiveAvgPool2d`` aceitam qualquer patch ímpar >= 3;
    a adaptação evita recalcular camadas densas ao alterar ``patch_size``.
    """

    def __init__(self, in_channels: int, classes: int, two_input: bool = False, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
        )
        _FusionClassifier.__init__(self, 32, classes, two_input, dropout)
        self.apply(apply_xavier_init)

    def forward(self, x: torch.Tensor, seg: torch.Tensor | None = None) -> torch.Tensor:
        return self.classify(self.features(x), seg)


class CNN3D(nn.Module, _FusionClassifier):
    """CNN espectral-espacial para cubos ``(N, 1, B, P, P)``.

    O primeiro kernel aprende ao longo de sete bandas e 3×3 pixels; o segundo
    amplia a representação. Pooling adaptativo deixa o classificador independente
    do número de bandas e do patch, mas CNN3D consome mais RAM que as demais.
    """

    def __init__(self, classes: int, two_input: bool = False, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3), padding=(3, 1, 1)), nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3), padding=(2, 1, 1)), nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten(),
        )
        _FusionClassifier.__init__(self, 16, classes, two_input, dropout)
        self.apply(apply_xavier_init)

    def forward(self, x: torch.Tensor, seg: torch.Tensor | None = None) -> torch.Tensor:
        return self.classify(self.features(x), seg)


def build_model(config, input_shape: int | tuple[int, ...], classes: int, device) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
    """Constrói arquitetura, loss e Adam conforme ``config.model_type``.

    ``input_shape`` vem de ``PreparedData.input_shape``. Aceitar inteiro mantém
    compatibilidade com chamadas antigas de CNN1D, onde ele significava ``Kb``.
    """
    if config.model_type == "cnn1d":
        seq_len = input_shape if isinstance(input_shape, int) else input_shape[-1]
        model: nn.Module = CNN1D(seq_len, classes, two_input=config.two_input, dropout=config.dropout)
    elif config.model_type == "cnn2d":
        model = CNN2D(input_shape[0], classes, two_input=config.two_input, dropout=config.dropout)
    elif config.model_type == "cnn3d":
        model = CNN3D(classes, two_input=config.two_input, dropout=config.dropout)
    else:
        raise ValueError(f"model_type inválido: {config.model_type!r}.")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=config.adam_eps)
    return model, criterion, optimizer
