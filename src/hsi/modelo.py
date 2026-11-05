"""CNN 1D com a fusão do id do superpixel, em duas arquiteturas.

- "projeto", a usada no trabalho: a do capítulo 3, com convoluções de 16, 32, 64 e 128
  filtros, max-pooling depois das três primeiras e densas de 256 e 128 neurônios;
- "original", a do notebook de Souza (2023): 16, 32, 64 e 64 filtros, sem pooling, e
  densas de 128 e 64. Fica para a comparação da linha de base.

Nas duas, os kernels são os do original (9, 3, 3 e 3), e a saída tem dropout nas duas
densas e devolve logits; o softmax fica na função de perda.
"""

import torch
from torch import nn

KERNELS = (9, 3, 3, 3)
ARQUITETURAS = {
    "projeto": {"filtros": (16, 32, 64, 128), "pooling": True, "densas": (256, 128)},
    "original": {"filtros": (16, 32, 64, 64), "pooling": False, "densas": (128, 64)},
}
# No projeto, o padding "same" mantém o comprimento e só os três poolings de 2 o
# encurtam: com menos de 8 componentes sobraria zero. O original, sem padding, consome
# 8 + 2 + 2 + 2 = 14 posições e pede ao menos 15.
MIN_COMPONENTS = {"projeto": 8, "original": 15}


class CNN1D(nn.Module):
    """Convoluções sobre o espectro do pixel; o id do superpixel só entra depois.

    A fusão é tardia: o id é concatenado ao vetor achatado das convoluções, e só as
    camadas densas veem as duas informações juntas. `fundir` é um passo separado
    porque a fusão concorrente protege apenas ele com a trava.
    """

    def __init__(self, n_components: int, n_classes: int, dropout: float, cnn: str = "projeto"):
        super().__init__()
        if n_components < MIN_COMPONENTS[cnn]:
            raise ValueError(f"a CNN {cnn!r} precisa de pelo menos {MIN_COMPONENTS[cnn]} componentes")
        arq = ARQUITETURAS[cnn]
        camadas, canais, comprimento = [], 1, n_components
        for i, (filtros, kernel) in enumerate(zip(arq["filtros"], KERNELS)):
            if arq["pooling"]:
                camadas += [nn.Conv1d(canais, filtros, kernel, padding="same"), nn.ReLU()]
                if i < 3:
                    camadas.append(nn.MaxPool1d(2))
                    comprimento //= 2
            else:
                camadas += [nn.Conv1d(canais, filtros, kernel), nn.ReLU()]
                comprimento -= kernel - 1
            canais = filtros
        self.convolucoes = nn.Sequential(*camadas, nn.Flatten())
        # As densas recebem o vetor achatado mais o id do superpixel, como um número só.
        entrada_densas = canais * comprimento + 1
        d1, d2 = arq["densas"]
        self.densas = nn.Sequential(
            nn.Linear(entrada_densas, d1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d1, d2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d2, n_classes),
        )
        # Inicialização do Keras (Xavier uniforme, viés zero), que o original usava.
        for camada in self.modules():
            if isinstance(camada, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(camada.weight)
                nn.init.zeros_(camada.bias)

    def fundir(self, caracteristicas: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        return torch.cat([caracteristicas, seg], dim=1)

    def forward(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        return self.densas(self.fundir(self.convolucoes(x), seg))


def build_model(config, dados: dict, device):
    """Rede, perda e otimizador para uma configuração e um conjunto de dados."""
    modelo = CNN1D(
        dados["espectros"].shape[1], dados["n_classes"], config.dropout, config.cnn,
    ).to(device)
    perda = nn.CrossEntropyLoss()
    otimizador = torch.optim.Adam(modelo.parameters(), lr=config.learning_rate,
                                  eps=config.adam_eps, weight_decay=config.weight_decay)
    return modelo, perda, otimizador
