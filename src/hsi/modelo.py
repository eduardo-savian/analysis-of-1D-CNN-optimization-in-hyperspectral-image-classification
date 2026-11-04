"""CNN 1D de Souza (2023) com a fusão do id do superpixel."""

import torch
import torch.nn as nn

# As quatro convoluções não têm padding e consomem 8 + 2 + 2 + 2 = 14 posições do
# espectro; com menos de 15 componentes a saída ficaria vazia.
MIN_COMPONENTS = 15


class CNN1D(nn.Module):
    """Convoluções sobre o espectro do pixel; o id do superpixel só entra depois.

    A fusão é tardia: o id é concatenado ao vetor achatado das convoluções, e só as
    camadas densas veem as duas informações juntas. `fundir` é um passo separado
    porque a fusão concorrente protege apenas ele com a trava.
    """

    def __init__(self, n_components: int, n_classes: int, dropout: float,
                 fusion: str = "scalar", n_segments: int = 0, embedding_dim: int = 8):
        super().__init__()
        if n_components < MIN_COMPONENTS:
            raise ValueError(f"a CNN 1D precisa de pelo menos {MIN_COMPONENTS} componentes")
        self.convolucoes = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3), nn.ReLU(),
            nn.Flatten(),
        )
        n_flat = 64 * (n_components - 14)
        # "embedding": uma linha aprendida por região no lugar do id cru.
        self.embedding = nn.Embedding(n_segments, embedding_dim) if fusion == "embedding" else None
        entrada_densas = n_flat + (embedding_dim if self.embedding is not None else 1)
        self.densas = nn.Sequential(
            nn.Linear(entrada_densas, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )
        # Inicialização do Keras (Xavier uniforme, viés zero), que o original usava.
        for camada in self.modules():
            if isinstance(camada, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(camada.weight)
                nn.init.zeros_(camada.bias)

    def fundir(self, caracteristicas: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        if self.embedding is not None:
            seg = self.embedding(seg.reshape(-1).long())
        return torch.cat([caracteristicas, seg], dim=1)

    def forward(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        return self.densas(self.fundir(self.convolucoes(x), seg))


def build_model(config, dados: dict, device):
    """Rede, perda e otimizador para uma configuração e um conjunto de dados."""
    modelo = CNN1D(
        dados["espectros"].shape[1], dados["n_classes"], config.dropout, config.fusion,
        # O maior id + 1, e não a contagem de ids: os ids têm buracos depois que os
        # pixels sem rótulo saem.
        n_segments=int(dados["segmentos"].max()) + 1, embedding_dim=config.embedding_dim,
    ).to(device)
    perda = nn.CrossEntropyLoss()
    otimizador = torch.optim.Adam(modelo.parameters(), lr=config.learning_rate,
                                  eps=config.adam_eps, weight_decay=config.weight_decay)
    return modelo, perda, otimizador
