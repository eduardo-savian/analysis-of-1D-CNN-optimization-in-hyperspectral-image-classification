"""CNN 1D que classifica cada pixel pelo seu espectro.

Arquitetura: quatro convoluções 1D sem padding sobre o vetor espectral, seguidas
de uma MLP com dropout. A última camada devolve *logits* — o softmax fica a cargo
da `nn.CrossEntropyLoss` no treino e é aplicado explicitamente na predição, o que
evita calcular log(softmax) duas vezes e é numericamente mais estável.

O espectro entra em layout *channels-first* `(N, 1, Kb)`, como o `nn.Conv1d` exige:
uma sequência de comprimento Kb com um único canal.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["CNN1D", "apply_xavier_init", "build_model"]


def apply_xavier_init(module: nn.Module) -> None:
    """Inicializa pesos com Xavier/Glorot uniforme e zera os biases.

    O padrão do PyTorch para Conv1d/Linear é Kaiming uniforme com bias também
    uniforme. A escolha aqui é deliberada e faz parte da definição do modelo: é
    a inicialização usada na implementação de referência do trabalho, e trocá-la
    muda a curva de convergência.
    """
    if isinstance(module, (nn.Conv1d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class CNN1D(nn.Module):
    """Quatro convoluções 1D sobre o espectro + MLP classificadora.

    two_input=False -> classifica só pelo espectro.
    two_input=True  -> concatena o id do superpixel ao vetor achatado, antes da
                       MLP: além de "com o que este pixel se parece", a rede passa
                       a saber "de que região homogênea ele veio".
    """

    def __init__(
        self,
        seq_len: int,
        classes: int,
        in_channels: int = 1,
        two_input: bool = False,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.two_input = two_input

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=9), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3), nn.ReLU(),
            nn.Flatten(),
        )

        # Forward simbólico em vez de calcular à mão o encolhimento da sequência
        # pelas convoluções sem padding (Kb=15 -> 7 -> 5 -> 3 -> 1; Kb=30 -> 16).
        with torch.no_grad():
            self.n_flat = self.features(torch.zeros(1, in_channels, seq_len)).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(self.n_flat + (1 if two_input else 0), 128), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, classes),  # logits: o softmax vem da CrossEntropyLoss
        )

        self.apply(apply_xavier_init)

    def forward(self, x: torch.Tensor, seg: torch.Tensor | None = None) -> torch.Tensor:
        h = self.features(x)
        if self.two_input:
            if seg is None:
                raise ValueError("Modelo com duas entradas: informe o tensor de segmentação.")
            h = torch.cat([h, seg], dim=1)
        return self.classifier(h)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(config, seq_len: int, classes: int, device) -> tuple[CNN1D, nn.Module, torch.optim.Optimizer]:
    """Modelo, critério e otimizador conforme a configuração.

    Deve ser chamado logo após `seed_everything`: a construção do modelo consome
    o RNG global do PyTorch, e é dela que sai a reprodutibilidade dos pesos.
    """
    model = CNN1D(
        seq_len=seq_len,
        classes=classes,
        in_channels=1,
        two_input=config.two_input,
        dropout=config.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, eps=config.adam_eps
    )
    return model, criterion, optimizer
