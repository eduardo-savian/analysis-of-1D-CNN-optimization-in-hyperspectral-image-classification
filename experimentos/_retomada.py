"""Retomada de experimentos longos: pula células já medidas.

Por que existe: as execuções longas morreram três vezes junto com a sessão, sempre
pela metade. Sem retomada, cada morte custa tudo de novo. Com ela, o script relê o
CSV consolidado, identifica o que já foi medido e continua de onde parou.

O CSV é a fonte de verdade — não um arquivo de estado à parte, que poderia
divergir dele.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def carregar_feito(caminho: Path, chaves: tuple[str, ...]) -> tuple[list[dict], set[tuple]]:
    """Devolve as linhas já medidas e o conjunto de células concluídas."""
    if not caminho.exists():
        return [], set()
    quadro = pd.read_csv(caminho)
    if quadro.empty or not all(c in quadro.columns for c in chaves):
        return [], set()
    linhas = quadro.to_dict(orient="records")
    feitas = {tuple(linha[c] for c in chaves) for linha in linhas}
    return linhas, feitas


def salvar(linhas: list[dict], caminho: Path) -> None:
    """Grava o consolidado a cada célula concluída, para sobreviver a interrupção."""
    pd.DataFrame(linhas).to_csv(caminho, index=False)
