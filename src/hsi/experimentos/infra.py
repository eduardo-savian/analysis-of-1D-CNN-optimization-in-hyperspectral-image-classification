"""Onde os experimentos gravam e como retomam uma execução interrompida.

O CSV de cada experimento é regravado a cada medição concluída. Ao recomeçar, o
experimento relê o CSV e pula as combinações que já estão nele.
"""

from pathlib import Path

import pandas as pd

from hsi.config import RESULTADOS


def caminho_resultado(nome: str) -> Path:
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    return RESULTADOS / nome


def carregar_feito(caminho: Path, chaves: tuple) -> tuple[list, set]:
    """Linhas já gravadas e o conjunto de combinações de `chaves` já medidas."""
    if not caminho.exists():
        return [], set()
    linhas = pd.read_csv(caminho).to_dict(orient="records")
    return linhas, {tuple(linha[c] for c in chaves) for linha in linhas}


def salvar(linhas: list, caminho: Path) -> None:
    pd.DataFrame(linhas).to_csv(caminho, index=False)
