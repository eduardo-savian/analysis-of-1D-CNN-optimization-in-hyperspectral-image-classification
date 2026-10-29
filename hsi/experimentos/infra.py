"""Caminhos de saída e retomada de execuções longas.

POR QUE A RETOMADA EXISTE. Três sessões de experimento morreram pela metade
(03/09, 10/09, 17/09) e custaram tudo de novo. Com a retomada, o experimento relê o
CSV consolidado, identifica o que já foi medido e continua de onde parou. O CSV é a
fonte de verdade — não um arquivo de estado à parte, que poderia divergir dele.

O LANÇADOR (`experimentos/executar.py`) roda um experimento sem um kernel aberto.
Ele fixa `torch.set_num_threads(1)` no nível do módulo porque o filho do `spawn`
reexecuta o lançador, e não a função do experimento, que só fixa a contagem no pai.
"""

import os
from pathlib import Path

import pandas as pd

from hsi.config import repo_root


def raiz_projeto() -> Path:
    """Diretório do projeto: a raiz do repositório, achada a partir do pacote.

    Não depende do diretório atual — o lançador grava em
    `experimentos/resultados/` de onde quer que seja chamado.
    """
    return repo_root()


def caminho_resultado(nome: str) -> Path:
    """Caminho de saída dentro de `experimentos/resultados/`, criando o diretório."""
    pasta = raiz_projeto() / "experimentos" / "resultados"
    pasta.mkdir(parents=True, exist_ok=True)
    return pasta / nome


def memoria_mb() -> float:
    """Memória residente do processo, em MB.

    Sobrou do experimento 07 (arquiteturas × segmentadores), removido quando o escopo
    ficou só na CNN 1D: hoje nenhuma célula o chama.
    """
    with open(f"/proc/{os.getpid()}/statm") as arquivo:
        return int(arquivo.read().split()[1]) * 4096 / 1e6


def carregar_feito(caminho: Path, chaves: tuple) -> tuple:
    """Devolve as linhas já medidas e o conjunto de células concluídas."""
    if not caminho.exists():
        return [], set()
    quadro = pd.read_csv(caminho)
    if quadro.empty or not all(c in quadro.columns for c in chaves):
        return [], set()
    linhas = quadro.to_dict(orient="records")
    feitas = {tuple(linha[c] for c in chaves) for linha in linhas}
    return linhas, feitas


def salvar(linhas: list, caminho: Path) -> None:
    """Grava o consolidado a cada célula concluída, para sobreviver a interrupção."""
    pd.DataFrame(linhas).to_csv(caminho, index=False)
