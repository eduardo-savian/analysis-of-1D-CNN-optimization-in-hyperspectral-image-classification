"""Carrega as definições do notebook para usar em scripts de experimento.

Por que isto existe: os experimentos precisam das mesmas funções do
`HSI_MLOps.ipynb` (run_experiment, automated_hpo, os segmentadores...), e
duplicá-las num pacote paralelo criaria duas verdades. Este módulo executa só as
células *declarativas* do notebook — imports, defs, classes e atribuições — e
ignora as que executam algo (as da seção 8), devolvendo um namespace pronto.

Uso:
    from _bootstrap import carregar
    ns = carregar()
    cfg = ns["ExperimentConfig"](dataset="IP")
    r = ns["run_experiment"](cfg, backend="thread", verbose=False)
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent.parent / "HSI_MLOps.ipynb"

_DECLARATIVO = (
    ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef,
    ast.ClassDef, ast.Assign, ast.AnnAssign, ast.Expr,
)


def _e_declarativa(fonte: str) -> bool:
    """A célula só define coisas? Chamadas de topo (display(), check_versions()) ficam de fora."""
    try:
        arvore = ast.parse(fonte)
    except SyntaxError:
        return False
    if not all(isinstance(no, _DECLARATIVO) for no in arvore.body):
        return False
    return not any(
        isinstance(no, ast.Expr) and isinstance(no.value, ast.Call) for no in arvore.body
    )


def carregar(destino: dict | None = None) -> dict:
    """Executa as células declarativas do notebook e devolve o namespace resultante.

    `destino` permite executar dentro dos globals do script chamador — necessário
    para o backend de processos, que precisa resolver `spatial_branch` como
    `__main__.spatial_branch` ao despicklar no processo filho.
    """
    ns = destino if destino is not None else {"__name__": "__main__"}
    caderno = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for indice, celula in enumerate(caderno["cells"]):
        if celula["cell_type"] != "code":
            continue
        fonte = "".join(celula["source"])
        if not _e_declarativa(fonte):
            continue
        exec(compile(fonte, f"<notebook célula {indice}>", "exec"), ns)
    return ns


def memoria_mb() -> float:
    """Memória residente do processo, em MB. Usado nos experimentos de memória."""
    with open(f"/proc/{os.getpid()}/statm") as arquivo:
        return int(arquivo.read().split()[1]) * 4096 / 1e6


def caminho_resultado(nome: str) -> Path:
    """Caminho de saída dentro de `experimentos/resultados/`, criando o diretório."""
    pasta = Path(__file__).resolve().parent / "resultados"
    pasta.mkdir(parents=True, exist_ok=True)
    return pasta / nome
