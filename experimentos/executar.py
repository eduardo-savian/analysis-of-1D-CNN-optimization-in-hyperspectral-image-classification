"""Lançador dos experimentos do TCC.

Uso:
    .venv/bin/python experimentos/executar.py 13 IP
    .venv/bin/python experimentos/executar.py 06 PU SA
    .venv/bin/python experimentos/executar.py --lista

POR QUE ESTE ARQUIVO EXISTE. Os experimentos 12 e 13 usam o executor de processos
com `spawn`. Com o código no pacote `hsi`, o filho encontra as funções importando o
pacote — de um script ou do notebook. O lançador continua sendo o jeito de rodar um
experimento longo fora do Jupyter, sem um kernel aberto.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")           # sem display: os experimentos só gravam arquivos
warnings.filterwarnings("ignore")

# Funciona mesmo sem `uv pip install -e .`: a raiz do repositório entra no path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from hsi.experimentos import EXPERIMENTOS  # noqa: E402

# No nível do módulo, e não dentro de `main()`: o filho do `spawn` reexecuta este
# arquivo como `__mp_main__` e herda a configuração de threads ao reexecutar esta linha.
torch.set_num_threads(1)        # uma thread por partição, não N × núcleos


def main() -> int:
    argumentos = sys.argv[1:]
    if not argumentos or argumentos[0] in {"--lista", "-l", "--help", "-h"}:
        print(__doc__.split("POR QUE")[0].strip())
        print("\nExperimentos disponíveis:\n")
        for chave, funcao in sorted(EXPERIMENTOS.items()):
            resumo = (funcao.__doc__ or "").strip().split("\n")[0]
            print(f"  {chave}  {resumo}")
        return 0

    numero, *cenas = argumentos
    numero = numero.zfill(2)
    if numero not in EXPERIMENTOS:
        print(f"experimento '{numero}' não existe. Use --lista para ver os disponíveis.")
        return 1
    EXPERIMENTOS[numero](*cenas)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
