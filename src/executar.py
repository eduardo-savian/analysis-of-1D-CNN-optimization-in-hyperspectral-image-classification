"""Roda um experimento fora do Jupyter.

    .venv/bin/python src/executar.py 08 IP
    .venv/bin/python src/executar.py 03 PU SA

Sem cena, cada experimento usa as suas cenas padrão.
"""

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
warnings.filterwarnings("ignore")
# Funciona sem instalar o pacote: a pasta src/ entra no caminho de importação.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from hsi.experimentos import EXPERIMENTOS  # noqa: E402

# No nível do módulo: os processos filhos do spawn reexecutam este arquivo.
torch.set_num_threads(1)

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1].zfill(2) not in EXPERIMENTOS:
        print(__doc__)
        print("experimentos:", ", ".join(EXPERIMENTOS))
        raise SystemExit(1)
    EXPERIMENTOS[sys.argv[1].zfill(2)](*sys.argv[2:])
