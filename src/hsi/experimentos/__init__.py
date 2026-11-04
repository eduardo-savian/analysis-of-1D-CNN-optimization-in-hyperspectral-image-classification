"""Os experimentos da monografia, um módulo por experimento.

    05  segmentadores com a CNN fixa (contraexemplo)
    06  segmentadores com a CNN ajustada para cada um
    08  busca aleatória, TPE e algoritmo genético
    10  configuração otimizada contra a herdada
    11  partição da cena
    13  pool frio contra pool quente
    14  fusão concorrente

Rodar: `.venv/bin/python src/executar.py 14 IP`. Os CSVs vão para experimentos/resultados/.
"""

from hsi.experimentos.e05_cnn_fixa import experimento_05
from hsi.experimentos.e06_segmentadores import experimento_06
from hsi.experimentos.e08_estrategias import experimento_08
from hsi.experimentos.e10_impacto_hpo import experimento_10
from hsi.experimentos.e11_particao import experimento_11
from hsi.experimentos.e13_pool_quente import experimento_13
from hsi.experimentos.e14_fusao_concorrente import experimento_14

EXPERIMENTOS = {
    "05": experimento_05, "06": experimento_06, "08": experimento_08, "10": experimento_10,
    "11": experimento_11, "13": experimento_13, "14": experimento_14,
}
