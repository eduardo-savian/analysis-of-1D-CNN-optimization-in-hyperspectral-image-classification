"""Os experimentos, um módulo por experimento.

    01  linha de base: hiperparâmetros originais, CNN do projeto e a do original
    02  segmentadores com a configuração fixa (contraexemplo)
    03  segmentadores com a CNN ajustada para cada um
    04  busca em grade contra otimização bayesiana
    05  configuração otimizada contra a original
    06  partição da cena
    07  pool frio contra pool quente
    08  fusão concorrente

Rodar: `.venv/bin/python src/executar.py 08 IP`. Os CSVs vão para experimentos/resultados/.
"""

from hsi.experimentos.e01_linha_base import experimento_01
from hsi.experimentos.e02_cnn_fixa import experimento_02
from hsi.experimentos.e03_segmentadores import experimento_03
from hsi.experimentos.e04_estrategias import experimento_04
from hsi.experimentos.e05_impacto_hpo import experimento_05
from hsi.experimentos.e06_particao import experimento_06
from hsi.experimentos.e07_pool_quente import experimento_07
from hsi.experimentos.e08_fusao_concorrente import experimento_08

EXPERIMENTOS = {
    "01": experimento_01, "02": experimento_02, "03": experimento_03, "04": experimento_04, "05": experimento_05,
    "06": experimento_06, "07": experimento_07, "08": experimento_08,
}
