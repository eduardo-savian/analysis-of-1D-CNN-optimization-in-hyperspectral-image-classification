"""Os experimentos do TCC, um módulo por experimento, e o índice que o lançador usa.

    infra.py                  caminhos de saída e retomada pelo CSV
    medicoes.py               números medidos fora de um CSV, com data e procedência
    e05_cnn_fixa.py           segmentadores com a CNN fixa (desenho inválido, guardado)
    e06_segmentadores.py      segmentadores sob otimização conjunta (pilar 4)
    e08_estrategias.py        estratégias de busca e peso dos eixos (pilar 3)
    e09_validacao_regiao.py   pixel contra região: quanto sobrevive sem vazamento
    e10_impacto_hpo.py        receita herdada contra receita otimizada (pilar 3)
    e11_particao.py           paralelismo por partição da cena (pilar 2)
    e12_particao_split.py     a partição sob split disjunto
    e13_pool_quente.py        pool frio contra quente (pilar 2)

`EXPERIMENTOS` é o que `experimentos/executar.py` consulta para rodar um deles.
"""

from hsi.experimentos.e05_cnn_fixa import experimento_05
from hsi.experimentos.e06_segmentadores import experimento_06
from hsi.experimentos.e08_estrategias import experimento_08
from hsi.experimentos.e09_validacao_regiao import experimento_09
from hsi.experimentos.e10_impacto_hpo import experimento_10
from hsi.experimentos.e11_particao import experimento_11
from hsi.experimentos.e12_particao_split import experimento_12
from hsi.experimentos.e13_pool_quente import experimento_13

EXPERIMENTOS = {
    "05": experimento_05,   # segmentadores com CNN fixa (contraexemplo inválido)
    "06": experimento_06,   # segmentadores sob otimização conjunta
    "08": experimento_08,   # estratégias de busca e peso dos eixos
    "09": experimento_09,   # validação cruzada por região
    "10": experimento_10,   # impacto do HPO contra a receita herdada
    "11": experimento_11,   # paralelismo por partição da cena
    "12": experimento_12,   # a partição sob split disjunto
    "13": experimento_13,   # pool quente: partida contra paralelismo
}
