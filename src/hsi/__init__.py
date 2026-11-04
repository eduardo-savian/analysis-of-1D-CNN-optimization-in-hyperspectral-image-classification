"""CNN 1D com fusão SEEDS (Souza, 2023) e as otimizações desta monografia.

Na ordem em que uma execução percorre o código:

    config.py               caminhos e ExperimentConfig, com a configuração herdada como padrão
    ambiente.py             versões, dispositivo, sementes e modo determinístico
    dados.py                cena, PCA, segmentadores, divisão e DataLoaders
    modelo.py               a CNN 1D com a fusão do id do superpixel
    treino.py               treino com parada antecipada e OA, AA e Kappa
    pipeline.py             run_experiment: da cena às métricas
    hpo.py                  espaço de busca e as três estratégias
    particao.py             a cena dividida em N partes, em threads ou processos
    fusao_concorrente.py    o esquema do orientador: T0 segmenta, T1..TN fundem
    experimentos/           os experimentos 05 a 14 da monografia
    figuras.py              as figuras do capítulo de resultados
"""
