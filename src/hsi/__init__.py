"""CNN 1D com fusão SEEDS (Souza, 2023) e as otimizações desta monografia.

As variáveis de execução ficam em config.toml, na raiz. Os módulos, na ordem de uma
execução:

    config.py               lê config.toml; caminhos e ExperimentConfig
    ambiente.py             versões, dispositivo, sementes e modo determinístico
    dados.py                cena, PCA, divisão e DataLoaders
    segmentadores.py        SEEDS, LSC e Watershed sobre a composição RGB
    modelo.py               a CNN 1D do projeto (e a do original) com a fusão do id
    treino.py               treino com parada antecipada e OA, AA e Kappa
    pipeline.py             run_experiment: da cena às métricas
    hpo.py                  busca de hiperparâmetros: grade e otimização bayesiana
    paralelismo/            partição da cena e fusão concorrente
    experimentos/           os experimentos 01 a 08
    figuras.py              as figuras dos resultados
"""
