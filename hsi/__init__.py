"""Classificação hiperespectral com CNN 1D — o trabalho de Souza (2023) otimizado.

O código do TCC, organizado na ordem em que uma execução o percorre. O notebook
`HSI_MLOps.ipynb` importa daqui e guarda só a narrativa e as células de execução.

    config.py               caminhos, `ExperimentConfig` e tabelas por cena
    ambiente.py             versões, GPU, determinismo e sementes
    dados/
        cenas.py            leitura dos `.mat` e tabela de pixels
        preprocessamento.py PCA e composição RGB
        segmentacao.py      SEEDS, SLIC, LSC e Watershed
        particoes.py        splits (pixel, bloco, região), `PreparedData`, DataLoaders
    paralelo/
        ramos.py            pilar 2: ramos espectral e espacial em paralelo
        particao.py         pilar 2: a cena partida em N pedaços
    modelo.py               a CNN1D com a fusão do id do superpixel
    treino.py               laço de treino e parada antecipada
    avaliacao.py            OA, AA, Kappa e matriz de confusão
    checkpoint.py           salvar e recarregar modelos
    pipeline.py             `run_experiment`: da cena às métricas
    variantes.py            linha de base contra a nossa implementação
    hpo/                    pilar 3: espaço, estratégias, busca, seleção e batch
    mlops/                  rastreio de runs, registro, campanhas, relatórios e painéis
    visualizacao/           diagnósticos, explicabilidade e dashboards
    experimentos/           os experimentos do TCC (05 a 13) e as medições avulsas

Os experimentos rodam pelo lançador: `.venv/bin/python experimentos/executar.py --lista`.
"""
