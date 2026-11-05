# Medições avulsas citadas na monografia

Números medidos fora das campanhas dos experimentos, com a data de cada medição. Até
04/11 ficavam em `src/hsi/experimentos/medicoes.py`, que nenhum código usava. Os de
antes de 04/11 foram medidos com a CNN do notebook de Souza (2023).

```python

# Cenas: forma do cubo, pixels rotulados e classes.
CENAS_RESUMO = {
    "IP": {"nome": "Indian Pines", "forma": (145, 145, 200), "rotulados": 10249, "classes": 16},
    "PU": {"nome": "University of Pavia", "forma": (610, 340, 103), "rotulados": 42776, "classes": 9},
    "SA": {"nome": "Salinas Valley", "forma": (512, 217, 204), "rotulados": 54129, "classes": 16},
}

# Pixels por classe em Indian Pines, da verdade terrestre.
CLASSES_IP = [46, 1428, 830, 237, 483, 730, 28, 478, 20, 972, 2455, 593, 205, 1265, 386, 93]

# Ramos do pré-processamento (03/09), segundos, média de 5 repetições. "sequential",
# "thread" e "process" são o bloco inteiro; "espectral" e "espacial", cada ramo sozinho.
RAMOS = {
    "IP": {"sequential": 0.166, "thread": 0.158, "process": 0.739,
           "espectral": 0.037, "espacial": 0.129},
    "PU": {"sequential": 0.971, "thread": 1.027, "process": 1.296,
           "espectral": 0.072, "espacial": 0.899},
    "SA": {"sequential": 0.822, "thread": 0.675, "process": 1.118,
           "espectral": 0.142, "espacial": 0.680},
}

# MB enviados ao processo filho quando o ramo espacial roda em processo (03/09).
SERIALIZACAO_MB = {"IP": 8.4, "PU": 42.7, "SA": 45.3}

# Estrutura dos segmentadores com os parâmetros padrão (04/11, LSC e Watershed já
# calibrados pelo SEEDS), com o OpenCV em uma thread. Tempo em ms: mediana de 14 chamadas,
# em duas passadas de 7 com uma chamada de aquecimento cada; mede só a segmentação, sem
# o PCA(3) e a composição RGB.
SEGMENTADORES = {
    "IP": {
        "seeds": {"regioes": 81, "area": 259.6, "pureza": 0.8919, "tempo_ms": 33.3},
        "lsc": {"regioes": 81, "area": 259.6, "pureza": 0.8825, "tempo_ms": 37.7},
        "watershed": {"regioes": 82, "area": 256.4, "pureza": 0.8330, "tempo_ms": 1.2},
    },
    "PU": {
        "seeds": {"regioes": 45, "area": 4608.9, "pureza": 0.9272, "tempo_ms": 599.1},
        "lsc": {"regioes": 45, "area": 4608.9, "pureza": 0.8194, "tempo_ms": 280.2},
        "watershed": {"regioes": 46, "area": 4508.7, "pureza": 0.7918, "tempo_ms": 10.6},
    },
    "SA": {
        "seeds": {"regioes": 96, "area": 1157.3, "pureza": 0.9886, "tempo_ms": 343.1},
        "lsc": {"regioes": 96, "area": 1157.3, "pureza": 0.9241, "tempo_ms": 151.0},
        "watershed": {"regioes": 97, "area": 1145.4, "pureza": 0.9563, "tempo_ms": 5.5},
    },
}

# ms por segmentação do LSC calibrado com 1 e com 32 threads do OpenCV (04/11). Com 32,
# mudam de 188 a 322 pixels entre chamadas com a mesma imagem em IP e de 142 a 330 em PU;
# com 1, nenhum.
LSC_THREADS_MS = {"IP": {1: 35.6, 32: 14.8}, "PU": {1: 289.7, 32: 83.6}}

# Ablação da fusão (03/09): Indian Pines, 3 sementes, OA. Cada linha altera uma
# propriedade do id do superpixel; a primeira é a fusão herdada.
ABLACAO_FUSAO = [
    ("id original", "tudo", 0.8981),
    ("id permutado", "agrupamento e escala; perde a ordem", 0.8552),
    ("id normalizado [0,1]", "tudo menos a magnitude", 0.8223),
    ("id aleatório por pixel", "só a escala; perde o agrupamento", 0.7898),
]

# Fusão escalar (id cru) contra embedding (vetor por região), OA, 3 sementes (03/09).
MODOS_FUSAO = {
    "IP": {"escalar": 0.8981, "embedding": 0.9471},
    "PU": {"escalar": 0.9785, "embedding": 0.9868},
}

# Fração do treino (10/09), Indian Pines, 3 sementes: fração, amostras de treino, OA,
# desvio da OA, AA, Kappa, segundos por execução e épocas.
FRACAO_TREINO = [
    (1.00, 8711, 0.9016, 0.0016, 0.8308, 0.8877, 34.3, 44),
    (0.50, 4356, 0.8643, 0.0140, 0.7310, 0.8448, 18.8, 46),
    (0.25, 2179, 0.8270, 0.0105, 0.6755, 0.8021, 11.4, 58),
    (0.10, 871, 0.7586, 0.0109, 0.6006, 0.7227, 6.0, 59),
    (0.05, 435, 0.6810, 0.0131, 0.5322, 0.6334, 4.0, 60),
]

# MB que cada trial da busca segurava quando guardava os próprios dados (10/09).
MEMORIA_POR_TRIAL = {"IP": 2.5, "PU": 5.1, "SA": 6.5}

# Threads internas do PyTorch (08/10), Indian Pines, 25 épocas: (threads, segundos).
THREADS_TORCH = [(16, 11.67), (8, 7.81), (4, 7.90), (2, 8.45), (1, 8.55)]

# Partição com pool frio (08/10), parede do bloco em segundos, antes do experimento 13.
EXECUTORES_FRIO = {4: {"sequencial": 12.70, "thread": 21.78, "process": 8.53},
                   8: {"sequencial": 14.39, "thread": 56.05, "process": 9.36}}

# GIL (08/10): 4 tarefas, sequencial contra 4 threads, em segundos.
GIL = [("laço Python puro", 1.01, 0.91), ("multiplicação de matrizes (C)", 0.80, 0.28)]
```
