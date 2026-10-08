"""Medições que não vivem em CSV de experimento, com a origem de cada uma.

Alguns números do trabalho saíram de medições pontuais feitas no terminal, e não de
um script com saída tabular: o benchmark dos três backends de pré-processamento, a
estrutura dos quatro segmentadores, a varredura de threads do PyTorch, a ablação da
fusão. Deixá-los soltos na conversa foi o que fez perdê-los três vezes.

Aqui eles ficam versionados, cada bloco com a data e como foi obtido, para que as
figuras sejam reprodutíveis e para que qualquer número no texto tenha procedência.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Cenas. Fonte: `load_scene` + contagem de rótulos não nulos.
CENAS = {
    "IP": {"nome": "Indian Pines", "forma": (145, 145, 200), "rotulados": 10249, "classes": 16},
    "PU": {"nome": "Pavia University", "forma": (610, 340, 103), "rotulados": 42776, "classes": 9},
    "SA": {"nome": "Salinas", "forma": (512, 217, 204), "rotulados": 54129, "classes": 16},
}

# Pixels por classe em Indian Pines, do ground truth. Explica a distância OA × AA.
CLASSES_IP = [46, 1428, 830, 237, 483, 730, 28, 478, 20, 972, 2455, 593, 205, 1265, 386, 93]

# ---------------------------------------------------------------------------
# Paralelismo dos dois ramos do pré-processamento (03/09).
# 5 repetições por combinação; tempos em segundos.
RAMOS = {
    "IP": {"sequential": 0.166, "thread": 0.158, "process": 0.739,
           "espectral": 0.037, "espacial": 0.129},
    "PU": {"sequential": 0.971, "thread": 1.027, "process": 1.296,
           "espectral": 0.072, "espacial": 0.899},
    "SA": {"sequential": 0.822, "thread": 0.675, "process": 1.118,
           "espectral": 0.142, "espacial": 0.680},
}

# Arrays enviados ao processo filho no backend `process`, em MB.
SERIALIZACAO_MB = {"IP": 8.4, "PU": 42.7, "SA": 45.3}

# ---------------------------------------------------------------------------
# Estrutura dos quatro segmentadores (10/09). Métricas determinísticas.
SEGMENTADORES = {
    "IP": {
        "seeds": {"regioes": 81, "area": 259.6, "borda": 0.2051, "pureza": 0.8831, "tempo": 0.149},
        "slic": {"regioes": 324, "area": 64.9, "borda": 0.2920, "pureza": 0.9581, "tempo": 0.051},
        "lsc": {"regioes": 324, "area": 64.9, "borda": 0.3503, "pureza": 0.9699, "tempo": 0.031},
        "watershed": {"regioes": 273, "area": 77.0, "borda": 0.5650, "pureza": 0.8543, "tempo": 0.004},
    },
    "PU": {
        "seeds": {"regioes": 45, "area": 4608.9, "borda": 0.1063, "pureza": 0.9212, "tempo": 1.854},
        "slic": {"regioes": 1428, "area": 145.2, "borda": 0.2132, "pureza": 0.9857, "tempo": 0.263},
        "lsc": {"regioes": 1428, "area": 145.2, "borda": 0.2440, "pureza": 0.9955, "tempo": 0.203},
        "watershed": {"regioes": 421, "area": 492.6, "borda": 0.2797, "pureza": 0.9313, "tempo": 0.030},
    },
}

# Dificuldade espacial do teste sob split por região, por segmentador (24/09):
# distância em pixels de cada amostra de teste ao treino mais próximo.
DISTANCIA_TESTE = {
    "seeds": {"regioes": 63, "media": 4.24, "maxima": 16.9, "colados": 0.29},
    "lsc": {"regioes": 214, "media": 2.40, "maxima": 8.3, "colados": 0.51},
    "slic": {"regioes": 229, "media": 2.35, "maxima": 8.7, "colados": 0.53},
    "watershed": {"regioes": 163, "media": 2.06, "maxima": 7.1, "colados": 0.62},
}

# ---------------------------------------------------------------------------
# Ablação da fusão (03/09). Indian Pines, split por pixel, 3 sementes.
ABLACAO_FUSAO = [
    ("id original", "tudo", 0.8981),
    ("id permutado", "agrupamento e escala; perde a ordem", 0.8552),
    ("id normalizado [0,1]", "tudo menos a magnitude", 0.8223),
    ("sem fusão", "piso", 0.8101),
    ("id aleatório por pixel", "só a escala; perde o agrupamento", 0.7898),
]

# Modo de fusão: escalar (id cru) contra embedding (vetor por região).
MODOS_FUSAO = {
    "IP": {"sem fusão": 0.8101, "escalar": 0.8981, "embedding": 0.9471},
    "PU": {"sem fusão": 0.9480, "escalar": 0.9785, "embedding": 0.9868},
}

# ---------------------------------------------------------------------------
# Fração do conjunto de treino (10/09). Indian Pines, 3 sementes, política completa.
FRACAO_TREINO = [
    # fração, amostras, OA, desvio OA, AA, Kappa, tempo por execução, épocas usadas
    (1.00, 8711, 0.9016, 0.0016, 0.8308, 0.8877, 34.3, 44),
    (0.50, 4356, 0.8643, 0.0140, 0.7310, 0.8448, 18.8, 46),
    (0.25, 2179, 0.8270, 0.0105, 0.6755, 0.8021, 11.4, 58),
    (0.10, 871, 0.7586, 0.0109, 0.6006, 0.7227, 6.0, 59),
    (0.05, 435, 0.6810, 0.0131, 0.5322, 0.6334, 4.0, 60),
]

# ---------------------------------------------------------------------------
# Memória retida por trial, em MB (10/09). Medida antes de o escopo virar só CNN1D;
# mantida porque justifica `release_result` e documenta o custo das arquiteturas 2D/3D.
MEMORIA_POR_TRIAL = {
    "cnn1d": {"IP": 2.5, "PU": 5.1, "SA": 6.5},
    "cnn2d": {"IP": 6.1, "PU": 25.7, "SA": 32.5},
    "cnn3d": {"IP": 410.0, "PU": 881.2, "SA": 2210.0},
}

# ---------------------------------------------------------------------------
# Threads internas do PyTorch (08/10). Indian Pines, CNN1D, 25 épocas.
THREADS_TORCH = [(16, 11.67), (8, 7.81), (4, 7.90), (2, 8.45), (1, 8.55)]

# Executores na partição da cena (08/10). Parede do bloco, em segundos.
EXECUTORES = {4: {"sequencial": 12.70, "thread": 21.78, "process": 8.53},
              8: {"sequencial": 14.39, "thread": 56.05, "process": 9.36}}

# Demonstração do GIL (08/10): 4 tarefas, sequencial contra 4 threads.
GIL = [("laço Python puro", 1.01, 0.91), ("multiplicação de matrizes (C)", 0.80, 0.28)]
