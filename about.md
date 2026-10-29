# About — o código do TCC, peça por peça

Este documento explica **todo** o código do pacote `hsi/`: o que cada termo
significa, o que cada função faz, por que ela existe e como as peças se encaixam.
Ele é organizado em camadas — do problema para o código, do código para os
pilares, dos pilares para os resultados. Leia na ordem na primeira vez; depois use
como referência.

Os números citados vêm de `Context.md` e de `hsi/experimentos/medicoes.py`, onde
cada um tem data e procedência. Quando o código e este texto discordarem, vale o
código — e este texto precisa ser corrigido.

---

## Sumário

1. [O problema em uma página](#1-o-problema-em-uma-página)
2. [O caminho de uma execução](#2-o-caminho-de-uma-execução)
3. [Mapa do pacote](#3-mapa-do-pacote)
4. [Configuração e ambiente](#4-configuração-e-ambiente)
5. [Dados: da cena aos tensores](#5-dados-da-cena-aos-tensores)
6. [O modelo: a CNN1D](#6-o-modelo-a-cnn1d)
7. [A fusão — o coração da arquitetura herdada](#7-a-fusão--o-coração-da-arquitetura-herdada)
8. [Treino e parada antecipada](#8-treino-e-parada-antecipada)
9. [Avaliação: OA, AA e Kappa](#9-avaliação-oa-aa-e-kappa)
10. [O pipeline que amarra tudo](#10-o-pipeline-que-amarra-tudo)
11. [Pilar 1 — Organização do código](#11-pilar-1--organização-do-código)
12. [Pilar 2 — Paralelismo](#12-pilar-2--paralelismo)
13. [Pilar 3 — Automação de hiperparâmetros](#13-pilar-3--automação-de-hiperparâmetros)
14. [Pilar 4 — Segmentadores](#14-pilar-4--segmentadores)
15. [O protocolo de avaliação e o vazamento espacial](#15-o-protocolo-de-avaliação-e-o-vazamento-espacial)
16. [MLOps: rastreio, campanhas e relatórios](#16-mlops-rastreio-campanhas-e-relatórios)
17. [Visualização e explicabilidade](#17-visualização-e-explicabilidade)
18. [Os experimentos 05 a 13](#18-os-experimentos-05-a-13)
19. [O notebook e o lançador](#19-o-notebook-e-o-lançador)
20. [Pontos de atenção conhecidos](#20-pontos-de-atenção-conhecidos)
21. [Glossário](#21-glossário)
22. [Roteiro de estudo e perguntas de checagem](#22-roteiro-de-estudo-e-perguntas-de-checagem)

---

## 1. O problema em uma página

**Imagem hiperespectral (HSI).** Uma foto comum tem 3 canais (vermelho, verde,
azul). Uma imagem hiperespectral tem **centenas**: cada pixel guarda a refletância
da superfície em ~100 a ~200 faixas estreitas de comprimento de onda (as
**bandas**). O resultado é um **cubo** `(H, W, B)` — altura, largura, bandas.

**Assinatura espectral.** O vetor de `B` valores de um pixel. Materiais diferentes
(milho, soja, asfalto, telhado) refletem a luz de forma diferente em cada banda, e
é essa "impressão digital" que permite classificá-los.

**Ground truth (GT).** Uma matriz `(H, W)` com a classe verdadeira de cada pixel.
O valor **0 é background** — pixel não rotulado, que não entra no treino nem na
avaliação. As classes reais são 1..C.

**A tarefa.** Classificação **por pixel**: dado o espectro de um pixel (e, aqui, a
região a que ele pertence), dizer a classe. Não é segmentação semântica da imagem
inteira; cada pixel rotulado é uma amostra independente.

**As três cenas:**

| Código | Cena | Cubo | Pixels rotulados | Classes |
|---|---|---|---|---|
| IP | Indian Pines | 145 × 145 × 200 | 10.249 | 16 |
| PU | Pavia University | 610 × 340 × 103 | 42.776 | 9 |
| SA | Salinas Valley | 512 × 217 × 204 | 54.129 | 16 |

Indian Pines é a cena de trabalho: pequena, **desbalanceada** (de 20 a 2.455 pixels
por classe) e difícil, porque milho e soja aparecem em três manejos cada —
espectros muito parecidos.

**O trabalho de referência.** Sydney Matheus de Souza (2023): uma CNN 1D que recebe
o espectro do pixel (reduzido por PCA) **e o id do superpixel SEEDS** a que o pixel
pertence. Esse segundo dado é a **fusão** (seção 7). O TCC não propõe método novo:
**otimiza esse trabalho em quatro pilares** e mede o ganho contra a configuração
dele, com fusão.

**Os quatro pilares** (a lista de atividades do orientador):

1. **Organização do código** — reimplementar de forma auditável e reprodutível.
2. **Paralelismo** — onde há ganho real e onde o teto é estrutural.
3. **Automação de hiperparâmetros (HPO)** — vale um hipertunador ou basta sorteio?
4. **Outros segmentadores** — o SEEDS é a melhor escolha ou só a única testada?

**A receita herdada** (notebook original, commit `3ee559e`, célula 33):

| Parâmetro | Valor |
|---|---|
| otimizador / learning rate | Adam, 0,001 |
| dropout | 0,5 |
| weight decay | 0 |
| batch | **32** |
| épocas | 60, parada antecipada em `val_loss`, patience 5 |
| PCA | 30 componentes em IP; 15 em PU e SA; com *whiten* |
| segmentador | SEEDS (350 superpixels pedidos em IP, 150 em PU/SA) |
| fusão | id escalar |

---

## 2. O caminho de uma execução

Quando você roda `run_experiment(config)`, acontece isto, nesta ordem:

```
ExperimentConfig ──────────────────────────────────────────────────────────┐
                                                                          │
load_scene ── cubo (H,W,B) + GT (H,W)                                     │
   │                                                                      │
to_pixel_table ── tabela (H·W, B) + rótulos (H·W)                         │
   │                                                                      │
   ├─────────────── RAMO ESPECTRAL ───────────┐   (os dois ramos são      │
   │ remove_background → PCA(Kb, whiten)      │    independentes: é aqui  │
   │                                          │    que entra o paralelismo│
   ├─────────────── RAMO ESPACIAL ────────────┤    de dois ramos)         │
   │ PCA(3) na cena inteira → RGB uint8 →     │                           │
   │ segmentador → id da região por pixel →   │                           │
   │ remove_background                        │                           │
   │                                          ▼                           │
   └──────────────► split (pixel | região | bloco) + subamostragem        │
                        │                                                 │
                    PreparedData ── validate_prepared_data (contrato)     │
                        │                                                 │
                    build_loaders ── DataLoaders treino / val / teste     │
                        │                                                 │
                    seed_everything → build_model (CNN1D + fusão, Adam)   │
                        │                                                 │
                    fit ── épocas de treino + validação + early stopping  │
                        │                                                 │
                    evaluate ── OA, AA, Kappa, matriz de confusão (teste) │
                        │                                                 │
                    ExperimentResult ◄────────────────────────────────────┘
```

Cada caixa é uma função de um módulo do pacote. As próximas seções descem em cada
uma.

---

## 3. Mapa do pacote

```
hsi/
├── __init__.py            mapa do pacote (comece por aqui)
├── config.py              caminhos, ExperimentConfig, tabelas por cena
├── ambiente.py            versões, GPU, determinismo, sementes
├── dados/
│   ├── cenas.py           leitura dos .mat, Scene, tabela de pixels, background
│   ├── preprocessamento.py PCA e composição RGB
│   ├── segmentacao.py     SEEDS, SLIC, LSC, Watershed — mesmo contrato de saída
│   └── particoes.py       splits, PreparedData, contrato de dados, DataLoaders
├── paralelo/
│   ├── ramos.py           PILAR 2: ramos espectral e espacial em paralelo
│   └── particao.py        PILAR 2: a cena partida em N pedaços
├── modelo.py              CNN1D + fusão (scalar | embedding)
├── treino.py              laço de treino, histórico, early stopping
├── avaliacao.py           predição, OA, AA, Kappa, matriz de confusão
├── checkpoint.py          salvar e recarregar a CNN1D
├── pipeline.py            run_experiment: da cena às métricas
├── variantes.py           BASELINE (sequencial) × OPTIMIZED (thread)
├── hpo/                   PILAR 3
│   ├── espaco.py          o que se busca (eixos, genes)
│   ├── estrategias.py     como se busca (random, TPE, genético)
│   ├── busca.py           automated_hpo: executa a busca, escolhe por validação
│   ├── selecao.py         retreina o vencedor com várias sementes (campanha)
│   ├── varredura_batch.py escada de batch sobre o vencedor
│   └── graficos.py        figuras do HPO
├── mlops/                 rastreio de runs, registro, campanhas, PDFs, painéis
├── visualizacao/          diagnósticos, explicabilidade, dashboards
└── experimentos/          um módulo por experimento (05…13) + medições avulsas
```

A regra de dependência: módulos de baixo (`config`, `dados`) não importam os de
cima (`pipeline`, `hpo`). Por isso as listas `MODEL_TYPES` e `SEGMENTERS` moram em
`config.py`: `ExperimentConfig` as valida e não pode importar `modelo.py` ou
`segmentacao.py`, que dependem dela.

---

## 4. Configuração e ambiente

### 4.1 `hsi/config.py`

**Caminhos.**

- `repo_root(start=None)` — sobe a árvore de diretórios a partir do **próprio
  pacote** até achar `.git`. Partir do pacote, e não do diretório atual, faz o
  notebook, o lançador e qualquer script acharem os mesmos `data/` e `outputs/`.
- `data_dir()` — onde estão os `.mat`. A variável de ambiente `HSI_DATA_DIR` tem
  precedência (útil na AWS); sem ela, `data/dataHSI/`.
- `output_dir()` — `outputs/` (ignorado pelo git): runs, campanhas, relatórios.

**Tabelas.**

- `DATASETS` — `{"IP": "Indian Pines", ...}`.
- `MODEL_TYPES = ("cnn1d",)` — só uma arquitetura (o escopo foi restrito à CNN 1D;
  as 2D/3D estão no commit `c4665d5`). A tupla continua existindo para que o
  `config.json` de cada run registre a arquitetura explicitamente.
- `SEGMENTERS = ("seeds", "slic", "lsc", "watershed")`.
- `PCA_COMPONENTS = {"IP": 30, "PU": 15, "SA": 15}` — IP precisa de mais
  componentes porque tem classes espectralmente próximas, não por ter mais bandas.

**`ExperimentConfig`** — um `dataclass(frozen=True)`: **todos** os parâmetros de
uma execução num objeto único, imutável e serializável (`to_dict()`). No notebook
original eram variáveis globais espalhadas. Ser *frozen* significa que, para mudar
um campo, você cria uma cópia com `dataclasses.replace(config, epochs=10)` — o HPO
faz isso o tempo todo. Campos, agrupados:

| Grupo | Campos | O que controlam |
|---|---|---|
| Cena e modelo | `dataset`, `model_type`, `pca_components` | qual cena; arquitetura; quantas componentes PCA (0 → tabela por cena) |
| Fusão | `two_input`, `segmenter`, `segmenter_overrides`, `fusion`, `segment_embedding_dim` | se o id da região entra; de qual segmentador; ajustes do segmentador vindos do HPO; como o id entra (`scalar`/`embedding`); tamanho do vetor do embedding |
| Split | `split_mode`, `split_block`, `split_folds`, `split_fold`, `test_ratio`, `valid_ratio`, `train_fraction` | como dividir treino/validação/teste (seção 15) |
| Treino | `epochs`, `batch_size`, `eval_batch_size`, `gradient_accumulation_steps`, `learning_rate`, `weight_decay`, `adam_eps`, `dropout` | o laço de otimização |
| Parada | `patience`, `early_stopping_monitor`, `early_stopping_min_delta`, `early_stopping_target`, `restore_best_weights` | quando parar e com quais pesos ficar |
| Reprodução | `seed`, `deterministic` | sementes e modo determinístico |
| Loader | `num_workers`, `prefetch_factor`, `persistent_workers` | workers do DataLoader (0 por padrão: os dados já estão em RAM) |

`__post_init__` valida tudo (um `ValueError` com a causa, em vez de um erro
obscuro no meio do treino) e preenche `pca_components` pela cena.

Propriedades úteis: `effective_batch_size = batch_size × gradient_accumulation_steps`;
`resolved_eval_batch_size`; `summary()` imprime a configuração legível.

Detalhes que importam:

- `adam_eps = 1e-7` é o padrão do **Keras**, que o original usava (o do PyTorch é
  1e-8). Mantido para comparabilidade.
- O padrão de parada **não** é o do original: aqui é `val_accuracy`, patience 10,
  Δ mínimo 0,001, restaurando a melhor época (política do `Context.md` §2.2). O
  original usava `val_loss` com patience 5.
- `test_ratio=0.15` e `valid_ratio=0.5` → **85% treino, 7,5% validação, 7,5% teste**.
  Essa é a divisão de **todos** os pixels rotulados, e acontece sempre.
- `train_fraction` vem **depois** e encolhe **só o treino**: com `0.10`, o modelo
  treina com 10% daqueles 85% (871 pixels em IP, ~8,5% dos rotulados), amostrados de
  forma estratificada por classe. Validação e teste não mudam — a régua é a mesma.
  O padrão é `1.0`. Quem usa 10% são os experimentos de segmentadores e HPO:

  | Experimentos | Treino | Parada |
  |---|---|---|
  | **05, 06, 08, 10** (segmentadores e HPO) | **10%** | meta `val_accuracy` 0,95, patience 10, teto 200 épocas (08 também 10 e 40) |
  | **09, 11, 12, 13** (vazamento e partição) | 100% | patience 10 |
  | `ExperimentConfig()` e célula 5 do notebook | 100% | patience 10 |

  Portanto os números de HPO (+8,40 pp), da comparação de segmentadores e o 0,8174 do
  SEEDS otimizado são **com 10% de treino**; a queda 0,8996 → 0,5897 sem vazamento é
  com 100%. Nenhum deles usa a parada do original (`val_loss`, patience 5).

### 4.2 `hsi/ambiente.py`

- `check_versions()` — compara o ambiente com `EXPECTED_VERSIONS` (as versões
  validadas) e avisa sobre divergências. Confere duas armadilhas: o build CUDA do
  PyTorch (RTX 50xx precisa de CUDA ≥ 12.8) e se o OpenCV é a variante **contrib**
  (só ela tem `cv.ximgproc`, onde estão SEEDS, SLIC e LSC).
- `select_device()` — escolhe `cuda` se houver, imprime a GPU e **roda uma
  convolução de teste** para provar que os kernels executam.
- `configure_determinism(deterministic=True)` — a troca explícita entre
  reprodutibilidade e velocidade:
  - `cudnn.deterministic=True`, `cudnn.benchmark=False` (o benchmark escolhe o
    kernel mais rápido por shape, e a escolha pode variar entre execuções);
  - **TF32 desligado** (TF32 acelera multiplicações na GPU truncando a mantissa —
    resultado diferente de FP32 puro);
  - `CUBLAS_WORKSPACE_CONFIG=:4096:8` e `use_deterministic_algorithms(True)`;
  - **OpenCV com 1 thread** (`configure_opencv_determinism`): o LSC paralelizado
    devolvia rótulos diferentes em ~390 pixels a cada chamada, o que invalidou uma
    campanha inteira (seção 20). Custo medido: zero.
- `seed_everything(seed)` — semeia NumPy, PyTorch CPU e todas as GPUs. Deve ser
  chamada **imediatamente antes de construir o modelo**, porque é a construção que
  sorteia os pesos iniciais.

---

## 5. Dados: da cena aos tensores

### 5.1 `hsi/dados/cenas.py`

- `SCENE_FILES` — nome do arquivo `.mat` e a chave do dicionário de cada cena.
- `Scene` — dataclass com `code`, `image` (cubo `H×W×B`) e `labels` (GT `H×W`).
  Propriedades: `shape`, `n_bands`, `n_classes` (o maior rótulo, sem contar o 0).
- `load_scene(dataset)` — lê os dois `.mat` com `scipy.io.loadmat`.
- `to_pixel_table(scene)` — **achata** o cubo `(H, W, B)` em uma tabela `(H·W, B)`:
  uma linha por pixel, em ordem *row-major* (linha por linha). Também achata o GT.
  É o formato de "amostras × características" que o aprendizado de máquina usa.
- `remove_background(x, y)` — fica só com as linhas de rótulo > 0 e **reindexa** as
  classes de 1..C para 0..C-1 (o PyTorch espera classes começando em 0).

Como a ordem é row-major, a posição de cada amostra se recupera do índice
achatado: `linha, coluna = divmod(índice, W)`. É assim que `prepare_data_parallel`
monta `coords`.

### 5.2 `hsi/dados/preprocessamento.py`

**PCA (Análise de Componentes Principais).** Encontra novas direções ("componentes")
que são combinações lineares das bandas, ordenadas pela variância que explicam. As
primeiras concentram quase toda a informação (em IP, 2 componentes já explicam
~90%). Reduzir 200 bandas para 30 componentes remove redundância — bandas vizinhas
são altamente correlacionadas — e ruído.

- `apply_pca(x, n_components, whiten=True)` — ajusta e aplica o PCA do scikit-learn.
  **`whiten=True`** divide cada componente pelo seu desvio, deixando **todas com
  variância 1**. Guarde isto: o espectro que chega à rede tem escala ~1, o que
  importa muito na fusão escalar (seção 7).
- `mat2gray(a)` — normaliza para [0, 1] (como o `mat2gray` do MATLAB).
- `rgb_composite(components, shape)` — pega as **3 primeiras componentes**,
  normaliza cada uma para 0..255 e empilha como uma imagem RGB `uint8 (H, W, 3)`.
  **É essa imagem — não o cubo — que os segmentadores recebem.** Os algoritmos de
  superpixel são feitos para imagens de 3 canais.

### 5.3 `hsi/dados/segmentacao.py`

**Superpixel.** Agrupamento de pixels vizinhos e parecidos numa região contígua.
Em vez de 21.025 pixels (IP), a imagem vira ~80 a ~320 regiões homogêneas. O
**id** da região de cada pixel é o que a fusão entrega à rede.

Os quatro algoritmos devolvem o **mesmo contrato**, `SegmentationResult`:

- `labels` — matriz `(H, W)` com o id da região de cada pixel;
- `contour_mask` — matriz `(H, W)` com as bordas entre regiões (para desenhar).

Por isso DataLoader, CNN e métricas não mudam quando o segmentador muda. E por
serem arrays puros (não o objeto do OpenCV, que não é serializável), o resultado
atravessa a fronteira de um processo — requisito do backend `process`.

`SegmentationArtifacts` acrescenta os intermediários visuais (`components`, `rgb`).

Os parâmetros por algoritmo e por cena ficam em dataclasses (`SeedsParams`,
`SlicParams`, `LscParams`, `WatershedParams`) e tabelas (`SEEDS_PARAMS`, ...). IP
usa regiões menores que PU/SA porque tem 145×145 e classes em faixas estreitas.

- `segment_superpixels(rgb, algorithm, params)` — o ponto único de execução. Cria o
  segmentador do OpenCV, itera e devolve `SegmentationResult`.
- `_watershed_segments` — Watershed com uma malha regular de sementes.
- `_boundary_mask` — deriva contornos comparando vizinhos (o Watershed do OpenCV
  não tem `getLabelContourMask`).
- `contour_overlay`, `colorize_segments` — só para visualização.

Como cada algoritmo funciona está na seção 14 (Pilar 4).

### 5.4 `hsi/dados/particoes.py`

**`Splits`** — três vetores de índices: `train`, `val`, `test`. Índices referentes
às linhas de `PreparedData`.

**`PreparedData`** — tudo o que o modelo precisa, já sem background e com PCA:

| Campo | Shape | O que é |
|---|---|---|
| `spectra` | `(N, Kb)` | componentes PCA de cada pixel rotulado (`Kb` = `pca_components`) |
| `labels` | `(N,)` | classe 0..C-1 |
| `segments` | `(N,)` ou `None` | id da região de cada pixel (a segunda entrada) |
| `splits` | `Splits` | índices de treino/val/teste |
| `scene_shape` | `(H, W)` | tamanho da cena |
| `coords` | `(N, 2)` | linha/coluna de cada amostra |

Propriedades: `n_features` (=Kb), `n_classes`, `two_input`, `input_shape()` →
`(1, Kb)`, e **`n_segments` = maior id + 1** — não a contagem de ids distintos,
porque os ids têm buracos depois de remover o background (em IP: 63 regiões
rotuladas, maior id 78). É o tamanho da tabela de embeddings.

**As funções de split** (detalhadas na seção 15):

- `split_indices` — estratificado por pixel, em duas etapas (`train_test_split`
  duas vezes). **Estratificado** = mantém a proporção de cada classe nas três
  partições; essencial com classes desbalanceadas.
- `split_indices_regions` — cada região inteira numa partição só.
- `split_indices_region_folds` — validação cruzada por região (`StratifiedGroupKFold`).
- `split_indices_blocks` — blocos retangulares da cena.
- `_split_by_groups` — o motor comum de região e bloco: sorteia grupos inteiros até
  completar a fração de teste, tenta 64 permutações e fica com a que deixa **mais
  classes no treino**.

**`subsample_train(splits, labels, fraction)`** — reduz só o treino (ex.: 10%),
**estratificado por classe**, garantindo ≥ 1 amostra por classe. Validação e
teste ficam intactos: comparar frações só faz sentido com a mesma régua. Roda
**depois** do split, sobre os 85% de treino (ver a tabela da seção 4.1 sobre quais
experimentos usam 10%). O custo medido de treinar com 10% em IP: −14,3 pp de OA e
−23,0 de AA (`FRACAO_TREINO`); a recomendação é buscar com 10% e retreinar o vencedor
com 100%.

**`segment_leakage(data)`** — mede o vazamento: que fração das amostras de teste
está numa região que também aparece no treino. ~1,0 no split por pixel; 0 no split
por região.

**`validate_prepared_data(data)`** — o **contrato de dados**, aplicado antes de
gastar GPU: arrays alinhados, sem NaN/infinito, classes contíguas 0..C-1, e
**nenhuma amostra em duas partições** (disjunção). Não exige cobertura total, porque
com `train_fraction < 1` sobram pixels fora de propósito.

**`_tensors(data, index)`** — converte uma partição em tensores PyTorch:

- espectro → `(N, 1, Kb)` float — o `1` é o **canal**: a Conv1d trabalha no formato
  *channels-first* `(batch, canais, comprimento)`;
- id → `(N, 1)` float;
- rótulo → `(N,)` int64.

**`build_loaders(data, config, device)`** — três `DataLoader`s:

- treino com `shuffle=True` e um `generator` semeado com `config.seed`: a ordem dos
  batches depende só da semente, não do estado global;
- validação e teste sem embaralhar, com `eval_batch_size`;
- `pin_memory=True` em CUDA, para a cópia CPU→GPU ser assíncrona.

---

## 6. O modelo: a CNN1D

Arquivo: `hsi/modelo.py`.

### 6.1 A ideia

Uma **CNN 1D** desliza filtros ao longo de **uma** dimensão — aqui, a sequência de
componentes espectrais de **um** pixel. Ela aprende padrões locais no espectro
("a componente 5 sobe enquanto a 6 desce"). Ela **não vê vizinhança espacial
nenhuma**: cada pixel é classificado isoladamente. Por isso a segunda entrada — a
região do pixel — tem efeito tão grande.

### 6.2 A arquitetura, camada por camada (Indian Pines, Kb = 30)

| Camada | Operação | Saída `(canais, comprimento)` | Parâmetros |
|---|---|---|---|
| entrada | espectro PCA | `(1, 30)` | — |
| Conv1d(1→16, k=9) + ReLU | 16 filtros de largura 9 | `(16, 22)` | 160 |
| Conv1d(16→32, k=3) + ReLU | | `(32, 20)` | 1.568 |
| Conv1d(32→64, k=3) + ReLU | | `(64, 18)` | 6.208 |
| Conv1d(64→64, k=3) + ReLU | | `(64, 16)` | 12.352 |
| Flatten | achata | `1.024` | — |
| **fusão** | concatena o id | `1.025` (scalar) ou `1.032` (embedding 8) | 0 ou 632 |
| Linear(→128) + ReLU + Dropout | | `128` | 131.328 |
| Linear(→64) + ReLU + Dropout | | `64` | 8.256 |
| Linear(→16) | **logits** | `16` | 1.040 |

Total: **160.912** parâmetros (fusão escalar, IP). Conferido executando o modelo.

Por que o comprimento encolhe: sem *padding*, uma convolução de kernel `k` reduz o
comprimento em `k − 1`. As quatro consomem `8 + 2 + 2 + 2 = 14` posições. Daí:

- `CNN1D_KERNELS = (9, 3, 3, 3)`;
- `CNN1D_MIN_COMPONENTS = 15` — com menos de 15 componentes a saída fica vazia. A
  CNN1D levanta um erro dizendo isso (antes, o PyTorch reclamava da convolução e
  escondia a causa — foi um bug real do HPO, seção 20);
- em PU/SA (Kb = 15) sobra **1** posição: o Flatten dá só 64 características, e o
  modelo tem 37.577 parâmetros.

`n_flat` é medido com uma **passagem a seco**: um tensor de zeros passa pelas
convoluções e o tamanho da saída é lido.

### 6.3 As peças

- `apply_xavier_init` — inicialização **Xavier/Glorot uniforme** com viés zero, o
  padrão do Keras usado no original (o padrão do PyTorch é outro e mudaria o ponto
  de partida).
- `_FusionClassifier` — a **cabeça densa** com a fusão (seção 7). É um *mixin*
  (classe auxiliar herdada junto com `nn.Module`), resto da época em que as CNNs 2D
  e 3D também o usavam.
- `CNN1D(nn.Module, _FusionClassifier)` — `features` (as 4 convoluções + Flatten) e
  `forward(x, seg)` = `classify(features(x), seg)`.
- `n_parameters` — conta os pesos treináveis.
- `build_model(config, input_shape, classes, device, n_segments)` — monta a
  CNN1D, a **loss** (`CrossEntropyLoss`) e o **otimizador** (`Adam` com `lr`,
  `eps` e `weight_decay` da config).

### 6.4 Termos do modelo

- **ReLU** — `max(0, x)`: a não-linearidade entre camadas.
- **Dropout(p)** — durante o treino, zera aleatoriamente uma fração `p` das
  ativações. Força redundância e reduz *overfitting*. Desligado em `eval()`.
  **p = 0,5 é o valor herdado e o pior medido** (seção 13).
- **Logits** — saídas cruas da última camada, uma por classe. O softmax as
  converte em probabilidades.
- **CrossEntropyLoss** — aplica softmax e mede o −log da probabilidade da classe
  correta. Recebe logits; por isso o modelo não tem softmax no fim.
- **Adam** — otimizador que adapta o passo de cada peso pela média móvel dos
  gradientes e dos seus quadrados. `lr` é o tamanho do passo; `weight_decay` é a
  penalização L2 (puxa os pesos para zero).

---

## 7. A fusão — o coração da arquitetura herdada

### 7.1 O que é

**Fusão** é juntar duas fontes de informação numa só decisão. Aqui:

- fonte 1: o **espectro** do pixel (o que ele é, espectralmente);
- fonte 2: o **id do superpixel** a que ele pertence (onde ele está, e com quem
  se agrupa).

A fusão é **tardia**: as convoluções nunca veem o id. Ele é concatenado ao vetor de
1.024 características **depois do Flatten**, e só a cabeça densa enxerga as duas
coisas juntas. Como as duas entradas são calculadas separadamente até ali, os ramos
espectral e espacial do pré-processamento são independentes — e paralelizáveis
(Pilar 2).

### 7.2 Onde está no código

1. **Produção do id** — `spatial_branch` (`paralelo/ramos.py`): PCA de 3
   componentes → `rgb_composite` → `segment_superpixels` → `remove_background`.
   Resultado: um id por pixel rotulado.
2. **Transporte** — `PreparedData.segments`; `_tensors` o converte em `(N, 1)` float;
   `unpack_batch` (`treino.py`) separa `(x, seg, y)`.
3. **Consumo** — `_FusionClassifier.classify(features, seg)` (`modelo.py`):

```python
if self.two_input:
    if self.segment_embedding is not None:          # fusão "embedding"
        seg = self.segment_embedding(seg.reshape(-1).long())   # id → vetor de 8
    features = torch.cat([features, seg], dim=1)    # concatena às 1.024 colunas
return self.classifier(features)
```

Com `two_input=False` o id não entra e o modelo é só espectral.

### 7.3 Os dois modos

**`scalar` (herdado).** O id cru vira **uma coluna a mais**: o vetor fica com 1.025
números, e o último é, por exemplo, `78.0`. Problemas:

- trata um **rótulo nominal** como número contínuo — a rede "vê" a região 78 como
  "78 vezes" a região 1, e regiões de ids próximos como parecidas;
- **escala**: o espectro *whitened* tem desvio ~1; o id vai até 78 em IP (até
  ~1.428 com SLIC em PU). Medido: a coluna do id gera **1,79× mais variação** na
  primeira camada densa do que as 1.024 colunas do espectro somadas. É por isso que
  segmentadores com muitas regiões exigem learning rate e dropout diferentes.

**`embedding`.** Uma `nn.Embedding(n_segments, 8)` é uma **tabela aprendida**: uma
linha de 8 números por região, ajustada pelo gradiente como qualquer peso. O id
vira um índice nessa tabela. Sem ordem falsa entre ids, sem depender da magnitude, e
a inicialização N(0, 1) deixa o vetor na mesma escala do espectro.

| Cena | scalar (herdado) | embedding |
|---|---|---|
| IP | 0,8981 | **0,9471 (+4,90 pp)** |
| PU | 0,9785 | **0,9868 (+0,83 pp)** |

(split por pixel, 3 sementes; `MODOS_FUSAO` em `medicoes.py`)

### 7.4 Por que a fusão funciona — a ablação

IP, split por pixel, 3 sementes (`ABLACAO_FUSAO`). Cada linha tira uma propriedade
do id:

| Manipulação do id | O que preserva | OA |
|---|---|---|
| original | tudo | 0,8981 |
| permutado (ids embaralhados entre regiões) | agrupamento e escala; perde a ordem | 0,8552 |
| normalizado para [0,1] | tudo menos a magnitude | 0,8223 |
| aleatório por pixel | só a escala; perde o agrupamento | 0,7898 |

Leituras:

1. **A informação é real**: ruído com a mesma escala é a pior linha (−10,83 pp).
2. **Parte do valor é posicional**: os ids crescem na varredura raster da imagem, então
   a numeração funciona como uma **coordenada grosseira**. Embaralhar os ids custa
   4,29 pp.
3. **A magnitude bruta importa**: normalizar mantém toda a informação e custa 7,58 pp
   — a rede se apoiou na escala exagerada.

### 7.5 O lado perigoso da fusão

O id carrega "região R". Se a mesma região tem pixels no treino e no teste — e no
split por pixel **100%** do teste compartilha região com o treino —, a rede pode
aprender a regra "região R → classe C" e aplicá-la ao teste. Isso não é
classificação espectral, é **memória de vizinhança**. Remover esse vazamento
derruba a configuração herdada de 0,8996 para 0,5897 (seção 15).

E o que prevê se um segmentador ajuda ou atrapalha não é a pureza das regiões, é
**quantas amostras de treino caem em cada região**: o SEEDS dá ~10,8 em IP; SLIC e
LSC, ~2,7. Com 2,7 pixels por região a rede não consegue estimar "região → classe",
e o id vira ruído de alta magnitude (o mesmo mecanismo da linha "aleatório por
pixel" da ablação).

**Premissa do trabalho:** a fusão é parte da arquitetura herdada e **não é variável
de estudo**. Não se compara "com × sem fusão"; todo ganho é medido contra a
configuração do Sydney, com fusão SEEDS. Entender a fusão é necessário para
interpretar os pilares 3 e 4 e o vazamento — não para propor tirá-la.

---

## 8. Treino e parada antecipada

Arquivo: `hsi/treino.py`.

### 8.1 Conceitos

- **Época** — uma passada completa pelo conjunto de treino.
- **Batch** — um lote de amostras processadas juntas. O gradiente é a média sobre o
  lote; um passo do otimizador por lote.
- **Forward / backward** — o *forward* calcula a saída e a loss; o *backward*
  (autograd) calcula o gradiente da loss em relação a cada peso.
- **`model.train()` / `model.eval()`** — liga/desliga o comportamento de treino
  (dropout).
- **Validação** — dados fora do treino usados para **decidir** (parar, escolher
  hiperparâmetros). **Teste** — dados usados **só para relatar** o resultado final.
  Escolher pelo teste é vazar o teste para a seleção.

### 8.2 As funções

- `unpack_batch(batch, device)` — transforma o batch em `(x, seg, y)` no device;
  `seg` é `None` sem fusão. `non_blocking=True` aproveita o `pin_memory`.
- `run_epoch(model, loader, criterion, device, optimizer=None, accum=1)` — uma
  passada:
  - com `optimizer`: treino (forward → loss → backward → `optimizer.step()`);
  - sem `optimizer`: só avaliação, com gradientes desligados;
  - devolve `(loss média, acurácia)`.
  - **Acumulação de gradiente** (`gradient_accumulation_steps > 1`): soma os
    gradientes de vários batches antes de um passo, simulando um batch maior sem
    precisar de mais memória. O código normaliza pelo número real de amostras da
    janela, para que a última janela incompleta continue correta. Com 1, o caminho
    numérico é exatamente o do baseline.
- `EarlyStopping` — a regra de parada:
  - **melhorou** se `val_accuracy > melhor + min_delta` (ou `val_loss < melhor −
    min_delta`). Melhorar zera a contagem e, se `restore_best_weights`, guarda uma
    **cópia profunda** dos pesos (sem a cópia, o "melhor estado" mudaria junto com o
    treino, porque `state_dict()` devolve referências);
  - **não melhorou** soma 1 em `wait`;
  - para quando `wait ≥ patience` **ou** quando a meta (`early_stopping_target`,
    ex.: 0,95) é atingida; ao parar, restaura os melhores pesos.
- `History` — as quatro séries por época com os nomes do Keras: `loss`,
  `accuracy`, `val_loss`, `val_accuracy`.
- `TrainResult` — histórico, tempos por época, se parou cedo, melhor época, motivo.
- `fit(model, loaders, criterion, optimizer, config, device)` — o laço:
  1. repõe a semente do gerador do loader de treino (inspecionar um batch antes
     adiantaria o gerador e mudaria o resultado em silêncio — bug real);
  2. para cada época: treino → validação → `cuda.synchronize()` (para o tempo
     medido ser real, a GPU é assíncrona) → registra → consulta o `EarlyStopping`.

**Atenção:** se o treino chega ao limite de épocas **sem** disparar a parada, o
modelo fica com os pesos da **última** época, mesmo com `restore_best_weights=True`
(seção 20).

---

## 9. Avaliação: OA, AA e Kappa

Arquivo: `hsi/avaliacao.py`.

**Matriz de confusão** `M` (C × C): `M[i, j]` = quantas amostras da classe real `i`
foram previstas como `j`. A diagonal são os acertos.

| Métrica | Fórmula | O que mede |
|---|---|---|
| **OA** (overall accuracy) | `traço(M) / soma(M)` | fração de pixels certos; cada **pixel** pesa igual |
| **AA** (average accuracy) | média de `M[i,i] / soma da linha i` | média da acurácia **por classe**; cada **classe** pesa igual |
| **Kappa** de Cohen | `(p_o − p_e) / (1 − p_e)` | acerto acima do esperado por acaso; `p_o` = OA, `p_e` = Σ (total previsto da classe × total real da classe) / N² |

Por que três: com classes desbalanceadas, um modelo que acerta só as classes
grandes tem OA alta e AA baixa. A AA pune quem ignora as classes raras. O Kappa
desconta o acerto que um chute proporcional às classes já teria.

Funções:

- `Metrics` — dataclass com OA, AA, Kappa, acurácia por classe, matriz, loss e
  relatório do scikit-learn.
- `predict_proba(model, loader, device)` — probabilidades (softmax explícito).
- `_average_accuracy(confusion)` — acurácia por classe e AA; classe sem suporte
  conta 0 sem dividir por zero.
- `evaluate(model, loader, criterion, device, y_true)` — previsões, matriz, métricas,
  e uma segunda passada para a loss de teste.
- `binary_readout` — reprodução de um cálculo **inválido** do original (fórmula
  binária aplicada ao canto 2×2 de uma matriz multiclasse). Mantido só para
  comparação; não use.

---

## 10. O pipeline que amarra tudo

Arquivo: `hsi/pipeline.py`. É **a única peça que conhece a ordem das etapas**.
Notebook, HPO e experimentos chamam estas funções, não cópias delas.

- `prepare_data(config, backend)` — atalho para `prepare_data_parallel` sem os
  tempos.
- `run_experiment(config, backend="sequential")`:
  1. `select_device` e `configure_determinism`;
  2. `prepare_data_parallel` → `PreparedData` + tempos dos ramos;
  3. `validate_prepared_data` (o contrato);
  4. `build_loaders`;
  5. `seed_everything` **imediatamente antes** de `build_model`;
  6. `fit`;
  7. `evaluate` no teste;
  8. devolve `ExperimentResult` (config, dados, modelo, treino, métricas, previsões,
     device, tempos, contrato).
- `result_footprint(result)` — quantos MB um resultado segura.
- `release_result(result)` — solta os arrays pesados (`spectra`, `segments`,
  `coords`, o modelo e as previsões) e chama o coletor de lixo. Sobrevive o que os
  relatórios usam: config, métricas, histórico, splits. É a técnica do orientador:
  **instanciar → treinar → extrair métricas → liberar** (seção 11).

`checkpoint.py`:

- `save_checkpoint` — grava o `state_dict` (os pesos) **e** o que define os shapes
  (Kb, classes, fusão, `n_segments`, dimensão do embedding). O PyTorch salva só
  pesos; a arquitetura precisa ser remontada antes de carregá-los.
- `load_checkpoint` — remonta a CNN1D, carrega os pesos, devolve em `eval()`.
  Aceita checkpoints antigos (sem as chaves de fusão).

`variantes.py`:

- `Variant`, `BASELINE` (backend `sequential`) e `OPTIMIZED` (backend `thread`) —
  **mesmos hiperparâmetros**, só muda como a preparação executa. Assim qualquer
  diferença de tempo vem da execução, e as métricas **têm** de sair idênticas — se
  não saírem, é bug, não ganho.
- `run_variant`, `VariantRun.speedup_over`, `VariantRun.matches`.

---

## 11. Pilar 1 — Organização do código

**O que foi feito:**

1. **Reimplementação em PyTorch** do notebook Keras original, reproduzindo-o **bit a
   bit**: mesmo histórico de perda por época, mesmos pesos finais, mesmas métricas
   (OA 0,976309 / AA 0,957918 / Kappa 0,968545 em PaviaU). Isso é **pré-requisito**:
   sem equivalência demonstrada, nenhuma comparação posterior significa nada.
2. **Configuração explícita** (`ExperimentConfig`) no lugar de globais soltas, e
   funções que recebem o que usam (nada de ler variável global escondida).
3. **Um módulo por responsabilidade** no pacote `hsi/`, cada nome definido uma vez
   só. O notebook importa daqui; a extração foi conferida bit a bit (mesmo treino
   pelo notebook antigo e pelo pacote → mesmas previsões e métricas).
4. **Determinismo** (seção 4.2) e **contrato de dados** (seção 5.4).
5. **Memória** (técnica do orientador): cada trial de HPO segurava sua cópia dos
   dados (2,5 MB em IP, 6,5 MB em SA). `release_result` +
   `automated_hpo(retain="winner")` mantêm só o campeão corrente. Medido numa
   campanha de 180 trials: a memória cresceu **16 MB** do início ao fim.
   *Dimensão honesta*: na CNN 1D o problema é pequeno; a técnica vale como garantia
   e boa prática, não como o que destravou o trabalho.
6. **Rastreabilidade** (MLOps, seção 16) e **retomada** dos experimentos longos
   (o CSV consolidado é a fonte de verdade; seção 18).

Pequenas correções de engenharia que fazem parte deste pilar: `repo_root`
independente do diretório atual; DataLoader com gerador próprio; `fit` repondo a
semente; checkpoint que remonta a arquitetura; erros com a causa real.

---

## 12. Pilar 2 — Paralelismo

### 12.1 Conceitos que você precisa dominar

**Concorrência × paralelismo.** Concorrência é ter várias tarefas em andamento;
paralelismo é executá-las **ao mesmo tempo** em núcleos diferentes.

**Thread** — fluxo de execução dentro de um processo. Threads compartilham a mesma
memória: criar é barato, não há cópia.

**Processo** — programa com memória própria. Isolado: para passar dados, é preciso
**serializar** (converter em bytes) e copiar.

**GIL (Global Interpreter Lock)** — no CPython, uma trava garante que **só uma
thread executa bytecode Python por vez**. Consequências:

- código Python puro em threads **não** roda em paralelo (medido: laço Python, 1,01 s
  sequencial → 0,91 s com 4 threads);
- código **nativo** (C/C++/Fortran) que **solta o GIL** roda em paralelo
  (multiplicação de matrizes: 0,80 s → 0,28 s). NumPy/BLAS, LAPACK (PCA) e OpenCV
  soltam o GIL durante o cálculo pesado.

**`ThreadPoolExecutor` / `ProcessPoolExecutor`** — pools do `concurrent.futures`.
Você submete tarefas (`submit`, `map`) e recebe *futures* (resultados prometidos).

**`pickle`** — o serializador do Python. Para mandar uma função a outro processo, o
pickle a envia **pelo nome** (`módulo.função`). Por isso as funções paralelizadas são
**funções de módulo**, nunca *closures* ou lambdas.

**`fork` × `spawn`** — como o processo filho nasce:

- `fork` (padrão no Linux): cópia do pai, herda memória e configuração. Rápido. O
  PyTorch **recusa** fork depois que o autograd rodou em threads;
- `spawn`: interpretador **limpo**, que reimporta os módulos. Lento para subir
  (importar PyTorch custa segundos) e não herda configurações — por isso o
  `initializer=_preparar_worker` refaz o que cada worker precisa (1 thread do
  PyTorch, OpenCV determinístico).

**Lei de Amdahl** — o ganho é limitado pela parte que não paraleliza. Se uma fração
`p` do tempo é paralelizável em `N` partes, o speedup máximo é
`1 / ((1 − p) + p/N)`. Corolários usados no trabalho: (a) paralelizar algo que é
1% do tempo muda no máximo 1%; (b) dois ramos desbalanceados têm teto
`(T₁ + T₂) / max(T₁, T₂)`.

**Speedup** = `tempo de referência / tempo paralelo`. **Overhead** = custo de
coordenar (criar pool, serializar, sincronizar). **Contenção** = tarefas paralelas
disputando o mesmo recurso (banda de memória, cache, GIL).

### 12.2 Paralelismo entre ramos — `hsi/paralelo/ramos.py`

Depois de carregar a cena, os ramos são independentes até a fusão:

- `spectral_branch(spectra, Kb, seed)` — PCA espectral dos pixels válidos;
- `spatial_branch(pixels, raw_labels, shape, params, seed, segmenter)` — PCA(3) da
  cena inteira (o segmentador precisa da imagem completa, com background) → RGB →
  segmentação → filtro de background.

`prepare_data_parallel(config, backend)` executa os dois conforme o backend
(`BACKENDS = ("sequential", "thread", "process")`):

| backend | como roda | custo |
|---|---|---|
| `sequential` | um depois do outro | linha de base |
| `thread` | 2 threads; compartilham os arrays (sem cópia) | funciona porque PCA e OpenCV soltam o GIL |
| `process` | ramo espacial num processo filho; espectral no pai | paga pool + serialização (8,4 MB IP; 42,7 MB PU; 45,3 MB SA) |

Depois dos ramos, faz o split conforme `split_mode`, a subamostragem do treino e
monta `PreparedData`. Detalhe: com `split_mode="region"` o ramo espacial roda
mesmo sem fusão, porque os grupos do split vêm da segmentação — `region_ids` (para o
split) e `segments` (entrada do modelo) são separados por isso.

Outras peças: `StageTimings` (tempo de cada ramo, `wall`, `branch_sum`, `overhead`,
`speedup_over`), `_segmenter_params` (parâmetros da cena + `segmenter_overrides` do
HPO), `_timed`.

**Resultados** (5 repetições):

| Cena | sequencial | thread | process | espectral | espacial | teto |
|---|---|---|---|---|---|---|
| IP | 0,166 s | 1,05× | 0,22× | 0,037 s | 0,129 s (78%) | 1,29× |
| PU | 0,971 s | 0,95× | 0,75× | 0,072 s | 0,899 s (93%) | 1,08× |
| SA | 0,822 s | **1,22×** | 0,74× | 0,142 s | 0,680 s (83%) | 1,21× |

Leitura: a segmentação domina (78–93%), então o teto é baixo. Em Salinas a
implementação entrega 1,22× contra teto de 1,21× — **no limite do desenho**. O
`process` sempre perde (overhead). E o número que encerra o assunto: o
pré-processamento é **0,9% do tempo total** (0,193 s contra 20,13 s de treino).
Por isso o esforço migrou para a partição da cena.

### 12.3 Paralelismo por partição da cena — `hsi/paralelo/particao.py`

A ideia: dividir a cena em **N partes de tamanho parecido** e processá-las ao mesmo
tempo. Partes equilibradas → teto ≈ N (em vez de 1,2×), e o que fica paralelo é o
**treino**, onde está o tempo.

**A segmentação é feita uma vez, na cena inteira, antes de particionar.** Segmentar
cada parte separadamente criaria superpixels diferentes nas fronteiras, e os ids
deixariam de ser comparáveis.

Funções:

- `particionar_cena(coords, shape, N, modo)`:
  - `"faixas"` — faixas horizontais contíguas, com fronteiras escolhidas para
    equilibrar o **número de pixels rotulados** (é ele que determina o tempo de
    treino);
  - `"aleatoria"` — embaralha sem olhar a posição. **É o controle**: mesmo custo
    computacional, sem contiguidade. Se um ganho de acurácia some aqui, vinha da
    posição.
- `_fatiar(data, máscara)` — recorta `PreparedData` para uma partição e **reindexa**
  os splits (os índices mudam após o recorte).
- `_treinar_e_inferir(tarefa)` — o trabalho de um worker: treina (se preciso) e
  infere uma partição, em CPU. Função de módulo, para ser serializável.
- `_preparar_worker()` — `initializer` de cada worker: `torch.set_num_threads(1)` e
  OpenCV determinístico. Sem ele, com `spawn`, cada worker subiria com 16 threads, e
  N partições virariam N×16 threads disputando os núcleos.
- `_aquecer_worker` — tarefa de aquecimento (sobe o worker, inicializa o autograd e o
  alocador) sem tocar nos dados.
- `PoolReutilizavel` — pool criado **uma vez**, aquecido (`aquecer()`), reusado
  (`mapear()`) e fechado (`fechar()` ou `with`). Guarda `tempo_criacao` e
  `tempo_aquecimento` à parte, para que a partida continue visível.
- `ResultadoParticionado` — tempos (`tempo_paralelo`, `tempos_por_particao`,
  `soma_das_partes`, `overhead`, `teto_de_amdahl`) e métricas.
- `run_experiment_particionado(config, N, modo, particao_modo, executor, pool)`.

**Dois modos:**

- `replicado` — treina **um** modelo com tudo e paraleliza só a inferência. OA
  **idêntica** ao monolítico (0,882965), mas a inferência é 0,4% da execução: não há o
  que ganhar. Amdahl de novo.
- `independente` — **um modelo por partição**, cada um com 1/N das amostras.
  Paraleliza o treino; o custo é cada modelo ver menos dados.

**Três executores:**

- `sequencial` — as partições uma a uma (o denominador honesto do speedup);
- `thread` — **não escala**: treinar uma CNN pequena em CPU é dominado por código
  Python, que disputa o GIL. E as threads disputam o **RNG global** do PyTorch, então
  a OA deixa de se repetir (3,51 pp de amplitude entre execuções idênticas);
- `process` (`spawn`) — escala e repete a OA exatamente: cada worker tem seu
  interpretador e seu gerador.

### 12.4 Pool frio × pool quente (experimento 13)

O primeiro desenho criava e destruía o pool **dentro** do cronômetro: cada medição
pagava N interpretadores subindo do zero. Isso é o pior caso, não o uso real (o
pool sobe uma vez e atende muitas execuções). Refeito medindo os dois.

Speedup contra o `sequencial`, IP, 25 épocas, `torch threads = 1`:

| Executor | N=2 | N=4 | N=8 | N=16 |
|---|---|---|---|---|
| thread | 1,61× | 0,61× | 0,23× | 0,23× |
| process, pool **frio** | 1,20× | 1,46× | 1,37× | 1,05× |
| **process, pool quente** | **1,86×** | **2,87×** | **3,68×** | **4,47×** |

Em N=16 o pool frio diz "não ganha nada" (1,05×); o quente entrega 4,47×. **Medir só
o pior caso não deixa o resultado conservador — deixa errado.** A partida chega a 78%
do tempo do bloco frio em N=16.

Por que não chega ao teto ≈ N: **contenção**. Uma partição que sozinha leva 0,80 s
leva 2,65 s com 16 rodando juntas (3,30× mais lenta) — disputa de banda de memória e
cache. **O desenho escala; a máquina não.** No `thread`, a contenção chega a 137×: o
GIL serializa tudo.

**Limite de memória:** N=32 com `process` não cabe. Cada worker `spawn` ocupa ~481 MB
(PSS); 32 × 481 MB ≈ 15,4 GB numa máquina de 15,8 GB. O paralelismo por processos
esbarra em **memória** antes de esbarrar em núcleos. E há um limite por acurácia: a
OA cai de 0,9155 (N=4) para 0,8505 (N=32), porque cada partição fica com ~272
amostras de treino.

### 12.5 O ganho de acurácia das faixas era ilusão

Particionar em faixas parecia **melhorar** a OA (+2 a +3 pp). Dois controles:

1. **Partição aleatória** (mesmo custo, sem contiguidade): a vantagem das faixas sobre
   ela chega a +31,51 pp em N=16 → o ganho é **posicional**.
2. **O mesmo arranjo sob split por região** (sem vazamento): o ganho **vira perda**
   (−5,07 pp em N=4). Era interpolação entre vizinhos, o mesmo vazamento da seção 15.

**Conclusão do pilar:** particionar a cena é uma decisão de **desempenho**, com custo
de acurácia a declarar — não uma técnica que melhora a classificação.

---

## 13. Pilar 3 — Automação de hiperparâmetros

### 13.1 Conceitos

**Hiperparâmetro** — o que você escolhe antes de treinar (learning rate, dropout,
batch...). **Parâmetro** — o que o treino aprende (pesos). **HPO** (*hyperparameter
optimization*) — procurar automaticamente a melhor combinação de hiperparâmetros.

**Trial** — um treino com uma combinação. **Espaço de busca** — o conjunto de
valores permitidos. **Orçamento** — quantos trials, com quantas épocas cada.

**Seleção só por validação.** O HPO escolhe pela melhor `val_accuracy` do histórico.
OA/AA/Kappa de teste aparecem nas tabelas como **relato**, nunca como critério.
Escolher pelo teste o contaminaria.

**Busca de baixa fidelidade.** Os trials rodam com poucas épocas (ex.: 10, com
patience encolhida para `max(2, epochs // 3)`), e o vencedor é retreinado com o
orçamento completo. O risco é o *low-fidelity bias*: o ranking com 10 épocas pode não
ser o ranking com 60.

### 13.2 O espaço — `hsi/hpo/espaco.py`

```python
HPO_SPACE = {
    "learning_rate": (3e-4, 1e-3, 3e-3),
    "dropout":       (0.2, 0.35, 0.5),
    "weight_decay":  (0.0, 1e-5, 1e-4),
    "batch_size":    (32, 64, 128),
}
```

Mais três eixos:

- `pca_components` — `_pca_options`: 0,5×, 1× e 1,5× a receita da cena, com piso
  `CNN1D_MIN_COMPONENTS = 15`. IP → (15, 30, 45); PU/SA → (15, 22).
- `segmenter` — os quatro.
- `override_index` — qual de três receitas do segmentador
  (`segmenter_hpo_overrides`): a do meio é a padrão; as outras, abaixo e acima no
  parâmetro de escala (nº de superpixels no SEEDS, `region_size` no SLIC/LSC,
  marcadores no Watershed).

Total em IP: 4 segmentadores × 3 receitas × 3⁴ eixos × 3 PCA = **2.916 combinações**.
Treinar todas é inviável; a busca é **amostrada**.

**Genes.** `hpo_genes` representa o espaço como uma lista ordenada de genes
`(nome, valores)`; um candidato é uma **tupla de índices**, um por gene. Exemplo:
`(0, 1, 1, 2, 0, 0, 1)` = (seeds, receita padrão, lr 1e-3, dropout 0,5, wd 0, batch
32, PCA 30). Essa representação é o denominador comum das três estratégias.
`hpo_decode` traduz a tupla em campos da `ExperimentConfig`. O receituário herdado
é um dos pontos do espaço.

### 13.3 As três estratégias — `hsi/hpo/estrategias.py`

Todas têm a interface **`ask()`** (próximo candidato) e **`tell(candidato, score)`**
(resultado). E todas começam com a mesma **cobertura**: os primeiros 4 trials são
forçados a segmentadores distintos, para nenhum ficar de fora por azar.

- **`_RandomSearcher` (random search)** — sorteio uniforme de cada gene, sem repetir
  candidatos. **Sem memória**: `tell` não faz nada. É a linha de base.
- **`_TPESearcher` (Tree-structured Parzen Estimator, via Optuna)** — um
  hipertunador **bayesiano**. Separa os trials observados em "bons" e "ruins",
  modela a densidade de cada grupo sobre o espaço (`l(x)` e `g(x)`) e propõe o ponto
  que maximiza `l(x) / g(x)` — onde os bons são mais prováveis que os ruins. Cada
  sugestão depende de tudo o que já foi visto. Segmentador e receita são
  **categóricos** (sem ordem); os outros eixos são inteiros **ordenados**. A fase
  aleatória inicial é a própria cobertura (o padrão do Optuna, 10 trials
  aleatórios, transformaria um HPO de 8 trials em random search puro).
- **`_GeneticSearcher` (algoritmo genético)** — evolução de uma população:
  - a 1ª geração é a cobertura + sorteios (coincide com o random search);
  - **seleção por torneio**: sorteia 3, vence o de maior validação;
  - **crossover uniforme**: cada gene do filho vem de um dos pais com chance 1/2;
  - **mutação**: cada gene é resorteado com probabilidade 0,25 (evita convergir cedo
    demais);
  - **elitismo**: os 2 melhores passam à geração seguinte **sem retreinar**.

`make_hpo_searcher(strategy, ...)` é o único ponto de escolha.

### 13.4 A busca — `hsi/hpo/busca.py`

- `automated_hpo(base_config, trials_per_cnn, search_epochs, strategy, retain)` —
  para cada trial: `ask` → `hpo_decode` → `replace` na config → encolhe épocas e
  patience → `run_experiment` → score = maior `val_accuracy` → `tell` → registra uma
  linha na tabela (com a `config` do trial, que reconstrói qualquer execução) →
  libera o resultado se não for o melhor (`retain="winner"`). Devolve `(tabela,
  vencedores)`.
- `hpo_candidates` — pré-gera os candidatos do random search sem treinar (para
  campanhas que precisam do plano inteiro, e do hash dele, antes do primeiro treino).
  Só existe para o random: TPE e genético são adaptativos.
- `compare_hpo_strategies` e `hpo_strategy_summary` — as três estratégias com
  orçamento idêntico.

### 13.5 Seleção, campanha final e batch

- `select_hpo_winners` (`hpo/selecao.py`) — melhor trial por CNN: maior validação;
  desempate por menor tempo e menor número de trial.
- `run_best_hpo_campaign(hpo_table, final_epochs, seeds)` — **retreina só o vencedor**,
  com várias sementes e orçamento completo, e congela a decisão em arquivos
  (`hpo_screening.json`, `selection.json`, `final_plan.json` com hash SHA-256). Assim
  é possível responder *por que* uma receita foi promovida. Média e intervalo entre
  sementes — não um run isolado — são o resultado científico.
- `run_audited_batch_sweep` (`hpo/varredura_batch.py`) — outra pergunta: fixada a
  receita vencedora, qual batch dá mais **throughput** (amostras/s) sem perder mais
  que 0,005 de validação? Escada uniforme 50, 100, …, 300.

### 13.6 Resultados

**Estratégias** (IP, 10% de treino, 3 sementes, 540 trials):

| Orçamento | genetic | random | tpe | amplitude | desvio típico |
|---|---|---|---|---|---|
| 10 épocas | 0,7334 | 0,7174 | 0,7226 | 1,60 pp | 1,58 pp |
| 40 épocas | 0,8058 | 0,7802 | 0,7924 | 2,56 pp | 1,89 pp |
| 200 épocas | 0,8288 | 0,8067 | 0,8266 | 2,21 pp | 2,73 pp |

**Nenhuma estratégia se separa do ruído entre sementes.** O que importa é o
**orçamento de épocas**.

**Quais eixos importam** (validação média por valor):

| Eixo | amplitude | melhor valor |
|---|---|---|
| dropout | 18,10 pp | 0,20 |
| learning rate | 15,26 pp | 3e-3 |
| PCA | 8,35 pp | 30 |
| weight decay | 4,55 pp | 0 |
| batch | 1,81 pp | 128 |

O dropout 0,5 herdado é o **pior** valor do eixo mais influente: com poucos dados,
regularizar demais é o pior erro.

**Ganho sobre a receita herdada** (experimento 10, IP, 5 sementes, 10% de treino):

| Condição | herdada | otimizada | ganho |
|---|---|---|---|
| **seeds** | 0,7334 | **0,8174** | **+8,40 pp** (único afirmável) |
| lsc | 0,6853 | 0,7566 | +7,13 pp (ruído) |
| slic | 0,6003 | 0,6411 | +4,08 pp (ruído) |
| watershed | 0,6583 | 0,5646 | −9,36 pp (ruído) |

**+8,40 pp é o ganho efetivo do pilar sobre o trabalho de referência** — com duas
ressalvas: o braço herdado usou batch 128 (o original é 32; precisa ser refeito) e foi
medido sob split por pixel, onde a posição sozinha dá 0,97 (seção 15).

---

## 14. Pilar 4 — Segmentadores

### 14.1 Os quatro algoritmos

Todos recebem a imagem RGB das 3 primeiras componentes PCA.

- **SEEDS** (*Superpixels Extracted via Energy-Driven Sampling*) — **o herdado**.
  Começa com uma grade de blocos e refina por *hill-climbing*: move blocos e depois
  pixels entre superpixels vizinhos quando isso aumenta a homogeneidade dos
  histogramas de cor. Trabalha em vários níveis de bloco (`num_levels`). Parâmetros:
  `num_superpixels` (**alvo, não garantia** — pede 350 e entrega 81 em IP), `prior`
  (suavização da forma), `num_histogram_bins`, `double_step`, `num_iterations`.
  Produz poucas regiões grandes (área média 4.609 px em PU).
- **SLIC** (*Simple Linear Iterative Clustering*), modo **SLICO** — k-means local:
  sementes numa grade de lado `region_size`, cada pixel atribuído à semente mais
  próxima numa distância que combina cor e posição, só numa janela local. O SLICO
  adapta a compacidade sozinho — por isso `ruler` **não tem efeito** (rótulos
  idênticos com 5, 10 e 20). Regiões regulares, quase quadradas.
- **LSC** (*Linear Spectral Clustering*) — mapeia cada pixel para um espaço de
  características de 10 dimensões em que um k-means ponderado aproxima o critério de
  *normalized cuts* (corte em grafo). Bordas mais fiéis à imagem. Parâmetros:
  `region_size`, `ratio` (compacidade). **Era não-determinístico** com várias threads
  (corrigido).
- **Watershed** — trata o gradiente da imagem como um relevo e "inunda" a partir de
  marcadores; onde águas de bacias diferentes se encontram, nasce uma fronteira. Aqui
  os marcadores são uma **malha regular** (`markers` é alvo). O OpenCV marca as
  linhas entre bacias com −1, que o código transforma num **id 0 espalhado pela cena**
  (21% dos pixels em IP).

### 14.2 O que cada um produz

| Cena | Segmentador | Regiões | Área média | Dens. borda | Pureza | Tempo |
|---|---|---|---|---|---|---|
| IP | seeds | 81 | 259,6 | 0,2051 | 0,8831 | 0,149 s |
| IP | slic | 324 | 64,9 | 0,2920 | 0,9581 | 0,051 s |
| IP | lsc | 324 | 64,9 | 0,3503 | 0,9699 | 0,031 s |
| IP | watershed | 273 | 77,0 | 0,5650 | 0,8543 | 0,004 s |
| PU | seeds | 45 | 4.608,9 | 0,1063 | 0,9212 | 1,854 s |
| PU | slic | 1.428 | 145,2 | 0,2132 | 0,9857 | 0,263 s |
| PU | lsc | 1.428 | 145,2 | 0,2440 | 0,9955 | 0,203 s |
| PU | watershed | 421 | 492,6 | 0,2797 | 0,9313 | 0,030 s |

- **Pureza** — fração da classe dominante dentro de cada região. É diagnóstico
  espacial, **não acurácia**.
- **Densidade de borda** — fração de pares de pixels vizinhos em regiões diferentes.

### 14.3 O desenho errado (experimento 05) e o certo (experimento 06)

**Errado: CNN fixa para os quatro.** Parece isolar o segmentador, mas mede o quanto
aquele conjunto fixo de hiperparâmetros por acaso serve a cada um. Em PU, o SLIC deu
Kappa 0,0346 — aleatório — contra 0,9360 do SEEDS. Motivo: 1.428 regiões → id
escalar até 1.428 contra um espectro de desvio 1 → precisa de outro learning rate e
outro dropout. Guardado como **contraexemplo**.

**Certo: otimização conjunta.** O segmentador é um eixo da otimização, como no uso
real. Para cada segmentador: HPO restrito a ele (TPE, 16 trials, seleção por
validação), depois retreino do vencedor com 5 sementes. O SEEDS passa pelo mesmo
tratamento e é a referência.

Diferença de OA para o SEEDS (ambos otimizados), split por pixel:

| Cena | SLIC | LSC\* | Watershed |
|---|---|---|---|
| IP | −17,63 | −7,28 | −25,28 |
| PU | +0,19 | −1,42 | +0,41 |
| SA | −1,31 | −1,62 | −1,35 |

\* LSC medido antes da correção do determinismo; refazer.

Conclusões:

1. **O segmentador é a decisão mais impactante do pipeline**: 25,3 pp separam o
   melhor do pior em IP, mais que qualquer hiperparâmetro isolado.
2. **Nenhum supera o SEEDS herdado de forma afirmável sob split por pixel.**
3. **Cada segmentador escolhe hiperparâmetros diferentes** (o SLIC pediu lr 3× menor e
   batch 128 para domar um id que vai até 324). O segmentador **reconfigura o treino**;
   não é um módulo plugável.
4. O que prevê o desempenho é **amostras de treino por região** (seção 7.5), não a
   pureza.

Sob split por região o ranking se inverte — mas por um motivo que não é mérito do
segmentador (seção 15).

---

## 15. O protocolo de avaliação e o vazamento espacial

Este é o achado transversal do trabalho.

### 15.1 Os modos de split

| `split_mode` | O que faz | Vazamento medido |
|---|---|---|
| `pixel` | sorteio estratificado por pixel — o padrão da literatura de HSI | **100%** |
| `block` | blocos retangulares da cena (`split_block`) | 66,5% a 100% |
| `region` | cada superpixel inteiro numa partição só | **0%** por construção |

**Vazamento espacial** — pixels vizinhos são quase idênticos (campos agrícolas
homogêneos). No split por pixel, todo pixel de teste tem um pixel de treino da mesma
classe a 1 ou √2 px. O teste vira "reconhecer o vizinho", não "classificar o espectro".

**Blocos não resolvem** quando o superpixel é grande: em IP os superpixels SEEDS têm
~16×16 px, do tamanho do bloco; o vazamento continuou em 100%.

**Validação cruzada por região** (`split_folds = k`): as regiões viram k grupos
(`StratifiedGroupKFold` — mantém cada região inteira num grupo e tenta preservar a
proporção das classes). Cada grupo é o teste uma vez; percorrendo `split_fold` de 0
a k−1, **cada região é testada exatamente uma vez**. Sem isso, uma partição única
deixava 3 ou 4 regiões no teste e desvio de ±0,25.

Decisão deliberada: a **validação não é disjunta por região** — é uma amostra por
pixel do treino. Separá-la também quadruplicava as classes ausentes do treino. O
invariante protegido é a disjunção **treino × teste**; uma validação otimista só pode
deixar o resultado final **pior**, que é o lado seguro de errar.

Limitação: em IP, 6 das 16 classes vivem em uma ou duas regiões SEEDS. Quando essa
região cai no teste, a classe é impossível de aprender (acurácia zero por
construção). O relatório do split devolve `classes_missing_from_train`.

### 15.2 Quanto sobra sem vazamento (IP, 5 folds × 5 sementes)

| Condição | OA split por pixel | OA split por região |
|---|---|---|
| **seeds** (herdado) | 0,8996 ± 0,0107 | **0,5897 ± 0,0635** |
| slic | 0,8440 | 0,7751 |
| lsc | 0,8668 | 0,7501 |
| watershed | 0,8055 | 0,6105 |

**A configuração herdada cai 31,0 pontos.** Todos caem, o SEEDS mais. O ranking se
inverte — mas não é mérito do SLIC: sob split por região **cada segmentador define a
própria partição**, e a do SLIC é muito mais fácil (teste a 2,35 px do treino em média,
contra 4,24 no SEEDS).

### 15.3 Homogeneidade: um classificador que só vê a posição

1-NN sobre as coordenadas (linha, coluna), **sem espectro nenhum**:

| Protocolo | IP | PU | SA |
|---|---|---|---|
| pixel, 100% do treino | 1,0000 | 1,0000 | 1,0000 |
| pixel, 10% do treino | 0,9711 | 0,9867 | 0,9975 |
| região SEEDS | 0,6997 | 0,5773 | 0,8568 |
| região SLIC | 0,8895 | 0,9360 | 0,9566 |

Consequências:

1. **Sob split por pixel, a OA não mede classificação espectral**: a posição sozinha
   (0,9711) supera a melhor CNN do trabalho em IP com 10% (0,8174). Como o id é uma
   coordenada grosseira, fusão e HPO medidos nesse protocolo podem estar medindo
   quanto o modelo explora a posição.
2. **Salinas está saturada**: 0,86 só com posição no protocolo mais difícil.
3. **A dificuldade do split por região depende do segmentador** (0,70 × 0,89 em IP).
   Comparar segmentadores sob split por região exige **a mesma partição para todos**
   (pendência).

---

## 16. MLOps: rastreio, campanhas e relatórios

**MLOps** — práticas para tornar experimentos de ML reprodutíveis e auditáveis:
saber exatamente qual configuração, com quais dados e qual código, produziu qual
modelo e qual resultado.

### 16.1 Rastreio — `hsi/mlops/rastreio.py`

Cada execução auditada vira um diretório imutável `outputs/runs/<run_id>/`:

| Arquivo | Conteúdo |
|---|---|
| `config.json` | a `ExperimentConfig` inteira |
| `environment.json` | versões, GPU, commit do git |
| `data.json` | **fingerprints** SHA-256 dos arrays que chegaram ao modelo (prova de quais dados) |
| `metrics.json` | métricas, histórico de treino, tempos |
| `manifest.json` | status e índice dos artefatos |
| `model.pt` | checkpoint |
| `report.pdf` | relatório |
| `failure.json` | só se falhou — falhas também ficam registradas |

Funções: `create_run` (abre antes de criar dados), `record_run` (fecha com sucesso),
`fail_run`, `array_fingerprint`, `data_fingerprint`, `result_metrics`.

Quem abre runs são as **campanhas** (`run_audited_experiment`). Os trials do HPO e a
execução avulsa da seção 8 do notebook **não** viram runs.

### 16.2 Campanhas — `hsi/mlops/campanha.py`

Uma campanha agrega runs por uma receita explícita, grava o plano **antes** de
treinar (com hash), atualiza resultados parciais após cada tentativa e calcula
estatísticas só a partir dos artefatos.

- `run_audited_experiment` — executa uma receita; uma exceção (até uma interrupção do
  kernel) vira falha registrada, e a campanha segue.
- `build_campaign_plan`, `run_audited_campaign` — campanha completa.
- `analyze_campaign` — agrega repetições por receita; ranking **só pela validação**.
- `_bootstrap_interval` — **intervalo bootstrap de 95%** da média: reamostra os
  valores com reposição 2.000 vezes (semente fixa) e pega os percentis 2,5 e 97,5 da
  média.
  Funciona mesmo com poucas sementes, sem supor normalidade.

### 16.3 Registro, observabilidade, relatórios

- `registro.py` — *registry*: guarda só o **campeão** de uma família comparável
  (mesma cena, arquitetura e protocolo). `promote_candidate` promove se for superior
  por AA, depois Kappa, depois OA. Promoção explícita; nenhuma célula a faz sozinha.
- `observabilidade.py` — lê os JSONs de `outputs/runs/` (`discover_runs`,
  `filter_runs`, `runs_summary`, `markdown_report`).
- `relatorios.py` — PDFs só com Matplotlib (`generate_pdf_report`,
  `generate_standalone_pdf_report`).
- `paineis.py` — painéis e PDF consolidado de **todos** os runs
  (`load_all_audited_runs`, `all_run_figures`, `generate_all_runs_pdf` →
  `outputs/reports/all_runs_report.pdf`). Cada ponto vem dos JSONs, sem recalcular.

---

## 17. Visualização e explicabilidade

### 17.1 Diagnósticos — `hsi/visualizacao/diagnosticos.py`

Figuras que **verificam** algo, não só ilustram: visão da cena e desbalanceamento
(`plot_scene_overview`), PCA (`plot_pca_diagnostics` — calcula sem *whiten* para
interpretar a variância), comparação dos segmentadores (`plot_segmentation_comparison`,
`segmentation_quality`, `plot_segmentation_overlay_suite`,
`plot_segmentation_quality_dashboard`), painel de avaliação (`plot_evaluation_dashboard`
— convergência, métricas, matriz de confusão, mapa de erros), métricas por classe,
custo de inferência (`profile_inference`) e perfil dos ramos paralelos
(`plot_parallel_profile`).

### 17.2 Explicabilidade — `hsi/visualizacao/explicabilidade.py`

**Hook** — função que o PyTorch chama durante o forward (ou backward) de uma camada,
para observar a saída sem alterar nada. É removida no fim.

- `capture_activations` + `plot_activation_atlases` — as ativações de cada convolução:
  uma imagem **filtro × posição**.
- **Grad-CAM** (`grad_cam`, `plot_grad_cam`) — na última Conv1d, o gradiente da
  pontuação da classe prevista em relação às ativações dá um **peso por filtro**
  (média do gradiente ao longo das posições); a soma ponderada das ativações, passada
  por ReLU e normalizada, mostra **quais posições mais empurraram a decisão**. Na
  CNN1D o mapa tem `Kb − 14` posições (16 em IP); a posição *i* resume as componentes
  *i* a *i + 14* (o campo receptivo).
- **Saliência** (`plot_input_saliency`) — `|d logit / d entrada|`: o quanto a
  pontuação da classe muda se cada componente PCA da entrada mudar um pouco.
- **Incerteza** (`plot_prediction_uncertainty`) — **confiança** (probabilidade máxima)
  e **entropia normalizada** (`−Σ p log p / log C`, de 0 = certeza a 1 = dúvida total)
  de cada pixel de teste, projetadas na cena.

Nenhuma dessas é prova de causalidade: indicam influência local.

### 17.3 Explicação inversa e dashboards

- `explicacao_inversa.py` — leva as decisões **de volta à imagem**: mapas H×W de
  previsão, acerto/erro, confiança e F1 (só nos pixels de teste — fora deles não há
  ground truth para medir); métricas agregadas **por superpixel**
  (`segment_decision_metrics`); e detecções representativas escolhidas por regras
  auditáveis (acerto mais confiante, maior incerteza, erro mais confiante, menor
  confiança) com o Grad-CAM de cada uma.
- `dashboards.py` — o mesmo protocolo de avaliação para qualquer resultado
  (métricas, por classe, custo, incerteza, Grad-CAM, saliência, estatísticas de todas
  as camadas), o que torna os dashboards comparáveis na compilação final.
- `glossario.py` — os cartões de definição exibidos junto aos gráficos.
- `originais.py` — as figuras do notebook original (títulos em inglês); não usadas
  nas células atuais.

---

## 18. Os experimentos 05 a 13

Um módulo por experimento em `hsi/experimentos/`; o índice `EXPERIMENTOS` em
`__init__.py`. Resultados em `experimentos/resultados/`.

| Exp. | Arquivo | Pergunta | Resposta curta |
|---|---|---|---|
| 05 | `e05_cnn_fixa.py` | segmentadores com CNN fixa | **desenho inválido**, guardado como contraexemplo (SLIC em PU: Kappa 0,03) |
| 06 | `e06_segmentadores.py` | qual segmentador, cada um otimizado para si | nenhum supera o SEEDS de forma afirmável sob split por pixel; o segmentador é a decisão de maior impacto |
| 08 | `e08_estrategias.py` | random × TPE × genético; orçamento de épocas; peso dos eixos | estratégias empatam no ruído; o orçamento importa; dropout e lr dominam |
| 09 | `e09_validacao_regiao.py` | quanto sobra sem vazamento | SEEDS cai 31 pp; ranking se inverte por causa da partição |
| 10 | `e10_impacto_hpo.py` | HPO contra a receita herdada | +8,40 pp no SEEDS (refazer com batch 32) |
| 11 | `e11_particao.py` | até onde a partição escala e quanto custa | escala; o "ganho" de acurácia das faixas é posicional |
| 12 | `e12_particao_split.py` | o ganho das faixas sobrevive sem vazamento? | não: vira perda |
| 13 | `e13_pool_quente.py` | pool frio × quente | quente: até 4,47×; frio subestimava e invertia a conclusão |

Infraestrutura:

- `infra.py` — `raiz_projeto`, `caminho_resultado` e a **retomada**: `carregar_feito`
  relê o CSV consolidado e pula células já medidas; `salvar` grava a cada célula
  concluída. Existe porque três sessões morreram pela metade.
- `medicoes.py` — números medidos fora de um CSV, cada bloco com data e
  procedência (`RAMOS`, `SEGMENTADORES`, `ABLACAO_FUSAO`, `MODOS_FUSAO`,
  `FRACAO_TREINO`, `THREADS_TORCH`, `GIL`, ...).
- `e13` também tem `conferir_memoria`: recusa uma célula `process` que não caberia na
  RAM, em vez de deixar o cronômetro medir swap.

---

## 19. O notebook e o lançador

**`HSI_MLOps.ipynb`** guarda a narrativa e as células de execução; o código está no
pacote. Cada célula de definição diz em que arquivo está o código da sua seção. A
primeira célula liga `%autoreload 2`: editar um arquivo de `hsi/` vale sem reiniciar
o kernel.

| Seção do notebook | O que faz | Onde está o código |
|---|---|---|
| 1–2 | ambiente e configuração | `ambiente.py`, `config.py` |
| 3 | dados, segmentadores, diagnósticos, glossário | `dados/`, `visualizacao/diagnosticos.py`, `glossario.py` |
| 4 | modelo | `modelo.py` |
| 5 | treino, explicabilidade, HPO, campanha, explicação inversa | `treino.py`, `avaliacao.py`, `pipeline.py`, `visualizacao/`, `hpo/`, `mlops/` |
| 6 | rastreio, registro, observabilidade | `mlops/` |
| 7 | variantes | `variantes.py` |
| 8 | **execução**: 8.1 painel, 8.2 explicabilidade, 8.4 HPO, 8.5 campanha, 8.6 explicação na imagem, 8.7 batch | chamadas às funções acima |
| 9 | todos os runs | `mlops/observabilidade.py`, `mlops/paineis.py` |
| 10 | experimentos do TCC | `experimentos/` |

**Lançador** (`experimentos/executar.py`) — roda um experimento sem kernel aberto:

```bash
.venv/bin/python experimentos/executar.py --lista   # lista os experimentos
.venv/bin/python experimentos/executar.py 13 IP     # roda o 13 em Indian Pines
```

Fixa `torch.set_num_threads(1)` no nível do módulo, porque o filho do `spawn`
reexecuta o lançador ao subir.

**Instalação:** `uv venv --python 3.12 && uv pip install -r requirements.txt && uv pip install -e . --no-deps`.

---

## 20. Pontos de atenção conhecidos

Coisas que o código faz e que você precisa saber para não ser pego de surpresa na
defesa:

1. **Batch da receita herdada.** O original treina com **32**. O experimento 10 usou
   **128** no braço "herdado" (o docstring de `e10_impacto_hpo.py` ainda descreve o
   desenho com 128, marcado com um aviso). O braço precisa ser refeito antes de citar
   os +8,40 pp.
2. **A configuração da célula 5 do notebook não é a do Sydney.** Ela usa batch 128,
   meta de validação 0,95 e patience 10. É uma configuração de trabalho, não a
   referência. `ExperimentConfig()` sem argumentos usa batch 32, mas a parada antecipada
   (`val_accuracy`, patience 10) também difere do original (`val_loss`, patience 5).
3. **`fit` só restaura os melhores pesos quando a parada dispara.** Se o treino chegar
   ao limite de épocas, o modelo fica com os pesos da última época.
4. **`confusion_matrix` sem `labels`.** Se uma classe não aparece nem no teste nem nas
   previsões, a matriz encolhe e os índices por classe deslocam. Raro no split por
   pixel; possível no split por região.
5. **Watershed com id 0 espalhado.** As linhas entre bacias viram uma "região" única
   que cobre 21% dos pixels de IP.
6. **`ruler` do SLIC não faz nada** no modo SLICO; as três receitas do HPO para o SLIC
   diferem só em `region_size`.
7. **Números de LSC anteriores a 24/09 são inválidos** (não-determinismo). Estão em
   `experimentos/resultados/invalidados_lsc/`.
8. **Speedup "contra o monolítico" do experimento 11 não vale**: compara GPU
   (monolítico) com CPU (partições). O que vale é partição contra partição (exp. 13).
9. **OA de partição com executor `thread` não é reprodutível** (RNG global
   compartilhado). Use o `process`.
10. **Sob split por pixel, a posição sozinha explica quase tudo** (seção 15.3). Todo
    resultado medido nesse protocolo deve ser lido com essa ressalva.
11. **`binary_readout`** reproduz um cálculo inválido do original; não é métrica deste
    problema.
12. **Backend `process` dos ramos usa `fork`** (padrão do Linux); o da partição usa
    `spawn`. São escolhas diferentes por motivos diferentes: o `fork` herda o OpenCV
    de 1 thread (determinismo do LSC); o `spawn` é obrigatório depois que o autograd
    rodou em threads.

---

## 21. Glossário

**Dados e sensoriamento**

- **Banda** — uma faixa de comprimento de onda medida pelo sensor.
- **Cubo hiperespectral** — `(H, W, B)`: duas dimensões espaciais e uma espectral.
- **Assinatura espectral** — o vetor de bandas de um pixel.
- **Ground truth (GT)** — rótulos verdadeiros; 0 = background.
- **Background** — pixel sem rótulo; descartado.
- **Row-major** — ordem de achatamento linha por linha.
- **Desbalanceamento** — classes com quantidades muito diferentes de amostras.

**Pré-processamento**

- **PCA** — projeção nas direções de maior variância.
- **Componente principal** — uma dessas direções; `Kb` é quantas a rede recebe.
- **Whitening** — escalar cada componente para variância 1.
- **Composição RGB** — 3 primeiras componentes como imagem de 3 canais.
- **Superpixel / região / segmento** — grupo contíguo de pixels parecidos.
- **Id da região** — o número do superpixel de cada pixel; a segunda entrada.
- **Pureza** — fração da classe dominante dentro de uma região.

**Aprendizado**

- **Amostra** — um pixel rotulado.
- **Treino / validação / teste** — aprender / decidir / relatar.
- **Estratificado** — preservando a proporção das classes.
- **Época, batch, passo** — passada completa, lote, atualização dos pesos.
- **Batch efetivo** — `batch_size × gradient_accumulation_steps`.
- **Learning rate** — tamanho do passo do otimizador.
- **Dropout** — desligar ativações ao acaso no treino.
- **Weight decay** — penalização L2 dos pesos.
- **Overfitting** — decorar o treino e generalizar mal.
- **Early stopping** — parar quando a validação para de melhorar.
- **Patience** — quantas épocas sem melhora se toleram.
- **Logits / softmax** — saídas cruas / probabilidades.
- **Cross-entropy** — a loss de classificação.
- **Adam** — otimizador adaptativo.
- **Xavier/Glorot** — inicialização que preserva a escala do sinal entre camadas.
- **Conv1d, kernel, filtro, canal** — convolução 1D; largura da janela; um detector
  aprendido; uma "trilha" de entrada/saída.
- **Flatten** — achatar `(canais, comprimento)` num vetor.
- **Embedding** — tabela aprendida que transforma um índice num vetor.
- **Fusão tardia** — juntar fontes depois da extração de características.
- **State dict** — o dicionário de pesos de um modelo.
- **Checkpoint** — pesos + o necessário para remontar o modelo.

**Métricas**

- **OA, AA, Kappa** — seção 9.
- **F1** — média harmônica de precisão e revocação.
- **IoU** — interseção sobre união, por classe.
- **Matriz de confusão** — real × previsto.
- **pp (pontos percentuais)** — diferença absoluta entre percentuais (0,80 → 0,85 =
  +5 pp).

**Avaliação**

- **Vazamento (leakage)** — informação do teste chegando ao treino ou à seleção.
- **Vazamento espacial** — vizinhos quase idênticos em treino e teste.
- **Split por pixel / bloco / região** — seção 15.
- **Validação cruzada (k-fold)** — k partições, cada uma o teste uma vez.
- **Semente (seed)** — valor que fixa a sequência aleatória.
- **Determinismo** — mesma entrada → mesma saída, bit a bit.
- **Bootstrap** — reamostragem com reposição para estimar incerteza.

**HPO**

- **Hiperparâmetro, trial, espaço, orçamento** — seção 13.1.
- **Random search** — sorteio uniforme.
- **TPE** — busca bayesiana por razão de densidades bons/ruins.
- **Algoritmo genético** — população, torneio, crossover, mutação, elitismo.
- **Gene** — um eixo do espaço; **indivíduo** — uma tupla de índices.
- **Baixa fidelidade** — avaliar com orçamento reduzido.
- **Throughput** — amostras processadas por segundo.

**Paralelismo**

- **Thread, processo, GIL, pool, worker, future** — seção 12.1.
- **Serialização / pickle** — converter objetos em bytes para enviar a outro processo.
- **fork / spawn** — clonar o pai / subir um interpretador limpo.
- **initializer** — função que cada worker roda uma vez ao nascer.
- **Pool frio / quente** — criado dentro da medição / criado e aquecido antes.
- **Speedup, teto, overhead, contenção, Lei de Amdahl** — seção 12.1.
- **RSS / PSS** — memória residente total / proporcional (divide as páginas
  compartilhadas); o PSS é o que precisa caber.

**Python e PyTorch**

- **dataclass / frozen** — classe de dados / imutável (mude com `replace`).
- **property** — método lido como atributo.
- **mixin** — classe auxiliar herdada junto com outra.
- **context manager** (`with`) — garante `__enter__`/`__exit__` (ex.: fechar o pool).
- **Tensor** — array do PyTorch, em CPU ou GPU.
- **autograd** — cálculo automático de gradientes.
- **`no_grad`** — desliga o rastreio de gradientes (inferência).
- **`nn.Module`** — classe base de camadas e modelos.
- **DataLoader / TensorDataset** — iterador de lotes / dataset sobre tensores.
- **`pin_memory`** — memória fixada para cópia CPU→GPU assíncrona.
- **cuDNN, TF32, cuBLAS** — bibliotecas e modos de cálculo da NVIDIA.
- **Hook** — função chamada durante o forward/backward de uma camada.

**MLOps**

- **Run** — uma execução rastreada. **Campanha** — conjunto planejado de runs.
- **Manifest** — índice e status dos artefatos.
- **Fingerprint** — hash SHA-256 que identifica um array.
- **Registry / campeão** — o melhor modelo de uma família comparável.

---

## 22. Roteiro de estudo e perguntas de checagem

**Ordem de leitura do código** (cada passo apoia o seguinte):

1. `hsi/__init__.py` e `hsi/config.py` — o vocabulário.
2. `hsi/dados/cenas.py` → `preprocessamento.py` → `segmentacao.py` → `particoes.py`.
3. `hsi/modelo.py` — releia `_FusionClassifier` com a seção 7 ao lado.
4. `hsi/treino.py` → `hsi/avaliacao.py` → `hsi/pipeline.py`.
5. `hsi/paralelo/ramos.py` → `hsi/paralelo/particao.py`.
6. `hsi/hpo/espaco.py` → `estrategias.py` → `busca.py` → `selecao.py`.
7. `hsi/experimentos/e09_validacao_regiao.py` e `e13_pool_quente.py` — os dois
   experimentos que mais mudaram as conclusões.
8. `mlops/` e `visualizacao/` por último: são apoio, não o núcleo.

Um exercício que vale mais que reler: abra o notebook, rode a seção 8 com
`epochs=3`, e acompanhe com o depurador uma chamada de `run_experiment` do começo ao
fim, conferindo os shapes da tabela da seção 6.2.

**Perguntas para checar se você domina** (as respostas estão neste documento):

1. Por que a CNN1D precisa de pelo menos 15 componentes PCA?
2. Por que o id escalar tem 1,79× mais influência que o espectro inteiro na primeira
   camada densa? O que o *whitening* tem a ver com isso?
3. O que o embedding resolve que o escalar não resolve? Quanto rende em IP?
4. Por que a fusão herdada é tão sensível ao vazamento espacial?
5. Qual é o teto do paralelismo de dois ramos em Salinas, e por quê? Por que vale
   tão pouco no tempo total?
6. Por que threads funcionam nos ramos e não funcionam no treino das partições?
7. Por que o pool frio subestimou o speedup, e em N=16 inverteu a conclusão?
8. O que impede N=32 com processos?
9. Por que a seleção do HPO usa validação e nunca teste?
10. Por que as três estratégias empatam? O que realmente importou?
11. Por que comparar segmentadores com a CNN fixa é um desenho inválido?
12. O que prevê se um segmentador ajuda: pureza ou amostras de treino por região?
13. Por que o split por blocos não eliminou o vazamento em IP?
14. Por que o SLIC "ganha" sob split por região, e por que isso não é mérito dele?
15. O que significa um classificador só de posição acertar 0,97 em IP?
16. Por que a validação da validação cruzada por região não é disjunta por região?
17. Qual o ganho efetivo do Pilar 3 sobre a referência, e quais as duas ressalvas?
