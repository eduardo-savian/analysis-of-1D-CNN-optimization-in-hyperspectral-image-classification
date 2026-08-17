# Classificação de imagens hiperespectrais com CNN 1D

Trabalho de Conclusão de Curso — Engenharia de Computação, UNIVALI.
Reimplementação em PyTorch e otimização do trabalho de Sydney Matheus de Souza.

| | |
|---|---|
| Implementação original | Sydney Matheus de Souza |
| Reimplementação e otimização | Eduardo Savian |
| Orientação | Felipe Viel |

O objetivo é classificar cada pixel de uma cena hiperespectral pela sua assinatura
espectral, combinando uma CNN 1D sobre o espectro com informação espacial vinda de uma
segmentação em superpixels — e, a partir dessa base, medir o efeito de paralelizar as
etapas independentes do pipeline.

---

## Sumário

- [O problema](#o-problema)
- [Arquitetura do pipeline](#arquitetura-do-pipeline)
- [Instalação rápida](#instalação-rápida)
- [Como rodar](#como-rodar)
- [Estrutura do repositório](#estrutura-do-repositório)
- [O pacote `hsi`](#o-pacote-hsi)
- [Paralelização](#paralelização)
- [Arquiteturas e segmentadores](#arquiteturas-e-segmentadores)
- [MLOps e batches na GPU](#mlops-e-batches-na-gpu)
- [Guia operacional MLOps](#guia-operacional-mlops)
- [Baseline × nossa implementação](#baseline--nossa-implementação)
- [Resultados de referência](#resultados-de-referência)
- [Reprodutibilidade](#reprodutibilidade)
- [Testes](#testes)
- [Datasets](#datasets)
- [Solução de problemas](#solução-de-problemas)

---

## O problema

Uma imagem hiperespectral não tem três canais como uma foto comum: tem centenas. Cada pixel
carrega um espectro — a resposta do material naquele ponto em cada faixa de comprimento de
onda. PaviaU, por exemplo, é uma cena de 610×340 pixels com **103 bandas** por pixel.

Isso muda a natureza do problema de classificação:

- **cada pixel é uma amostra**, e o vetor de features é o espectro dele;
- as bandas são **altamente correlacionadas** — daí o PCA reduzir 103 bandas para 15 sem
  perda relevante;
- as classes são **muito desbalanceadas** (em PaviaU, a maior classe tem 20× mais amostras
  que a menor), por isso os splits são estratificados e a métrica AA importa tanto quanto a OA;
- a maior parte da cena é **background** (rótulo 0), descartado antes do treino.

Classificar só pelo espectro ignora que pixels vizinhos tendem a ser da mesma classe. A
segunda entrada da rede resolve isso: um algoritmo de superpixels (SEEDS, SLIC ou LSC) agrupa a cena em
regiões homogêneas, e o **id da região** entra na rede junto com o espectro. Além de "com o
que este pixel se parece", a rede passa a saber "de que região homogênea ele veio".

## Arquitetura do pipeline

```
                       ┌───────────────────────────────────────┐
   cena .mat           │  ramo espectral                       │
  (H, W, B)            │  PCA -> Kb componentes                │──┐
      │                └───────────────────────────────────────┘  │
      ├── tabela de                                               │ fusão
      │   pixels ──┐                                              │   │
      │            │   ┌───────────────────────────────────────┐  │   ▼
      │            └──▶│  ramo espacial                        │  │  CNN 1D
      │                │  PCA -> 3 comp. -> RGB -> superpixels │──┘  + MLP
      │                │  -> id de superpixel por pixel        │      │
      │                └───────────────────────────────────────┘      ▼
      └── ground truth ──────────────────── split estratificado ─▶  OA/AA/kappa
```

Os dois ramos são **independentes até a fusão** — é essa independência que a
[paralelização](#paralelização) explora.

O baseline (CNN 1D):

| bloco | camadas |
|---|---|
| extração | `Conv1d(1→16, k=9)` → `Conv1d(16→32, k=3)` → `Conv1d(32→64, k=3)` → `Conv1d(64→64, k=3)`, ReLU em cada, e `Flatten` |
| fusão | `torch.cat([features, id_do_superpixel])` — só quando `two_input=True` |
| classificação | `Linear(→128)` → `Dropout(0.5)` → `Linear(→64)` → `Dropout(0.5)` → `Linear(→classes)` |

São 37.577 parâmetros treináveis para PaviaU — um modelo pequeno, o que tem consequências
diretas de desempenho (veja [Paralelização](#paralelização)).

As variantes 2D e 3D usam a mesma cabeça classificadora e a mesma fusão opcional, mas mudam
o tensor de entrada: CNN2D recebe um patch das três componentes espaciais; CNN3D recebe um
cubo de bandas e vizinhança espacial. Veja [Arquiteturas e segmentadores](#arquiteturas-e-segmentadores).

## Instalação rápida

Detalhes completos, incluindo AWS e as armadilhas de CUDA, em [SETUP.md](SETUP.md).

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.13.0 \
    --index-url https://download.pytorch.org/whl/cu130     # ou .../whl/cpu
uv pip install --python .venv/bin/python -r requirements.txt

.venv/bin/python scripts/download_data.py                  # datasets (~67 MB)
```

Dois pontos que quebram o ambiente se ignorados:

- **`opencv-contrib`, não `opencv-python`** — o SEEDS (`cv.ximgproc`) só existe no pacote
  *contrib*, e os dois colidem no mesmo namespace `cv2`.
- **RTX 50xx é Blackwell (`sm_120`)** — precisa de wheel CUDA **12.8+**. Um wheel cu126
  importa sem erro e falha no primeiro kernel.

## Como rodar

### Linha de comando

```bash
# execução completa (PaviaU, espectro + segmentação)
.venv/bin/python scripts/run_experiment.py

# outras cenas e modos
.venv/bin/python scripts/run_experiment.py -d IP --no-two-input
.venv/bin/python scripts/run_experiment.py -d SA --backend thread

# comparações espaciais e espectral-espaciais
.venv/bin/python scripts/run_experiment.py -d PU --model cnn2d --segmenter slic --patch-size 5
.venv/bin/python scripts/run_experiment.py -d IP --model cnn3d --segmenter lsc --patch-size 5 --batch-size 64

# métricas e histórico em JSON
.venv/bin/python scripts/run_experiment.py --json outputs/pu.json
```

### Notebook

```bash
.venv/bin/python -m jupyterlab     # abrir src/TCC3_SydneySouza_PyTorch.ipynb
```

Ou sem interface:

```bash
.venv/bin/python -m jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=2400 \
    --output-dir outputs --output executed.ipynb \
    src/TCC3_SydneySouza_PyTorch.ipynb
```

### Como biblioteca

```python
from hsi import ExperimentConfig, run_experiment

result = run_experiment(ExperimentConfig(dataset="PU", two_input=True), backend="thread")
print(result.metrics.summary())
print(result.prep_timings.summary())
```

O notebook, o CLI e a API chamam **as mesmas funções** — por construção, produzem o mesmo
resultado.

## Estrutura do repositório

```
├── src/
│   ├── hsi/                          o pacote (toda a lógica)
│   └── TCC3_SydneySouza_PyTorch.ipynb   notebook: roteiro que executa o pacote
│   └── TCC3_SydneySouza_01-10-2023.ipynb  notebook original, preservado como referência
├── scripts/
│   ├── download_data.py              baixa e valida os datasets (sha256 + conteúdo)
│   ├── run_experiment.py             execução completa sem notebook
│   ├── benchmark_parallel.py         mede os backends de paralelização
│   ├── benchmark_batches.py          mede throughput por batch físico na GPU
│   └── compare_variants.py           baseline × nossa implementação
├── tests/test_hsi.py                 testes unitários, sem GPU nem datasets
├── tcc/                              texto do TCC (LaTeX/Typst)
├── data/dataHSI/                     os .mat (gerado, fora do versionamento)
├── outputs/                          curvas, checkpoints (gerado)
├── SETUP.md                          ambiente, CUDA, AWS
└── requirements.txt                  versões exatas de validação
```

## O pacote `hsi`

Cada módulo tem uma responsabilidade só, e as dependências vão numa direção só
(`config` ← `data`/`preprocess`/`segmentation` ← `parallel` ← `pipeline` ← `variants`):

| módulo | responsabilidade |
|---|---|
| `config` | `ExperimentConfig` — todos os hiperparâmetros num objeto imutável |
| `paths` | raiz do repositório, diretório de dados e de saídas |
| `env` | conferência de versões, seleção de GPU, determinismo, sementes |
| `data` | carga dos `.mat`, tabela de pixels, remoção do background |
| `preprocess` | PCA e composição RGB |
| `segmentation` | superpixels SEEDS, SLIC e LSC e seus artefatos |
| `patches` | extração vetorizada de patches para CNN2D e CNN3D |
| `dataset` | split estratificado e DataLoaders |
| `parallel` | execução concorrente dos ramos independentes + medição |
| `model` | `CNN1D`, `CNN2D`, `CNN3D` e a cabeça de fusão compartilhada |
| `train` | laço de treino, histórico e parada antecipada |
| `evaluate` | predição, OA/AA/kappa, matriz de confusão |
| `plots` | curvas e figuras |
| `tracking` | cria e fecha a trilha local de cada execução (configuração, ambiente, dados, métricas e artefatos) |
| `pipeline` | encadeia tudo: `prepare_data`, `run_experiment` |
| `variants` | baseline (Souza, 2023) × nossa implementação |

### Como isso difere do notebook original

O notebook original funcionava, mas era um script único com estado global. As mudanças
estruturais, e o porquê de cada uma:

| antes | agora | motivo |
|---|---|---|
| `dataset`, `Kb`, `test_ratio`, `NUM_EPOCHS`... espalhados por células | um `ExperimentConfig` | funções liam variáveis globais definidas em outra célula; mudar um parâmetro exigia caçar onde ele era usado |
| `X_re1`, `X_re2`, `X_new`, `X_norm`, `X_pca`... | nomes por significado, escopo local | não dava para saber em que estágio cada array estava |
| espectro e segmentação empilhados em `(N, Kb, 1, 1, 2)` para caber num `train_test_split` | o **índice** é dividido uma vez e cada array é indexado por ele | o empilhamento existia só para contornar a API, e era desfeito logo depois com fatiamento posicional |
| `if dataset == 'IP': ... elif ...` repetido | tabelas (`SCENE_FILES`, `SEEDS_PARAMS`, `PCA_COMPONENTS`) | adicionar uma cena exigia editar vários pontos |
| `Google Drive` + `os.getcwd()` | raiz do repositório descoberta na árvore, `HSI_DATA_DIR` | quebrava conforme o diretório em que o Jupyter subia |
| prints de debug dentro das funções de cálculo | funções retornam dados; quem imprime é o notebook/CLI | permitiu reusar as mesmas funções em três frontends |
| nenhum teste | testes sem GPU nem datasets | a refatoração precisava de rede de segurança |

**A reimplementação reproduz o original bit a bit**: mesmo histórico de perda por época,
mesmos pesos finais e mesmos OA/AA/kappa (veja [Reprodutibilidade](#reprodutibilidade)).

## Paralelização

Depois do PCA, os dois ramos da preparação não dependem um do outro — só a fusão precisa dos
dois. Executá-los concorrentemente troca `T_espectral + T_espacial` por
`max(T_espectral, T_espacial) + overhead`.

Três backends, selecionáveis por `backend=` ou `--backend`:

| backend | como executa | característica |
|---|---|---|
| `sequential` | um ramo depois do outro | linha de base da comparação |
| `thread` | `ThreadPoolExecutor` | funciona apesar do GIL: a SVD do PCA (BLAS) e o `iterate` do SEEDS (OpenCV) o liberam. Não copia nada |
| `process` | `ProcessPoolExecutor` | imune ao GIL, mas paga serialização da tabela de pixels |

### Resultados medidos

RTX 5070 Laptop, mediana de 5 repetições
(`scripts/benchmark_parallel.py`), tempo do **bloco de preparação**:

| cena | sequential | thread | process | speedup (thread) | teto teórico |
|---|---|---|---|---|---|
| IP — Indian Pines | 0,508 s | **0,266 s** | 0,771 s | **1,91×** | 1,58× |
| PU — Pavia University | 1,218 s | **1,140 s** | 1,397 s | 1,07× | 1,11× |
| SA — Salinas Valley | 1,128 s | **0,946 s** | 1,525 s | 1,19× | 1,25× |

Três conclusões, e nenhuma delas é "paralelizou, ficou rápido":

1. **O ganho depende do balanço entre os ramos.** Em Indian Pines os dois ramos custam quase
   o mesmo (0,18 s e 0,32 s) e o speedup chega perto do teto. Em PaviaU o ramo espacial
   domina (0,13 s contra 1,10 s): sobrepor um ramo curto a um longo quase não ajuda.
2. **`process` é mais lento que `sequential` nas três cenas.** O ganho de sobrepor as etapas
   não paga a criação do processo mais a serialização da tabela de pixels (43 MB em PaviaU).
   O projeto original previa `multiprocessing`; a medição mostra que, nesta escala, threads
   são a escolha certa — o trabalho pesado está em código nativo que libera o GIL.
3. **A preparação é uma fração pequena do total.** No fluxo completo de PaviaU, ela é **2,1%**
   do tempo de ponta a ponta (1,3 s contra ~59 s de treino). Pela lei de Amdahl, acelerar só
   a preparação não pode render mais que **1,02×** no total — e o ruído entre execuções do
   treino (±3 s) é maior que o ganho.

O gargalo real deste pipeline é o treino, e dentro dele o **lançamento de kernel**, não a
computação: com batch de 32 amostras × 15 floats num modelo de 37 mil parâmetros, a GPU passa
mais tempo sendo alimentada do que calculando. É por isso que `num_workers=0` é intencional e
que instâncias AWS maiores não reduzem o tempo por época.

```bash
.venv/bin/python scripts/benchmark_parallel.py -n 5
.venv/bin/python scripts/benchmark_parallel.py -d PU --with-training
```

## Arquiteturas e segmentadores

As arquiteturas são alternativas experimentais, não resultados diretamente comparáveis por
padrão: CNN1D, CNN2D e CNN3D consomem contextos diferentes e possuem números de parâmetros
distintos. Registre `model_type`, `patch_size`, `segmenter`, batch e seed no mesmo experimento
antes de tirar conclusões.

| modelo | tensor por amostra | contexto aprendido | custo relativo |
|---|---|---|---|
| `cnn1d` | `(1, Kb)` | somente bandas do pixel | baixo; baseline reproduzível |
| `cnn2d` | `(3, P, P)` | vizinhança espacial das 3 componentes principais | moderado |
| `cnn3d` | `(1, B, P, P)` | bandas e vizinhança espacial simultaneamente | alto em RAM/GPU |

`Kb` é o número de componentes PCA do ramo espectral, `B` é o número original de bandas da
cena e `P` é `--patch-size`. Patches usam padding por reflexão nas bordas. A seleção dos pixels
rotulados preserva a ordem da tabela de pixels, portanto os patches permanecem alinhados aos
rótulos e aos ids de superpixel.

### CNN 1D — baseline espectral

Recebe um espectro por pixel em `(N, 1, Kb)`. Quatro `Conv1d` percorrem apenas a dimensão de
bandas; após `Flatten`, o embedding pode receber o id do superpixel antes da MLP. É o único
modo cuja arquitetura reproduz o notebook de referência e deve ser usado para validar qualquer
mudança estrutural.

```bash
.venv/bin/python scripts/run_experiment.py -d PU --model cnn1d \
  --segmenter seeds --backend thread --batch-size 512
```

### CNN 2D — contexto espacial PCA

Recebe patches `(N, 3, P, P)` formados pelas três componentes principais da cena inteira. Duas
`Conv2d` aprendem textura e vizinhança; `AdaptiveAvgPool2d` permite trocar `P` sem mudar a MLP.
Mesmo com `--no-two-input`, CNN2D ainda calcula PCA espacial para criar os patches — o que é
desligado nesse modo é apenas a concatenação do id do superpixel.

```bash
.venv/bin/python scripts/run_experiment.py -d PU --model cnn2d --segmenter slic \
  --patch-size 5 --backend thread --batch-size 256
```

### CNN 3D — contexto espectral-espacial

Recebe cubos `(N, 1, B, P, P)` do dado cru normalizado para `[0, 1]`. `Conv3d` aprende filtros
em bandas e em espaço ao mesmo tempo; é a variante mais expressiva e também a que mais consome
memória. Comece com `P=5` e batches pequenos (por exemplo, 32 ou 64), então execute
`benchmark_batches.py` para encontrar um valor seguro na GPU alvo.

```bash
.venv/bin/python scripts/run_experiment.py -d IP --model cnn3d --segmenter lsc \
  --patch-size 5 --batch-size 64 --backend thread
```

### Segmentadores disponíveis

Todos recebem a mesma composição RGB de três componentes PCA e retornam um mapa `(H, W)` de
ids mais a máscara de contorno. Isso isola a comparação: mudar `--segmenter` não altera split,
arquitetura, loss ou métricas.

| opção | algoritmo | parâmetros por cena | quando comparar |
|---|---|---|---|
| `seeds` | Superpixels Extracted via Energy-Driven Sampling | superpixels-alvo, níveis, histograma, prior e iterações | baseline original |
| `slic` | SLICO (SLIC com compacidade adaptativa) | tamanho de região, ruler e iterações | regiões compactas com pouca calibração |
| `lsc` | Linear Spectral Clustering | tamanho de região, ratio e iterações | alternativa de superpixel guiada por cor/estrutura |

SLIC e LSC, assim como SEEDS, exigem `opencv-contrib-python[-headless]`. O modo de ablação sem
segmentação é `--no-two-input`, e não um quarto segmentador: dessa forma não há id artificial
injetado na fusão.

Para uma comparação justa, fixe o modelo, patch, batch, seed e backend; altere apenas o
segmentador. Para comparar arquiteturas, mantenha cena, split e seed fixos, mas reporte memória,
tempo por época e métricas separadamente.

## MLOps e batches na GPU

O projeto aplica MLOps como disciplina de experimento, não como uma ferramenta externa
obrigatória. Cada execução do CLI cria `outputs/runs/<run-id>/`, sem sobrescrever a anterior:

```
config.json + environment.json + data.json
                  │
                  ▼
preparação (PCA espectral || PCA-3/RGB/superpixels, em threads de CPU)
                  │
                  ▼
DataLoader em RAM -> batches físicos na GPU -> CNN 1D/2D/3D -> fusão -> métricas
                  │
                  ▼
metrics.json + checkpoint + curvas + manifest.json
```

`data.json` contém hashes do espectro pós-PCA, rótulos, segmentação e splits; o ambiente
registra Python, PyTorch, CUDA, GPU e commit Git. Assim, uma métrica ou checkpoint pode ser
associado à configuração, versão de código e dados que o produziram. `--no-track` desativa esse
registro quando necessário. Para times maiores, esse diretório é uma fronteira simples para
subir artefatos a MLflow, DVC/S3 ou um registry, sem acoplar o treino a esses serviços.

### O que é — e o que não é — paralelismo antes da fusão

O paralelismo de `--backend thread` é corretamente aplicado **uma vez, na preparação**:
PCA espectral e PCA-3/RGB/superpixels são independentes e executam código nativo que libera o GIL.
Após isso, os ids de superpixel são estáticos. Não faz sentido criar uma thread de segmentação por
batch e outra da CNN: a segmentação terminaria repetidamente o mesmo trabalho, enquanto a CNN
só inicia depois que os tensores e seus ids existem. Essa estratégia também disputaria CPU/RAM
com o DataLoader sem criar paralelismo de GPU.

A unidade de paralelismo da CNN é o **batch**: um único `Conv1d` recebe `(B, 1, Kb)` e o CUDA
executa as amostras de `B` em paralelo nos SMs. O tamanho correto depende da GPU e deve ser
medido no dataset real:

```bash
# mede épocas reais, descarta candidatos sem memória e aponta o maior throughput
.venv/bin/python scripts/benchmark_batches.py -d PU \
  --batches 32 64 128 256 512 1024 --json outputs/batches-pu.json

# aplica o batch vencedor; avaliação pode usar batch ainda maior
.venv/bin/python scripts/run_experiment.py -d PU --backend thread \
  --batch-size 512 --eval-batch-size 1024
```

Há duas escolhas diferentes, ambas registradas em `config.json`:

- `--batch-size`: batch **físico** que entra no `Conv1d`; é o principal controle de ocupação da GPU e de redução de lançamentos de kernel.
- `--accumulate-steps`: acumula gradientes de vários batches antes do Adam atualizar os pesos. Ele simula batch efetivo maior quando a memória limita o batch físico, mas **não** substitui aumentar o batch físico para reduzir lançamentos.

`num_workers=0` continua o padrão porque os tensores já estão materializados em RAM. Só use
`--num-workers`, `--prefetch-factor` e `--persistent-workers` após perfil de desempenho: para
este modelo, workers extras frequentemente custam mais IPC do que economizam.

## Guia operacional MLOps

Esta seção transforma os conceitos de MLOps em uma rotina concreta para este repositório.
O objetivo não é adicionar uma ferramenta por moda: é tornar cada decisão de desempenho e cada
métrica verificável depois.

### 1. Fluxo recomendado de experimento

1. **Validar ambiente e dados.** Instale as versões documentadas e execute os testes. O script
   de download também confere hash, shape e número de classes dos arquivos `.mat`.
2. **Medir a preparação.** Use `benchmark_parallel.py` para comparar `sequential`, `thread` e
   `process` na máquina que realmente executará o experimento. Não assuma que processos são
   mais rápidos: em PaviaU, a cópia dos arrays costuma anular o ganho.
3. **Medir batch físico.** Use `benchmark_batches.py` com a cena e a GPU alvo. Escolha o maior
   throughput em `samples_per_second`, não simplesmente o maior batch que cabe na memória.
4. **Treinar uma execução rastreada.** Passe explicitamente o backend e os batches escolhidos.
   Antes de montar DataLoaders, o contrato valida shapes, NaN/infinito, classes, patches e splits.
5. **Promover conscientemente.** Use `--promote` somente com checkpoint salvo. O registry local
   compara AA, kappa e OA com o campeão de condições compatíveis e registra a decisão.

```bash
# 1. segurança contra regressões
.venv/bin/python tests/test_hsi.py

# 2. decidir o paralelismo pré-fusão
.venv/bin/python scripts/benchmark_parallel.py -d PU -n 5

# 3. decidir o batch físico da GPU
.venv/bin/python scripts/benchmark_batches.py -d PU \
  --batches 32 64 128 256 512 1024 --json outputs/benchmarks/pu-batches.json

# 4. experimento reproduzível e rastreado
.venv/bin/python scripts/run_experiment.py -d PU --backend thread \
  --batch-size 512 --eval-batch-size 1024 --promote --json outputs/pu-resumo.json
```

### 2. Contrato de cada execução

Sem `--out`, os artefatos ficam todos juntos em `outputs/runs/<run-id>/`:

| arquivo | quando é criado | conteúdo e uso |
|---|---|---|
| `config.json` | antes do treino | hiperparâmetros e opções de carregamento; é a receita da execução |
| `environment.json` | antes do treino | Python, PyTorch, CUDA, GPU e commit Git; identifica o ambiente que pode afetar desempenho e números |
| `data.json` | sucesso | shape, classes e SHA-256 de espectros, segmentação, patches 2D/3D e índices dos splits; inclui o resultado do contrato de dados |
| `metrics.json` | sucesso | OA, AA, kappa, loss, histórico e tempos da preparação/treino |
| `manifest.json` | início e término | estado `running`, `completed` ou `failed` e os caminhos dos artefatos |
| `failure.json` | falha | tipo e mensagem do erro, sem perder a evidência de uma execução interrompida |
| `*_torch.pt` e `*_curve_torch.pdf` | sucesso, salvo `--no-save` | modelo e curvas associados àquele `run-id` |

`--out outro/diretorio` envia somente checkpoint e curvas para esse local; o manifesto conserva
o caminho absoluto deles. `--json` é um resumo adicional solicitado pelo usuário. `--no-track`
mantém o modo antigo, sem criar o diretório de rastreio.

### 3. Referência completa da linha de comando

| opção | efeito | quando usar |
|---|---|---|
| `--dataset IP\|PU\|SA` | seleciona cena e PCA padrão | sempre declare ao comparar cenas |
| `--model cnn1d\|cnn2d\|cnn3d` | seleciona contexto espectral, espacial ou espectral-espacial | `cnn1d` para baseline; 2D/3D para ablação espacial |
| `--patch-size N` | lado ímpar dos patches de CNN2D/CNN3D | comece em 5; aumente só com memória/benchmark suficientes |
| `--segmenter seeds\|slic\|lsc` | algoritmo que produz o id usado na fusão | altere um por vez em estudos de segmentação |
| `--no-two-input` | remove o id de superpixel e a fusão | ablação sem segmentação |
| `--backend sequential\|thread\|process` | escolhe execução dos ramos antes da fusão | `thread` após benchmark; `sequential` para baseline |
| `--batch-size N` | tamanho físico de `(N, 1, Kb)` na GPU | principal alavanca de throughput |
| `--eval-batch-size N` | batch de validação/teste; `0` reutiliza o de treino | aumente se houver memória livre |
| `--accumulate-steps N` | atualiza o otimizador a cada N batches | somente quando o batch físico não cabe; batch efetivo = físico × N |
| `--num-workers N` | subprocessos do DataLoader | comece em `0`; meça antes de aumentar |
| `--prefetch-factor N` | batches preparados por worker | vale somente com `num-workers > 0` |
| `--persistent-workers` | reaproveita workers entre épocas | válido somente com `num-workers > 0` |
| `--fast` | habilita kernels rápidos/TF32 e desabilita determinismo estrito | benchmark de desempenho, não comparação bit a bit |
| `--no-save` / `--no-track` | desabilita artefatos / rastreio | testes rápidos e diagnósticos |
| `--promote` | tenta atualizar o campeão em `outputs/registry/` | exige tracking e checkpoint; use após analisar o experimento |

### 4. Leitura dos resultados e promoção

O registry local fica em `outputs/registry/<dataset>/<model_type>/champion.json`. A promoção é
explícita via `--promote`, mas a decisão é auditável: a chave exige mesma cena, arquitetura,
modo de entrada, patch, PCA, split e seed; o candidato vence por ordem **AA → kappa → OA**.
Segmentador, backend e batch podem variar, pois são os fatores que se quer comparar. Se a chave
for diferente, o registry não sobrescreve o campeão.

```bash
.venv/bin/python scripts/run_experiment.py -d PU --model cnn2d --segmenter slic \
  --patch-size 5 --batch-size 256 --backend thread --promote
```

Além do registry, anote no TCC `run-id`, commit, tempo e caminho do checkpoint. Nunca compare
como equivalentes execuções cujo `data.json`, split, `two_input` ou determinismo sejam diferentes.

Quando o projeto precisar de armazenamento compartilhado, o diretório de uma execução pode ser
copiado integralmente para S3/EFS ou importado por MLflow/DVC. A estrutura local evita depender
de credenciais e servidor durante o desenvolvimento, mas preserva os metadados que essas
ferramentas exigem.

### 5. Referência das funções adicionadas

| função | responsabilidade |
|---|---|
| `create_run(config)` | cria `run-id`, diretório e metadados que existem antes do treino |
| `array_fingerprint(array)` | gera hash SHA-256 de shape, dtype e bytes do array |
| `data_fingerprint(data)` | reúne hashes dos tensores que chegam ao modelo e dos splits |
| `result_metrics(result)` | converte o resultado Python para um dicionário JSON estável |
| `record_run(context, result, artifacts)` | grava dados/métricas e conclui o manifesto com sucesso |
| `fail_run(context, error)` | conclui o manifesto como falho, preservando a causa |
| `run_epoch(..., gradient_accumulation_steps=N)` | executa treino/avaliação e, no treino, normaliza corretamente o gradiente da janela acumulada |
| `build_loaders(data, config, device)` | materializa tensores, habilita memória pinada em CUDA e constrói loaders de treino/validação/teste |
| `validate_prepared_data(data)` | bloqueia dados desalinhados, incompletos ou não finitos antes do treino |
| `promote_candidate(context, result, checkpoint)` | atualiza o campeão local somente quando métricas e condições permitirem |

### 6. Automação contínua

O workflow [ci.yml](.github/workflows/ci.yml) executa em todo `push` e `pull request`. Ele usa
PyTorch CPU e roda a suíte sintética: não baixa datasets, não exige GPU e verifica contrato de
dados, modelos 1D/2D/3D, segmentadores, checkpoint, tracker e registry. Mantenha esse workflow
verde antes de aceitar mudança de arquitetura ou de MLOps.

## Baseline × nossa implementação

`hsi.variants` define as duas versões que o trabalho compara:

- **`BASELINE`** — o fluxo da implementação de referência (Souza, 2023): estritamente
  sequencial, com os hiperparâmetros do notebook original;
- **`OPTIMIZED`** — a nossa: mesmo modelo e mesmos hiperparâmetros, preparação paralelizada.

Os hiperparâmetros são idênticos **de propósito**. É isso que torna a comparação
interpretável: qualquer diferença de tempo vem da estratégia de execução, e as métricas de
qualidade têm de sair iguais. Se divergirem, a otimização mudou o que o código calcula — é
bug, não ganho, e o script sinaliza isso com código de saída diferente de zero.

```bash
.venv/bin/python scripts/compare_variants.py -d PU -d IP --json outputs/comparacao.json
```

Medição em PaviaU (60 épocas máx., early stopping na 21ª):

| etapa | baseline | nossa impl. | speedup |
|---|---|---|---|
| preparação | 1,344 s | 1,080 s | 1,24× |
| treino | 58,880 s | 60,109 s | 0,98× |
| **total** | **64,242 s** | **61,636 s** | **1,04×** |

| métrica | baseline | nossa impl. |
|---|---|---|
| OA | 0,976309 | 0,976309 |
| AA | 0,957918 | 0,957918 |
| kappa | 0,968545 | 0,968545 |

Classificação idêntica — como tem de ser. A diferença no treino é ruído de medição, não
efeito da mudança: as duas variantes treinam exatamente o mesmo modelo, na mesma ordem de
batches, e chegam aos mesmos pesos.

## Resultados de referência

PaviaU, `two_input=True`, `seed=42`, modo determinístico, RTX 5070 Laptop:

```
Early stopping na época 21 de 60 (melhor val_loss: 0.0561)
Tempo total de treino: ~59s | média por época: ~2.8s

OA:    0.9763092269326683
AA:    0.9579177506866148
kappa: 0.9685447509130496
```

Saídas em `outputs/runs/<run-id>/`: curvas de perda e acurácia em PDF, checkpoint `.pt`,
métricas, configuração, hash dos dados e manifesto da execução.

## Reprodutibilidade

Duas execuções na mesma máquina dão resultado **bit a bit idêntico** — verificado comparando
histórico de perda por época, OA/AA/kappa e os tensores de peso. Isso depende de quatro
coisas, todas configuradas no código:

- `random_state=42` no PCA e nos splits, `seed_everything` em NumPy e PyTorch;
- `seed_everything` chamado **imediatamente antes** de construir o modelo — é a construção que
  consome o RNG global e define os pesos iniciais;
- `generator` semeado no DataLoader de treino, e `fit` repõe essa semente antes da primeira
  época. Sem isso, apenas *inspecionar* um batch antes de treinar mudaria a ordem dos batches
  e, com ela, o resultado;
- `deterministic=True`: `cudnn.deterministic=True`, `cudnn.benchmark=False` e TF32 desligado.
  Com `benchmark=True` o cuDNN escolhe o kernel mais rápido por shape e a escolha pode variar;
  TF32 é determinístico, mas trunca a mantissa.

Entre GPUs de arquiteturas diferentes ainda há variação nas últimas casas decimais, porque a
ordem de redução dos kernels muda. Ao comparar execuções, mantenha fixos cena, semente,
`two_input` e máquina.

## Testes

```bash
.venv/bin/python tests/test_hsi.py     # ou: python -m pytest tests/
```

43 testes, em segundos, sem GPU e sem os `.mat` baixados — usam cenas sintéticas. Cobrem o
que era fácil de quebrar na refatoração: filtro de background, split estratificado (partição
sem sobreposição, determinismo, estratificação), shapes que atravessam o `Conv1d`, a lógica
da parada antecipada, as métricas, os ramos do paralelismo e a equivalência das variantes.

## Datasets

Obtidos por `scripts/download_data.py`, com sha256 pinado e verificação de conteúdo (shape e
número de classes contra a especificação publicada), para que um mirror que devolva página de
erro seja rejeitado.

| código | cena | dimensões | classes | PCA |
|---|---|---|---|---|
| `IP` | Indian Pines (corrected) | 145×145×200 | 16 | 30 |
| `PU` | Pavia University | 610×340×103 | 9 | 15 |
| `SA` | Salinas (corrected) | 512×217×204 | 16 | 15 |

Fonte canônica: [Hyperspectral Remote Sensing Scenes](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes),
Grupo de Inteligencia Computacional, UPV/EHU. A EHU costuma responder HTTP 403 a download
automatizado; o script tenta a EHU primeiro e cai para um mirror no GitHub.

```bash
.venv/bin/python scripts/download_data.py            # os três (~67 MB)
.venv/bin/python scripts/download_data.py -d PU      # só PaviaU
.venv/bin/python scripts/download_data.py --check    # verificar integridade
```

Para dados num volume separado (EBS/EFS), sem editar código:

```bash
export HSI_DATA_DIR=/mnt/dados/dataHSI
```

## Solução de problemas

| sintoma | causa e solução |
|---|---|
| `AttributeError: module 'cv2' has no attribute 'ximgproc'` | está instalado `opencv-python` em vez de `opencv-contrib-python-headless`. Desinstale o primeiro |
| `no kernel image is available for execution on the device` | wheel CUDA antigo demais para a GPU. RTX 50xx precisa de cu128+ |
| `pacote hsi não encontrado` no notebook | o Jupyter foi iniciado fora do repositório |
| arquivos `.mat` ausentes | `python scripts/download_data.py`; o pipeline também baixa sozinho |
| resultados diferentes entre execuções | `deterministic=False`, semente alterada, ou GPU de arquitetura diferente |
| a célula de versões acusa divergência | informativo, não bloqueia. `uv pip install -r requirements.txt` para igualar |

## Licença e créditos

Trabalho acadêmico — UNIVALI, Escola Politécnica, Engenharia de Computação.
A implementação original é de Sydney Matheus de Souza (2023); o notebook dela está preservado
em `src/TCC3_SydneySouza_01-10-2023.ipynb`.
