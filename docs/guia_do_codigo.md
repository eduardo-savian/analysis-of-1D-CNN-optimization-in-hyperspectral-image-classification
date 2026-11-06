# Guia do código, parte 1: da configuração ao resultado de uma execução

Esta parte cobre os arquivos que toda execução percorre: `config.toml`, `src/hsi/__init__.py`, `config.py`, `ambiente.py`, `dados.py`, `segmentadores.py`, `modelo.py`, `treino.py`, `pipeline.py` e `src/executar.py`. A parte 2 trata de `hpo.py`, `paralelismo/`, `experimentos/` e `figuras.py`.

Todas as formas e contagens abaixo foram medidas no código atual (commit 57f4b6d), rodando o pacote nesta máquina, que tem GPU CUDA.

---

## Mapa rápido

| Arquivo | O que faz |
|---|---|
| `config.toml` (raiz) | Guarda todo número que muda uma execução: caminhos, PCA, hiperparâmetros, segmentadores, busca e experimentos |
| `hsi/config.py` | Lê o toml uma única vez e define `ExperimentConfig`, a "ficha" de uma execução |
| `hsi/ambiente.py` | Confere versões, escolhe GPU ou CPU, fixa sementes e liga o modo determinístico |
| `hsi/dados.py` | Leva da cena em disco aos DataLoaders: PCA, segmentação, divisão |
| `hsi/segmentadores.py` | SEEDS, LSC e Watershed sobre uma composição RGB da cena |
| `hsi/modelo.py` | A CNN 1D (do projeto e do original) com a fusão do id do superpixel |
| `hsi/treino.py` | Laço de épocas, parada antecipada, predição, OA/AA/Kappa |
| `hsi/pipeline.py` | `run_experiment`: liga tudo, da configuração às métricas |
| `src/executar.py` | Roda um experimento pelo terminal, sem Jupyter |

---

## Uma execução do começo ao fim

```python
from hsi.config import ExperimentConfig
from hsi.pipeline import run_experiment
r = run_experiment(ExperimentConfig(dataset="IP"), verbose=True)
```

Essa chamada tem a mesma configuração da linha "IP, projeto" do experimento 01 (semente 42). Por isso, na mesma máquina, o resultado esperado é **OA 0,9142, AA 0,8625, Kappa 0,9020, 52 épocas, melhor época 42**, conforme `experimentos/resultados/01_linha_base.csv`. O notebook diz que leva cerca de 1 minuto.

```
ExperimentConfig(dataset="IP")
  └─ pipeline.run_experiment
       ├─ ambiente.select_device, configure_determinism
       ├─ dados.prepare_data
       │    ├─ load_scene ......... cubo (145,145,200) uint16, gt (145,145)
       │    ├─ pixel_table ........ pixels (21025,200), rótulos (21025,)
       │    ├─ só rotulados ....... espectros (10249,200), classes 0..15
       │    ├─ spectral_branch .... PCA 30 -> (10249,30)
       │    ├─ spatial_branch ..... PCA 3 -> RGB (145,145,3) uint8 -> SEEDS -> mapa (145,145), 81 regiões
       │    ├─ split_indices ...... treino 8711 / val 769 / teste 769
       │    └─ check_data
       ├─ dados.build_loaders ..... lote: x (32,1,30), seg (32,1), y (32,)
       ├─ ambiente.seed_everything(42)
       ├─ modelo.build_model ...... CNN1D + CrossEntropyLoss + Adam
       ├─ treino.fit .............. até 200 épocas, parada antecipada
       ├─ treino.predict .......... classes do teste (769,)
       └─ treino.metrics .......... OA, AA, Kappa, matriz 16x16
```

**Passo 1, `config.py:ExperimentConfig`.** Os campos que você não passa são preenchidos com a seção `[execucao]` do toml: `segmenter="seeds"`, `cnn="projeto"`, `batch_size=32`, `learning_rate=1e-3`, `dropout=0.5`, `weight_decay=0.0`, `epochs=200`, `patience=10`, `min_delta=1e-3`, `adam_eps=1e-8`, `seed=42` e `target=None`. Como `pca_components` vem 0, o `__post_init__` coloca 30, que é o valor de IP em `[dados.pca]`. Atenção: a cena padrão do toml é **PU**, então `ExperimentConfig()` sem argumentos roda Pavia.

**Passo 2, `pipeline.py:run_experiment`.** Se nenhum `device` for passado, chama `ambiente.select_device` (usa `cuda` se houver GPU). Depois chama `ambiente.configure_determinism()`.

**Passo 3, `dados.py:prepare_data(config)`.** O backend padrão é `"sequential"`.
- `load_scene("IP")` lê dois `.mat` de `data/dataHSI`. Entram dois arquivos e saem o cubo `(145, 145, 200)` uint16 (altura, largura, bandas) e a verdade terrestre `(145, 145)` com valores de 0 a 16, em que 0 significa pixel sem rótulo.
- `pixel_table` achata a imagem em uma linha por pixel: `(21025, 200)` e `(21025,)`. A linha `i` corresponde ao pixel `(i // 145, i % 145)`.
- `np.flatnonzero(rotulos_cena > 0)`, sobre a verdade terrestre achatada, separa os **10.249** pixels rotulados. Saem `espectros (10249, 200)` e `rotulos` de 0 a 15, em int64 (o código subtrai 1 da classe).
- `spectral_branch` chama `apply_pca(espectros, 30, 42)` e devolve `(10249, 30)` em float64. Cada coluna é uma componente principal com variância 1 (PCA branqueado).
- `spatial_branch` faz um PCA de 3 componentes sobre **todos** os 21.025 pixels, o que dá `(21025, 3)`. Em seguida, `rgb_composite` monta `(145, 145, 3)` uint8 e `segment(rgb, "seeds", params)` devolve o mapa `(145, 145)` int32 com **81 regiões** (ids de 0 a 80).
- `split_indices(rotulos, 0.15, 0.5, 42)` devolve 8.711 índices de treino, 769 de validação e 769 de teste. Esses índices apontam para o vetor de 10.249 amostras, não para a imagem.
- `subsample_train` não faz nada com `train_fraction=1.0`.
- O resultado é um dicionário com:

| chave | forma (IP) | conteúdo |
|---|---|---|
| `espectros` | (10249, 30) | espectros reduzidos |
| `rotulos` | (10249,) | classe 0..15 |
| `segmentos` | (10249,) | id do superpixel de cada pixel rotulado (aparecem ids de 0 a 79; 63 das 81 regiões têm pixel rotulado) |
| `treino`, `val`, `teste` | (8711,), (769,), (769,) | índices |
| `coords` | (10249, 2) | (linha, coluna) de cada amostra, usado na partição, na fusão concorrente e nas figuras |
| `forma` | (145, 145) | tamanho da cena |
| `n_classes` | 16 | |

- Por fim, `check_data` confere o dicionário (detalhes na seção do arquivo).

**Passo 4, `dados.py:build_loaders`.** Monta um DataLoader por partição. Cada lote é uma tupla `(x, seg, y)`:
- `x (32, 1, 30)` float32. O `unsqueeze(1)` cria o eixo de canal, porque o `Conv1d` espera o formato (lote, canais, comprimento);
- `seg (32, 1)` float32, o id como número;
- `y (32,)` int64.

O treino tem 273 lotes por época (8711/32, arredondado para cima) e é embaralhado. Validação e teste não são embaralhados.

**Passo 5, `ambiente.py:seed_everything(42)`.** É chamado logo antes de criar o modelo, porque é a criação do modelo que consome o gerador aleatório do PyTorch (pesos iniciais).

**Passo 6, `modelo.py:build_model`.** Cria `CNN1D(30, 16, 0.5, "projeto")` com 166.416 parâmetros, junto com `CrossEntropyLoss` e `Adam(lr=1e-3, eps=1e-8, weight_decay=0)`. O caminho de um lote pela rede (as quatro convoluções usam padding "same"):

| camada | saída |
|---|---|
| entrada | (32, 1, 30) |
| Conv1d 16, kernel 9 + ReLU | (32, 16, 30) |
| MaxPool 2 | (32, 16, 15) |
| Conv1d 32, kernel 3 + ReLU | (32, 32, 15) |
| MaxPool 2 | (32, 32, 7) |
| Conv1d 64, kernel 3 + ReLU | (32, 64, 7) |
| MaxPool 2 | (32, 64, 3) |
| Conv1d 128, kernel 3 + ReLU | (32, 128, 3) |
| Flatten | (32, 384) |
| `fundir`: concatena o id | **(32, 385)** |
| Linear 256 + ReLU + Dropout 0,5 | (32, 256) |
| Linear 128 + ReLU + Dropout 0,5 | (32, 128) |
| Linear 16 | (32, 16), os **logits** |

**Passo 7, `treino.py:fit`.** Cada época roda `run_epoch` no treino (273 passos de gradiente) e depois na validação (sem gradiente e sem dropout). O treino para quando passam 10 épocas sem a acurácia de validação melhorar mais de 0,001 e volta aos pesos da melhor época. Em IP, a melhor época foi a 42 e o treino parou na 52.

**Passo 8, `treino.py:predict`.** Aplica softmax e argmax e devolve as classes do teste, `(769,)`, na mesma ordem de `dados["teste"]`.

**Passo 9, `treino.py:metrics`.** Calcula a matriz de confusão 16x16, OA, AA e Kappa.

**Passo 10.** `run_experiment` devolve um dicionário com `config`, `dados`, `modelo`, `previsto`, `historico`, `epocas`, `tempo_treino`, `motivo_parada`, `melhor_epoca`, `OA`, `AA`, `Kappa` e `confusao`.

### As três cenas lado a lado (CNN do projeto)

| cena | cubo | rotulados | classes | PCA | treino / val / teste | regiões SEEDS | entrada das densas | parâmetros |
|---|---|---|---|---|---|---|---|---|
| IP | 145x145x200 | 10.249 | 16 | 30 | 8.711 / 769 / 769 | 81 | 385 | 166.416 |
| PU | 610x340x103 | 42.776 | 9 | 15 | 36.359 / 3.209 / 3.208 | 45 | 129 | 99.977 |
| SA | 512x217x204 | 54.129 | 16 | 15 | 46.009 / 4.060 / 4.060 | 96 | 129 | 100.880 |

Com 15 componentes, o vetor achatado em PU e SA tem só 128 valores (128 filtros x 1 posição).

---

## Arquivo por arquivo

### `config.toml`

É o único lugar com números de execução. As seções são:
- `[caminhos]`: dados, resultados e a pasta de figuras da monografia.
- `[dados]`: `semente = 42` e `pca = { IP = 30, PU = 15, SA = 15 }`. Essa semente vale **só para o PCA e para a divisão**.
- `[execucao]`: os padrões de `ExperimentConfig`, inclusive a cena (`PU`), `test_ratio`, `valid_ratio` e `train_fraction`. Juntam os hiperparâmetros do notebook de Souza (2023) (lote 32, taxa 1e-3, dropout 0,5), a CNN do projeto e a parada deste trabalho (200 épocas, paciência 10, `min_delta` 1e-3). A chave `seed` daqui muda só o treino: pesos iniciais, ordem dos lotes e dropout.
- `[segmentadores]`: a lista dos três segmentadores e os parâmetros do SEEDS por cena, iguais aos do notebook original. LSC e Watershed têm listas de três valores (0,7x, 1x e 1,35x as regiões do SEEDS): no LSC, `region_size` por cena e `ratio` (`num_iterations` fica fixo em 10); no Watershed, `markers` por cena. O do meio é o padrão e a busca usa os três.
- `[hpo]` e `[experimentos.NN]`: ver parte 2.

### `src/hsi/__init__.py`

Só tem a docstring com o mapa do pacote, sem nenhum import. Por isso `import hsi.config` não carrega o PyTorch nem o OpenCV.

### `src/hsi/config.py`

- `RAIZ = Path(__file__).resolve().parents[2]` sobe de `src/hsi/config.py` até a raiz do repositório. Os caminhos funcionam de qualquer diretório, seja pelo notebook ou pelo terminal.
- `CONFIG = tomllib.load(...)` lê o toml **uma vez**, na importação (`tomllib` faz parte do Python desde a versão 3.11).
- Constantes: `DADOS`, `RESULTADOS`, `FIGURAS`, `DATASETS`, `SEGMENTERS`, `PCA_COMPONENTS`, `SEMENTE_DADOS`.
- `ExperimentConfig` é uma dataclass **congelada** (`frozen=True`): depois de criada, não pode ser alterada. Assim nenhuma parte do código muda a configuração no meio de uma execução, e a busca de hiperparâmetros cria uma configuração nova a cada trial (com `dataclasses.replace`).
  - `__post_init__` valida a cena, o segmentador e a CNN, e troca `pca_components=0` pelo valor da cena. Usa `object.__setattr__` porque a classe é congelada.
  - `segmenter_overrides` são ajustes sobre os parâmetros padrão do segmentador, preenchidos pela busca.
  - `target` vem de `_PADRAO.get("target")`. Sem a chave no toml, o valor é `None` e a meta fica desligada.

### `src/hsi/ambiente.py`

- `check_versions()` imprime as versões instaladas contra `EXPECTED_VERSIONS` (torch 2.13.0, scikit-learn 1.9.0, opencv-contrib-python-headless 5.0.0.93, optuna 4.9.0...) e devolve a lista das que diferem. Ignora o sufixo `+cu130` do torch de GPU e avisa se faltar `cv2.ximgproc`: SEEDS e LSC só existem no **opencv-contrib**.
- `select_device()` escolhe `cuda` se houver GPU e `cpu` caso contrário.
- `configure_determinism()`, linha a linha:
  - `cv.setNumThreads(1)`: o LSC com várias threads devolvia rótulos diferentes para a **mesma imagem**. Com 32 threads mudavam de 188 a 322 pixels entre chamadas em IP e de 142 a 330 em PU; com 1 thread, nenhum. O custo foi medido (`experimentos/resultados/medicoes_avulsas.md`, `LSC_THREADS_MS`): IP passa de 14,8 para 35,6 ms e PU de 83,6 para 289,7 ms por segmentação, o que é desprezível perto do treino.
  - `CUBLAS_WORKSPACE_CONFIG` (`:4096:8`, só se ainda não estiver definida): exigida pelo cuBLAS para fazer multiplicações de matriz determinísticas na GPU.
  - `cudnn.benchmark = False` e `cudnn.deterministic = True`: impedem o cuDNN de escolher, por cronometragem, um algoritmo que pode mudar de uma execução para outra.
  - TF32 desligado (nas multiplicações de matriz e no cuDNN): o TF32 reduz a precisão das multiplicações em GPUs recentes, e com ele desligado a conta fica em float32 cheio.
  - `use_deterministic_algorithms(True, warn_only=True)`: se alguma operação não tiver versão determinística, o PyTorch avisa em vez de falhar.
- `seed_everything(seed)` semeia o NumPy e o PyTorch (CPU e GPU).

**As duas sementes.** `SEMENTE_DADOS` (42, fixa) define o PCA e quais pixels caem em treino, validação e teste. `config.seed` define pesos iniciais, ordem dos lotes, dropout e, se houver, a subamostra do treino. Quando um experimento roda 5 sementes, o conjunto de teste é sempre o mesmo e só o treino varia. Assim, segmentadores e configurações são comparados nas mesmas amostras de teste.

### `src/hsi/dados.py`

- `SCENE_FILES`, `load_scene` e `pixel_table`: leitura e achatamento, como descrito no passo 3.
- `apply_pca(x, n, seed)` **emula o `PCA(whiten=True)` do sklearn anterior à 1.5**, que era a versão do Colab do Sydney. Os passos são: centralizar em float64, `randomized_svd` com `flip_sign=True` (o sinal de cada componente é decidido por U, como no sklearn antigo), projetar e dividir pelo desvio de cada componente (`sqrt(s²/(n-1))`), que é o branqueamento. A emulação é necessária porque, a partir da 1.5, o sklearn passou a escolher outro algoritmo para dados "altos" (muitas linhas, poucas colunas) e a decidir o sinal por V. Algumas componentes saem com o sinal trocado. Para a rede, uma componente invertida não é informação pior, mas muda os números em relação ao Colab. No PCA de 3 componentes, o efeito é mais sério: um canal invertido vira `255 − valor` na composição RGB e o SEEDS segmenta de outro jeito em IP e PU. Com a emulação, os dados ficam **idênticos aos do Colab**.
- `spectral_branch` aplica o PCA só nos pixels rotulados. `spatial_branch` aplica o PCA de 3 componentes na cena inteira, porque o segmentador precisa da imagem completa. Os dois ramos são calculados antes da divisão e não usam rótulos, como no original.
- `split_indices` sorteia em duas etapas, como no original:
  1. separa 15% das amostras, estratificando pelas classes inteiras;
  2. divide esses 15% ao meio, **estratificando pelo one-hot**, e o primeiro retorno vira o teste.

  O original fazia o one-hot antes da segunda divisão. Quando recebe `y` em duas dimensões, o sklearn transforma cada linha em texto (`"0.0 1.0 0.0 ..."`) para definir as classes. Os grupos continuam os mesmos, mas a **ordem** das classes muda (ordem de texto em vez de ordem numérica). Como o sorteio estratificado consome o gerador aleatório classe por classe, nessa ordem, outra ordem dá outros pixels. Estratificando pelos inteiros, a validação e o teste seriam diferentes dos do Colab. A conferência foi feita em PU: são as mesmas 3.208 amostras de teste.
- `subsample_train` fica com uma fração do treino, sorteada por classe (no mínimo 1 amostra por classe) com `config.seed`. Validação e teste não mudam. É usada no experimento 04 (10%).
- `prepare_data(config, backend)`: com `backend="thread"`, os dois ramos rodam ao mesmo tempo: o espacial numa thread do pool e o espectral na thread que chamou. Isso funciona porque a álgebra linear do PCA (NumPy e SciPy) e o OpenCV liberam o GIL no trabalho pesado. Essa opção é usada por `paralelismo/particao.py`; `run_experiment` usa a versão sequencial.
- `check_data` faz quatro conferências baratas antes do treino: nenhum NaN ou infinito nos espectros; classes de 0 a C-1 sem buracos; uma linha por amostra em espectros, segmentos e rótulos; e nenhuma amostra repetida entre treino, validação e teste.
- `build_loaders`:
  - o id vira float `(N, 1)`;
  - o DataLoader de treino tem um **gerador próprio** semeado com `config.seed`, para a ordem dos lotes não depender de nada mais;
  - `pin_memory` fica ligado na GPU;
  - validação e teste não são embaralhados, e é isso que faz `predict` sair alinhado com `rotulos[teste]`.

### `src/hsi/segmentadores.py`

- `SEGMENTER_PARAMS` é montado na importação a partir do toml. O SEEDS usa os parâmetros por cena. LSC e Watershed usam o **valor do meio** (`[1]`) de cada lista; por exemplo, em IP o LSC fica com `region_size=16`, `ratio=0.075` e 10 iterações, e o Watershed com 116 marcadores.
- `rgb_composite` normaliza cada componente pelo mínimo e máximo (min-max) e converte para uint8. Sai `(H, W, 3)`.
- `segmenter_params(config)` junta os parâmetros padrão com `segmenter_overrides`; os overrides ganham.
- `segment(rgb, algoritmo, p)` devolve o mapa `(H, W)` int32:
  - SEEDS: `createSuperpixelSEEDS(w, h, canais, num_superpixels, num_levels, prior, num_histogram_bins, double_step)` e depois `iterate(rgb, num_iterations)`;
  - LSC: `createSuperpixelLSC(rgb, region_size, ratio)` e depois `iterate(num_iterations)`;
  - Watershed: `_watershed(rgb, markers)`.
- `_watershed` distribui os marcadores numa grade com a proporção da cena. Os marcadores da moldura se perdem, porque o `cv.watershed` marca a borda da imagem com -1. As linhas entre bacias (-1) viram a região 0, e `np.unique(..., return_inverse=True)` renumera os ids em sequência. O RGB é convertido para BGR porque o OpenCV espera BGR.
- **Calibração.** LSC e Watershed foram ajustados para gerar o mesmo número de regiões que o SEEDS. Assim, a comparação entre segmentadores mede a **forma** das regiões, não a quantidade, e a escala do id, que entra cru na rede, fica comparável. Contagem medida agora:

| cena | SEEDS | LSC | Watershed (inclui a região 0 das linhas) |
|---|---|---|---|
| IP | 81 | 81 | 82 |
| PU | 45 | 45 | 46 |
| SA | 96 | 96 | 97 |

  Na medição registrada em `medicoes_avulsas.md` (`SEGMENTADORES`, 04/11, mediana de 14 chamadas, OpenCV com 1 thread), uma segmentação leva de 1,2 a 10,6 ms no Watershed, de 37,7 a 280,2 ms no LSC e de 33,3 a 599,1 ms no SEEDS, conforme a cena. O ramo espacial custa pouco perto do treino.
- O número de regiões do SEEDS anda em degraus (em IP: 16, 81 ou 324). Por isso a busca varia também as iterações e o `prior` (parte 2).

### `src/hsi/modelo.py`

- `KERNELS = (9, 3, 3, 3)` são os kernels do original, usados nas duas arquiteturas. `ARQUITETURAS` define as duas:

| | projeto (usada no trabalho) | original (só no exp. 01) |
|---|---|---|
| filtros | 16, 32, 64, 128 | 16, 32, 64, 64 |
| pooling | max-pooling 2 após as três primeiras | nenhum |
| padding | "same" | nenhum |
| densas | 256, 128 | 128, 64 |
| parâmetros IP / PU / SA | 166.416 / 99.977 / 100.880 | 160.912 / 37.577 / 38.032 |

- **Por que padding "same" na CNN do projeto.** Sem padding, cada convolução também encurtaria o espectro e, somada aos poolings, o comprimento zeraria:
  - IP (30): 30 → conv9 → 22 → pool → 11 → conv3 → 9 → pool → 4 → conv3 → 2 → pool → 1 → conv3 → **negativo**.
  - PU/SA (15): 15 → conv9 → 7 → pool → 3 → conv3 → 1 → pool → **0**.

  Com "same", só os três poolings encurtam o espectro: 30 → 15 → 7 → 3, ou 15 → 7 → 3 → 1. Daí vem `MIN_COMPONENTS`: o projeto precisa de pelo menos 8 componentes (8 → 4 → 2 → 1), e o original de pelo menos 15, porque consome 8 + 2 + 2 + 2 = 14 posições. Abaixo do mínimo, `CNN1D` levanta `ValueError`. O capítulo 3 prevê filtros, pooling e densas, mas não fixa kernels nem padding; essas são decisões de implementação.
- `CNN1D.__init__` monta as convoluções, calcula o comprimento final e dimensiona a primeira densa como `canais * comprimento + 1`. O "+1" é o id. Fica 385 em IP no projeto e 1.025 no original.
- **Como o id entra na rede (fusão tardia).** As convoluções veem só o espectro. O id do superpixel é **um número cru** (0 a 80 em IP), sem normalização e sem one-hot, concatenado ao vetor achatado por `fundir` (`torch.cat(..., dim=1)`). Só as densas veem as duas informações juntas. É a fusão do original, mantida porque os ganhos são medidos contra a configuração do Sydney com fusão SEEDS. Nos três segmentadores, os ids nascem de uma grade percorrida linha a linha, então regiões vizinhas tendem a ter ids próximos e o id funciona como uma posição grosseira na cena. `fundir` é um método separado porque a fusão concorrente (parte 2) protege só esse passo com a trava.
- O dropout vem depois de cada densa oculta. A saída são **logits**, sem softmax, porque `CrossEntropyLoss` já aplica o log-softmax.
- A inicialização é Xavier uniforme com viés zero, o padrão do Keras usado pelo original.
- `build_model` cria a perda e o Adam. O `weight_decay` do Adam do PyTorch é a regularização L2 somada ao gradiente, não a do AdamW. O `eps` é 1e-8, o padrão do PyTorch; o do Keras é 1e-7.

### `src/hsi/treino.py`

- `run_epoch(modelo, loader, perda, device, otimizador=None)` treina se houver otimizador e só avalia se não houver. `modelo.train(treinando)` liga ou desliga o dropout. A perda é acumulada ponderada pelo tamanho do lote. A função devolve (perda média, acurácia).
- `fit`:
  - ressemeia o gerador do loader (`manual_seed(config.seed)`), porque qualquer iteração sobre o loader antes do treino (por exemplo, para conferir a forma de um lote) adiantaria o sorteio;
  - conta como melhora só `val_acc > melhor + min_delta`. Uma melhora de até 0,001 não zera a paciência nem salva pesos;
  - guarda os melhores pesos com `copy.deepcopy(state_dict())`. Sem a cópia, o "melhor" seria só um apontador para os pesos que continuam treinando;
  - para por `patience` épocas sem melhora ou quando atinge `target`, se houver meta. Nos dois casos **restaura os melhores pesos**. Se chegar a 200 épocas sem parar, ficam os da última, mas na prática a paciência dispara bem antes (52 épocas em IP);
  - devolve `historico` (loss, accuracy, val_loss, val_accuracy por época), `tempo` (só o treino, sem preparo dos dados), `melhor_epoca` e `motivo`.
- **Por que parar pela acurácia de validação.** O original parava pela perda de validação, com paciência de 5 e teto de 60 épocas. Aqui, a mesma métrica, a acurácia de validação, governa a parada e a escolha de hiperparâmetros na busca.
- `predict` coloca o modelo em `eval`, desliga o gradiente e aplica softmax e argmax. O softmax não muda o argmax; foi mantido só para repetir o cálculo do original.
- `metrics` calcula:
  - OA: acertos / total;
  - AA: média das acurácias por classe, que pesa as classes pequenas tanto quanto as grandes;
  - Kappa: concordância descontado o acaso.

  Uma classe prevista que não tem amostras no teste entra na matriz com suporte 0 e conta como acurácia 0 na AA.

### `src/hsi/pipeline.py`

`run_experiment(config, verbose=False, device=None)` segue a ordem dos passos 1 a 10. Dois detalhes:
- `seed_everything` vem **depois** de preparar os dados. O preparo não usa o gerador global do PyTorch, e semear logo antes do modelo faz os pesos dependerem só de `config.seed`.
- Cada chamada refaz PCA e segmentação. Na busca, cada trial refaz o ramo espacial, o que custa de milissegundos a menos de 1 s.

Quem chama `run_experiment`: o notebook, `hpo.py` (e, por ele, as buscas dos experimentos 03 e 04), os experimentos 01, 02, 03, 05, 06 e 08 e `figuras.py`.

### `src/executar.py`

- Uso: `.venv/bin/python src/executar.py 03 PU SA`. `zfill(2)` aceita "3" como "03". Sem cena, o experimento usa as cenas de `[experimentos.NN]`. Com um número inválido, imprime o uso e a lista de experimentos.
- Coloca `src/` no caminho de importação, então funciona sem instalar o pacote.
- `matplotlib.use("Agg")` gera figuras sem tela.
- `warnings.filterwarnings("ignore")` também esconde os avisos do `warn_only` do modo determinístico.
- `torch.set_num_threads(1)` fica no nível do módulo porque os processos filhos (spawn) reexecutam este arquivo, e cada worker fica com uma thread. Os experimentos que precisam de mais threads ajustam isso por conta própria; o 08, por exemplo, sobe para `threads_treino = 16` só durante o treino do modelo. Na GPU o treino vai para a placa de qualquer forma.

---

## Como o `config.toml` é lido e como mudar um parâmetro

**Leitura.** O `config.toml` é lido uma única vez, quando `hsi.config` é importado. Outros módulos copiam suas seções também na importação: `segmentadores.SEGMENTER_PARAMS`, `hpo._H` e `hpo.ORIGINAL`, e o `C = CONFIG["experimentos"]["NN"]` dos experimentos 02 a 08 (o 01 lê a sua seção de `CONFIG` ao rodar). Os padrões de `ExperimentConfig` também ficam fixados nesse momento. **Por isso, depois de editar o toml, é preciso reiniciar o kernel do Jupyter** (o `autoreload` do notebook só recarrega arquivos `.py`), ou abrir um processo novo no terminal.

**Onde mudar cada coisa:**

| quero... | faça |
|---|---|
| mudar algo em uma execução | `ExperimentConfig(dataset="IP", dropout=0.35, seed=11)` |
| derivar de uma configuração existente | `dataclasses.replace(config, learning_rate=3e-4)` (`config.dropout = ...` dá `FrozenInstanceError`) |
| trocar o segmentador ou um parâmetro dele | `ExperimentConfig(dataset="IP", segmenter="lsc", segmenter_overrides={"region_size": 19})` |
| mudar o PCA | `pca_components=20` na execução, ou `[dados.pca]` para todas (mínimo 8 no projeto e 15 no original) |
| ligar a meta de validação | `target=0.95` na execução, ou `target = 0.95` em `[execucao]` |
| mudar cenas, sementes ou trials de um experimento | `[experimentos.NN]` |
| mudar o padrão de todas as execuções | `[execucao]` |

**Cuidado:** `hpo.ORIGINAL` lê os eixos da busca a partir de `[execucao]`. Mexer em taxa, dropout, weight decay ou lote ali muda o que a busca e o experimento 05 tratam como "configuração original", e um valor que não esteja na lista de `[hpo.espaco]` faz a busca em grade falhar. Mantenha `[execucao]` com os hiperparâmetros do Sydney.

---

## Perguntas que o orientador pode fazer

1. **"Os dados são os mesmos do trabalho do Sydney?"**
   Sim. São os mesmos `.mat`, com PCA emulando o sklearn do Colab (mesmo algoritmo e mesma regra de sinal) e divisão 85/7,5/7,5 com a mesma semente e a mesma estratificação por one-hot. Em PU caem no teste as mesmas 3.208 amostras, e a CNN original tem os mesmos 37.577 parâmetros do modelo Keras.

2. **"O original é Keras e você usa PyTorch. Os números batem?"**
   A arquitetura, a inicialização e os dados batem. Com a CNN original, o experimento 01 deu OA de 0,8960 em IP, 0,9853 em PU e 0,9951 em SA, contra 0,9142, 0,9928 e 0,9965 reportados. A diferença vem de quatro coisas: os frameworks partem de pesos diferentes, a regra de parada mudou (acurácia de validação com paciência 10, contra perda de validação com paciência 5), o teto de épocas é outro (200 contra 60; em IP a CNN original treinou 63 épocas) e o `eps` do Adam também (1e-8 contra 1e-7). É uma execução de cada lado.

3. **"Qual CNN é a linha de base, e ela é melhor que a do Sydney?"**
   A linha de base é a CNN do capítulo 3 com os hiperparâmetros do original. Com a semente 42, em IP ela ganha 1,8 pontos de OA (0,9142 contra 0,8960) e 2,1 de Kappa, mas perde 0,9 de AA (0,8625 contra 0,8717). Em PU e SA fica cerca de 0,12 ponto de OA abaixo. Ainda é uma execução por cena; os experimentos com 5 sementes (02, 03 e 05) é que vão dar a variação.

4. **"Como o superpixel entra na rede?"**
   Pela fusão tardia: o id da região é um único número cru, concatenado ao vetor das convoluções, e só as densas o veem. Uma ablação de 03/09, feita com a CNN do Sydney e 3 sementes e servindo só como indício, mostrou que a rede usa o id: em IP a OA cai de 0,8981 para 0,8552 com os ids permutados, para 0,8223 com o id normalizado e para 0,7898 com um id aleatório por pixel. Um embedding por região chegou a dar 0,9471 em IP, mas mudaria a fusão de referência; por isso saiu do código e pode entrar como trabalho futuro.

5. **"O OpenCV com uma thread não vai contra o pilar de paralelismo?"**
   Não. É uma escolha de reprodutibilidade: o LSC paralelo mudava de 142 a 330 pixels entre chamadas com a mesma imagem. O custo é de milissegundos por segmentação. O paralelismo do trabalho está em outro nível: partição da cena, fusão concorrente e ramos espectral e espacial em threads (parte 2).

6. **"Por que padding 'same' e max-pooling?"**
   O max-pooling está previsto no capítulo 3. Sem padding, convoluções mais poolings zerariam o espectro: com 30 componentes o comprimento ficaria negativo na quarta convolução, e com 15 zeraria no segundo pooling. Com "same", só os poolings encurtam o espectro, e a rede funciona a partir de 8 componentes.

7. **"Como você garante que duas execuções iguais dão o mesmo resultado?"**
   Com quatro mecanismos: a configuração congelada em um objeto só; o modo determinístico (sementes, cuDNN sem benchmark, sem TF32, OpenCV com 1 thread); um gerador próprio para a ordem dos lotes; e `check_data` antes de cada treino. Duas sementes separam o que é fixo (PCA e divisão) do que varia entre repetições (treino).

8. **"O PCA e a segmentação veem as amostras de teste?"**
   Os dois são calculados antes da divisão e não usam rótulos. O PCA espectral roda sobre todos os pixels rotulados e a segmentação sobre a cena inteira, exatamente como no notebook do Sydney. Foi mantido assim para os dados serem os mesmos da referência.

---

## Divergências entre estes arquivos e o texto atual da monografia

- `4_desenvolvimento.tex` diz que o ε = 10⁻⁷ do Adam foi replicado do Keras, mas o `config.toml` usa `adam_eps = 1e-8` desde o commit f84ead6. É preciso alinhar o texto ou o config.
- O mesmo trecho cita teto de 60 épocas. Hoje o padrão é 200 (25 nos experimentos 06 e 07).
- A subseção do treino em `4_desenvolvimento.tex` ainda usa a numeração antiga dos experimentos (ver `experimentos/resultados/NUMERACAO.md`), põe meta de 0,95 em todos os experimentos de segmentadores e de busca e teto de 60 épocas na fusão concorrente. No `config.toml`, só o experimento 04 tem `target = 0.95` (com `train_fraction = 0.10`), e o 08 treina com o teto padrão de 200.
- `tab:linhaBase` (`5_resultados.tex`) ainda mostra os números antigos (IP 0,9051/0,8237/0,8914), que também aparecem citados em `5_resultados.tex` e `apendice_c.tex`. Os novos estão em `experimentos/resultados/01_linha_base.csv`, com as duas CNNs.

---

# Guia do código, parte 2: os três pilares e os experimentos

Esta parte cobre o que foi construído sobre o trabalho do Sydney: a busca de hiperparâmetros, os segmentadores dentro da busca, as duas formas de paralelismo e os oito experimentos. Também mostra como rodar tudo. Os números que vêm de execuções antigas aparecem como **indício**, com a configuração em que foram medidos. Eles não entram na monografia.

## Onde está cada pilar

| Pilar do projeto (cap. 3) | Código | Experimentos |
| --- | --- | --- |
| Otimização de hiperparâmetros | `src/hsi/hpo.py` | 03 (usa a busca), 04, 05 |
| Outros segmentadores | `src/hsi/segmentadores.py` e `[segmentadores]` / `[hpo.seeds]` em `config.toml` | 02, 03 |
| Paralelismo | `src/hsi/paralelismo/particao.py`, `src/hsi/paralelismo/fusao_concorrente.py` e `prepare_data(..., "thread")` em `dados.py` | 06, 07, 08 |
| Linha de base | `pipeline.run_experiment` | 01 |

Todos os experimentos ficam em `src/hsi/experimentos/eNN_*.py`. `infra.py` cuida da gravação e da retomada, e `figuras.py` gera as figuras da monografia a partir dos CSVs.

---

## Pilar 1: busca de hiperparâmetros (`hpo.py`)

### O espaço de busca

São 7 eixos com 3 valores cada, ou seja, 3^7 = **2.187 candidatos**. O valor da configuração original está na coluna "Original" (nem sempre é o do meio).

| Eixo (nome no código) | Valores | Original |
| --- | --- | --- |
| `segmenter` | seeds, lsc, watershed | seeds |
| `override_index` (configuração do segmentador) | 0 = menor, 1 = padrão, 2 = maior | 1 |
| `learning_rate` | 3e-4, 1e-3, 3e-3 | 1e-3 |
| `dropout` | 0,2; 0,35; 0,5 | 0,5 |
| `weight_decay` | 0; 1e-5; 1e-4 | 0 |
| `batch_size` | 32, 64, 128 | 32 |
| `pca_components` | IP 15/30/45; PU e SA 8/15/22 | IP 30; PU e SA 15 |

- Taxa, dropout, weight decay e lote vêm de `[hpo.espaco]`. Os segmentadores vêm de `lista` em `[segmentadores]`.
- O PCA sai de `pca_options`, que aplica os fatores 0,5/1/1,5 (`fatores_pca`) sobre o valor da cena sem descer do mínimo da CNN (`MIN_COMPONENTS`: 8 na CNN do projeto, por causa dos três poolings).

### Como um candidato é representado

Um candidato é uma **tupla de índices**, um por eixo, na ordem de `genes()`:

```
(segmenter, override_index, learning_rate, dropout, weight_decay, batch_size, pca_components)
```

Exemplo em Indian Pines: `(0, 1, 1, 2, 0, 0, 1)` significa SEEDS, configuração padrão, taxa 1e-3, dropout 0,5, wd 0, lote 32 e PCA 30. É a configuração original.

- `genes(dataset, segmenters, cnn)` monta a lista de pares `(nome, opções)`. O segmentador precisa ser o eixo 0, porque a cobertura inicial da bayesiana fixa essa posição.
- `decode(lista_genes, candidato, dataset)` converte índices em campos de `ExperimentConfig`. O `override_index` é resolvido com `segmenter_options(segmentador, cena)`, e o dicionário escolhido vai para `segmenter_overrides`.
- Trabalhar com índices permite que grade e bayesiana usem a mesma função `avaliar(candidato)`. Também faz com que eixos de natureza diferente (categóricos, inteiros, floats em escala log) virem todos 0, 1 ou 2.

### Segmentadores dentro da busca: três opções e a calibração

Os parâmetros do SEEDS são os do notebook do Sydney. LSC e Watershed foram **calibrados para gerar o mesmo número de regiões que o SEEDS** com esses parâmetros: IP 81, PU 45 e SA 96. Assim a comparação entre segmentadores não mistura o algoritmo com a granularidade. As variações menor e maior miram 0,7x e 1,35x esse número.

A contagem abaixo foi feita com o código atual (OpenCV com 1 thread):

| Cena | Segmentador | menor | padrão | maior |
| --- | --- | --- | --- | --- |
| IP | SEEDS (superpixels / iterações / prior) | 244/120/1 → 81 | 350/200/2 → 81 | 472/250/3 → 319 |
| IP | LSC (region_size / ratio) | 19/0,04 → 56 | 16/0,075 → 81 | 14/0,12 → 100 |
| IP | Watershed (marcadores) | 86 → 57 | 116 → 82 | 152 → 111 |
| PU | SEEDS | 105/120/1 → 45 | 150/150/1 → 45 | 202/250/3 → 45 |
| PU | LSC | 76/0,04 → 32 | 65/0,075 → 45 | 57/0,12 → 60 |
| PU | Watershed | 56 → 33 | 72 → 46 | 102 → 61 |
| SA | SEEDS | 105/120/1 → 24 | 150/150/1 → 96 | 202/250/3 → 96 |
| SA | LSC | 40/0,04 → 65 | 34/0,075 → 96 | 29/0,12 → 126 |
| SA | Watershed | 98 → 66 | 136 → 97 | 172 → 127 |

- **SEEDS anda em degraus.** O número de regiões não segue o pedido. Em IP, a grade inicial tem 16, 81 ou 324 regiões; na variação maior, 5 das 324 somem nas iterações e sobram 319. Por isso as variações mudam também as iterações e o prior, que alteram a forma das fronteiras mesmo quando a contagem não muda (em PU as três opções dão 45 regiões).
- **LSC:** `region_size` é o lado do superpixel. Quanto menor o lado, mais regiões. O `ratio` (compacidade) varia junto, e `num_iterations` fica fixo em 10.
- **Watershed:** os marcadores formam uma grade com a proporção da cena. Os da moldura se perdem, porque o `cv.watershed` escreve -1 na borda. As linhas entre bacias viram a região 0.
- Esse eixo é **categórico** na bayesiana. "Maior" no SEEDS não significa "mais regiões" de forma regular, então não há ordem confiável entre os índices.

### Busca em grade (`grid_search`)

- Ponto de partida: `_indice_original`, o candidato da configuração original.
- Varia só o segmentador e os eixos de `eixos_da_grade` (taxa e dropout). São 3 × 3 × 3 = **27 pontos**. Os demais eixos ficam no valor original (configuração padrão do segmentador, wd 0, lote 32, PCA da cena).
- Não usa a semente para escolher pontos: a grade é sempre a mesma. A semente só muda o treino.
- A grade **contém o ponto original**, então o vencedor dela nunca fica abaixo do original na validação. A bayesiana não tem essa garantia.
- Por que esses dois eixos: na campanha anterior, eles tiveram a maior amplitude de validação média (dropout 18,1 pp, taxa 15,3 pp). Isso é indício: foi medido com a CNN do Sydney, 10% do treino e TPE/aleatória/genético, no log `experimentos/resultados/08_IP.log` da numeração antiga (não o de `invalidados_lsc/`). O experimento 04 mede de novo.

### Otimização bayesiana com processo gaussiano (`bayesian_search`)

- Usa o `optuna.samplers.GPSampler`, com a semente da execução.
- **Tipos de eixo:**
  - `segmenter` e `override_index` são `suggest_categorical`.
  - Os demais são `suggest_int(0, 2)` sobre o índice. Nesses eixos o índice seguinte é de fato um valor maior. Na taxa, os valores crescem cerca de 3x a cada passo, então o índice funciona como uma escala log.
- **Cobertura inicial:**
  - Antes de tudo, enfileira um trial por segmentador (`enqueue_trial({"segmenter": i})`), só com o segmentador fixo. Os outros eixos são sorteados.
  - `n_startup_trials = max(bayes_trials_iniciais = 4, nº de segmentadores)`.
  - No 04, com 3 segmentadores, os trials 1 a 3 cobrem um segmentador cada, o 4 é sorteado e o GP escolhe a partir do 5.
  - No 03, com 1 segmentador, os trials 1 a 4 são sorteados e o GP escolhe do 5 ao 16.
  - Motivo: sem isso, o sorteio inicial pode não visitar um segmentador, e o GP não teria nenhuma observação dessa categoria.
- **Candidatos repetidos:**
  - O espaço é discreto, e o GP às vezes propõe uma tupla já treinada.
  - Ela recebe o valor já medido (`estudo.tell` com o valor do dicionário `medidos`), não treina de novo e não conta no orçamento.
  - O laço faz no máximo `8 × n_trials` propostas. Num caso extremo, a busca termina com menos trials que o orçamento. Vale conferir que cada busca do 04 tem 27 linhas.
- O GPSampler ainda é marcado como experimental no Optuna 4, e o aviso é silenciado.

### O que cada trial mede (`avaliar` em `automated_hpo`)

1. `decode` gera a configuração, e o teto de épocas vira `search_epochs`.
2. A paciência cai para `min(10, max(2, search_epochs // 3))`: 3 com 10 épocas, 10 com 40 ou 200. Com 10 épocas e paciência 10, a parada nunca dispararia.
3. `run_experiment(config)` faz uma execução completa: dados, treino com parada antecipada e teste.
4. O **objetivo** é `max(historico["val_accuracy"])`, a melhor acurácia de validação do histórico.
5. Cada trial vira uma linha com:
   - estratégia e número do trial;
   - segmentador e parâmetros do segmentador (JSON);
   - taxa, dropout, weight decay, lote e PCA;
   - `val_accuracy (objetivo)`;
   - `OA/AA/Kappa teste (relato)`;
   - tempo de treino e épocas executadas;
   - `melhor até agora`, o máximo acumulado, que dá a curva de convergência;
   - `config`, o objeto inteiro, usado pelo 03 e descartado no CSV do 04.

Detalhe: com orçamento curto, o treino pode terminar pelo limite de épocas sem disparar a parada. Nesse caso ficam os pesos da última época, e o teste relatado pode corresponder a uma época diferente da do objetivo. Isso não afeta a escolha, que usa só a validação.

### Como o vencedor é escolhido e por que o teste só é relatado

- O vencedor é a linha de maior `val_accuracy (objetivo)`. Em empate, fica o trial mais antigo (`idxmax`).
- A validação (7,5%) e o teste (7,5%) são fixos para todos os trials, porque a divisão usa a semente de dados 42. Os trials disputam no mesmo conjunto.
- O teste nunca entra na escolha. Escolher pelo teste o transformaria numa segunda validação, e a OA final sairia otimista, porque o vencedor seria justamente o que acertou mais naquele conjunto. Por isso as colunas de teste se chamam "(relato)".
- No experimento 03, o número que vai para a monografia não vem da busca. O vencedor é **retreinado com 5 sementes** (11, 29, 42, 73, 101) e a OA relatada é a média ± desvio no teste.

---

## Pilar 2: paralelismo

### A. Partição da cena (`particao.py`)

```
cena → prepare_data(config, "thread")      PCA e segmentação em 2 threads, na cena inteira, uma vez
     → partition_scene(coords, N, modo)    índice 0..N-1 para cada amostra rotulada
     → slice_data(dados, grupo == k)       N pedaços com treino/val/teste renumerados
     → executor: N × train_and_predict     cada tarefa treina (ou não) e prevê o teste do pedaço
     → concatena as previsões              OA/AA/Kappa sobre o teste inteiro
```

- **A segmentação vem antes de partir.** Segmentar cada pedaço daria superpixels diferentes nas fronteiras.
- **`partition_scene`:**
  - `"faixas"` ordena as amostras pela linha da cena e corta em N partes do mesmo tamanho. As partes ficam equilibradas, e o teto do speedup fica perto de N.
  - `"aleatoria"` sorteia as amostras. É o controle: mesmo custo, sem contiguidade.
- **Modos:**
  - `"independente"`: cada pedaço treina e testa o próprio modelo. O treino é paralelizado.
  - `"replicado"`: um único modelo, treinado na cena inteira (no dispositivo padrão), desce para a CPU, e só a inferência é dividida. A OA é idêntica à do modelo monolítico. Com processos, esse modo é recusado (`ValueError`), porque exigiria enviar o modelo a cada processo.
- **Executores:**
  - `"sequencial"` é o denominador do speedup.
  - `"thread"` usa `ThreadPoolExecutor`.
  - `"process"` usa `ProcessPoolExecutor`.
  - As partições sempre rodam na **CPU**, para que os três executores se comparem no mesmo dispositivo.
- **`init_worker`** coloca 1 thread do PyTorch e 1 do OpenCV em cada worker. O número de threads do PyTorch vale por thread, e um processo novo sobe com o padrão da máquina (32 núcleos). Sem isso, N workers disputariam N × 32 threads.
- **Por que spawn e não fork:** o PyTorch recusa fork depois que o autograd rodou em threads. O spawn sobe um interpretador limpo e reimporta o PyTorch, o que custa alguns segundos por processo. `train_and_predict` fica no nível do módulo porque o pickle só serializa funções pelo nome. Em `src/executar.py`, o `torch.set_num_threads(1)` também fica no nível do módulo, porque os filhos do spawn reexecutam o módulo principal.
- **Pool frio e pool quente:**
  - Frio: o pool é criado e fechado dentro da medição e paga a partida dos workers.
  - Quente: `new_pool` + `warm_up` antes da medição. O `_aquecer` roda uma convolução com `backward` em cada worker para inicializar o autograd, e um `sleep(0,25 s)` mantém o worker ocupado para o pool criar todos em vez de reusar um.
- **O que `run_experiment_particionado` devolve:**
  - `tempo_paralelo`: a parede do bloco concorrente;
  - `soma_partes`: a soma dos tempos de cada tarefa;
  - `teto = soma/max`: o speedup possível dado o desequilíbrio entre as partes;
  - `overhead = parede − max`.
  - Com executor concorrente, cada parte é medida sob disputa. Por isso o **speedup honesto é parede do sequencial ÷ parede do executor**, e não soma ÷ parede (o próprio código avisa isso).
- **Determinismo:**
  - Sequencial e processos repetem a OA.
  - Threads no modo independente não repetem, porque as partições disputam o gerador aleatório global do PyTorch (dropout e pesos iniciais).
  - O replicado com threads repete, porque só faz inferência.

Paralelismo entre etapas nos dados: `prepare_data(config, backend="thread")` roda o ramo espectral (PCA) e o espacial (PCA de 3 componentes, RGB e segmentador) em duas threads. PCA e OpenCV liberam o GIL. A partição usa esse caminho, e o pipeline normal usa o sequencial.

### B. Fusão concorrente (`fusao_concorrente.py`)

A ideia: as convoluções de um pixel não dependem do mapa de superpixels. Só a concatenação depende. Então T0 segmenta enquanto T1..TN já passam os pixels pelas convoluções. O esquema vale para a **inferência** de um modelo treinado. No treino, a segmentação roda uma vez e não há o que sobrepor.

```
                       +------------------- T0 (threading.Thread "T0") --------------------+
  cena inteira ------> |  PCA(3) -> RGB -> SEEDS -> rótulos (H x W)                         |
                       +-----------------------------------+--------------------------------+
                                                           | with trava:
                                                           |   mapa["rotulos"][...] = rótulos
                                                           |   mapa["publicado"] = True
                                                           |   trava.notify_all()
                                                           v
                     mapa = { rotulos: H x W com -1, publicado: False, trava: Condition }
                                                           ^
                                                           | with trava: wait_for(publicado)
                                                           |   ids = rotulos[linha, coluna]
  pixels rotulados                                         |
  em blocos de `bloco` ---> fila do pool T1..TN            |
                            tarefa de um bloco:            |
                            [convoluções] -----------------+--> [concatena id] --> [densas] --> classe
                            (não precisa do mapa)               (região crítica)
```

Linha do tempo esperada:

```
tempo  ------------------------------------------------------------------------>
T0     |======== PCA(3) + SEEDS ========|publica
T1     |conv b1|.......espera........|fusão b1|conv b4|fusão b4|conv b7|fusão b7|
T2     |conv b2|.......espera........|fusão b2|conv b5|fusão b5|
T3     |conv b3|.......espera........|fusão b3|conv b6|fusão b6|

desacoplado (todas as convoluções entram na fila antes de todas as fusões):
T1     |conv b1|conv b4|conv b7|.|fusão b1|fusão b4|fusão b7|
T2     |conv b2|conv b5|conv b8|.|fusão b2|fusão b5|
```

- **T0** é uma `threading.Thread` à parte. **T1..TN** são as threads de um `ThreadPoolExecutor`; o pool as nomeia `T_0..T_{N-1}`, e a figura renumera para T1..TN.
- **`threading.Condition`, e não só um mutex:**
  - Um Lock garante exclusão mútua, mas não a ordem: uma tarefa poderia pegar a trava antes de T0 e ler -1.
  - A condição com a flag `publicado` faz quem chega antes esperar. `wait_for` solta a trava enquanto espera, e `notify_all` acorda todos quando T0 publica.
- **`secao`, a extensão da região crítica:**
  - `"desenho"`, como no esquema: T0 segura a trava durante toda a segmentação, e cada tarefa a segura durante a leitura dos ids e a concatenação (`modelo.fundir`).
  - `"minima"`: T0 segmenta sem trava e só trava para publicar. Cada tarefa trava só para copiar os ids e concatena fora.
  - As densas rodam fora da trava nos dois casos. `CNN1D.fundir` é um método separado justamente para isso.
- **`bloco`:** número de posições por tarefa (1, 64 ou 1.024 no experimento). Com bloco 1 há uma tarefa por pixel (10.249 em IP), e o custo de agendamento domina.
- **Variante desacoplada (`desacoplado=True`):**
  - Submete todas as convoluções e depois todas as fusões. Cada fusão espera o futuro da sua convolução.
  - Não entra em impasse: a fila é atendida em ordem e todas as convoluções estão à frente das fusões.
  - A diferença para o esquema: no esquema, uma thread que termina as convoluções antes de T0 publicar fica parada esperando o mapa e não pega outra tarefa. Na desacoplada, ela segue convolvendo os blocos seguintes.
- **Pool aquecido (`warm_pool`):**
  - Faz uma passada do modelo em cada thread antes da medição.
  - Uma `Barrier(n)` obriga o executor a criar as N threads.
  - O pool, criado por `new_thread_pool`, herda `torch.set_num_threads` do chamador no momento da criação (1 no experimento). Sem isso, cada thread subiria com 32 threads do PyTorch, e uma convolução de 64 pixels levaria cerca de 100 ms em vez de 0,2 ms.
  - `torch.inference_mode()` também vale por thread, então é ligado dentro de cada tarefa.
- **Referências (`executar_sequencial`):**
  - Sem `bloco`, é o **lote**: segmenta e infere tudo de uma vez, como o pipeline. Medido com 1 e 16 threads do PyTorch.
  - Com `bloco`, percorre as posições na mesma granularidade das tarefas concorrentes. É o denominador justo do speedup das threads.
- **Conferências de cada execução:**
  - `ids_iguais_ao_treino`: os ids lidos do mapa são exatamente `dados["segmentos"]`, os do treino. Isso prova que o mapa de T0 é o mesmo e que a leitura por (linha, coluna) está certa.
  - `classes_iguais_ao_lote`: as classes previstas são idênticas às da inferência em lote com 1 thread. A concorrência não mudou o resultado.
  - Se uma das duas falhar, o experimento imprime `DIVERGÊNCIA`.

Teto teórico: a fórmula do capítulo 3 é `(Tseg + Tcnn) / [max(Tseg, Tcnn) + Toverhead]`; sem sobrecarga, o teto é `(Tseg + Tcnn) / max(Tseg, Tcnn)`.
- Indício de `antes_da_fidelidade/14_fusao_concorrente.csv` (CNN do Sydney): em IP, o lote com 1 thread levava 0,33 s, dos quais 0,09 s eram de segmentação. Só a sobreposição daria no máximo cerca de 1,4x.
- Em PU, a segmentação ocupava 0,88 s de 1,05 s.
- Passar disso exige que as N threads também dividam as convoluções.

---

## Os experimentos 01 a 08

### Resumo e tempo de máquina

As estimativas usam os logs e CSVs antigos medidos nesta máquina (RTX 5070 Laptop, 32 núcleos, 15,8 GB). O mapa da numeração antiga para a nova está em `experimentos/resultados/NUMERACAO.md`.

| Exp. | Pergunta | Estado | Tempo estimado | Base da estimativa |
| --- | --- | --- | --- | --- |
| 01 | Onde a CNN do projeto parte, com os hiperparâmetros originais? | **feito** | ~5 min | `01_projeto.log` |
| 02 | Com a CNN fixa, qual segmentador vai melhor? | pendente | ~1 h | épocas do 01 e tempo por época do 03 |
| 03 | Com a CNN ajustada para cada segmentador, quem ganha? | **parcial** (IP: SEEDS e LSC) | falta ~2 a 2,5 h | `03_resumo.csv` (8,2 e 12,2 min) e `tpe_e_slic/06_resumo.csv` (PU 15 a 29 min e SA 15 a 20 min por segmentador) |
| 04 | Grade ou bayesiana, com o mesmo orçamento? | pendente | ~40 min a 1 h | `antes_da_fidelidade/08_trials.csv` (540 trials, 16,5 min só de treino) |
| 05 | Quanto a otimização acrescenta sobre o original? | pendente (precisa do 03 de IP) | ~20 min | 30 treinos de ~35 s |
| 06 | A partição reduz o tempo? Quanto custa em acurácia? | pendente | ~10 a 15 min | `cnn_do_sydney/11_paralelismo_particao.csv` (9,9 min) |
| 07 | Pool frio ou quente? Threads ou processos? | pendente | ~40 min | `cnn_do_sydney/13_pool_quente.csv` (24,7 min de blocos, mais preparo e treinos do replicado) |
| 08 | A fusão concorrente acelera a inferência? | pendente | IP ~10 min; PU mais ~6 min | `14_IP_PU_SA.log` e `antes_da_fidelidade/14_*.csv` |

O que falta soma **cerca de 5 a 6 h de máquina**. Cabe numa noite.

### 01: linha de base (`e01_linha_base.py`)
- **Pergunta:** com os hiperparâmetros do Sydney (lote 32, taxa 1e-3, dropout 0,5, SEEDS, 85% de treino) e a parada deste trabalho (até 200 épocas e paciência 10), onde chegam a CNN do projeto e a CNN original?
- **Desenho:** 3 cenas × 2 CNNs (`projeto` e `original`), semente 42, uma execução cada.
- **Grava:** `01_linha_base.csv` (OA, AA, Kappa, épocas, melhor época e motivo da parada) e `01_linha_base_por_classe.csv` (amostras de teste e acertos por classe). Regrava tudo a cada execução e não tem retomada, o que não pesa porque leva minutos.
- **Resultado:**

  | Cena | CNN do projeto | CNN original | Diferença |
  | --- | --- | --- | --- |
  | IP | 0,9142 | 0,8960 | +1,8 pp |
  | PU | 0,9841 | 0,9853 | −0,1 pp |
  | SA | 0,9938 | 0,9951 | −0,1 pp |

- **Atenção:** é uma semente só. As diferenças de PU e SA estão abaixo do ruído entre sementes visto no 03 (desvio de ~0,5 pp em IP).

### 02: segmentadores com a configuração fixa (`e02_cnn_fixa.py`)
- **Pergunta:** com a CNN e os hiperparâmetros originais fixos, como SEEDS, LSC e Watershed se comparam? É o desenho do projeto e fica como contraexemplo: a configuração fixa foi ajustada para o SEEDS, então mede o quanto ela combina com cada segmentador, e não a qualidade do segmentador.
- **Desenho:** 3 cenas × 3 segmentadores × 5 sementes = 45 treinos. Também calcula o número de regiões e a **pureza** de cada segmentação: a fração da classe dominante em cada região, ponderada pelos pixels rotulados.
- **Grava:** `02_impacto_segmentadores.csv`, **só no fim**.
- **Atenção:**
  - Não tem retomada: interromper perde tudo.
  - Regiões e pureza só vão para a tela. Rode com a saída redirecionada para um log.

### 03: segmentadores com hiperparâmetros otimizados (`e03_segmentadores.py`)
- **Pergunta:** com os hiperparâmetros ajustados para cada segmentador, LSC ou Watershed superam o SEEDS?
- **Desenho:** para cada cena e segmentador (SEEDS primeiro):
  1. busca bayesiana de 16 trials restrita àquele segmentador (`segmenters=(seg,)`), com semente 42, treino completo e até 200 épocas;
  2. escolha do vencedor pela validação;
  3. retreino do vencedor com 5 sementes.
- **Grava:**
  - `03_resumo.csv`, uma linha por (cena, segmentador), regravado a cada segmentador concluído. Traz `val_busca`, OA/AA/Kappa com desvio, os hiperparâmetros vencedores, `overrides`, `treino_s`, `épocas` e `minutos_totais`. A coluna `ganho_OA_pp` é o ganho sobre o **SEEDS otimizado** da mesma cena.
  - `03_por_seed_<cenas>.csv`, só no fim.
- **Resultado parcial (IP):**

  | Segmentador | Validação da busca | OA (5 sementes) | Vencedor |
  | --- | --- | --- | --- |
  | SEEDS | 0,9480 | 0,9324 ± 0,0062 | taxa 3e-4, dropout 0,2, wd 0, lote 64, PCA 15, SEEDS padrão |
  | LSC | 0,9389 | 0,9152 ± 0,0045 | mesmos hiperparâmetros, LSC padrão |

- **Atenção:** o arquivo por semente de IP SEEDS e LSC não foi gravado, porque a execução parou antes do fim. O 05 recupera esses valores: o braço "otimizada" do 05 refaz esses treinos com a mesma configuração e as mesmas sementes, e o código fixa o modo determinístico. Confira que a média desse braço bate com a OA do `03_resumo.csv`.

### 04: busca em grade × otimização bayesiana (`e04_estrategias.py`)
- **Pergunta:** com o mesmo orçamento de 27 trials, qual estratégia acha a melhor validação? Como o teto de épocas por trial muda isso? Quais eixos mais importam?
- **Desenho:** Indian Pines, 10% do treino (validação e teste inteiros), meta de validação 0,95. Orçamentos de 10, 40 e 200 épocas × 2 estratégias × 3 sementes = 18 buscas × 27 trials = **486 trials**.
  - Grade: 27 pontos fixos; a semente só muda o treino.
  - Bayesiana: 27 trials no espaço inteiro; a semente governa a busca e o treino.
- **Grava:** `04_trials.csv` com **todos** os trials, regravado a cada busca. Imprime a média e o desvio da melhor validação entre sementes e a amplitude da validação média por valor de cada eixo (sem o segmentador). A figura `eixos_hpo` recalcula essa amplitude a partir do CSV, incluindo o segmentador.
- **Indício (numeração antiga 08, CNN do Sydney, aleatória/TPE/genético):** as diferenças entre estratégias ficaram dentro do desvio entre sementes. Amplitude por eixo: dropout 18,1 pp, taxa 15,3, PCA 8,4, wd 4,6, lote 1,8.

### 05: configuração otimizada × original (`e05_impacto_hpo.py`)
- **Pergunta:** quanto a busca acrescenta sobre a configuração do Sydney, por segmentador?
- **Desenho:** Indian Pines × 3 segmentadores × 2 configurações × 5 sementes = 30 treinos, todos com a CNN do projeto e 85% do treino.
  - Original (no CSV, `herdada`): taxa 1e-3, dropout 0,5, wd 0, lote 32, PCA da cena e segmentador padrão.
  - Otimizada: a vencedora do 03 para a mesma cena e segmentador.
- **Grava:** `05_impacto_hpo.csv`, uma linha por treino, com retomada. Imprime o ganho em pp e diz se ele é "afirmável", isto é, maior que 2 × o desvio combinado.
- **Atenção:**
  - Lê `03_resumo.csv`. Falta de uma linha (cena, segmentador) gera `IndexError`, então rode o 03 de IP inteiro antes.
  - Na monografia, chame o braço de "configuração original", e não pelo rótulo do código.
- A linha **SEEDS original** é a configuração do Sydney com fusão SEEDS (com a CNN do projeto). É contra ela que se mede o ganho.

### 06: partição da cena (`e06_particao.py`)
- **Pergunta:** dividir a cena em N faixas processadas ao mesmo tempo reduz o tempo? Quanto custa em acurácia?
- **Desenho:** Indian Pines, N = 2, 4, 8 e 16, 3 sementes, 25 épocas, lote 64, executor de threads, 1 thread do PyTorch. Arranjos:
  - monolítico (run_experiment normal, **na GPU**);
  - independente · faixas;
  - independente · aleatória;
  - replicado · faixas.

  Ao todo, 39 linhas.
- **Grava:** `06_paralelismo_particao.csv`, com retomada.
- **Atenção:**
  - A coluna `speedup_bloco` é `soma_partes / parede`. Com threads, ela sai otimista, porque cada parte foi medida sob disputa (no arquivo antigo, 14x com N = 16, enquanto o 07 mostrou que threads ficam abaixo de 1x). Use o 06 para **acurácia** e o 07 para **speedup**.
  - O monolítico roda na GPU e as partes na CPU, então o tempo contra o monolítico mistura dispositivo com paralelismo.
- **Indício (CNN do Sydney, dados antes da correção):**
  - O replicado manteve a OA idêntica ao monolítico.
  - O independente · faixas ficou 2,7 a 3,5 pp acima: cada faixa tem menos classes, e o modelo se especializa na sua região.
  - O controle aleatório caiu de 4 a 29 pp: cada modelo recebe só 1/N do treino da cena toda.

### 07: pool frio × pool quente (`e07_pool_quente.py`)
- **Pergunta:** quanto do tempo é a partida dos workers? Com a partida fora da medição, threads ou processos dão speedup real?
- **Desenho:** Indian Pines, semente 42, 25 épocas, lote 64, N = 2, 4, 8, 16 e 32.
  - Arranjos: independente com sequencial, process e thread; replicado com sequencial e thread.
  - Frio com 2 repetições, quente com 3.
  - Processos só rodam se `1,25 × N × 481 MB` couber na memória livre. Nesta máquina, N = 32 com processos é recusado.
- **Grava:** `07_pool_quente.csv` com `bloco_paralelo`, `soma_partes`, `overhead`, `teto`, `tempo_aquecimento` e `OA`. Imprime o speedup sobre o sequencial, por modo, estado e N.
- **Indício (CNN do Sydney, dados antes da correção):**
  - Processos com pool quente: 1,9x (N = 2) a 4,5x (N = 16).
  - Processos com pool frio: no máximo 1,5x. A partida custava cerca de 5 s.
  - Threads: 1,6x com N = 2; de N = 4 em diante, ficam abaixo de 1x.
  - A OA de threads no modo independente variou 3,5 pp entre execuções, como esperado.

### 08: fusão concorrente (`e08_fusao_concorrente.py`)
- **Pergunta:** o esquema concorrente (T0 segmenta enquanto T1..TN convolvem) reduz o tempo de inferência em relação ao lote? Com qual tamanho de bloco e qual região crítica?
- **Desenho:** treina uma vez a configuração original (SEEDS, CNN do projeto, 16 threads do PyTorch), passa o modelo para a CPU e infere todos os pixels rotulados. Arranjos:
  - lote com 1 e 16 threads;
  - sequencial com blocos 1, 64 e 1.024;
  - concorrente e desacoplado com N = 1, 2, 4, 8 e 16 × blocos 1, 64 e 1.024 × seções "desenho" e "minima".

  São 5 repetições (3 com bloco 1, só em IP), 283 linhas em IP.
- **Grava:**
  - `08_fusao_concorrente.csv` com `parede`, `segmentacao`, `espera_total`, `espera_maxima`, `conv_antes_do_mapa` (fração das posições convolvidas antes de o mapa sair), `OA_teste` e as duas conferências;
  - `08_linha_do_tempo_<cena>_{concorrente,desacoplado}.csv`, do arranjo (N = 4, bloco 64, mínima), primeira repetição.
  - Na variante desacoplada, `espera_total` inclui o tempo em fila e não mede bloqueio. Ali, leia `parede` e `conv_antes_do_mapa`.
- **Indício (CNN do Sydney, dados antes da correção):**
  - Em IP, o melhor ficou em cerca de 0,16 s (bloco 1.024, N = 8): 2,0x sobre o lote com 1 thread e 1,2x sobre o lote com 16.
  - Com bloco 1, o concorrente e o desacoplado ficaram de 7 a 35x mais lentos que o lote com 1 thread.
  - Todas as execuções passaram nas duas conferências.
- **Atenção:** a figura `fusao_linha_do_tempo` lê os arquivos de **PU**, porque a segmentação ali leva cerca de 0,9 s e a sobreposição aparece. O `config.toml` só lista IP, então rode `08 IP PU`.

### Figuras (`figuras.py`)

| Figura | Depende de |
| --- | --- |
| `segmentacoes_ip` | só segmenta |
| `particao_faixas` | prepara os dados de IP e particiona, sem treino |
| `curvas_treino`, `mapa_classificacao`, `matriz_confusao` | treinam a linha de base de IP uma vez (cerca de 1 min) |
| `eixos_hpo` | `04_trials.csv` (o `ylim` é fixo em 0,4 a 0,75; confira depois do 04 novo) |
| `fusao_linha_do_tempo` | `08_linha_do_tempo_PU_*.csv` |

As cinco primeiras já foram regeradas com o código atual em 05/11. `eixos_hpo` e `fusao_linha_do_tempo` em `assets/resultados/` ainda são da rodada antiga e só podem ser refeitas depois do 04 e do 08. Até lá, gerar todas falha com `FileNotFoundError` em `fusao_linha_do_tempo`, porque o CSV `08_linha_do_tempo_PU_*` ainda não existe. Por isso, gere cada uma pelo nome.

---

## Como rodar

### Pelo notebook (`Eduardo_Savian_2026_CNN1D_SEEDS_otimizado.ipynb`)
1. Rode a célula 1 (caminho e imports) e a 3 (versões, dispositivo e determinismo).
2. As seções 2 a 7 mostram cada peça. As demoradas começam desligadas:
   - `BUSCAR = True`: 8 trials bayesianos curtos;
   - `PARTICIONAR = True`: 4 partes com os três executores.

   A seção 6 sempre roda: faz a fusão concorrente com N = 2, 4 e 8 e mostra se as classes ficam iguais às do lote. Ela usa a cena da seção 2 e o modelo da seção 4, que precisam rodar antes.
3. Na célula 17, troque `EXECUTAR = None` por `"03"` (com dois dígitos: `"3"` dá `KeyError`) e `CENAS = ("IP",)` por `("IP", "PU", "SA")`. Com `GERAR_FIGURAS = True`, regera as figuras (mesma ressalva da seção Figuras).
4. Campanhas longas rodam melhor no terminal: se o kernel cai, a saída da tela se perde (o CSV fica).

### Pelo terminal

Na raiz do repositório:

```bash
.venv/bin/python src/executar.py 03 IP PU SA       # "3" também funciona; sem cena, usa as do config.toml
nohup .venv/bin/python src/executar.py 03 IP PU SA >> experimentos/resultados/03_projeto.log 2>&1 &
tail -f experimentos/resultados/03_projeto.log
.venv/bin/python -m hsi.figuras                     # todas as figuras; ou: ... -m hsi.figuras f_eixos
```

Ordem sugerida para o que falta:

```bash
.venv/bin/python src/executar.py 03 IP            # Watershed de IP, ~10 min
.venv/bin/python src/executar.py 05 IP            # precisa do 03 de IP completo, ~20 min
.venv/bin/python src/executar.py 03 PU SA         # ~2 h
.venv/bin/python src/executar.py 02               # ~1 h, sem retomada: não interromper
.venv/bin/python src/executar.py 04               # ~40 min a 1 h
# a partir daqui os experimentos medem tempo: máquina sem nenhuma outra carga
.venv/bin/python src/executar.py 06
.venv/bin/python src/executar.py 07
.venv/bin/python src/executar.py 08 IP PU
.venv/bin/python -m hsi.figuras
```

### Retomada de uma execução interrompida

Rode o **mesmo comando** de novo. `infra.carregar_feito` relê o CSV e pula as combinações que já estão nele.

| Exp. | Unidade da retomada | O que se perde ao interromper |
| --- | --- | --- |
| 01 | nenhuma (regrava tudo) | tudo (minutos) |
| 02 | nenhuma (grava no fim) | tudo |
| 03 | (cena, segmentador) | o segmentador em curso: busca e retreinos |
| 04 | (cena, orçamento, estratégia, semente) | a busca em curso |
| 05 | (cena, segmentador, configuração, semente) | o treino em curso |
| 06 | (cena, arranjo, N, semente) | a medição em curso |
| 07 | (cena, modo, executor, N, estado), com todas as repetições | o grupo em curso |
| 08 | (cena, arranjo, N, bloco, seção, threads, repetição) | as repetições do grupo em curso (o CSV é salvo por grupo) |

Cuidados:
- **A retomada não confere a configuração.** Ela só compara as chaves. Depois de mudar o `config.toml` ou o código, mova ou apague o CSV, senão linhas antigas se misturam às novas.
- Para refazer uma medição específica, apague a linha correspondente do CSV.
- O 03 grava `03_por_seed_<cenas>.csv` só com o que rodou naquela chamada, e chamadas com as mesmas cenas sobrescrevem o arquivo.
- O texto da seção 8 do notebook diz que todos os experimentos retomam. Isso não vale para o 01 nem para o 02.

---

## Perguntas que o orientador pode fazer

**1. O projeto previa Ray Tune. Por que Optuna?**
O método é o mesmo: otimização bayesiana com processo gaussiano. O `BayesOptSearch` do Ray só aceita eixos contínuos, e o espaço deste trabalho tem eixos categóricos (o segmentador e a configuração dele) e discretos. O `GPSampler` do Optuna trata os dois tipos. A busca em grade é feita no próprio código.

**2. Uma grade de 27 pontos contra uma bayesiana de 27 trials é uma comparação justa?**
É de mesmo orçamento, que é o que se compara. A grade completa teria 2.187 pontos. Por isso ela ficou com o segmentador e os dois eixos de maior efeito medido (dropout e taxa), com os demais eixos no valor original. A bayesiana gasta os mesmos 27 trials no espaço inteiro. A grade tem a vantagem de sempre conter o ponto original. Cada busca é repetida com 3 sementes para separar a estratégia do ruído.

**3. Como você garante que o teste não influenciou a escolha?**
O objetivo de cada trial é a melhor acurácia de validação, e o vencedor é escolhido só por ela. O teste aparece apenas em colunas de relato. O número final do 03 é a média de 5 retreinos do vencedor no teste, que é fixo e separado da validação desde a divisão.

**4. O ganho é medido contra o quê?**
Contra a configuração do Sydney com fusão SEEDS, em duas partes:
- troca de CNN: experimento 01, mesma configuração com as duas CNNs;
- otimização: experimento 05, configuração original contra a otimizada, mesma CNN e 5 sementes.

Indício parcial em IP: o SEEDS otimizado chega a 0,9324 ± 0,0062, contra 0,8960 da CNN do Sydney e 0,9142 da CNN do projeto com os hiperparâmetros originais. O 01 tem uma semente só. Se o texto for afirmar o ganho da troca de CNN, vale repetir o 01 com 5 sementes (cerca de 10 min em IP). Para isso é preciso mudar o código, porque `[experimentos.01]` não tem lista de sementes. O lado da CNN do projeto já sai com 5 sementes no braço original do SEEDS no 05.

**5. Os segmentadores foram comparados de forma justa?**
Sim, de duas formas. LSC e Watershed foram calibrados para gerar o mesmo número de regiões que o SEEDS em cada cena. E o 03 ajusta os hiperparâmetros para cada um, porque com a configuração fixa (o 02) a comparação favorece o SEEDS, para o qual aquela configuração foi feita.

**6. Por que threads perdem e processos ganham na partição?**
A CNN é pequena e os lotes também, então o tempo vai em sobrecarga do Python, que segura o GIL. As threads acabam se revezando, e no modo independente ainda disputam o gerador aleatório global do PyTorch. Processos escapam do GIL, mas pagam a partida: spawn, interpretador novo e reimportação do PyTorch, cerca de 5 s. O pool quente tira esse custo da medição, e aí o speedup aparece (indício: 4,5x com N = 16). O spawn é obrigatório porque o PyTorch recusa fork depois que o autograd rodou em threads.

**7. Por que `Condition` e não um mutex? O que muda entre "desenho" e "minima"?**
O mutex garante exclusão, mas não a ordem: uma tarefa poderia ler o mapa antes de T0 escrever. A condição com a flag `publicado` faz a tarefa esperar até a publicação, e `wait_for` solta a trava enquanto espera.
- "desenho" segue o esquema: T0 trava durante toda a segmentação, e a concatenação acontece dentro da trava.
- "minima" trava só a publicação e a cópia dos ids.

Isso mede quanto a extensão da região crítica pesa.

**8. Como você sabe que a concorrência não alterou o resultado, e o esquema cumpre o RNF07 (speedup > 1)?**
Toda execução confere duas coisas: os ids lidos são idênticos aos do treino, e as classes são idênticas às da inferência em lote. Na rodada antiga, todas passaram. Quanto ao speedup:
- pela fórmula do capítulo 3, sem a sobrecarga, o teto da pura sobreposição é `(Tseg + Tcnn) / max(Tseg, Tcnn)`. Em IP, onde a segmentação é pequena, isso dá cerca de 1,4x;
- com N threads e blocos de 1.024, o indício antigo deu 2,0x sobre o lote com 1 thread e 1,2x sobre o lote com 16;
- com bloco 1, o esquema fica de 7 a 35x mais lento que o lote.

Os números definitivos saem do 08 novo.