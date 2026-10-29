# Context

Este é o registro único do trabalho **Análise da Otimização de CNNs 1D na
Classificação de Imagens Hiperespectrais** — Eduardo Savian, UNIVALI, orientação de
Felipe Viel. Reúne o que o trabalho assume, como executa, como mede, o que achou e por
que cada medição foi desenhada daquele jeito. O trabalho otimiza uma CNN 1D para
classificação de pixels hiperespectrais e avalia criticamente o ganho em quatro
eixos — organização do código, paralelismo, automação de hiperparâmetros e
segmentadores — além da rastreabilidade das execuções.

---

## 1. Premissas da pesquisa

### 1.1 De onde o trabalho parte

O ponto de partida é a implementação de **Sydney Matheus de Souza (2023)**: uma CNN
1D que classifica pixels hiperespectrais usando, além do espectro, o **id do
superpixel** a que o pixel pertence (segmentação SEEDS). O notebook original está
preservado no histórico do git, sem alterações, no commit `3ee559e`.

**A contribuição deste trabalho não é um método novo.** É a otimização e a avaliação
crítica de um trabalho existente. Todo resultado aqui só faz sentido contra aquilo
que o original entregava.

### 1.2 A pergunta

Quanto se consegue extrair de um pipeline de classificação hiperespectral existente
otimizando-o em quatro eixos — e **quanto do ganho medido é real**, em vez de
artefato do protocolo de avaliação?

A segunda metade da pergunta não estava no plano inicial. Ela apareceu quando a
medição de vazamento espacial mostrou que o protocolo padrão da literatura de HSI
infla todos os números (seção 8), e acabou virando a contribuição mais forte.

### 1.3 Os quatro pilares

São a lista de atividades do orientador, e organizam tudo:

1. **Organização do código** — reimplementar de forma auditável e reprodutível;
2. **Paralelismo** — onde há ganho real e onde o teto é estrutural;
3. **Automação de hiperparâmetros (HPO)** — vale um hipertunador, ou basta sorteio?
4. **Teste de outros segmentadores** — SEEDS é a melhor escolha, ou a única testada?

### 1.4 A linha de base herdada

Toda comparação é contra a **configuração do Sydney**, conferida no notebook original
(commit 3ee559e, célula 33):

| Parâmetro | Valor herdado |
|---|---|
| learning rate | 0,001 (Adam) |
| dropout | 0,5 |
| weight decay | 0 |
| batch | 32 |
| épocas | 60, parada antecipada em `val_loss` com patience 5 |
| componentes de PCA | 30 em IP, 15 em PU e SA |
| segmentador | SEEDS |
| fusão | id escalar |

Um número sem essa referência não é atribuível a nada. A fusão faz parte da
arquitetura herdada e **não é variável de estudo**: o trabalho não compara "com fusão ×
sem fusão". Todo ganho é medido contra a configuração herdada completa — CNN1D com
fusão SEEDS e a receita acima —, porque a premissa é otimizar os pilares para obter
um ganho efetivo sobre ela.

**Atenção:** o experimento 10 usou batch 128 como receita herdada; o original treina
com 32. O braço herdado precisa ser refeito com 32 (seção 10.2).

### 1.5 As cenas

| Cena | Forma | Pixels rotulados | Classes |
|---|---|---|---|
| Indian Pines (IP) | 145 × 145 × 200 | 10.249 | 16 |
| Pavia University (PU) | 610 × 340 × 103 | 42.776 | 9 |
| Salinas (SA) | 512 × 217 × 204 | 54.129 | 16 |

Indian Pines é a cena de trabalho: pequena, desbalanceada (de 20 a 2.455 pixels por
classe) e a mais difícil das três. O desbalanceamento é o que separa OA de AA — a
acurácia média por classe pune quem acerta só as classes grandes.

---

## 2. Método

### 2.1 Implementação

- **O código vive no pacote `hsi/`**, um módulo por responsabilidade (mapa na seção
  11), com cada nome definido uma vez só. O notebook `HSI_MLOps.ipynb` importa dali e
  guarda a narrativa, os diagnósticos e as células de execução.
- **Os experimentos moram no mesmo pacote**, em `hsi/experimentos/`, um módulo por
  experimento (`experimento_05` … `experimento_13`). Assim não existem duas
  verdades: quem implementa e quem mede usam as mesmas funções. O notebook roda um
  deles pela seção 10; fora do Jupyter, o lançador `experimentos/executar.py`
  — ver seção 3.
- Escopo restrito à **CNN 1D**, como no título: saiu do notebook o que não é objeto
  do trabalho. O código anterior está no commit `c4665d5`.
- Toda execução longa é **retomável**: o CSV consolidado é a fonte de verdade e o
  experimento pula o que já mediu. Isso existe porque três sessões de experimento
  foram perdidas pela metade.

### 2.2 Política de treino

Explícita e auditável, igual em todos os experimentos salvo indicação contrária:

- meta de parada: `val_accuracy ≥ 0,95`;
- parada por estagnação: 10 épocas sem ganho de 0,001 em validação;
- os pesos da melhor época são restaurados ao parar;
- **a seleção usa exclusivamente a validação.** OA, AA, Kappa, F1 e IoU no teste são
  relato final, nunca critério de escolha.

Defaults do contrato: `test_ratio` 0,15, batch 32, lr 1e-3, dropout 0,5, weight decay
0, 60 épocas, semente 42, modo determinístico.

### 2.3 Protocolo de divisão dos dados

Três modos, e a diferença entre eles é o achado da seção 8:

| Modo | O que faz | Vazamento medido |
|---|---|---|
| `pixel` | sorteio estratificado por pixel — o padrão da literatura | **100%** |
| `block` | blocos retangulares da cena | 66,5% a 100% |
| `region` | cada superpixel inteiro numa partição só | **0%** por construção |

Com `split_folds = k`, as regiões viram k grupos e cada região é testada exatamente
uma vez — é o que torna o número confiável, já que uma partição única por região
deixa 3 ou 4 regiões no teste e desvio de ±0,25.

### 2.4 Regras metodológicas adotadas

Estas regras governam o desenho de quase todo experimento deste trabalho e explicam
por que cada medição tem a forma que tem.

1. **Sempre medir contra a configuração herdada.** Número sem a referência do Sydney
   não é atribuível — e ela já inclui a fusão SEEDS: um braço "sem fusão" não é
   referência deste trabalho.
2. **Sempre calcular o teto.** 1,22× parece fracasso; 1,22× contra um teto de 1,21×
   é a implementação no limite do que o desenho permite.
3. **Sempre ter um controle sem o efeito.** A partição aleatória tem custo
   computacional idêntico ao das faixas e destrói a contiguidade espacial: se o ganho
   some nela, era posição, não especialização.
4. **Nunca comparar B mantendo A fixo, quando A e B são otimizados juntos no uso
   real.** É o critério que separa a seção 7.2 da 7.3.
5. **Toda afirmação carrega o regime em que foi medida.** "A vantagem do hipertunador
   some com orçamento alto" valia com 100% dos dados e se inverteu com 10%.
6. **Antes de culpar o método, verificar se a medição é reprodutível.** Duas vezes o
   culpado era uma biblioteca não-determinística, não o método (seção 9).
7. **Benchmark mede o regime permanente; a primeira execução é outro número** — desde
   que o custo de partida continue reportado à parte (seção 5.3).

### 2.5 Determinismo

`configure_determinism` fixa: cuDNN determinístico com `benchmark=False`, TF32
desligado, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `use_deterministic_algorithms` e
**OpenCV com uma thread** — esta última porque o LSC era não-determinístico e
invalidou uma campanha inteira (seção 9).

### 2.6 Ambiente

Python 3.12, PyTorch 2.13 (CUDA 13.0), RTX 5070 Laptop, 32 núcleos, 15,8 GB de RAM,
WSL2. Ambiente criado com `uv venv`; versões fixas em `requirements.txt` e conferidas
por `check_versions()` na seção 2 do notebook.

---

## 3. Como executar

### 3.1 Ambiente e dados

1. Prepare o ambiente: `uv venv --python 3.12 && uv pip install -r requirements.txt`
   e instale o pacote em modo editável: `uv pip install -e . --no-deps`.
2. Coloque os arquivos `.mat` das cenas em `data/dataHSI/` ou defina `HSI_DATA_DIR`.
3. Abra `HSI_MLOps.ipynb` e execute as seções 1 a 7 na ordem indicada pelo roteiro.
4. Na seção 8, habilite apenas a execução que deseja fazer (`RUN_EXPERIMENT`,
   `RUN_HPO`, `RUN_AUDITED_CAMPAIGN` ou `RUN_BATCH_SWEEP`).

As seções 1 a 7 só importam de `hsi/`; cada célula diz em que arquivo está o código
da sua seção. O `%autoreload 2` da primeira célula faz uma edição nos arquivos valer
sem reiniciar o kernel.

### 3.2 Rodar os experimentos

Os experimentos do TCC estão em **`hsi/experimentos/`**, um módulo por experimento,
como as funções `experimento_05` … `experimento_13`. Dentro do notebook, a seção 10
escolhe um pelo número (`EXECUTAR`).

Fora do kernel, o lançador headless faz o mesmo:

```bash
.venv/bin/python experimentos/executar.py 13 IP
```

Ele importa `hsi.experimentos` e chama `experimento_13("IP")`, sem precisar de um
kernel aberto. Com o código num pacote, o executor de processos (`spawn`) funciona
tanto daqui quanto do notebook: o filho encontra as funções importando `hsi`.

Os logs e CSVs de todos os experimentos ficam em `experimentos/resultados/`.

### 3.3 Política de treino

A receita padrão é explícita e auditável:

- Meta: `val_accuracy >= 0.95`.
- Parada por estagnação: 10 épocas sem ganho mínimo de `0.001` em validação.
- Pesos: o estado da melhor época é restaurado ao atingir a meta ou a patience.
- Teste: OA, AA, Kappa, F1 e IoU são somente relato final; a seleção de HPO usa
  validação.

Os parâmetros `early_stopping_target`, `patience` e `early_stopping_min_delta` ficam
juntos em `ExperimentConfig` (`hsi/config.py`) e devem ser ajustados como uma única
política experimental.

### 3.4 Espaço de hiperparâmetros

A busca é semeada e tem três estratégias (random, TPE e genético, em
`hsi/hpo/estrategias.py`). `HPO_SPACE` (`hsi/hpo/espaco.py`) define os eixos — learning rate, dropout, weight decay, batch e componentes de PCA —
e cada trial ainda sorteia um segmentador e um conjunto dos seus parâmetros internos.
Os primeiros trials cobrem obrigatoriamente os quatro segmentadores.

O produto cartesiano desse espaço passa de mil combinações; sortear cobre mais eixos
que uma grade pequena pelo mesmo número de treinos, e continua reprodutível porque o
gerador é semeado pela seed da configuração. A seleção usa exclusivamente a
validação. `plot_hpo_space` mostra a validação obtida em cada valor de cada eixo, o
que permite descartar eixos irrelevantes na campanha seguinte.

Trocar a estratégia de busca (hipertunador bayesiano, algoritmo evolutivo) exige
substituir apenas `hpo_candidates`: é o único ponto que decide quais configurações
treinar.

### 3.5 Batches

O benchmark usa uma escada uniforme `50, 100, 150, 200, 250, 300`. O incremento fixo
de 50 isola o efeito do tamanho do batch, revela o ganho marginal de throughput e
evita saltos arbitrários até o limite de memória. O vencedor é o maior throughput
dentro de 0,005 da melhor validação.

### 3.6 Rastreabilidade e PDFs

Cada execução auditada grava configuração, ambiente, fingerprint dos dados, métricas,
histórico e checkpoint em `outputs/runs/`. As campanhas ficam em `outputs/campaigns/`.

Os relatórios PDF são gerados automaticamente para execuções e campanhas auditadas.
Eles incluem métricas, curvas, metadados e a transcrição dos documentos
textuais/estruturados gerados (`.json`, `.csv`, `.md` e `.txt`); artefatos binários
também são inventariados. Os PDFs ficam no diretório da respectiva execução ou
campanha.

Ao fim de campanhas e sweeps, o notebook atualiza `outputs/reports/all_runs_report.pdf`:
uma visão inteiramente visual de **todo o conjunto de runs**, com qualidade,
estabilidade, custo, Pareto, HPO, early stopping, batches, evolução das curvas de
treino, linha do tempo, acurácia por classe, validação×teste, atingimento da meta e
repetibilidade por seed. Cada gráfico é acompanhado de uma nota sobre origem, legenda
e interpretação.

Na seção de diagnósticos, os quatro segmentadores (SEEDS, SLIC, LSC e Watershed)
também são mostrados como regiões, contornos sobrepostos ao mesmo PCA/RGB e gráficos
de quantidade, escala, distribuição de área, densidade de borda e pureza. Os
dashboards da CNN exibem curvas, métricas, matriz de confusão, camadas, MaxPooling,
Grad-CAM, saliência, incerteza e a projeção inversa das decisões na cena.

---

## 4. Pilar 1 — Organização do código

**O código está organizado no pacote `hsi/`** (28/10): 44 módulos em seis grupos —
dados, paralelismo, modelo e treino, HPO, MLOps e visualização —, mais um módulo por
experimento. A extração das células do notebook foi conferida bit a bit: o mesmo
treino (IP, 3 épocas, semente 42) dá o mesmo histórico, as mesmas previsões e as
mesmas OA/AA/Kappa pelo notebook antigo e pelo pacote. O mapa está na seção 11.

**A reimplementação em PyTorch reproduz o original bit a bit**: mesmo histórico de
perda por época, mesmos pesos finais e mesmas métricas — OA 0,976309 / AA 0,957918 /
Kappa 0,968545 em PaviaU. Isso é pré-requisito, não resultado: sem equivalência
demonstrada, nenhuma comparação posterior significa alguma coisa.

**Memória — a técnica do orientador.** Cada trial de HPO cria um `ExperimentResult`
que carrega sua própria cópia dos dados preparados, e a campanha guardava todos numa
coluna da tabela. A assimetria que resolve: do trial interessam uns kilobytes (OA,
AA, Kappa, config); ele segura o conjunto inteiro.

| Cena | Retido por trial | 20 trials |
|---|---|---|
| Indian Pines | 2,5 MB | 49 MB |
| Pavia | 5,1 MB | 102 MB |
| Salinas | 6,5 MB | 130 MB |

A técnica é instanciar → treinar → **extrair as métricas** → **liberar**, implementada
como `release_result()` e `automated_hpo(retain="winner")`, hoje o padrão: só o
campeão corrente sobrevive, e quando um trial melhor aparece o anterior é liberado.
Como a tabela passou a guardar a `config` de cada trial, liberar não perde
rastreabilidade — qualquer execução pode ser reconstruída.

Medido numa campanha real de 180 trials: a memória do processo cresceu **16 MB** do
início ao fim (`1894 → 1910 MB`, depois da inicialização do PyTorch).

**Dimensão honesta do resultado:** na CNN 1D o problema é pequeno. Com 2,5 MB por
trial, uma campanha de 180 trials sem a técnica custaria algo como 450 MB — incômodo,
não impeditivo. A técnica vale como garantia de campanha plana e como boa prática de
MLOps, e **não deve ser apresentada como se tivesse destravado este trabalho**.

---

## 5. Pilar 2 — Paralelismo

### 5.1 Entre etapas do pré-processamento

Depois do PCA, o ramo espectral e o ramo espacial são independentes até a fusão.
Três executores, cinco repetições, nas três cenas:

| Cena | sequencial | thread | process | ramo espectral | ramo espacial | teto |
|---|---|---|---|---|---|---|
| Indian Pines | 0,166 s | 1,05× | 0,22× | 0,037 s | 0,129 s (78%) | 1,29× |
| Pavia | 0,971 s | 0,95× | 0,75× | 0,072 s | 0,899 s (93%) | 1,08× |
| Salinas | 0,822 s | 1,22× | 0,74× | 0,142 s | 0,680 s (83%) | 1,21× |

**O ganho é estruturalmente limitado.** A segmentação é 78–93% do pré-processamento,
então dois ramos desbalanceados nunca passam de ~1,1–1,3×. Em Salinas a implementação
entrega 1,22× contra teto de 1,21×: está no limite do desenho, não é falha de código.

O `process` perde sempre — criar o pool, serializar os arrays (8,4 MB em IP, 42,7 em
PU, 45,3 em SA) e disputar núcleos com o pai.

**O número que encerra a discussão:** numa execução real, o pré-processamento leva
0,193 s contra 20,13 s de treino. É **0,9% do tempo total**. Speedup infinito ali
mudaria menos de 1% da execução. Por isso o esforço migrou para particionar a cena.

### 5.2 Por partição da cena

Dividir a imagem em N partes processadas ao mesmo tempo. Duas decisões de desenho:

- **A segmentação é feita uma vez, na cena inteira, antes de particionar.** Segmentar
  cada parte isoladamente produziria superpixels diferentes nas fronteiras e os ids
  deixariam de ser comparáveis entre partições. Custa 0,15 a 1,9 s e mantém o contrato.
- **Dois modos:** `replicado` treina um modelo com tudo e paraleliza só a inferência;
  `independente` treina um modelo por partição.

As partes têm tamanho parecido, então o teto passa a ser ≈ N em vez de 1,2×, e o que
fica paralelo é o **treino** — onde o tempo está.

### 5.3 Pool frio contra pool quente

**O erro corrigido.** A primeira medição criava e destruía o pool **dentro**
do cronômetro. Com `spawn`, cada medição pagava N interpretadores limpos subindo do
zero — e, na época destas medições, cada filho ainda recarregava as definições do
notebook antes de treinar (hoje ele só importa o pacote `hsi`). Isso é o pior caso: a primeira e única execução de um processo
recém-nascido. O orientador cortou, e com razão: num uso real o pool sobe uma vez e
atende muitas execuções.

Refeito medindo o **mesmo bloco** nas duas situações. Indian Pines, 25 épocas,
semente 42, `torch threads=1`, 2 repetições frias e 3 quentes por célula. Speedup
contra o executor `sequencial` processando **as mesmas partições**:

| Executor | N=2 | N=4 | N=8 | N=16 | N=32 |
|---|---|---|---|---|---|
| thread | 1,61× | 0,61× | 0,23× | 0,23× | 0,22× |
| process, pool **frio** | 1,20× | 1,46× | 1,37× | **1,05×** | — |
| **process, pool quente** | **1,86×** | **2,87×** | **3,68×** | **4,47×** | não coube |

Tempo absoluto do bloco e quanto dele era partida:

| N | frio | quente | partida | fração do frio |
|---|---|---|---|---|
| 2 | 11,08 s | 7,31 s | 3,77 s | 34% |
| 4 | 9,12 s | 4,51 s | 4,61 s | 51% |
| 8 | 9,01 s | 3,34 s | 5,67 s | 63% |
| 16 | 12,95 s | 2,86 s | 10,09 s | **78%** |

**Em N=16 o erro inverte a conclusão.** O pool frio diz 1,05×, ou seja "passar de 8
partições não ganha nada"; o arranjo entrega 4,47×. Medir o pior caso não deixa o
resultado conservador — deixa errado.

Três apoios que sustentam a leitura:

- **Controle:** o `sequencial` não tem pool, e frio e quente empatam (±0,34 s). Sem
  isso, a diferença poderia ser deriva da máquina.
- **O aquecimento não é tempo escondido:** 4,92 s em média no `process` (cresce com
  N, são mais workers a subir) contra 0,26 s no `thread`, registrado em coluna
  própria do CSV e pago uma vez.
- **A OA não muda:** `sequencial` e `process` repetem o valor exatamente em todos os
  N. O pool muda quem executa, não o que é executado.

**Por que não escala até o teto.** O teto de Amdahl é ≈ N, mas a concorrência cobra:

| executor | N | partição sozinha | sob concorrência | contenção | teto | obtido |
|---|---|---|---|---|---|---|
| process | 2 | 6,79 s | 7,24 s | 1,07× | 1,99× | 1,86× |
| process | 4 | 3,24 s | 4,45 s | 1,37× | 3,95× | 2,87× |
| process | 8 | 1,54 s | 3,15 s | 2,05× | 7,59× | 3,68× |
| process | 16 | 0,80 s | 2,65 s | **3,30×** | 14,88× | 4,47× |
| thread | 8 | 1,54 s | 53,35 s | 34,7× | 7,88× | 0,23× |
| thread | 32 | 0,41 s | 56,72 s | **137×** | 29,62× | 0,22× |

Treinos simultâneos disputam banda de memória e cache. **O desenho escala; a máquina
não.** No `thread` o mesmo número chega a 137×: o GIL serializa tudo, e o tempo do
bloco estaciona em ~55-61 s a partir de N=8 — ir de 8 para 32 threads não muda nada
porque nunca houve paralelismo ali. Demonstração isolada do GIL: laço Python puro
1,01 s → 0,91 s com 4 threads; multiplicação de matrizes (que solta o GIL) 0,80 s →
0,28 s. Threads internas do PyTorch, à parte: o melhor é 8 (7,81 s); 16 piora (11,67 s).

### 5.4 O limite de memória

A máquina tem 32 núcleos, mas **N=32 com `process` não roda**. Cada worker do `spawn`
sobe um interpretador limpo que reimporta o PyTorch e recarrega as definições do
notebook. Medido com `smaps_rollup`: **RSS 714 MB, PSS 481 MB, privado 422 MB** por
worker. O RSS engana porque conta as páginas compartilhadas em cada processo; o PSS é
o que precisa caber.

| N | PSS projetado | cabe em 10,7 GB disponíveis? |
|---|---|---|
| 8 | 3,8 GB | sim |
| 16 | 7,7 GB | sim, apertado |
| 32 | **15,4 GB** | **não** — a máquina tem 15,8 GB no total |

O script recusa a célula em vez de tentar: empurrar 32 workers para swap faria o
cronômetro medir paginação de disco. **O paralelismo por processos esbarra em memória
antes de esbarrar em núcleos**, e isso é específico do `spawn` — um executor
`forkserver` compartilharia as páginas do PyTorch por *copy-on-write*. Não medido.

**E o limite útil é menor que o do hardware, por acurácia:** a OA cai de 0,9155 (N=4)
para 0,8999 (N=16) e **0,8505 (N=32)**. Com 32 partições sobram ~272 amostras de
treino em cada uma, e vale o que a seção 6.4 mediu sobre escassez de dados.

### 5.5 O modo replicado

Treina uma CNN com a cena inteira e paraleliza só a inferência.

- **A OA é 0,882965 nas dez células, idêntica à do monolítico.** A partição e a
  remontagem estão corretas, e nenhuma predição muda.
- **Mas não há o que paralelizar:** o bloco paralelo mede 0,04 s contra ~9 s de
  treino. A inferência é **0,4% da execução**. Com `thread` ele fica mais lento que o
  sequencial a partir de N=4 (0,30× a 0,41×), porque coordenar custa mais que o
  trabalho. É a lei de Amdahl de novo, agora do outro lado.

**Leitura conjunta dos dois modos:** o `replicado` é o arranjo correto por preservar
a acurácia, não por ganhar tempo; o `independente` é onde há tempo a ganhar, e o
ganho de acurácia que ele aparentava ter não sobrevive (seção 8).

---

## 6. Pilar 3 — Automação de hiperparâmetros

### 6.1 As estratégias

Três buscas sobre **o mesmo espaço** (2.916 combinações), com **o mesmo orçamento de
trials** e **a mesma cobertura inicial** dos quatro segmentadores. Só muda como o
candidato é escolhido. A representação comum é uma tupla de índices, um por gene.

- `random` — sorteio uniforme;
- `tpe` — Optuna 4.9.0, `TPESampler`;
- `genetic` — torneio (3), crossover uniforme, mutação 25%, elitismo 2 (os melhores
  passam sem retreinar).

**Resultado com 3 sementes por célula** (IP, 10% de treino, 540 trials), validação:

| Orçamento | genetic | random | tpe | amplitude | desvio típico | veredito |
|---|---|---|---|---|---|---|
| 10 épocas | 0,7334 | 0,7174 | 0,7226 | 1,60 pp | 1,58 pp | dentro do ruído |
| 40 épocas | 0,8058 | 0,7802 | 0,7924 | 2,56 pp | 1,89 pp | dentro do ruído |
| 200 épocas | 0,8288 | 0,8067 | 0,8266 | 2,21 pp | 2,73 pp | dentro do ruído |

**Nenhuma estratégia se separa do ruído.** O que importa é o **orçamento de épocas**
(0,72 → 0,81 → 0,83). Sem repetição por semente, a conclusão teria sido "o genético
vence" — e a execução anterior, com uma semente só, chegou a dar vencedores
diferentes a cada rodada por causa do bug do LSC (seção 9).

As épocas realmente executadas, com teto de 200: média 63,8, mínimo 12, máximo 157. O
early stopping governa bem antes do teto — não há motivo para testar 500 épocas.

### 6.2 Quais eixos importam

Validação média por valor, sobre os 540 trials:

| Eixo | amplitude | melhor valor |
|---|---|---|
| dropout | 18,10 pp | 0,20 |
| learning rate | 15,26 pp | 3e-3 |
| componentes de PCA | 8,35 pp | 30 |
| weight decay | 4,55 pp | 0 |
| batch | 1,81 pp | 128 |

Dropout e learning rate dominam; batch e weight decay são quase irrelevantes. **Com
dados escassos, regularizar demais é o pior erro** — o modelo já tem pouca informação,
e o dropout 0,5 da receita herdada é justamente o valor pior.

### 6.3 Impacto contra a receita herdada

É a comparação que o pilar existe para fazer. IP, 5 sementes, 10% de treino:

| Condição | receita herdada | otimizada | ganho | veredito |
|---|---|---|---|---|
| **seeds** | 0,7334 | **0,8174** | **+8,40 pp** | **afirmável** |
| lsc | 0,6853 | 0,7566 | +7,13 pp | ruído (desvio 5,22) |
| slic | 0,6003 | 0,6411 | +4,08 pp | ruído (desvio 25,54) |
| watershed | 0,6583 | 0,5646 | −9,36 pp | ruído (desvio 22,70) |

**Na configuração herdada (SEEDS), o HPO entrega +8,40 pp — o único ganho afirmável
da tabela.** É o ganho efetivo do pilar sobre o trabalho de referência. Antes de citar
o número no texto, refazer o braço herdado com batch 32 (seção 1.4).

### 6.4 Quanto custa treinar com menos dados

Amostragem estratificada por classe; validação e teste intactos (comparar frações só
faz sentido com a régua idêntica). IP, CNN 1D + SEEDS, 3 sementes:

| Fração | Amostras | OA | AA | Tempo/run | Épocas |
|---|---|---|---|---|---|
| 1,00 | 8.711 | 0,9016 ± 0,0016 | 0,8308 | 34,3 s | 44 |
| 0,50 | 4.356 | 0,8643 ± 0,0140 | 0,7310 | 18,8 s | 46 |
| 0,25 | 2.179 | 0,8270 ± 0,0105 | 0,6755 | 11,4 s | 58 |
| **0,10** | 871 | **0,7586 ± 0,0109** | **0,6006** | 6,0 s | 59 |
| 0,05 | 435 | 0,6810 ± 0,0131 | 0,5322 | 4,0 s | 60 |

Os 10% custam **14,3 pp de OA e 23,0 de AA**. A AA cai quase o dobro porque
subamostrar tira desproporcionalmente das classes minoritárias. O ganho de tempo é
5,7× e não 10×, porque há custo fixo por execução que não escala com as amostras.

**Recomendação de protocolo:** os 10% servem para a **busca** (o HPO só precisa
ranquear), mas o capítulo de resultados não deveria sair de 10% sem declaração
explícita. O defensável é **buscar com 10% e retreinar o vencedor com 100%** — mesma
lógica de buscar com 10 épocas e retreinar com 60, só que no eixo dos dados.

---

## 7. Pilar 4 — Segmentadores e fusão

### 7.1 O que cada um produz

Métricas estruturais são determinísticas, então uma medição basta. Pureza = fração da
classe dominante dentro de cada região; **não é acurácia**, é diagnóstico espacial.

| Cena | Segmentador | Regiões | Área média | Dens. borda | Pureza | Tempo |
|---|---|---|---|---|---|---|
| IP | seeds | 81 | 259,6 | 0,2051 | 0,8831 | 0,149 s |
| IP | slic | 324 | 64,9 | 0,2920 | 0,9581 | 0,051 s |
| IP | lsc | 324 | 64,9 | 0,3503 | **0,9699** | 0,031 s |
| IP | watershed | 273 | 77,0 | 0,5650 | 0,8543 | **0,004 s** |
| PU | seeds | 45 | 4.608,9 | 0,1063 | 0,9212 | 1,854 s |
| PU | slic | 1.428 | 145,2 | 0,2132 | 0,9857 | 0,263 s |
| PU | lsc | 1.428 | 145,2 | 0,2440 | **0,9955** | 0,203 s |
| PU | watershed | 421 | 492,6 | 0,2797 | 0,9313 | 0,030 s |

**SEEDS entrega muito menos regiões do que se pede** (350 pedidas, 81 entregues em IP;
150 pedidas, 45 em PU): `num_superpixels` é alvo, não garantia. Suas regiões são
enormes — 4.609 pixels em PU contra 145 do LSC, diferença de 32×. Watershed fragmenta:
densidade de borda de 0,5650 significa que 56% dos pares de pixels vizinhos estão em
regiões diferentes.

### 7.2 O desenho inválido, guardado como contraexemplo

A primeira comparação manteve os hiperparâmetros da CNN **fixos** para os quatro
segmentadores, achando que isso isolava a variável. Isola a variável errada: mede o
quanto aquele conjunto fixo por acaso serve a cada segmentador.

Avaliação do orientador: *"não faz sentido neste trabalho analisar ele
separado"*. Os dados provaram:

| Cena | Condição | OA | Kappa | Regiões |
|---|---|---|---|---|
| PU | seeds | 0,9519 ± 0,0026 | 0,9360 | 45 |
| PU | lsc | 0,5398 ± **0,1886** | 0,1970 | 1.428 |
| PU | **slic** | **0,4485** | **0,0346** | 1.428 |

Kappa 0,0346 é praticamente aleatório. O padrão: quanto mais regiões, pior o resultado
e maior o desvio — um id escalar indo até 1.428, contra um espectro de desvio 1,0,
exige learning rate e dropout completamente diferentes dos fixados. O desvio de ±0,19
é a assinatura: algumas sementes treinam, outras colapsam.

**Guardado porque é a evidência de que a otimização conjunta não é capricho:** sem
ela, a conclusão seria "o SLIC não serve para Pavia" (Kappa 0,0346 contra 0,9360 do
SEEDS). Otimizado para si, ele empata com o SEEDS ali (seção 7.3).

### 7.3 Sob otimização conjunta

A pergunta certa: **qual é o melhor resultado que cada segmentador alcança quando a
CNN é ajustada para ele?** HPO restrito a cada segmentador (TPE, 16 trials, seleção só
por validação), depois retreino do vencedor com 5 sementes. O SEEDS, segmentador
herdado, passa pelo mesmo tratamento e é a referência: a pergunta é se trocá-lo rende
um ganho efetivo.

**Indian Pines:**

| Condição | OA | Δ para o SEEDS | lr | dropout | batch | PCA |
|---|---|---|---|---|---|---|
| **seeds** (herdado) | **0,8174 ± 0,0112** | — | 1e-3 | 0,35 | 64 | 15 |
| lsc | 0,7446 ± 0,0284 | −7,28 pp | 1e-3 | 0,35 | 64 | 30 |
| slic | 0,6411 ± **0,1856** | −17,63 pp | **3e-4** | 0,20 | **128** | 30 |
| watershed | 0,5646 ± **0,2255** | −25,28 pp | 1e-3 | 0,35 | 64 | 15 |

**Diferença de OA para o SEEDS herdado, nas três cenas** (ambos otimizados):

| Cena | SLIC | LSC | Watershed |
|---|---|---|---|
| Indian Pines | −17,63 | −7,28 \* | −25,28 |
| Pavia | +0,19 | −1,42 \* | +0,41 |
| Salinas | −1,31 | −1,62 \* | −1,35 |

\* LSC medido antes da correção do determinismo — **refazer**. Em IP, depois
da correção, o LSC otimizado deu 0,7566 (experimento 10), −6,08 pp do SEEDS.

Três conclusões:

1. **O segmentador é a decisão mais impactante do pipeline.** 25,3 pontos de OA
   separam o melhor do pior em IP, com o mesmo orçamento de busca para cada um — mais
   amplitude do que qualquer hiperparâmetro isolado produziu (lr 15,3 pp; dropout
   18,1 pp).
2. **Nenhum segmentador supera o SEEDS herdado de forma afirmável sob split por
   pixel.** Em IP todos ficam atrás mesmo otimizados para si (16 trials cada; não é
   falta de ajuste), SLIC e Watershed com desvio enorme — algumas sementes colapsam.
   Em Pavia, SLIC e Watershed empatam (+0,19 e +0,41 pp, dentro do desvio); em
   Salinas, todos ficam 1,3 a 1,6 pp atrás. Sob split por região o quadro muda
   (seção 8.2).
3. **Cada segmentador escolhe hiperparâmetros diferentes.** O SLIC foi o único a pedir
   learning rate três vezes menor e batch 128 — exatamente o que se espera de quem
   precisa domar um id que vai até 324. **O segmentador reconfigura o treino; não é um
   módulo plugável.** Esse achado é invisível para a comparação com CNN fixa, e é o
   melhor argumento a favor do desenho conjunto.

### 7.4 O que prevê o ganho

Não é a pureza:

| Segmentador | Regiões | Pureza | **Amostras de treino por região** | Δ para o SEEDS |
|---|---|---|---|---|
| seeds | 81 | 0,8831 | **10,8** | — |
| lsc | 324 | **0,9710** | 2,7 | −7,28 pp |
| slic | 324 | 0,9581 | 2,7 | −17,63 pp |
| watershed | 273 | 0,8543 | 3,2 | −25,28 pp |

O LSC tem a maior pureza e fica 7 pontos atrás do SEEDS, que tem a menor entre os
superpixels. **O que prevê é quantas amostras de treino caem em cada região.** Com
~2,7 pixels por região o modelo não estima "região → classe", e o id vira ruído de
alta magnitude entrando na cabeça densa — o mesmo mecanismo do "id aleatório por
pixel" da ablação abaixo, a pior linha dela.

**Métrica de projeto que falta implementar:** *amostras de treino por região*. É o
número que prevê, antes de treinar, se aquele segmentador vai ajudar ou atrapalhar
naquele protocolo. Hoje só se sabe depois de rodar.

### 7.5 Como o id entra — ablação da fusão

IP, split por pixel, 3 sementes:

| Manipulação | O que preserva | OA |
|---|---|---|
| id original | tudo | 0,8981 |
| id permutado | agrupamento e escala; perde a ordem | 0,8552 |
| id normalizado [0,1] | tudo menos a magnitude | 0,8223 |
| id aleatório por pixel | só a escala; perde o agrupamento | 0,7898 |

A referência é o id original, a fusão herdada. Três coisas saem daí: **a informação
é real** (ruído com a mesma escala é a pior linha, −10,83 pp); **parte do que o id
entrega vem de a numeração funcionar como coordenada espacial grosseira** (os ids
crescem na varredura raster; permutá-los custa 4,29 pp); e **a magnitude bruta
importa** — normalizar preserva toda a informação e mesmo assim custa 7,58 pp. Medida de alavancagem: a coluna do id gera **1,79× mais variação**
entre pixels na primeira camada densa do que as 1.024 colunas do espectro somadas.

**O modo de codificar importa mais do que o esperado:**

| Cena | escalar (herdado) | embedding |
|---|---|---|
| Indian Pines | 0,8981 | **0,9471 (+4,90)** |
| Pavia | 0,9785 | **0,9868 (+0,83)** |

Trocar o id cru por `nn.Embedding` rende +4,9 pontos em IP com a mesma informação de
entrada. O id é um rótulo nominal; tratá-lo como número contínuo obriga a rede a
fingir que a região 78 é "78 vezes" a região 1.

---

## 8. O protocolo de avaliação

É o achado transversal do trabalho.

### 8.1 O vazamento

No split aleatório por pixel, **100% das amostras de teste compartilham superpixel com
o treino**, nas duas cenas medidas. Não é suposição: `segment_leakage` conta.

Primeira tentativa de correção, por **blocos espaciais**: mediu **100% de vazamento
mesmo assim** — em IP os superpixels do SEEDS têm ~16×16 px, do tamanho do bloco
escolhido. Bloco maior baixou para 66,5%, ainda longe de disjunto. **A intuição de
"basta separar espacialmente" falha quando o superpixel é grande em relação à cena.**
Só o split por região resolve.

### 8.2 Quanto sobra sem vazamento

Validação cruzada por região (StratifiedGroupKFold, 5 folds × 5 sementes), IP:

| Condição | OA split por pixel | OA split por região | Δ SEEDS pixel | Δ SEEDS região |
|---|---|---|---|---|
| **seeds** (herdado) | 0,8996 ± 0,0107 | **0,5897 ± 0,0635** | — | — |
| slic | 0,8440 ± 0,0118 | 0,7751 ± 0,0386 | −5,57 pp | **+18,53 pp** |
| lsc | 0,8668 ± 0,0106 | 0,7501 ± 0,0284 | −3,28 pp | +16,04 pp |
| watershed | 0,8055 ± 0,0191 | 0,6105 ± 0,1736 | −9,41 pp | +2,07 pp |

**A configuração herdada desaba 31,0 pontos** ao remover o vazamento (0,8996 →
0,5897). Todos os segmentadores caem — pixels vizinhos são quase idênticos, e o split
por pixel coloca quase-duplicatas em treino e teste, **inflando todos os números da
literatura de HSI** —, mas o SEEDS cai mais: −31,0 pp, contra −6,9 do SLIC, −11,7 do
LSC e −19,5 do Watershed.

O ranking se inverte: SEEDS lidera sob split por pixel, mas SLIC e LSC entregam
16 a 19 pontos a mais sob split por região. **Isso não é ganho do segmentador:** sob
split por região cada segmentador define a própria partição, e a do SLIC é muito mais
fácil — o teste do SEEDS fica a 4,24 px do treino em média, contra 2,35 do SLIC
(`DISTANCIA_TESTE`), e um classificador que só vê a posição do pixel acerta 0,8895 na
partição do SLIC contra 0,6997 na do SEEDS (seção 8.4). A diferença entre as
partições, 19,0 pp, é do tamanho da vantagem do SLIC. Comparar segmentadores sob
split por região exige **a mesma partição para todos**.

### 8.3 O mesmo teste aplicado à partição da cena

O modo `independente` parecia ganhar acurácia. Dois controles mataram a ideia:

**Controle 1 — partição aleatória** (mesmo custo computacional, sem contiguidade):

| N | faixas | aleatória | diferença |
|---|---|---|---|
| 2 | 0,9107 | 0,8418 | +6,89 pp |
| 4 | 0,9150 | 0,7859 | +12,92 pp |
| 8 | 0,9172 | 0,7078 | +20,94 pp |
| 16 | 0,9094 | 0,5943 | +31,51 pp |

A vantagem das faixas é **inteiramente posicional**, não especialização do modelo.

**Controle 2 — o mesmo arranjo sob split por região:**

| N | ganho sob split por pixel | ganho sob split por região | quanto sobra |
|---|---|---|---|
| 2 | +2,31 pp | −1,35 pp | −58% |
| 4 | +2,86 pp | **−5,07 pp** | −177% |
| 8 | +2,39 pp | −4,38 pp | −183% |

**O ganho não sobrevive: vira perda.** Era interpolação entre vizinhos — o mesmo
vazamento da seção 8.1. A vantagem espacial da partição aleatória cai de +13,89 pp
para −0,71 pp ao remover o vazamento.

**Consequência para o trabalho:** particionar a cena é uma decisão de **desempenho**,
com custo de acurácia a declarar, e não uma técnica que melhora a classificação.

### 8.4 Homogeneidade das cenas

As três cenas são **campos homogêneos**: o ground truth rotula o interior de talhões e
deixa as divisas como fundo. Medido em 28/10 (`load_scene` + `prepare_data_parallel`;
classificador 1-NN sobre as coordenadas (linha, coluna), **sem nenhum espectro**):

| Cena | Pixels interiores¹ | Campo do pixel típico | Vizinho / mesma classe² | Teste em borda³ |
|---|---|---|---|---|
| IP | 73,2% | 491 px | 0,54 | 0,4% |
| PU | 63,6% | 5.332 px | 0,34 | 0,0% |
| SA | 87,6% | 4.828 px | 0,61 | 0,1% |

¹ os 8 vizinhos têm o mesmo rótulo. ² distância espectral mediana (bandas
padronizadas) ao vizinho de 4-conectividade da mesma classe, dividida pela distância a
um pixel da mesma classe sorteado: abaixo de 1, o vizinho é quase uma cópia. ³ pixels
de teste com um vizinho rotulado de outra classe, no split por pixel.

**OA de um classificador que só vê a posição:**

| Protocolo | IP | PU | SA |
|---|---|---|---|
| pixel, 100% do treino | 1,0000 | 1,0000 | 1,0000 |
| pixel, 10% do treino (5 sementes) | 0,9711 ± 0,0041 | 0,9867 ± 0,0015 | 0,9975 ± 0,0010 |
| região SEEDS (5 folds) | 0,6997 ± 0,0642 | 0,5773 ± 0,1567 | 0,8568 ± 0,0639 |
| região SLIC (5 folds) | 0,8895 ± 0,0211 | 0,9360 ± 0,0012 | 0,9566 ± 0,0069 |

Todo pixel de teste do split por pixel tem um pixel de treino da mesma classe a 1 ou
√2 px. Consequências:

1. **Sob split por pixel, a OA não mede classificação espectral.** A posição sozinha
   supera qualquer CNN deste trabalho (IP, 10% do treino: 0,9711 contra 0,8174 do
   melhor resultado). O id do superpixel é uma coordenada grosseira (seção 7.5), então
   a fusão e o HPO medidos nesse protocolo — experimentos 06, 08 e 10, inclusive os
   +8,40 pp da seção 6.3 — podem estar medindo quanto o modelo explora a posição.
2. **Salinas está saturada:** 0,86 só com posição mesmo no protocolo mais difícil.
   Diferenças de 1–2 pp entre configurações ali dizem pouco.
3. **A dificuldade do split por região depende do segmentador** (0,70 contra 0,89 em
   IP): é a ressalva da seção 8.2.
4. **O SEEDS em PU deixa 45 regiões**: com 5 folds o desvio da posição chega a ±0,16.

---

## 9. Bugs e não-determinismos

Encontrados nos próprios experimentos, e registrados porque cada um quase contaminou
um resultado em silêncio.

**Piso de componentes de PCA.** `_pca_options` gerava opções escalando a
receita da cena por 0,5/1,0/1,5, com piso de 3. O piso 3 vinha da **composição RGB da
segmentação**, que é um PCA diferente. O que alimenta as convoluções da CNN 1D precisa
de pelo menos 15: os kernels (9, 3, 3, 3) consomem 14 posições, e abaixo disso a saída
fica vazia. Cenas com receita 15 (PaviaU, Salinas) geravam a opção 8 e o treino
quebrava dentro da primeira convolução, com a mensagem do PyTorch apontando para a
convolução em vez da causa.

Corrigido derivando o piso dos próprios kernels (`CNN1D_MIN_COMPONENTS`) e fazendo a
CNN 1D levantar um erro que diz a causa. **Indian Pines não foi afetada** — a receita 30
nunca gerou opção abaixo de 15. PU e SA foram relançadas.

**Contrato de dados recusando subamostragem.** `validate_prepared_data` exigia
que os splits cobrissem todas as amostras. Com `train_fraction < 1` isso é falso de
propósito. O invariante que protege o experimento é a **disjunção** (nenhuma amostra em
duas partições), não a cobertura. Trocada a checagem; o contrato agora reporta
`coverage` (1,0 no protocolo cheio, ~0,19 com 10% de treino).

**LSC não-determinístico, e o estrago que causou.** O `createSuperpixelLSC`
do OpenCV é paralelizado e seu desempate depende da ordem das threads: duas chamadas
com a **mesma imagem e os mesmos parâmetros** devolviam rótulos diferentes em ~390
pixels de Indian Pines. SEEDS, SLIC e Watershed não têm o problema.

A consequência era silenciosa e cara. Com o LSC no espaço de busca, o score de um
mesmo trial variava entre execuções, e as estratégias adaptativas amplificavam a
variação até trocarem de vencedor: **a mesma campanha de TPE deu 0,8713 numa execução
e 0,7711 em outra.** Duas execuções seguidas no mesmo processo davam 0,8440 e 0,8713.

Como foi rastreado, e a ordem importa:

1. Treino com config fixa: **determinístico** (idêntico até 1e-6, com 16 e com 1 thread).
2. Amostradores com scores fixos: **os três reprodutíveis** (random, TPE, genético).
3. Logo o problema estava nos *scores* → comparar a saída dos segmentadores.
4. Segmentação chamada duas vezes: **LSC difere em 388 pixels**; os outros três, zero.

Correção: `cv.setNumThreads(1)`, chamado por `configure_opencv_determinism()` dentro
de `configure_determinism`. **Custo nulo** — 38,6 ms por segmentação com uma thread
contra 39,1 ms com 32; as cenas são pequenas demais para a paralelização compensar.

Verificado depois da correção: LSC idêntico em duas execuções e TPE reproduzindo
trial a trial.

**O que isso invalida.** Qualquer medição feita antes desta correção que envolva LSC
ou estratégia adaptativa: o ranking de estratégias da seção 6.1 e os números de LSC
da seção 7.3 precisam ser refeitos. Os resultados de SEEDS, SLIC e Watershed sem busca
adaptativa continuam válidos. As medições invalidadas estão em
`experimentos/resultados/invalidados_lsc/` e **não devem ser usadas**.

**O monolítico treina em GPU e as partições em CPU.** `run_experiment` chama
`select_device()`, que devolve `cuda`; `_treinar_e_inferir` fixa `torch.device("cpu")`.
A coluna "speedup contra o monolítico" do experimento 11 compara, portanto, **GPU
contra CPU** — mede a diferença de dispositivo junto com a de paralelismo, e não
vale. Conferido: o monolítico roda em `cuda` em 8,98 s com OA 0,882965, que
são os 9,29 s e 0,8825 daquele experimento.

Nada disso invalida a comparação **entre executores**, que é partição contra partição
no mesmo dispositivo — é o que o experimento 13 mede, com o `sequencial` no
denominador. Mas as linhas "speedup" do `11_paralelismo_particao.csv` não devem ir
para o TCC.

O mesmo descasamento quebrava o modo `replicado` de forma barulhenta: o modelo único
treinava em CUDA e as partições tentavam inferir com entrada em CPU, levantando
`Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor)`. Corrigido
descendo o modelo para a CPU antes de replicá-lo — a alternativa, mover as entradas
para a GPU, faria N workers disputarem um só dispositivo e o bloco mediria a fila da
GPU. A OA não mudou: 0,882965, igual à do monolítico.

**As partições em threads disputam o RNG global do PyTorch.**
`_treinar_e_inferir` chama `seed_everything`, que semeia o gerador **global**. Com N
threads num só interpretador, as partições consomem esse gerador em ordem
imprevisível — na construção dos pesos e em cada máscara de dropout durante o treino.

Medido: `independente` + `thread` deu **3,51 pp de amplitude de OA** entre execuções
que deveriam ser idênticas (0,8296 a 0,8648 com N=32), enquanto `sequencial` e
`process` repetiram o valor exatamente (0,00 pp) em todos os N. A amplitude cresce
com o número de threads, que é a assinatura esperada de disputa pelo gerador.

Não tem conserto por semente: enquanto as partições compartilharem o interpretador,
compartilham o RNG. Processos resolvem por construção, cada um com o seu. É o
**segundo** não-determinismo encontrado nestes experimentos, depois do LSC; desta vez
foi detectado pela conferência de repetibilidade embutida no próprio experimento.
**Número de acurácia de partição deve sair do executor `process`.**

**Consequência para o experimento 11:** as colunas de OA do executor `thread` no modo
`independente` carregam essa variação dentro do desvio entre sementes, e ela é grande
o bastante para engoli-las — os ΔOA de +2,82 a +3,47 pp ficam **dentro** da amplitude
de 3,51 pp do próprio não-determinismo. Não são afirmáveis a partir daquele executor.
O experimento 12 já os havia derrubado por outro motivo, e o número que vale vem do
`process`, onde a OA se repete exatamente.

**DataLoader adiantando o gerador semeado.** Iterar o loader de treino para inspecionar
um batch sorteava uma permutação e mudava silenciosamente o resultado. `fit()` repõe a
semente antes da primeira época.

**`pkill -f` matando o próprio shell.** `pkill -9 -f "06_segmentadores"` casou
com a linha de comando do bash que o executava, e o comando morreu antes de relançar o
experimento. Usar `setsid` para lançar e padrões específicos para matar.

---

## 10. Limitações e pendências

### 10.1 Limitações a declarar no texto

1. **Validação cruzada por região só em Indian Pines.** Pavia e Salinas faltam.
2. **Sob split por região, a partição passa a depender do segmentador**, o que confunde
   a comparação *entre* segmentadores: cada um define grupos diferentes. Não há saída
   limpa; é limitação, não resultado.
3. **Os números de LSC sob otimização conjunta são anteriores à correção do
   determinismo** e precisam ser refeitos nas três cenas.
4. **O split por região deixa classes inteiras fora do treino** em alguns folds (o log
   avisa), o que puxa a AA para baixo de forma não comparável ao split por pixel.
5. **A medição de paralelismo por partição é só em Indian Pines e só em CPU.**
6. **O ranking de estratégias de HPO não é estável entre protocolos** — com 100% dos
   dados a dispersão cai com o orçamento; com 10%, cresce. É resultado metodológico
   legítimo sobre HPO multi-fidelidade, mas impede afirmar "a estratégia X é melhor".

### 10.2 Pendências

- Refazer o braço "receita herdada" do experimento 10 com batch 32, o do original
  (seção 1.4); o experimento usou 128.
- Refazer o LSC sob otimização conjunta, pós-correção, nas três cenas.
- Comparar segmentadores sob split por região com **uma partição comum** a todos
  (ex.: as regiões do SEEDS, ou blocos grandes), para que só a fusão mude (seção 8.4).
- Medir o ganho do HPO (experimento 10) também sob split por região: sob split por
  pixel a posição sozinha já dá 0,97 em IP.
- Validação cruzada por região em PU e SA.
- Pool quente em PU e SA — partições maiores devem mudar a conta da contenção.
- Medir o executor `forkserver`: é o caminho para passar de 16 processos.
- Medir o regime que mais interessa ao HPO: um pool quente atendendo **trials
  sucessivos**, que são independentes entre si — o ganho deve ser quase linear.
- Implementar a métrica *amostras de treino por região* como diagnóstico anterior ao
  treino.
- Refazer as figuras (ver seção 11.3).

---

## 11. Mapa dos arquivos

### 11.1 Estrutura

O código está no pacote `hsi/`; o notebook importa dele e guarda a narrativa e as
células de execução. Este `Context.md` é o registro escrito.

```
hsi/
├── config.py              caminhos, ExperimentConfig, tabelas por cena
├── ambiente.py            versões, GPU, determinismo, sementes
├── dados/
│   ├── cenas.py           leitura dos .mat, tabela de pixels
│   ├── preprocessamento.py  PCA e composição RGB
│   ├── segmentacao.py     SEEDS, SLIC, LSC, Watershed
│   └── particoes.py       splits (pixel, bloco, região), PreparedData, DataLoaders
├── paralelo/
│   ├── ramos.py           pilar 2: ramos espectral e espacial (prepare_data_parallel)
│   └── particao.py        pilar 2: a cena partida em N pedaços
├── modelo.py              CNN1D com a fusão do id do superpixel
├── treino.py              laço de treino e parada antecipada
├── avaliacao.py           OA, AA, Kappa, matriz de confusão
├── checkpoint.py          salvar e recarregar modelos
├── pipeline.py            run_experiment: da cena às métricas
├── variantes.py           linha de base contra a nossa implementação
├── hpo/                   pilar 3: espaco, estrategias, busca, selecao, varredura_batch, graficos
├── mlops/                 rastreio, registro, observabilidade, campanha, relatorios, paineis
├── visualizacao/          diagnosticos, glossario, explicabilidade, dashboards, explicacao_inversa
└── experimentos/          infra, medicoes, e05 … e13 e o índice EXPERIMENTOS
```

| O que | Onde |
|---|---|
| Código (implementação e experimentos) | `hsi/` |
| Narrativa, diagnósticos e execução interativa | `HSI_MLOps.ipynb` |
| Lançador headless dos experimentos | `experimentos/executar.py` |
| Instalação do pacote | `pyproject.toml` (`uv pip install -e . --no-deps`) |
| Logs e CSVs | `experimentos/resultados/` |
| Medições invalidadas pelo LSC | `experimentos/resultados/invalidados_lsc/` |
| Cenas de entrada | `data/` (arquivos `.mat` em `data/dataHSI/`) |
| Artefatos, relatórios, checkpoints e campanhas | `outputs/` |
| Versões validadas das dependências | `requirements.txt` |
| Texto do TCC e material acadêmico | `tcc/` |

### 11.2 Experimento → o que mede → seção deste documento

| Experimento | O que mede | Seção |
|---|---|---|
| `experimento_05` | o desenho inválido, preservado | 7.2 |
| `experimento_06` | cada segmentador com seus hiperparâmetros | 7.3 |
| `experimento_08` | estratégias de busca, 3 sementes | 6.1 |
| `experimento_09` | quanto do resultado sobrevive sem vazamento | 8.2 |
| `experimento_10` | otimizado contra a receita herdada | 6.3 |
| `experimento_11` | partição da cena, escala e acurácia | 5.2 |
| `experimento_12` | a partição sob split por região | 8.3 |
| `experimento_13` | pool frio contra quente, até N=32 | 5.3, 5.4 |

### 11.3 Figuras

As 28 figuras, o gerador `graficos.py` e o estilo `estilo.py` foram apagados para
limpar o ambiente. Estão no git e precisam ser refeitas com os números atuais:

```bash
git checkout 614b31d -- experimentos/figuras experimentos/graficos.py experimentos/estilo.py
```

---
