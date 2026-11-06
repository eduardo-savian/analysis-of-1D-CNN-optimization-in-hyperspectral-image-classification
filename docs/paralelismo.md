# Paralelismo no TCC: como cada forma foi implementada

Este documento explica, com o código do repositório, as quatro camadas de paralelismo do trabalho:

| Nível | O que roda ao mesmo tempo | Onde está | Experimento que mede |
|---|---|---|---|
| 0 | Operações dentro de uma etapa (GPU e threads internas) | PyTorch, OpenCV, [`ambiente.py`](../src/hsi/ambiente.py) | (infraestrutura) |
| 1 | Os dois ramos do pré-processamento | [`dados.py`](../src/hsi/dados.py), `prepare_data` | só medição avulsa antiga |
| 2 | Partição da cena em N faixas | [`paralelismo/particao.py`](../src/hsi/paralelismo/particao.py) | 06 e 07 |
| 3 | Fusão concorrente (segmentação × convoluções) | [`paralelismo/fusao_concorrente.py`](../src/hsi/paralelismo/fusao_concorrente.py) | 08 |

Os níveis 2 e 3 são o pilar de paralelismo do TCC. Os níveis 0 e 1 são a base sobre a qual eles rodam.

Os trechos de código abaixo são copiados do pacote, às vezes encurtados com `...`.

---

## Nível 0: paralelismo dentro de cada operação

### GPU

No treino normal (`pipeline.run_experiment`), o DataLoader entrega lotes de 32 a 128 pixels, e cada lote vai inteiro para a GPU. Os núcleos CUDA calculam as convoluções e as camadas densas de todos os pixels do lote ao mesmo tempo. Isso vem pronto do PyTorch; o trabalho só escolhe o dispositivo:

```python
# src/hsi/ambiente.py
def select_device(verbose: bool = True) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ...
```

### Threads internas da CPU, controladas de propósito

O PyTorch e o OpenCV dividem cada operação entre várias threads. O trabalho limita isso em dois lugares.

O primeiro é o OpenCV, que fica com uma thread, porque o LSC paralelo devolvia rótulos diferentes a cada chamada com a mesma imagem:

```python
# src/hsi/ambiente.py
def configure_determinism() -> None:
    cv.setNumThreads(1)
    ...
```

O segundo são os workers dos pools (níveis 2 e 3), que usam uma thread do PyTorch cada. O número de threads do PyTorch vale por thread, e um processo novo sobe com o padrão da máquina (32). Sem o limite, N workers usariam N × 32 threads nos mesmos núcleos:

```python
# src/hsi/paralelismo/particao.py
def init_worker():
    """Roda uma vez em cada thread ou processo do pool."""
    torch.set_num_threads(1)
    cv.setNumThreads(1)
```

---

## Nível 1: os dois ramos do pré-processamento

A preparação dos dados tem dois ramos que não dependem um do outro:

```
ramo espectral:  pixels rotulados ─► PCA (30 ou 15 componentes) ─► espectros reduzidos ─┐
                                                                                        ├─► divisão e DataLoaders
ramo espacial:   cena inteira ─► PCA(3) ─► composição RGB ─► SEEDS ─► mapa de regiões ──┘
```

Com `backend="thread"`, os dois rodam em duas threads:

```python
# src/hsi/dados.py
def prepare_data(config, backend: str = "sequential") -> dict:
    ...
    args_espectral = (espectros, config.pca_components, SEMENTE_DADOS)
    args_espacial = (pixels, gt.shape, config.segmenter, segmenter_params(config), SEMENTE_DADOS)
    if backend == "thread":
        # PCA e OpenCV liberam o GIL no trabalho pesado, então as duas threads rodam
        # de fato ao mesmo tempo, compartilhando os arrays sem cópia.
        with ThreadPoolExecutor(max_workers=2) as pool:
            espacial = pool.submit(spatial_branch, *args_espacial)
            reduzidos = spectral_branch(*args_espectral)
            mapa = espacial.result()
    else:
        reduzidos = spectral_branch(*args_espectral)
        mapa = spatial_branch(*args_espacial)
```

**Por que threads funcionam aqui apesar do GIL:** o GIL (Global Interpreter Lock) impede duas threads de executarem código Python ao mesmo tempo. Porém, o NumPy e o OpenCV soltam o GIL enquanto fazem a conta em C. Então o PCA de uma thread e o SEEDS da outra rodam de fato em paralelo, e ainda compartilham os arrays sem copiar.

**Quem usa:** a partição da cena (nível 2) prepara os dados com `prepare_data(config, "thread")`. O treino normal usa o modo sequencial.

---

## Nível 2: partição da cena (experimentos 06 e 07)

### A ideia

Dividir a cena em N faixas horizontais e processar as faixas ao mesmo tempo. Cada faixa tem um número parecido de pixels rotulados, então o teto teórico do speedup é perto de N.

```
cena inteira ─► prepare_data (uma vez) ─► partition_scene ─► faixas 0..N-1
                                                                  │
                         ┌────────────────┬───────────────────────┼──────────────┐
                         ▼                ▼                       ▼              ▼
                  train_and_predict  train_and_predict   ...  train_and_predict
                     (faixa 0)          (faixa 1)                (faixa N-1)
                         └────────────────┴──────────┬────────────┴──────────────┘
                                                     ▼
                              junta as previsões ─► OA, AA e Kappa no teste inteiro
```

### Passo 1: segmentar uma vez e cortar em faixas

A segmentação roda uma vez na cena inteira, **antes** de partir. Segmentar cada pedaço daria superpixels diferentes nas fronteiras. Depois, as amostras são ordenadas pela linha da cena e cortadas em N partes do mesmo tamanho:

```python
# src/hsi/paralelismo/particao.py
def partition_scene(coords: np.ndarray, n: int) -> np.ndarray:
    """Índice da partição de cada amostra: as amostras ordenadas pela linha da cena e
    cortadas em N faixas do mesmo tamanho; uma fronteira pode cair no meio de uma linha.
    """
    ordem = np.argsort(coords[:, 0], kind="stable")
    particao = np.empty(len(coords), dtype=np.int64)
    for k, bloco in enumerate(np.array_split(ordem, n)):
        particao[bloco] = k
    return particao
```

### Passo 2: recortar os dados de cada faixa

Cada faixa recebe só os seus pixels. Os índices de treino, validação e teste são renumerados para apontar dentro do pedaço:

```python
def slice_data(d: dict, mascara: np.ndarray) -> dict:
    """Recorta os dados para as amostras de uma partição, renumerando os índices."""
    indices = np.flatnonzero(mascara)
    nova_posicao = np.full(len(mascara), -1)
    nova_posicao[indices] = np.arange(len(indices))

    def renumerar(parte):
        posicoes = nova_posicao[parte]
        return posicoes[posicoes >= 0]

    return {**d, "espectros": d["espectros"][indices], "rotulos": d["rotulos"][indices],
            "segmentos": d["segmentos"][indices], "coords": d["coords"][indices],
            "treino": renumerar(d["treino"]), "val": renumerar(d["val"]),
            "teste": renumerar(d["teste"])}
```

### Passo 3: o trabalho de uma faixa

```python
def train_and_predict(tarefa):
    """Trabalho de uma partição: treina (se não vier modelo) e prevê o teste dela.

    Fica no nível do módulo porque o executor de processos a envia por pickle, que
    só serializa funções pelo nome.
    """
    k, pedaco, config, modelo = tarefa
    inicio = time.perf_counter()
    loaders = build_loaders(pedaco, config, CPU)
    if modelo is None:
        seed_everything(config.seed)
        modelo, perda, otimizador = build_model(config, pedaco, CPU)
        fit(modelo, loaders, perda, otimizador, config, CPU)
    previsto = predict(modelo, loaders["teste"], CPU)
    return k, previsto, pedaco["rotulos"][pedaco["teste"]], time.perf_counter() - inicio
```

Repare que a função precisa ficar no nível do módulo: o executor de processos manda a tarefa para outro interpretador por `pickle`, e `pickle` só sabe serializar funções pelo nome. É também por isso que a partição em processos não funciona com funções definidas dentro do notebook: o processo novo não as enxerga, e o notebook chama a partição a partir do pacote.

### Os dois modos

| Modo | O que é paralelo | Como funciona |
|---|---|---|
| `independente` | Treino e inferência | Cada faixa treina e testa o próprio modelo (`modelo=None` na tarefa) |
| `replicado` | Só a inferência | Um modelo é treinado uma vez com a cena inteira, descido para a CPU e enviado a todas as faixas |

```python
def run_experiment_particionado(config, n: int, modo: str = "independente",
                                executor: str = "thread", pool=None) -> dict:
    if modo == "replicado" and executor == "process":
        raise ValueError("o modo replicado com processos exigiria enviar o modelo a cada um")
    configure_determinism()
    dados = prepare_data(config, "thread")
    grupo = partition_scene(dados["coords"], n)
    pedacos = [slice_data(dados, grupo == k) for k in range(n)]

    modelo = None
    if modo == "replicado":
        # Treina uma vez com tudo, no dispositivo padrão, e desce o modelo para a CPU,
        # onde todas as partições inferem.
        device = select_device(verbose=False)
        ...
        fit(modelo, loaders, perda, otimizador, config, device)
        modelo = modelo.to(CPU)

    tarefas = [(k, pedacos[k], config, modelo) for k in range(n)]
    ...
```

### Os três executores

Todas as faixas rodam na **CPU**, para que os três executores sejam comparados no mesmo dispositivo:

```python
def new_pool(executor: str, n: int):
    """Pool de `n` workers, ou None para o executor sequencial."""
    if executor == "sequencial":
        return None
    if executor == "thread":
        return ThreadPoolExecutor(max_workers=n, initializer=init_worker)
    # spawn, e não fork: o PyTorch recusa fork depois que o autograd rodou em threads.
    # O preço é que cada processo sobe um interpretador novo e reimporta o PyTorch.
    return ProcessPoolExecutor(max_workers=n, mp_context=multiprocessing.get_context("spawn"),
                               initializer=init_worker)
```

| Executor | Vantagem | Custo ou limitação |
|---|---|---|
| `sequencial` | É a referência do speedup | Nenhum paralelismo |
| `thread` | Barato de criar; compartilha memória | O treino tem muito código Python, que disputa o GIL. As faixas também dividem o gerador aleatório global do PyTorch, e a OA muda entre execuções iguais |
| `process` | Cada processo tem o próprio interpretador e o próprio GIL: paralelismo real | Subir o processo custa caro (Python novo, PyTorch reimportado, cerca de 481 MB de memória cada). Usa `spawn` porque o PyTorch recusa `fork` depois de usar threads |

O laço que distribui as tarefas é o mesmo para os três:

```python
    pool_frio = pool is None and executor != "sequencial"
    inicio = time.perf_counter()
    if pool_frio:
        pool = new_pool(executor, n)
    if executor == "sequencial":
        saidas = [train_and_predict(t) for t in tarefas]
    else:
        saidas = list(pool.map(train_and_predict, tarefas))
    if pool_frio:
        pool.shutdown()
    tempo_paralelo = time.perf_counter() - inicio
```

### Pool frio e pool quente

- **Pool frio:** o pool é criado e fechado **dentro** da medição. O tempo inclui subir os processos.
- **Pool quente:** o pool é criado e aquecido **antes**, e só o trabalho é medido. É o caso de um serviço que deixa os workers prontos.

O aquecimento força cada worker a terminar de subir e a inicializar o autograd:

```python
def _aquecer(_):
    # Força o worker a terminar de subir e a inicializar o autograd antes da medição.
    # O sleep segura o worker ocupado, para o pool criar todos em vez de reusar um.
    x = torch.zeros(8, 1, 64, requires_grad=True)
    nn.Conv1d(1, 16, 9)(x).sum().backward()
    time.sleep(0.25)


def warm_up(pool, n: int) -> float:
    """Sobe e aquece os `n` workers de um pool; devolve quanto isso levou."""
    inicio = time.perf_counter()
    if pool is not None:
        list(pool.map(_aquecer, range(n)))
    return time.perf_counter() - inicio
```

### O que a função devolve

```python
    return {
        **metrics(verdade, previsto),
        "tempo_paralelo": tempo_paralelo,          # parede do bloco concorrente
        "soma_partes": sum(tempos),                # soma do tempo de cada faixa
        "teto": sum(tempos) / max(tempos),         # speedup máximo possível com essa divisão
        "overhead": tempo_paralelo - max(tempos),  # o que passou da faixa mais lenta
    }
```

- **Speedup honesto:** é o tempo do executor `sequencial` dividido pelo do executor paralelo, com as mesmas faixas. Não é `soma_partes / tempo_paralelo`, porque com executor concorrente cada parte já foi medida sob disputa.
- **Teto:** com faixas desiguais, a mais lenta limita tudo (lei de Amdahl).

### Experimento 06: escala e custo em acurácia

[`experimentos/e06_particao.py`](../src/hsi/experimentos/e06_particao.py) treina o monolítico e depois os dois modos com N = 2, 4, 8 e 16, em 3 sementes, 25 épocas e lote 64:

```python
MODOS = ("independente", "replicado")
...
        for modo in MODOS:
            for n in PARTICOES:
                for semente in SEMENTES_PARTICAO:
                    ...
                    executor = "process" if modo == "independente" else "thread"
                    r = run_experiment_particionado(_config_particao(cena, semente), n, modo,
                                                    executor=executor)
```

O modo independente roda em processos para a OA se repetir entre execuções iguais. O replicado só infere, sem nenhum sorteio, e pode usar threads.

**A pergunta do 06:** quanto se perde de acurácia ao treinar N modelos com 1/N dos dados cada (independente), e quanto se perde ao só dividir a inferência (replicado)?

### Experimento 07: executores, pool frio e quente

[`experimentos/e07_pool_quente.py`](../src/hsi/experimentos/e07_pool_quente.py) mede o tempo dos três executores com N de 2 a 32, com pool frio e quente:

```python
def _medir(cena, modo, executor, n, estado) -> list[dict]:
    if executor == "process" and 1.25 * n * MB_POR_PROCESSO > _memoria_livre_mb():
        raise MemoryError(f"{n} processos não cabem na memória livre")
    ...
    if estado == "quente":
        pool = new_pool(executor, n)
        aquecimento = warm_up(pool, n)
    ...
        for rep in range(REPETICOES_POOL[estado]):
            r = run_experiment_particionado(config, n, modo, executor, pool=pool)
```

Um pool de processos que não cabe na memória livre, com 25% de folga, é recusado antes de começar. Assim a medição não cronometra a máquina usando a área de troca (*swap*). Com 15 GB de RAM, 32 processos não cabem.

**A pergunta do 07:** quanto cada executor acelera, e quanto do ganho dos processos o custo de criá-los consome?

---

## Nível 3: fusão concorrente (experimento 08)

### A ideia

Vale para a **inferência** de um modelo já treinado, que é o uso real: classificar uma cena nova. A CNN tem duas partes que não dependem uma da outra:

- as **convoluções** usam só o espectro do pixel;
- o **id do superpixel** só entra depois, na concatenação antes das camadas densas.

Por isso, as convoluções podem rodar enquanto a segmentação ainda está calculando o mapa:

```
T0:      PCA(3) ─► RGB ─► SEEDS ──────────────────► publica o mapa (notify_all)
                                                           │
T1..TN:  bloco de pixels ─► convoluções ─► espera o mapa ──┴─► lê o id ─► concatena ─► densas ─► classe
                            (não precisam do mapa)
```

A rede foi escrita com essas partes separadas justamente para isso:

```python
# src/hsi/modelo.py
    def fundir(self, caracteristicas: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        return torch.cat([caracteristicas, seg], dim=1)

    def forward(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        return self.densas(self.fundir(self.convolucoes(x), seg))
```

### O mapa compartilhado e a trava

O mapa começa com −1, um valor que nenhum segmentador produz, e é protegido por uma `threading.Condition`:

```python
# src/hsi/paralelismo/fusao_concorrente.py
    mapa = {"rotulos": np.full(entrada["forma"], -1, dtype=np.int32),
            "publicado": False, "trava": threading.Condition()}
```

**Por que uma `Condition`, e não um mutex simples:** um mutex só garante que uma thread de cada vez mexa no mapa. Ele não garante que T0 escreva **antes** de alguém ler. Uma thread de trabalho rápida poderia pegar a trava primeiro e ler o mapa ainda vazio. Com a `Condition`, quem chega antes dorme em `wait_for` até T0 avisar com `notify_all`:

```python
def _publicar(mapa: dict, rotulos: np.ndarray) -> None:
    # Chamar com a trava em mãos.
    mapa["rotulos"][...] = rotulos
    mapa["publicado"] = True
    mapa["trava"].notify_all()
```

### T0: a thread que segmenta

```python
def _t0(entrada, mapa, secao, origem, relogio):
    relogio["t0_inicio"] = time.perf_counter() - origem
    if secao == "desenho":
        with mapa["trava"]:
            _publicar(mapa, _segmentar(entrada))
    else:
        rotulos = _segmentar(entrada)
        with mapa["trava"]:
            _publicar(mapa, rotulos)
    relogio["t0_fim"] = time.perf_counter() - origem
```

### T1..TN: convoluções e fusão

```python
def _convolucoes(modelo, entrada, posicoes, origem):
    inicio = time.perf_counter() - origem
    # O modo sem gradiente vale por thread: precisa ser ligado dentro da tarefa.
    with torch.inference_mode():
        x = torch.from_numpy(entrada["espectros"][posicoes]).float().unsqueeze(1)
        caracteristicas = modelo.convolucoes(x)
    return posicoes, caracteristicas, inicio, time.perf_counter() - origem


def _fundir(modelo, entrada, mapa, convolucoes, secao, origem):
    """Espera o mapa, lê o id de cada posição, concatena e passa pelas densas."""
    posicoes, caracteristicas, inicio, fim_convolucoes = convolucoes
    linhas, colunas = entrada["coords"][posicoes, 0], entrada["coords"][posicoes, 1]
    with torch.inference_mode():
        with mapa["trava"]:
            mapa["trava"].wait_for(lambda: mapa["publicado"])
            pegou_mapa = time.perf_counter() - origem
            ids = mapa["rotulos"][linhas, colunas].copy()
            if secao == "desenho":
                fundido = modelo.fundir(caracteristicas, torch.from_numpy(ids).float().reshape(-1, 1))
        if secao == "minima":
            fundido = modelo.fundir(caracteristicas, torch.from_numpy(ids).float().reshape(-1, 1))
        previsto = modelo.densas(fundido).argmax(dim=1).numpy()
    ...
```

### As duas extensões da região crítica

| Seção | T0 segura a trava | Cada tarefa segura a trava |
|---|---|---|
| `desenho` | Durante a segmentação inteira | Durante a leitura dos ids e a concatenação |
| `minima` | Só para publicar o mapa pronto | Só para copiar os ids |

Na `minima`, a concatenação fica fora da trava, porque usa só dados locais da tarefa. Assim as tarefas não fazem fila umas atrás das outras.

### Blocos, pool e a variante desacoplada

- **Blocos:** os pixels são agrupados em blocos de 1, 64 ou 1.024 por tarefa (`_blocos`). Com bloco 1, cada pixel vira uma tarefa, e o custo de agendar domina.
- **Pool de threads:** `new_thread_pool` cria as threads com o mesmo número de threads do PyTorch da chamadora (uma). `warm_pool` faz uma passada do modelo em cada thread antes da medição, com uma barreira que obriga o executor a criar todas as threads.
- **Variante desacoplada:** no esquema original, uma thread que termina as convoluções antes do mapa fica parada esperando, sem pegar outro trabalho. Na desacoplada, todas as convoluções entram na fila antes das fusões:

```python
def executar_concorrente(modelo, entrada: dict, n: int, bloco: int = 1, secao: str = "minima",
                         desacoplado: bool = False, pool=None) -> dict:
    ...
    t0 = threading.Thread(target=_t0, args=(entrada, mapa, secao, origem, relogio), name="T0")
    t0.start()
    blocos = _blocos(len(entrada["espectros"]), bloco)
    if desacoplado:
        convolucoes = [pool.submit(_convolucoes, modelo, entrada, p, origem) for p in blocos]
        futuros = [pool.submit(_tarefa_fusao, modelo, entrada, mapa, c, secao, origem)
                   for c in convolucoes]
    else:
        futuros = [pool.submit(_tarefa, modelo, entrada, mapa, p, secao, origem) for p in blocos]
    saidas = [f.result() for f in futuros]
    t0.join()
```

### A referência: inferência em lote

`executar_sequencial` faz o mesmo trabalho numa thread só: segmenta e depois infere, tudo de uma vez (lote) ou na mesma granularidade das tarefas (blocos). O experimento 08 compara o esquema concorrente contra o lote com 1 e com 16 threads do PyTorch.

### Conferência de correção

Toda execução do 08 confere duas coisas. A primeira é se os ids lidos do mapa são os mesmos do treino. A segunda é se as classes previstas são idênticas às da inferência em lote. Um ganho de tempo nunca esconde um erro de sincronização.

```python
# src/hsi/experimentos/e08_fusao_concorrente.py
        "ids_iguais_ao_treino": bool(np.array_equal(r["ids"], dados["segmentos"])),
        "classes_iguais_ao_lote": bool(np.array_equal(r["previsto"], referencia)),
```

### O limite do esquema

O esquema só esconde as **convoluções** atrás da **segmentação**. Com tempo de segmentação S e tempo de inferência I, o tempo concorrente nunca fica abaixo de S. O speedup máximo sobre o lote fica perto de (S + I) / S:
- quando a segmentação domina, como em Pavia e Salinas com o SEEDS, sobra pouco a esconder;
- em Indian Pines, que tem a segmentação mais rápida, há mais margem.

---

## Como rodar

**No notebook** (`Eduardo_Savian_2026_CNN1D_SEEDS_otimizado.ipynb`):
- **Seção 6:** a fusão concorrente sobre o modelo treinado na seção 4, com N = 2, 4 e 8. Confere se as classes são iguais às do lote.
- **Seção 7** (`PARTICIONAR = True`): a partição em 4 faixas com executor sequencial, threads e processos.

**No terminal:**

```bash
.venv/bin/python src/executar.py 06 IP        # partição: escala e acurácia
.venv/bin/python src/executar.py 07 IP        # executores, pool frio e quente
.venv/bin/python src/executar.py 08           # fusão concorrente, nas três cenas (config.toml)
```

Os parâmetros (N, sementes, épocas, blocos, threads, repetições) ficam em `config.toml`, nas seções `[experimentos.06]`, `[experimentos.07]` e `[experimentos.08]`. Os CSVs vão para `experimentos/resultados/`. Uma execução interrompida continua de onde parou.

---

## O que já se sabe e o que falta medir

Os três experimentos estão no fim da fila, para rodarem com a máquina livre. Os números abaixo são de **rodadas antigas**, com a CNN de Souza e os dados de antes da correção de fidelidade. Servem só de indício do comportamento:

| Medida antiga | Valor |
|---|---|
| Partição, processos, pool quente, N = 16 | 4,47× sobre o sequencial |
| Partição, processos, pool frio, N = 16 | 1,05× |
| Partição, threads, pool frio, N = 8 | 0,23× (mais lento que o sequencial) |
| Fusão concorrente, contra o lote com 16 threads | de 1,01× (PU) a 1,20× (IP) |
| Pré-processamento e inferência numa execução de IP | cerca de 1% do tempo; o treino domina |

**Leitura provisória:**
- O ganho grande vem da partição com processos e pool quente.
- Threads perdem para o GIL quando há muito código Python, como o laço de treino.
- A fusão concorrente é correta, mas o ganho é limitado pela segmentação.
- Os números definitivos, com a CNN do projeto, saem dos experimentos 06, 07 e 08.
