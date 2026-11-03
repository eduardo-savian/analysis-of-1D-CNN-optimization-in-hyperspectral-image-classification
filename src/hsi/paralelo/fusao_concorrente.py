"""Fusão concorrente: T0 segmenta enquanto T1…TN extraem características e fundem.

É o desenho do orientador (02/11), passado para código:

    HSI ─PCA(3)→ RGB ─T0: SEEDS→ mapa de segmentação (MS) ──────────┐
                                                                    ▼
    pixel ─T1…TN: CNN, só as convoluções→ vetor de características ─(+)─ densas → classe

- **T0** roda o ramo espacial (PCA de 3 componentes, composição RGB e SEEDS) e
  publica o mapa de segmentação `(H, W)` numa estrutura compartilhada.
- **T1…TN** são um pool de threads. Cada tarefa recebe posições do mapa — uma só,
  com `bloco=1`, que é o "cada thread recebe uma posição do MS" do desenho —, passa
  o espectro desses pixels pelas convoluções, lê o id do segmento de cada posição
  e concatena (`CNN1D.fuse`). A cabeça densa fecha a tarefa.

As convoluções não dependem do mapa: é isso que deixa as threads trabalharem
enquanto T0 ainda segmenta. Só a concatenação espera por ele.

A TRAVA. O desenho põe um mutex em volta da segmentação, em T0, e em volta da
concatenação, em T1…TN. Um mutex sozinho, porém, não ordena ninguém: se uma thread
de trabalho pegar a trava antes de T0, ela lê o mapa ainda vazio. Por isso a trava
é a de uma `threading.Condition` com a marca `publicado`: quem chega antes espera
(`wait_for`, que solta a trava enquanto espera) até T0 publicar. Duas extensões da
seção crítica, comparáveis porque o resto é idêntico:

``"desenho"``
    Como no papel: T0 segura a trava durante a segmentação inteira, e cada tarefa a
    segura durante a concatenação.
``"minima"``
    T0 só trava para publicar o mapa pronto, e cada tarefa só para ler os ids. Depois
    de publicado o mapa só é lido, e leitura concorrente de um array imutável não
    tem condição de corrida — travar a concatenação serializa as threads sem
    proteger nada.

O LIMITE DO DESENHO LITERAL. Uma thread que termina as convoluções antes de T0
publicar fica parada esperando o mapa, e ocupa a vaga dela no pool. Enquanto T0
segmenta, no máximo N × bloco posições avançam — com `bloco=1`, quase nada. A
variante `desacoplado=True` separa convolução e fusão em tarefas distintas e
enfileira todas as convoluções antes, o que dá a sobreposição máxima que o esquema
permite. As duas são medidas no experimento 14.

O QUE NÃO ESTÁ AQUI: O TREINO. No treino a perda de cada batch já depende do id do
segmento, desde o primeiro, e o gradiente atravessa as convoluções; não há o que
sobrepor à segmentação, que de todo modo roda uma vez só. O esquema vale para a
passagem para a frente de um modelo treinado — a inferência.

AS THREADS DO POOL PRECISAM DA MESMA CONTAGEM DE THREADS DO PYTORCH. O número de
threads do OpenMP é estado de cada thread: `torch.set_num_threads(1)` na thread
principal não vale nas threads do pool, que sobem com o padrão (32 nesta máquina). Uma
convolução de um bloco de 64 pixels passava de 0,2 ms para ~100 ms, só de montar e
sincronizar uma equipe de 32 threads por operação — medido em 02/11, e foi o que
invalidou a primeira rodada do experimento 14. Por isso todo pool daqui é criado com
`_herdar_threads`, que repete nas threads de trabalho a contagem de quem as criou.

O resultado é **idêntico** ao do pipeline sequencial em lote: os mesmos ids e as
mesmas classes, conferidos a cada execução no experimento 14. Concorrência muda
quando cada passo roda, não o que ele calcula.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import torch

from hsi.paralelo.ramos import spatial_branch

SECOES = ("desenho", "minima")


class MapaSegmentacao:
    """O mapa de segmentação (MS) que T0 escreve e T1…TN leem.

    Começa preenchido com −1, que nenhum segmentador produz, para que uma leitura
    antes da publicação seja visível em vez de silenciosa.
    """

    def __init__(self, shape: tuple[int, int]) -> None:
        self.labels = np.full(shape, -1, dtype=np.int32)
        self.condicao = threading.Condition(threading.Lock())
        self.publicado = False

    def publicar(self, labels: np.ndarray) -> None:
        """Copia o mapa pronto e acorda quem espera. Chamar com a trava em mãos."""
        self.labels[...] = labels
        self.publicado = True
        self.condicao.notify_all()

    def esperar(self) -> None:
        """Bloqueia até a publicação. Chamar com a trava em mãos."""
        self.condicao.wait_for(lambda: self.publicado)


@dataclass(frozen=True)
class EntradaFusao:
    """O que a inferência concorrente consome, já fora do disco.

    `espectros` são os pixels rotulados já reduzidos pelo PCA espectral, na mesma
    ordem de `coords` — é a entrada da CNN. `pixels` e `raw_labels` são a cena
    inteira, que o ramo espacial de T0 precisa (os superpixels usam a imagem
    completa, background incluído).
    """

    pixels: np.ndarray
    raw_labels: np.ndarray
    scene_shape: tuple[int, int]
    params: object
    seed: int
    segmenter: str
    espectros: np.ndarray
    coords: np.ndarray


@dataclass
class Tarefa:
    """Linha do tempo de uma tarefa de T1…TN, em segundos desde o início do bloco."""

    thread: str
    n_posicoes: int
    inicio: float
    fim_convolucoes: float
    pegou_mapa: float
    fim: float

    @property
    def espera(self) -> float:
        """Do fim das convoluções até a leitura do mapa: espera por T0 e fila na trava."""
        return self.pegou_mapa - self.fim_convolucoes


@dataclass
class ResultadoFusao:
    """Previsões e tempos de uma execução, concorrente ou de referência."""

    arranjo: str
    n_threads: int
    bloco: int
    secao: str
    previsto: np.ndarray
    ids: np.ndarray
    parede: float
    t0_inicio: float = 0.0
    t0_fim: float = 0.0
    tarefas: list[Tarefa] = field(default_factory=list)

    @property
    def espera_total(self) -> float:
        return sum(t.espera for t in self.tarefas)

    @property
    def espera_maxima(self) -> float:
        return max((t.espera for t in self.tarefas), default=0.0)

    @property
    def convolucoes_antes_do_mapa(self) -> float:
        """Fração das posições cujas convoluções terminaram antes de T0 publicar.

        É a sobreposição que o desenho busca: 0 significa que as threads só
        começaram depois da segmentação; perto de 1, que fizeram quase tudo durante.
        """
        if not self.tarefas:
            return 0.0
        antes = sum(t.n_posicoes for t in self.tarefas if t.fim_convolucoes <= self.t0_fim)
        return antes / sum(t.n_posicoes for t in self.tarefas)


def _herdar_threads(n_threads_torch: int) -> None:
    """Inicializador das threads do pool: a mesma contagem de threads do PyTorch do criador."""
    torch.set_num_threads(n_threads_torch)


def _novo_pool(n_threads: int) -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=n_threads, thread_name_prefix="T",
                              initializer=_herdar_threads,
                              initargs=(torch.get_num_threads(),))


def _blocos(n: int, bloco: int) -> list[np.ndarray]:
    """Divide as `n` posições em fatias contíguas de `bloco` (a última pode ser menor)."""
    return [np.arange(i, min(i + bloco, n)) for i in range(0, n, bloco)]


def _segmentar(entrada: EntradaFusao) -> np.ndarray:
    """Ramo espacial inteiro — PCA(3), RGB e segmentador — devolvendo o mapa `(H, W)`."""
    _, _, _, segmentacao = spatial_branch(
        entrada.pixels, entrada.raw_labels, entrada.scene_shape,
        entrada.params, entrada.seed, entrada.segmenter,
    )
    return segmentacao.labels


def _t0(entrada: EntradaFusao, mapa: MapaSegmentacao, secao: str,
        origem: float, relogio: dict) -> None:
    """Corpo de T0: segmenta e publica, com a trava na extensão pedida."""
    relogio["t0_inicio"] = time.perf_counter() - origem
    if secao == "desenho":
        with mapa.condicao:
            mapa.publicar(_segmentar(entrada))
    else:
        labels = _segmentar(entrada)
        with mapa.condicao:
            mapa.publicar(labels)
    relogio["t0_fim"] = time.perf_counter() - origem


def _convolucoes(modelo, entrada: EntradaFusao, posicoes: np.ndarray, origem: float):
    """Primeira metade de uma tarefa: o espectro pelas convoluções. Não toca no mapa.

    O `inference_mode` fica aqui dentro, e não em volta do pool: o modo de gradiente
    do PyTorch é por thread, e um `no_grad` aberto na thread principal não vale nas
    threads do pool.
    """
    inicio = time.perf_counter() - origem
    with torch.inference_mode():
        x = torch.from_numpy(entrada.espectros[posicoes]).float().unsqueeze(1)
        caracteristicas = modelo.features(x)
    return posicoes, caracteristicas, inicio, time.perf_counter() - origem


def _fundir(modelo, entrada: EntradaFusao, mapa: MapaSegmentacao, convolucoes: tuple,
            secao: str, origem: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, Tarefa]:
    """Segunda metade: espera o mapa, lê os ids, concatena e passa pelas densas."""
    posicoes, caracteristicas, inicio, fim_conv = convolucoes
    linhas, colunas = entrada.coords[posicoes, 0], entrada.coords[posicoes, 1]
    with torch.inference_mode():
        if secao == "desenho":
            with mapa.condicao:
                mapa.esperar()
                pegou = time.perf_counter() - origem
                ids = mapa.labels[linhas, colunas].copy()
                fundido = modelo.fuse(caracteristicas,
                                      torch.from_numpy(ids).float().reshape(-1, 1))
        else:
            with mapa.condicao:
                mapa.esperar()
                pegou = time.perf_counter() - origem
                ids = mapa.labels[linhas, colunas].copy()
            fundido = modelo.fuse(caracteristicas, torch.from_numpy(ids).float().reshape(-1, 1))
        previsto = modelo.classifier(fundido).argmax(dim=1).numpy()
    fim = time.perf_counter() - origem
    registro = Tarefa(threading.current_thread().name, len(posicoes),
                      inicio, fim_conv, pegou, fim)
    return posicoes, ids, previsto, registro


def _tarefa(modelo, entrada, mapa, posicoes, secao, origem):
    """Uma tarefa do desenho: convoluções e fusão na mesma thread, em sequência.

    A thread que termina as convoluções antes de T0 publicar fica parada na espera
    do mapa — e com ela a vaga no pool. Por isso, enquanto T0 segmenta, no máximo
    N × bloco posições avançam; o resto da fila espera thread livre.
    """
    return _fundir(modelo, entrada, mapa, _convolucoes(modelo, entrada, posicoes, origem),
                   secao, origem)


def _tarefa_fusao(modelo, entrada, mapa, futuro_convolucoes, secao, origem):
    """A fusão da variante desacoplada: pega as convoluções já feitas e funde.

    Não há impasse em esperar outro futuro de dentro do pool: todas as tarefas de
    convolução são submetidas antes de qualquer fusão, e a fila do executor é FIFO.
    Quando uma fusão é retirada da fila, a convolução dela já saiu antes — está
    rodando ou terminou.
    """
    return _fundir(modelo, entrada, mapa, futuro_convolucoes.result(), secao, origem)


def executar_concorrente(modelo, entrada: EntradaFusao, n_threads: int, bloco: int = 1,
                         secao: str = "minima", desacoplado: bool = False,
                         pool: ThreadPoolExecutor | None = None) -> ResultadoFusao:
    """Executa o desenho: T0 segmenta, `n_threads` threads extraem e fundem.

    `desacoplado=False` é o desenho literal: cada tarefa faz convoluções e fusão.
    `desacoplado=True` separa as duas metades em tarefas distintas e enfileira todas
    as convoluções primeiro, para que nenhuma thread fique parada esperando o mapa
    enquanto ainda há convolução por fazer. É a sobreposição máxima que o desenho
    permite, e mede o teto dele.

    `pool`, quando dado, precisa ter `n_threads` workers e já ter rodado uma vez: as
    threads do executor nascem sob demanda, e a primeira operação do PyTorch em cada
    uma inicializa estado preguiçoso. É a mesma separação entre partida e
    paralelismo do experimento 13, agora com threads.
    """
    if secao not in SECOES:
        raise ValueError(f"secao inválida: {secao!r}. Use uma de {SECOES}.")
    modelo.eval()
    mapa = MapaSegmentacao(entrada.scene_shape)
    relogio: dict[str, float] = {}
    proprio = pool is None
    pool = pool or _novo_pool(n_threads)

    origem = time.perf_counter()
    try:
        # T0 sobe antes de qualquer tarefa ser submetida. Isso não é o que garante a
        # ordem — quem garante é `MapaSegmentacao.esperar` —, mas deixa a trava do
        # modo "desenho" com T0 desde o começo, como no papel.
        t0 = threading.Thread(target=_t0, args=(entrada, mapa, secao, origem, relogio),
                              name="T0")
        t0.start()
        blocos = _blocos(len(entrada.espectros), bloco)
        if desacoplado:
            convolucoes = [pool.submit(_convolucoes, modelo, entrada, posicoes, origem)
                           for posicoes in blocos]
            futuros = [pool.submit(_tarefa_fusao, modelo, entrada, mapa, c, secao, origem)
                       for c in convolucoes]
        else:
            futuros = [pool.submit(_tarefa, modelo, entrada, mapa, posicoes, secao, origem)
                       for posicoes in blocos]
        saidas = [f.result() for f in futuros]
        t0.join()
    finally:
        if proprio:
            pool.shutdown()
    parede = time.perf_counter() - origem

    previsto = np.empty(len(entrada.espectros), dtype=np.int64)
    ids = np.empty(len(entrada.espectros), dtype=np.int32)
    for posicoes, ids_bloco, previsto_bloco, _ in saidas:
        previsto[posicoes] = previsto_bloco
        ids[posicoes] = ids_bloco
    return ResultadoFusao(
        arranjo="desacoplado" if desacoplado else "concorrente",
        n_threads=n_threads, bloco=bloco, secao=secao,
        previsto=previsto, ids=ids, parede=parede,
        t0_inicio=relogio["t0_inicio"], t0_fim=relogio["t0_fim"],
        tarefas=[s[3] for s in saidas],
    )


def executar_sequencial(modelo, entrada: EntradaFusao, bloco: int | None = None) -> ResultadoFusao:
    """Referência: segmenta primeiro, depois infere, tudo na thread que chamou.

    Com `bloco=None` a inferência é um lote só — é o que o pipeline faz hoje, e o
    número contra o qual o desenho precisa ganhar. Com um `bloco`, percorre as
    posições na mesma granularidade das tarefas concorrentes: é o denominador justo
    do speedup das threads, porque só a concorrência muda.
    """
    modelo.eval()
    origem = time.perf_counter()
    labels = _segmentar(entrada)
    t0_fim = time.perf_counter() - origem
    n = len(entrada.espectros)
    previsto = np.empty(n, dtype=np.int64)
    ids = labels[entrada.coords[:, 0], entrada.coords[:, 1]].astype(np.int32)
    with torch.inference_mode():
        for posicoes in _blocos(n, bloco or n):
            x = torch.from_numpy(entrada.espectros[posicoes]).float().unsqueeze(1)
            seg = torch.from_numpy(ids[posicoes]).float().reshape(-1, 1)
            previsto[posicoes] = modelo.classify(modelo.features(x), seg).argmax(dim=1).numpy()
    parede = time.perf_counter() - origem
    return ResultadoFusao(
        arranjo="lote" if bloco is None else "sequencial", n_threads=1,
        bloco=bloco or n, secao="-", previsto=previsto, ids=ids, parede=parede,
        t0_inicio=0.0, t0_fim=t0_fim,
    )


def aquecer_pool(modelo, entrada: EntradaFusao, n_threads: int) -> ThreadPoolExecutor:
    """Cria o pool e roda uma passagem em cada thread, fora de qualquer medição.

    Uma barreira de `n_threads` partes obriga o executor a criar todas as threads:
    sem ela, uma thread rápida pegaria várias tarefas de aquecimento e as outras
    nasceriam frias, já dentro do bloco medido.
    """
    pool = _novo_pool(n_threads)
    barreira = threading.Barrier(n_threads)
    x = torch.from_numpy(entrada.espectros[:8]).float().unsqueeze(1)
    seg = torch.zeros(len(x), 1)

    def passagem() -> None:
        with torch.inference_mode():
            modelo.classify(modelo.features(x), seg)
        barreira.wait()

    for futuro in [pool.submit(passagem) for _ in range(n_threads)]:
        futuro.result()
    return pool
