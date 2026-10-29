"""Dividir a imagem em N partes e processá-las concorrentemente.

POR QUE ISTO EXISTE. O paralelismo de dois ramos (`prepare_data_parallel`) tem teto
estrutural: a segmentação é 78–93% do pré-processamento, então dois ramos
desbalanceados nunca passam de ~1,2× — medido nas três cenas. Pior, o
pré-processamento inteiro é 0,9% do tempo de execução, e o treino é o resto.

Particionar a cena ataca o lado certo. As partes têm tamanho parecido, então o teto
é ≈ N em vez de 1,2×, e o que fica paralelo é o **treino**, que é onde o tempo está.

A SEGMENTAÇÃO É FEITA UMA VEZ, NA CENA INTEIRA, antes de particionar. Isso não é
detalhe: segmentar cada parte isoladamente produziria superpixels diferentes nas
fronteiras, e os ids deixariam de ser comparáveis entre partições. Fazer uma vez
custa pouco (0,15 a 1,9 s) e mantém o contrato.

DOIS MODOS, porque respondem a perguntas diferentes:

- ``"replicado"`` — um modelo treinado com todos os dados, replicado para inferir as
  partições em paralelo. O resultado é **idêntico** ao monolítico; o que se mede é
  puro ganho de velocidade na inferência.
- ``"independente"`` — cada partição treina seu próprio modelo, com os dados dela.
  Paraleliza o treino, que é o caro, mas cada modelo vê 1/N das amostras. Como
  medimos que cortar o treino de 100% para 10% custa 14 pontos de OA, o custo em
  acurácia aqui é esperado e precisa ser quantificado, não presumido.

EXECUTORES. O `thread`, que funciona nos dois ramos da preparação porque PCA e
OpenCV soltam o GIL, não escala aqui: treinar uma CNN pequena em CPU é dominado por
código Python, que disputa o GIL, e passadas poucas partições ele fica mais lento
que o sequencial. No modo `independente` as partições ainda disputam o RNG global do
PyTorch, e a OA deixa de se repetir. Só o `process` escala e repete o resultado, com
um interpretador e um gerador por worker — ver o experimento 13.
"""

import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from hsi.ambiente import (
    configure_determinism,
    configure_opencv_determinism,
    seed_everything,
    select_device,
)
from hsi.avaliacao import Metrics, _average_accuracy, evaluate
from hsi.config import ExperimentConfig
from hsi.dados.particoes import (
    PreparedData,
    Splits,
    build_loaders,
    validate_prepared_data,
)
from hsi.modelo import build_model
from hsi.paralelo.ramos import prepare_data_parallel
from hsi.treino import fit


def particionar_cena(coords: np.ndarray, scene_shape: tuple[int, int],
                     n_particoes: int, modo: str = "faixas",
                     seed: int = 42) -> np.ndarray:
    """Atribui cada amostra a uma partição.

    ``"faixas"`` corta a cena em faixas horizontais. Faixas, e não um xadrez: elas
    preservam a contiguidade espacial de cada parte, que é como o processamento
    seria distribuído num cenário real — cada nó recebe um pedaço contíguo da
    imagem. O corte segue a ordem das linhas, com fronteiras escolhidas para
    equilibrar a **quantidade de pixels rotulados**, e não a área, porque é o número
    de amostras que determina o tempo de treino. Por isso uma fronteira pode cair no
    meio de uma linha.

    ``"aleatoria"`` embaralha os pixels e distribui sem olhar a posição. **É o
    controle do experimento, não uma alternativa de produção**: como o custo
    computacional é idêntico ao das faixas, qualquer diferença de acurácia entre os
    dois modos é atribuível à localização espacial, e não à partição em si. Serve
    para testar se um eventual ganho das faixas vem da posição ou de partir o
    problema. Se vier da posição, ainda falta separar especialização legítima de
    atalho de vizinhança, como o vazamento espacial do split por pixel — é o que o
    experimento 12 faz.
    """
    if modo not in {"faixas", "aleatoria"}:
        raise ValueError("modo de partição deve ser 'faixas' ou 'aleatoria'.")
    total = len(np.asarray(coords))
    if modo == "faixas":
        ordem = np.argsort(np.asarray(coords)[:, 0], kind="stable")
    else:
        ordem = np.random.default_rng(seed).permutation(total)
    particao = np.empty(total, dtype=np.int64)
    for indice, bloco in enumerate(np.array_split(ordem, n_particoes)):
        particao[bloco] = indice
    return particao


def _fatiar(data: "PreparedData", mascara: np.ndarray) -> "PreparedData":
    """Recorta `PreparedData` para um subconjunto de amostras, reindexando os splits.

    Os splits guardam posições absolutas; depois do recorte elas mudam. O mapa
    `posicao` traduz índice antigo → novo e descarta o que saiu da partição.
    """
    indices = np.flatnonzero(mascara)
    posicao = -np.ones(len(mascara), dtype=np.int64)
    posicao[indices] = np.arange(len(indices))
    traduz = lambda bloco: posicao[bloco][posicao[bloco] >= 0]
    return PreparedData(
        spectra=data.spectra[indices],
        labels=data.labels[indices],
        segments=None if data.segments is None else data.segments[indices],
        splits=Splits(train=traduz(data.splits.train), val=traduz(data.splits.val),
                      test=traduz(data.splits.test)),
        coords=None if data.coords is None else data.coords[indices],
        scene_shape=data.scene_shape,
    )


def _treinar_e_inferir(tarefa: tuple) -> tuple[int, np.ndarray, np.ndarray, float]:
    """Treina (quando preciso) e infere uma partição. Devolve `(k, previsto, real, s)`.

    Função de módulo, e não closure, porque precisa ser serializável para o backend
    de processos — mesma razão de `spatial_branch` na preparação dos dados. Com
    `spawn`, o filho a encontra importando `hsi.paralelo.particao`; por isso o
    executor `process` roda tanto do lançador quanto do notebook.
    """
    k, pedaco, config, n_classes, n_segments, modelo_pronto = tarefa
    marca = time.perf_counter()
    device = torch.device("cpu")
    loaders = build_loaders(pedaco, config, device)
    if modelo_pronto is None:
        seed_everything(config.seed)
        modelo, criterio, otimizador = build_model(
            config, pedaco.input_shape(), n_classes, device, n_segments=n_segments)
        fit(modelo, loaders, criterio, otimizador, config, device, verbose=False)
    else:
        modelo = modelo_pronto
    verdadeiro = pedaco.labels[pedaco.splits.test]
    _, previsto = evaluate(modelo, loaders["test"], nn.CrossEntropyLoss(),
                           device, verdadeiro)
    return k, previsto, verdadeiro, time.perf_counter() - marca


def _preparar_worker() -> None:
    """Estado que todo worker precisa antes da primeira tarefa. Roda uma vez por worker.

    Com `spawn` o filho sobe um interpretador limpo e herdaria os defaults do PyTorch
    e do OpenCV, não os da medição: 16 threads intra-operação **por worker**, o que
    transformaria N partições em N×16 threads disputando os mesmos núcleos. Aqui a
    conta volta a ser uma thread por partição, que é o que o experimento mede.
    """
    torch.set_num_threads(1)
    configure_opencv_determinism(verbose=False)


def _aquecer_worker(_: int) -> float:
    """Tarefa de aquecimento: paga o custo de partida, que não é custo de paralelismo.

    O que ela força a acontecer **antes** de a medição começar:

    - no `process`, o filho termina de subir. Importar PyTorch e NumPy é a maior
      parte do custo do `spawn`, e o filho ainda reexecuta `experimentos/executar.py`
      e importa o pacote `hsi` — segundos por worker, pagos uma vez só;
    - em qualquer executor, a primeira passagem para trás instancia o pool de threads
      do autograd e o alocador de CPU, ambos preguiçosos;
    - o `sleep` mantém o worker ocupado enquanto as outras tarefas de aquecimento são
      submetidas. É o que garante que o pool crie **todos** os workers em vez de
      reaproveitar um que já ficou ocioso — `ThreadPoolExecutor` e
      `ProcessPoolExecutor` só criam um worker novo quando não há nenhum livre.
      Criar não é aquecer, porém: no `process` um worker leva segundos para subir,
      e o primeiro de pé pode levar várias tarefas enquanto outro ainda importa o
      PyTorch. Nada aqui garante uma tarefa por worker.

    Deliberadamente não toca nos dados da cena: aquecer com a tarefa real adiantaria
    o trabalho que se quer medir, em vez de só pagar a partida.
    """
    marca = time.perf_counter()
    entrada = torch.zeros(8, 1, 64, requires_grad=True)
    nn.Conv1d(1, 16, 9)(entrada).sum().backward()
    time.sleep(0.25)
    return time.perf_counter() - marca


class PoolReutilizavel:
    """Pool criado uma vez, aquecido e reusado em várias execuções.

    POR QUE EXISTE. `run_experiment_particionado` cria e destrói o pool **dentro** do
    bloco cronometrado, então cada medição paga a partida inteira. Isso é o pior
    caso: a primeira e única execução de um processo que acabou de nascer. Num uso
    real o pool sobe uma vez e atende muitas execuções, e a partida se dilui.

    As duas coisas são resultados diferentes e precisam ser medidas separadamente. O
    pool frio responde "quanto custa a primeira execução"; o pool quente responde
    "quanto custa cada uma depois dela". Reportar só o frio atribui ao paralelismo um
    custo que é de inicialização — foi o erro que esta classe existe para corrigir.

    O tempo de criação e o de aquecimento ficam guardados em `tempo_criacao` e
    `tempo_aquecimento`, fora de qualquer medição, para que o custo de partida
    continue visível em vez de desaparecer. O de criação sai perto de zero: os
    executores sobem os workers sob demanda, e quem paga a partida é `aquecer()`.
    """

    def __init__(self, executor: str = "thread", max_workers: int = 4):
        if executor not in {"sequencial", "thread", "process"}:
            raise ValueError("executor deve ser 'sequencial', 'thread' ou 'process'.")
        self.executor = executor
        self.max_workers = max_workers
        self.tempo_aquecimento = 0.0
        marca = time.perf_counter()
        if executor == "sequencial":
            self._pool = None
        elif executor == "thread":
            self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                            initializer=_preparar_worker)
        else:
            self._pool = ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_preparar_worker,
            )
        self.tempo_criacao = time.perf_counter() - marca

    def aquecer(self, rodadas: int = 1) -> float:
        """Sobe todos os workers e paga o custo preguiçoso de cada um.

        Devolve o tempo acumulado de aquecimento, que fica fora das medições.
        """
        marca = time.perf_counter()
        for _ in range(rodadas):
            self.mapear(_aquecer_worker, range(self.max_workers))
        self.tempo_aquecimento += time.perf_counter() - marca
        return self.tempo_aquecimento

    def mapear(self, funcao, tarefas) -> list:
        """Aplica `funcao` às tarefas no pool existente, sem criá-lo nem destruí-lo."""
        if self._pool is None:
            return [funcao(t) for t in tarefas]
        return list(self._pool.map(funcao, tarefas))

    def fechar(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def __enter__(self) -> "PoolReutilizavel":
        return self

    def __exit__(self, *_) -> None:
        self.fechar()


@dataclass
class ResultadoParticionado:
    """Saída de uma execução particionada, com os tempos que justificam o desenho."""

    config: ExperimentConfig
    modo: str
    particao_modo: str
    executor: str
    n_particoes: int
    metrics: "Metrics"
    predicted: np.ndarray
    tempo_preparo: float
    #: O ramo espacial inteiro da preparação — PCA de 3 componentes, RGB e
    #: segmentação —, que a segmentação domina.
    tempo_segmentacao: float
    #: Tempo de parede do bloco que roda concorrente. No modo "independente" inclui
    #: treino e inferência; no "replicado", só a inferência — o treino é único e
    #: sequencial por definição, e somá-lo aqui mascararia o speedup real.
    tempo_paralelo: float
    #: Treino único do modo "replicado". Zero no modo "independente", onde o treino
    #: já está dentro de `tempo_paralelo`.
    tempo_treino_sequencial: float
    tempos_por_particao: list[float]
    amostras_por_particao: list[int]
    parametros_totais: int
    #: O bloco paralelo rodou num pool já existente e aquecido (`PoolReutilizavel`)?
    #: Quando falso, o tempo medido inclui criar e destruir o pool — com `spawn`,
    #: subir N interpretadores. Comparar um frio com um quente mistura as duas coisas.
    pool_reusado: bool = False

    @property
    def soma_das_partes(self) -> float:
        """Soma dos tempos individuais das partições.

        **Atenção ao interpretar.** Quando o executor é concorrente, cada tempo foi
        cronometrado já sob disputa com os outros. Se as N partições começam e
        terminam juntas, `soma ÷ parede` tende a N por construção — e não porque o
        paralelismo esteja funcionando. Para speedup honesto, compare a parede deste
        arranjo com a de uma execução `executor="sequencial"`, que mede o custo real
        de processar as partições uma a uma.
        """
        return sum(self.tempos_por_particao)

    @property
    def tempo_total(self) -> float:
        """Soma dos blocos cronometrados: preparo + treino sequencial + bloco paralelo."""
        return self.tempo_preparo + self.tempo_treino_sequencial + self.tempo_paralelo

    @property
    def overhead(self) -> float:
        """Custo de coordenar: parede menos a partição mais lenta."""
        return self.tempo_paralelo - max(self.tempos_por_particao, default=0.0)

    @property
    def teto_de_amdahl(self) -> float:
        """Teto do speedup: soma das partes dividida pela mais lenta.

        Com partições equilibradas isto tende a N — é a diferença em relação ao
        paralelismo de dois ramos, onde o desequilíbrio fixava o teto em ~1,2×.
        """
        maior = max(self.tempos_por_particao, default=0.0)
        return self.soma_das_partes / maior if maior else float("nan")

    def resumo(self) -> str:
        linhas = [
            f"modo: {self.modo} | partição: {self.particao_modo} | "
            f"executor: {self.executor} ({'quente' if self.pool_reusado else 'frio'})"
            f" | {self.n_particoes} partes",
            f"  segmentação (uma vez):  {self.tempo_segmentacao:7.3f}s",
            f"  bloco paralelo:         {self.tempo_paralelo:7.3f}s"
            + (f"  (+ treino sequencial {self.tempo_treino_sequencial:.3f}s)"
               if self.tempo_treino_sequencial else ""),
            f"  soma das partes:        {self.soma_das_partes:7.3f}s  "
            f"(speedup {self.soma_das_partes / max(self.tempo_paralelo, 1e-9):.2f}x, "
            f"teto {self.teto_de_amdahl:.2f}x)",
            f"  overhead:               {self.overhead:+7.3f}s",
            f"  tempo total:            {self.tempo_total:7.3f}s",
            f"  amostras por partição:  {self.amostras_por_particao}",
        ]
        return "\n".join(linhas)


def run_experiment_particionado(
    config: ExperimentConfig,
    n_particoes: int = 4,
    modo: str = "independente",
    particao_modo: str = "faixas",
    executor: str = "thread",
    data_dir: Path | None = None,
    backend: str = "thread",
    verbose: bool = True,
    pool: "PoolReutilizavel | None" = None,
) -> ResultadoParticionado:
    """Executa o pipeline dividindo a cena em `n_particoes` processadas em paralelo.

    Devolve métricas calculadas sobre o conjunto de teste **inteiro**, remontado a
    partir das predições de cada partição — e não a média das partições, que seria
    outra coisa e não comparável com o pipeline monolítico.
    """
    if modo not in {"replicado", "independente"}:
        raise ValueError("modo deve ser 'replicado' ou 'independente'.")
    if pool is not None:
        # O executor passa a ser o do pool: ele já existe, e recriá-lo aqui anularia
        # justamente o que medir com pool quente quer isolar.
        executor = pool.executor
        if pool.max_workers < n_particoes:
            raise ValueError(
                f"pool tem {pool.max_workers} workers para {n_particoes} partições: "
                "as excedentes seriam enfileiradas e a medição deixaria de ser de "
                "paralelismo."
            )
    if executor not in {"sequencial", "thread", "process"}:
        raise ValueError("executor deve ser 'sequencial', 'thread' ou 'process'.")
    if executor == "process" and modo == "replicado":  # noqa: SIM102
        raise ValueError(
            "modo 'replicado' com executor 'process' exigiria serializar o modelo "
            "treinado para cada worker; use 'thread' ou o modo 'independente'."
        )
    if n_particoes < 2:
        raise ValueError("n_particoes deve ser pelo menos 2.")

    device = select_device(verbose=False)
    configure_determinism(config.deterministic, verbose=False)

    inicio = time.perf_counter()
    data, _, prep_timings = prepare_data_parallel(config, data_dir, backend=backend)
    tempo_preparo = time.perf_counter() - inicio
    validate_prepared_data(data)

    if data.coords is None:
        raise ValueError("partição espacial exige coordenadas em PreparedData.")
    particao = particionar_cena(data.coords, data.scene_shape, n_particoes,
                                particao_modo, config.seed)
    pedacos = [_fatiar(data, particao == k) for k in range(n_particoes)]

    if modo == "replicado":
        # Treina uma vez com a cena inteira; o paralelismo é só da inferência.
        marca = time.perf_counter()
        loaders = build_loaders(data, config, device)
        seed_everything(config.seed)
        modelo, criterio, otimizador = build_model(
            config, data.input_shape(), data.n_classes, device, n_segments=data.n_segments,
        )
        fit(modelo, loaders, criterio, otimizador, config, device, verbose=False)
        tempo_treino_unico = time.perf_counter() - marca
        # O modelo desce para a CPU antes de ser replicado, porque é onde as
        # partições inferem (`_treinar_e_inferir` fixa CPU). Sem isto a inferência
        # quebra — entrada em CPU, pesos em CUDA. E mover as entradas para a GPU no
        # lugar disso seria pior: N workers disputando um só dispositivo fazem o
        # bloco medir a fila da GPU, não o paralelismo. É também o que mantém este
        # bloco comparável com o do modo `independente`, medido na mesma CPU.
        modelo = modelo.to(torch.device("cpu"))
        modelos = [modelo] * n_particoes
    else:
        tempo_treino_unico = 0.0

    tarefas = [
        (k, pedacos[k], config, data.n_classes, data.n_segments,
         modelos[k] if modo == "replicado" else None)
        for k in range(n_particoes)
    ]
    marca = time.perf_counter()
    if pool is not None:
        saidas = sorted(pool.mapear(_treinar_e_inferir, tarefas))
    elif executor == "sequencial":
        saidas = sorted(_treinar_e_inferir(t) for t in tarefas)
    elif executor == "thread":
        with ThreadPoolExecutor(max_workers=n_particoes) as pool:
            saidas = sorted(pool.map(_treinar_e_inferir, tarefas))
    else:
        # `spawn`, e não o `fork` padrão do Linux: o PyTorch recusa fork depois que
        # o autograd rodou em threads ("Unable to handle autograd's threading in
        # combination with fork-based multiprocessing"). O preço do spawn é que cada
        # worker sobe um interpretador limpo e reimporta tudo — custo que aparece
        # inteiro na parede e precisa ser reportado, não escondido. O `initializer` é
        # o mesmo do pool quente: sem ele, um filho lançado a partir do notebook
        # subiria com as threads padrão do PyTorch em vez de uma por partição.
        contexto = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_particoes, mp_context=contexto,
                                 initializer=_preparar_worker) as pool:
            saidas = sorted(pool.map(_treinar_e_inferir, tarefas))
    tempo_paralelo = time.perf_counter() - marca

    previstos = np.concatenate([s[1] for s in saidas])
    verdadeiros = np.concatenate([s[2] for s in saidas])
    tempos = [s[3] for s in saidas]

    confusao = np.zeros((data.n_classes, data.n_classes), dtype=np.int64)
    np.add.at(confusao, (verdadeiros, previstos), 1)
    por_classe, media = _average_accuracy(confusao)
    acertos = int(np.trace(confusao))
    total = int(confusao.sum())
    esperado = float((confusao.sum(0) * confusao.sum(1)).sum() / (total * total))
    oa = acertos / total
    metrics = Metrics(
        overall_accuracy=oa, average_accuracy=media,
        kappa=(oa - esperado) / (1 - esperado) if esperado < 1 else 0.0,
        per_class_accuracy=por_classe, confusion=confusao,
        test_loss=float("nan"), test_accuracy=oa,
        report=f"partições={n_particoes} modo={modo} amostras de teste={total}",
    )

    resultado = ResultadoParticionado(
        config=config, modo=modo, particao_modo=particao_modo, executor=executor,
        n_particoes=n_particoes,
        metrics=metrics, predicted=previstos,
        tempo_preparo=tempo_preparo,
        tempo_segmentacao=prep_timings.stages.get("espacial", 0.0),
        tempo_paralelo=tempo_paralelo,
        tempo_treino_sequencial=tempo_treino_unico,
        tempos_por_particao=tempos,
        amostras_por_particao=[len(p.labels) for p in pedacos],
        parametros_totais=sum(m.n_parameters for m in
                              (modelos if modo == "replicado" else [])) or 0,
        pool_reusado=pool is not None,
    )
    if verbose:
        print(resultado.resumo())
        print(f"  OA={metrics.overall_accuracy:.4f} AA={metrics.average_accuracy:.4f} "
              f"Kappa={metrics.kappa:.4f}")
    return resultado
