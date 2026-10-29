"""Estratégias de busca do HPO: random search, TPE (Optuna) e algoritmo genético.

Todas recebem o mesmo espaço (`hpo/espaco.py`) e a mesma semente; `make_hpo_searcher`
é o único ponto de escolha, usado por `automated_hpo` (`hpo/busca.py`).
"""

import numpy as np

from hsi.config import MODEL_TYPES
from hsi.hpo.espaco import _HPO_MAX_DRAWS


class _RandomSearcher:
    """Sorteio uniforme com cobertura garantida de segmentadores e sem repetição.

    Não usa o resultado dos trials anteriores — é a linha de base contra a qual as
    duas estratégias guiadas são comparadas.
    """

    name = "random"

    def __init__(self, genes, rng, coverage):
        self.genes, self.rng, self.coverage = genes, rng, list(coverage)
        self.seen = set()

    def _draw(self):
        return tuple(int(self.rng.integers(len(values))) for _, values in self.genes)

    def ask(self):
        forced = self.coverage.pop(0) if self.coverage else None
        individual = self._draw()
        for _ in range(_HPO_MAX_DRAWS):
            candidate = individual if forced is None else (forced,) + individual[1:]
            if candidate not in self.seen:
                break
            individual = self._draw()
        self.seen.add(candidate)
        return candidate

    def tell(self, individual, score):
        """Random search é sem memória: o resultado não altera os sorteios seguintes."""


class _TPESearcher:
    """Hipertunador bayesiano (Tree-structured Parzen Estimator) via Optuna.

    O TPE modela duas densidades sobre o espaço — a dos trials bons e a dos ruins —
    e propõe pontos que maximizam a razão entre elas. Diferente do random search,
    cada sugestão depende de tudo o que já foi observado.

    `segmenter` e `override_index` são declarados como categóricos (não têm ordem
    natural); os demais eixos como inteiros ordenados, porque neles "o próximo
    valor" realmente significa "um pouco maior".
    """

    name = "tpe"

    def __init__(self, genes, seed, coverage):
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.genes = genes
        # A fase aleatória do TPE é a própria cobertura (mínimo 4). O padrão do
        # Optuna, 10 trials aleatórios antes de modelar, faria do HPO de 8 trials da
        # seção 8.4 um random search do início ao fim.
        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=int(seed), n_startup_trials=max(4, len(coverage))),
        )
        # A mesma cobertura do random search, para que as estratégias partam da mesma
        # informação inicial sobre os segmentadores. Só o segmentador é fixado: os
        # demais genes desses trials vêm do sorteio do próprio TPE e não coincidem
        # com os do random search.
        for index in coverage:
            self.study.enqueue_trial({"segmenter": int(index)}, skip_if_exists=False)
        self._current = None

    def ask(self):
        self._current = self.study.ask()
        indices = []
        for name, values in self.genes:
            if name in ("segmenter", "override_index"):
                indices.append(int(self._current.suggest_categorical(name, list(range(len(values))))))
            else:
                indices.append(int(self._current.suggest_int(name, 0, len(values) - 1)))
        return tuple(indices)

    def tell(self, individual, score):
        self.study.tell(self._current, float(score))


class _GeneticSearcher:
    """Algoritmo genético elitista sobre o espaço discreto de índices.

    Uma geração inteira é avaliada antes de produzir a seguinte. Operadores:

    - **seleção por torneio**: `tournament` indivíduos sorteados, vence o de maior
      validação. Mantém pressão seletiva sem descartar a diversidade de imediato.
    - **crossover uniforme**: cada gene do filho vem de um dos dois pais com
      probabilidade 1/2. Adequado aqui porque os genes são independentes — não há
      vizinhança a preservar, como haveria numa codificação binária de um número.
    - **mutação por gene**: com probabilidade `mutation_rate`, um gene é resorteado.
      É o que impede a convergência prematura para um único segmentador.
    - **elitismo**: os `elite` melhores passam à geração seguinte **sem retreinar**,
      o que economiza orçamento e garante que o melhor nunca piore.

    A primeira geração é a cobertura completada por sorteios — do mesmo gerador e
    com a mesma semente do random search —, então coincide com os primeiros
    `population` trials dele (salvo se ele precisar resortear uma repetição).
    Seleção e cruzamento só entram da segunda geração em diante, e cada geração nova
    custa `population - elite` trials. Filhos repetidos não são descartados: uma
    configuração já avaliada pode ser treinada de novo.
    """

    name = "genetic"

    def __init__(self, genes, rng, coverage, population=8, elite=2, mutation_rate=.25, tournament=3):
        self.genes, self.rng = genes, rng
        self.population_size = max(4, int(population))
        self.elite = max(1, min(int(elite), self.population_size - 1))
        self.mutation_rate, self.tournament = float(mutation_rate), max(2, int(tournament))
        self.generation, self.previous = [], []
        self.queue = [(index,) + self._draw()[1:] for index in coverage]
        while len(self.queue) < self.population_size:
            self.queue.append(self._draw())

    def _draw(self):
        return tuple(int(self.rng.integers(len(values))) for _, values in self.genes)

    def _tournament(self):
        pool = [self.previous[int(self.rng.integers(len(self.previous)))] for _ in range(self.tournament)]
        return max(pool, key=lambda item: item[0])[1]

    def _breed(self):
        ranked = sorted(self.generation, key=lambda item: item[0], reverse=True)
        self.previous = ranked
        elites = ranked[:self.elite]
        # Elites entram na próxima geração já avaliados: não voltam para a fila.
        self.generation = list(elites)
        children = []
        while len(children) < self.population_size - len(elites):
            first, second = self._tournament(), self._tournament()
            child = tuple(
                first[position] if self.rng.random() < .5 else second[position]
                for position in range(len(self.genes))
            )
            child = tuple(
                int(self.rng.integers(len(values))) if self.rng.random() < self.mutation_rate else gene
                for gene, (_, values) in zip(child, self.genes)
            )
            children.append(child)
        return children

    def ask(self):
        if not self.queue:
            self.queue = self._breed()
        return self.queue.pop(0)

    def tell(self, individual, score):
        self.generation.append((float(score), individual))


#: Estratégias disponíveis. "random" é a linha de base; "tpe" é o hipertunador
#: bayesiano; "genetic" é o algoritmo evolutivo.
HPO_STRATEGIES = ("random", "tpe", "genetic")


def make_hpo_searcher(strategy, genes, segmenters, seed, model_type, population=8):
    """Instancia a estratégia de busca, dando a todas a mesma cobertura inicial.

    A semente é deslocada pelo índice da arquitetura em `MODEL_TYPES`, para que
    CNNs diferentes não repitam os mesmos sorteios; com só a CNN1D o deslocamento é
    zero e vale a semente recebida.
    """
    coverage = list(range(len(tuple(segmenters))))
    offset = MODEL_TYPES.index(model_type)
    rng = np.random.default_rng(seed + offset)
    if strategy == "random":
        return _RandomSearcher(genes, rng, coverage)
    if strategy == "tpe":
        return _TPESearcher(genes, seed + offset, coverage)
    if strategy == "genetic":
        return _GeneticSearcher(genes, rng, coverage, population=population)
    raise ValueError(f"estratégia inválida: {strategy!r}. Use uma de {HPO_STRATEGIES}.")
