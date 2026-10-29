"""Divisão treino/validação/teste e construção dos DataLoaders.

O notebook original empilhava espectro e segmentação num tensor `(N, Kb, 1, 1, 2)`
só para conseguir passar os dois pelo mesmo `train_test_split`, e depois desfazia
o empilhamento com fatiamento posicional. Aqui divide-se o **índice** uma vez e
cada array é indexado por ele — mesma partição, sem o malabarismo de shape.
"""

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class Splits:
    """Índices das três partições, referentes às linhas de `PreparedData`."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def sizes(self) -> dict[str, int]:
        return {"treino": len(self.train), "validação": len(self.val), "teste": len(self.test)}


@dataclass(frozen=True)
class PreparedData:
    """Dados prontos para o modelo, já sem background e com PCA aplicado.

    spectra:  (N, Kb) float64 — componentes principais do espectro
    labels:   (N,)    int64   — classes reindexadas para 0..C-1
    segments: (N,)    ou None — id do superpixel de cada pixel (segunda entrada)
    """

    spectra: np.ndarray
    labels: np.ndarray
    segments: np.ndarray | None
    splits: Splits
    scene_shape: tuple[int, int]
    #: (N, 2) linha/coluna de cada amostra na cena. Necessário para o split por
    #: blocos espaciais e para projetar previsões de volta na imagem.
    coords: np.ndarray | None = None

    @property
    def n_features(self) -> int:
        return self.spectra.shape[1]

    @property
    def n_classes(self) -> int:
        return int(self.labels.max()) + 1

    @property
    def two_input(self) -> bool:
        return self.segments is not None

    @property
    def n_segments(self) -> int:
        """Quantas linhas a tabela de embeddings precisa ter.

        Usa `max + 1`, e não a contagem de valores distintos, porque os ids saem
        do segmentador antes da remoção do background e ficam com buracos: em
        Indian Pines há 63 regiões rotuladas, mas o maior id é 78. Linhas não
        usadas simplesmente nunca recebem gradiente.
        """
        return 0 if self.segments is None else int(np.max(self.segments)) + 1

    def input_shape(self, model_type: str = "cnn1d") -> tuple[int, ...]:
        """Shape de uma amostra no layout channels-first da CNN1D."""
        return (1, self.n_features)


def split_indices(labels: np.ndarray, test_ratio: float, valid_ratio: float, seed: int = 42) -> Splits:
    """Split estratificado em duas etapas, como no original.

    Primeiro separa treino de (validação + teste); depois reparte esse resto.
    `stratify` mantém a proporção de cada classe nas três partições — necessário
    porque as cenas são bastante desbalanceadas.
    """
    index = np.arange(len(labels))
    train, rest = train_test_split(
        index, test_size=test_ratio, random_state=seed, stratify=labels
    )
    # Atenção à ordem: no original o primeiro retorno desta segunda chamada é o
    # conjunto de *teste* e o segundo é o de validação.
    test, val = train_test_split(
        rest, test_size=valid_ratio, random_state=seed, stratify=labels[rest]
    )
    return Splits(train=train, val=val, test=test)


def _split_by_groups(labels, group_id, test_ratio, valid_ratio, seed=42, attempts=64):
    """Sorteia grupos inteiros para cada partição, maximizando classes no treino.

    Um "grupo" é um bloco espacial ou um superpixel: a unidade que não pode ser
    dividida entre treino e teste. Estratificação exata é impossível — uma classe
    rara pode viver dentro de um único grupo — então a busca tenta várias
    permutações e devolve a que cobre mais classes no treino, com um relatório do
    que ficou de fora.
    """
    labels = np.asarray(labels)
    group_id = np.asarray(group_id)
    groups, sizes = np.unique(group_id, return_counts=True)
    size_of = dict(zip(groups.tolist(), sizes.tolist()))
    n_samples, classes = len(labels), np.unique(labels)
    rng = np.random.default_rng(seed)

    best = None
    for _ in range(attempts):
        order = rng.permutation(groups)
        held, accumulated = [], 0
        for candidate in order:
            if accumulated >= test_ratio * n_samples:
                break
            held.append(candidate)
            accumulated += size_of[candidate]
        validation, accumulated_val = [], 0
        for candidate in held:
            if accumulated_val >= valid_ratio * accumulated:
                break
            validation.append(candidate)
            accumulated_val += size_of[candidate]
        train_mask = ~np.isin(group_id, held)
        val_mask = np.isin(group_id, validation)
        test_mask = ~train_mask & ~val_mask
        if not (train_mask.any() and val_mask.any() and test_mask.any()):
            continue
        score = (len(np.unique(labels[train_mask])),
                 -abs(test_mask.mean() + val_mask.mean() - test_ratio))
        if best is None or score > best[0]:
            best = (score, train_mask, val_mask, test_mask, len(held))
    if best is None:
        raise ValueError(
            "Não foi possível formar três partições com grupos disjuntos. "
            "Use grupos menores (reduza `split_block`) ou aumente `test_ratio`."
        )

    _, train_mask, val_mask, test_mask, held_groups = best
    report = {
        "groups_total": int(len(groups)), "groups_held_out": int(held_groups),
        "train_fraction": float(train_mask.mean()), "val_fraction": float(val_mask.mean()),
        "test_fraction": float(test_mask.mean()),
        "classes_missing_from_train": sorted(
            set(classes.tolist()) - set(np.unique(labels[train_mask]).tolist())
        ),
    }
    splits = Splits(train=np.flatnonzero(train_mask), val=np.flatnonzero(val_mask),
                    test=np.flatnonzero(test_mask))
    return splits, report


def split_indices_regions(labels, region_ids, test_ratio, valid_ratio, seed=42) -> tuple[Splits, dict]:
    """Split disjunto por superpixel: uma região inteira vai para uma partição só.

    É o teste exato do vazamento espacial. No split por pixel, o mesmo superpixel
    aparece em treino e em teste, então um modelo com fusão pode memorizar
    "região R -> classe C" e aplicar isso a pixels de teste da mesma região —
    o que infla a acurácia de qualquer configuração com fusão, inclusive a herdada. Aqui `segment_leakage` devolve zero
    por construção.

    Limitação a declarar no texto: a partição passa a depender do segmentador,
    então comparar segmentadores diferentes sob este modo compara também splits
    diferentes. O experimento 09 lê a diferença entre segmentadores sob este modo
    junto com `DISTANCIA_TESTE`, que mede essa dificuldade.
    """
    splits, report = _split_by_groups(labels, region_ids, test_ratio, valid_ratio, seed)
    return splits, {"mode": "region", **report}


def split_indices_region_folds(labels, region_ids, n_folds: int, fold: int,
                               valid_ratio: float = 0.15, seed: int = 42) -> tuple[Splits, dict]:
    """Validação cruzada por região: o grupo `fold` é o teste, disjunto por região.

    Por que isto existe. `split_indices_regions` sorteia **uma** partição, e com
    `test_ratio=0,15` sobram 3 ou 4 regiões no teste — trocar uma região por outra
    muda quase tudo, e o desvio entre sementes chegou a ±0,25 em Pavia. O número era
    inutilizável. Percorrendo `fold` de 0 a n_folds-1, **cada região é testada
    exatamente uma vez** e a média usa a cena inteira.

    Usa `StratifiedGroupKFold`: mantém cada região inteira numa partição só e ainda
    assim tenta preservar a proporção das classes. Um `GroupKFold` puro ignora as
    classes e deixa mais delas de fora do treino.

    **Decisão deliberada: a validação NÃO é disjunta por região.** Ela é uma amostra
    estratificada por pixel do conjunto de treino. O motivo é medido: separar também
    a validação por região quadruplicou as classes ausentes do treino (de 4 para 16
    ocorrências ao longo dos 5 folds em Indian Pines), porque ela rouba justamente as
    regiões das classes raras.

    O invariante que protege o número reportado é a disjunção **treino × teste**, e
    essa está garantida. A validação compartilhar regiões com o treino a torna
    otimista, o que pode fazer o early stopping parar num ponto que memoriza mais —
    mas esse modelo então vai **pior** no teste disjunto. O viés é pessimista sobre o
    resultado final, nunca otimista, que é o lado seguro de errar.

    **Limitação que o protocolo não resolve.** Em Indian Pines, com as regiões do
    SEEDS, 6 das 16 classes vivem em uma ou duas regiões apenas — as classes 0, 6,
    8 e 12 estão inteiramente dentro de *uma* região. Quando essa região cai no
    teste, a classe é impossível de aprender e sua acurácia é zero por construção,
    derrubando a AA. O relatório devolve `classes_missing_from_train` para que a
    métrica seja lida com essa informação, em vez de parecer um fracasso do modelo.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    labels = np.asarray(labels)
    region_ids = np.asarray(region_ids)
    regioes = np.unique(region_ids)
    if len(regioes) < n_folds:
        raise ValueError(
            f"{len(regioes)} regiões não dão para {n_folds} folds. "
            "Aumente o número de superpixels do segmentador ou reduza n_folds."
        )
    vazio = np.zeros(len(labels))
    divisor = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    resto_idx, teste_idx = list(divisor.split(vazio, labels, groups=region_ids))[fold % n_folds]

    # Validação por pixel dentro do treino (ver docstring): estratificada, para não
    # deixar uma classe rara sem amostra de validação — e limitada a `len(pool) - 1`,
    # para que nenhuma classe saia inteira do treino.
    rng = np.random.default_rng(seed + 1)
    escolhidos = []
    for klass in np.unique(labels[resto_idx]):
        pool = resto_idx[labels[resto_idx] == klass]
        quantos = min(len(pool) - 1, max(1, int(round(len(pool) * valid_ratio))))
        if quantos > 0:
            escolhidos.append(rng.choice(pool, size=quantos, replace=False))
    val_idx = np.concatenate(escolhidos) if escolhidos else resto_idx[:0]
    treino_idx = np.setdiff1d(resto_idx, val_idx)

    faltando = sorted(set(np.unique(labels).tolist())
                      - set(np.unique(labels[treino_idx]).tolist()))
    relatorio = {
        "mode": "region_folds", "n_folds": int(n_folds), "fold": int(fold),
        "regioes_total": int(len(regioes)),
        "regioes_treino": int(len(np.unique(region_ids[treino_idx]))),
        "regioes_teste": int(len(np.unique(region_ids[teste_idx]))),
        "train_fraction_px": float(len(treino_idx) / len(labels)),
        "test_fraction_px": float(len(teste_idx) / len(labels)),
        "classes_missing_from_train": faltando,
        "classes_in_test": int(len(np.unique(labels[teste_idx]))),
        "val_disjunta_por_regiao": False,
    }
    splits = Splits(train=np.sort(treino_idx), val=np.sort(val_idx), test=np.sort(teste_idx))
    return splits, relatorio


def split_indices_blocks(labels, coords, scene_shape, test_ratio, valid_ratio,
                         block: int = 0, seed: int = 42) -> tuple[Splits, dict]:
    """Split por blocos retangulares da cena, independente do segmentador.

    Vantagem sobre `split_indices_regions`: a mesma partição serve para comparar
    segmentadores diferentes. Desvantagem: a disjunção é **aproximada** — regiões
    que cruzam a fronteira do bloco continuam compartilhadas. Isso importa muito
    em Indian Pines, onde os superpixels do SEEDS têm ~16x16 px e um bloco pequeno
    não os contém. Confira sempre `segment_leakage` em vez de supor.
    """
    coords = np.asarray(coords)
    block = block or max(4, min(scene_shape) // 4)
    grid_cols = int(np.ceil(scene_shape[1] / block))
    block_id = (coords[:, 0] // block) * grid_cols + (coords[:, 1] // block)
    splits, report = _split_by_groups(labels, block_id, test_ratio, valid_ratio, seed)
    return splits, {"mode": "block", "block_size": int(block), **report}


def subsample_train(splits: Splits, labels, fraction: float, seed: int = 42) -> tuple[Splits, dict]:
    """Reduz o conjunto de treino para uma fração, preservando a proporção das classes.

    Amostragem estratificada por classe, e não sorteio uniforme: com 10% uniforme,
    uma classe de 20 pixels teria em média 2 amostras e, em cerca de 1 sorteio a
    cada 8, nenhuma — e uma classe ausente do treino tem acurácia zero por
    construção. Estratificar garante ao menos uma amostra por classe presente no
    treino.

    Validação e teste não são tocados. Isso é deliberado: comparar `train_fraction`
    diferentes só faz sentido se o conjunto de avaliação for idêntico.
    """
    labels = np.asarray(labels)
    if fraction >= 1:
        return splits, {"train_fraction": 1.0, "train_size": int(len(splits.train))}
    rng = np.random.default_rng(seed)
    chosen = []
    for klass in np.unique(labels[splits.train]):
        pool = splits.train[labels[splits.train] == klass]
        keep = max(1, int(round(len(pool) * fraction)))
        chosen.append(rng.choice(pool, size=keep, replace=False))
    train = np.sort(np.concatenate(chosen))
    report = {
        "train_fraction": float(fraction),
        "train_size": int(len(train)),
        "train_size_full": int(len(splits.train)),
        "classes_in_train": int(len(np.unique(labels[train]))),
        # Menor contagem entre as classes presentes; o `and` devolve 0 com treino
        # vazio, sem chegar ao `np.min` de um array vazio.
        "min_samples_per_class": int(np.bincount(labels[train]).max(initial=0) and
                                     np.min(np.bincount(labels[train])[np.bincount(labels[train]) > 0])),
    }
    return Splits(train=train, val=splits.val, test=splits.test), report


def segment_leakage(data: PreparedData) -> dict:
    """Quanto do teste compartilha superpixel com o treino — o vazamento residual.

    No split por pixel esse número fica perto de 1,0 por construção; no split por
    região, é zero. No split por blocos ele mede só as regiões que cruzam a
    fronteira entre blocos, e é o número que torna honesta qualquer afirmação sobre
    resultados com fusão.
    """
    if data.segments is None:
        return {"applicable": False}
    train_ids = np.unique(data.segments[data.splits.train])
    test_ids = data.segments[data.splits.test]
    unique_test = np.unique(test_ids)
    return {
        "applicable": True,
        "test_samples": int(len(test_ids)),
        "test_samples_sharing_region_with_train": float(np.isin(test_ids, train_ids).mean()),
        "regions_in_test": int(len(unique_test)),
        "regions_also_in_train": int(np.isin(unique_test, train_ids).sum()),
    }


def validate_prepared_data(data: PreparedData) -> dict[str, int | bool]:
    """Aplica o contrato de dados antes de alocar GPU ou iniciar o treino.

    A validação concentra falhas comuns de experimentos: arrays desalinhados,
    NaN/infinito após PCA, classes fora de ``0..C-1`` e splits que se sobrepõem.
    Retorna um resumo serializável para o tracker — incluindo a cobertura, que é
    menor que 1 quando ``train_fraction < 1`` — e levanta ``ValueError`` com uma
    causa objetiva quando o contrato não é atendido.
    """
    n_samples = len(data.labels)
    if data.spectra.ndim != 2 or len(data.spectra) != n_samples:
        raise ValueError("spectra deve ser uma matriz com uma linha por rótulo.")
    if data.labels.ndim != 1 or n_samples == 0:
        raise ValueError("labels deve ser um vetor não vazio.")
    if not np.isfinite(data.spectra).all():
        raise ValueError("spectra contém NaN ou infinito.")
    expected_classes = np.arange(data.n_classes)
    if not np.array_equal(np.unique(data.labels), expected_classes):
        raise ValueError("labels deve conter classes contíguas de 0 até C-1.")
    if data.segments is not None and (len(data.segments) != n_samples or not np.isfinite(data.segments).all()):
        raise ValueError("segments deve ser finito e ter uma entrada por rótulo.")
    all_indices = np.concatenate([data.splits.train, data.splits.val, data.splits.test])
    # O invariante que importa é a DISJUNÇÃO — nenhuma amostra em duas partições.
    # A cobertura total não é exigida: com `train_fraction < 1` o treino é uma
    # amostra estratificada, e sobram pixels rotulados fora das três partições
    # de propósito. Exigir cobertura aqui recusaria um experimento válido.
    if len(np.unique(all_indices)) != len(all_indices):
        raise ValueError("splits se sobrepõem: uma amostra aparece em mais de uma partição.")
    if all_indices.min(initial=0) < 0 or all_indices.max(initial=-1) >= n_samples:
        raise ValueError("splits contém índice fora dos dados preparados.")
    return {
        "valid": True,
        "samples": n_samples,
        "samples_used": int(len(all_indices)),
        "coverage": float(len(all_indices) / n_samples),
        "classes": data.n_classes,
        "has_segments": data.segments is not None,
    }


def _tensors(data: PreparedData, index: np.ndarray, model_type: str = "cnn1d") -> tuple[torch.Tensor, ...]:
    """Fatia uma partição e converte para os tensores que o modelo consome.

    O espectro vira ``(N, 1, Kb)``. A segmentação vai como um float por amostra,
    ``(N, 1)``: a fusão `scalar` o concatena depois do achatamento das convoluções,
    e a `embedding` o converte de volta em índice da tabela.
    """
    features = torch.from_numpy(np.ascontiguousarray(data.spectra[index])).float().unsqueeze(1)
    labels = torch.from_numpy(np.ascontiguousarray(data.labels[index]))
    if data.segments is None:
        return features, labels
    segments = torch.from_numpy(np.ascontiguousarray(data.segments[index])).float().reshape(-1, 1)
    return features, segments, labels


def build_loaders(data: PreparedData, config, device) -> dict[str, DataLoader]:
    """DataLoaders de treino, validação e teste.

    O loader de treino recebe um `generator` semeado explicitamente: a ordem dos
    batches passa a depender só de `config.seed`, e não do estado global do RNG.
    Criar um iterador sobre esse loader (para inspecionar um batch, por exemplo)
    adianta o gerador — por isso `fit` repõe a semente antes da primeira época.

    Em CUDA, ``pin_memory=True`` torna a transferência CPU→GPU não bloqueante em
    ``unpack_batch``. ``prefetch_factor`` e ``persistent_workers`` só são
    passados quando há workers, pois o PyTorch os rejeita/ignora com zero workers.
    Validação e teste podem usar batch maior porque não armazenam gradientes.
    """
    pin = device.type == "cuda"
    datasets = {
        name: TensorDataset(*_tensors(data, index, config.model_type))
        for name, index in (
            ("train", data.splits.train),
            ("val", data.splits.val),
            ("test", data.splits.test),
        )
    }
    worker_options = {}
    if config.num_workers:
        worker_options = {
            "prefetch_factor": config.prefetch_factor,
            "persistent_workers": config.persistent_workers,
        }

    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=config.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(config.seed),
            pin_memory=pin, num_workers=config.num_workers, drop_last=False,
            **worker_options,
        )
    }
    for name in ("val", "test"):
        loaders[name] = DataLoader(
            datasets[name], batch_size=config.resolved_eval_batch_size, shuffle=False,
            pin_memory=pin, num_workers=config.num_workers,
            **worker_options,
        )
    return loaders
