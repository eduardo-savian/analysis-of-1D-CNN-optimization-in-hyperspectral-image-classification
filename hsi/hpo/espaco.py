"""Espaço de busca do HPO: eixos, opções de PCA e de segmentador, genes."""

from hsi.config import PCA_COMPONENTS, SEGMENTERS
from hsi.dados.segmentacao import (
    LSC_PARAMS,
    SEEDS_PARAMS,
    SLIC_PARAMS,
    WATERSHED_PARAMS,
)
from hsi.modelo import CNN1D_MIN_COMPONENTS

#: Eixos de treino do espaço, além do segmentador e de seus parâmetros internos.
#: Cada um inclui o valor da receita herdada (lr 1e-3, dropout 0,5, weight decay 0,
#: batch 32 — o do notebook original; o experimento 10 usou 128 por engano): com o
#: PCA da cena e o SEEDS padrão, que também estão no espaço, a receita original é
#: um dos candidatos possíveis.
#: `pca_components` não aparece aqui porque depende da cena; é construído por
#: `_pca_options`, que aplica o piso imposto pelos kernels da CNN1D.
HPO_SPACE = {
    "learning_rate": (3e-4, 1e-3, 3e-3),
    "dropout": (0.2, 0.35, 0.5),
    "weight_decay": (0.0, 1e-5, 1e-4),
    "batch_size": (32, 64, 128),
}


#: Tentativas de resorteio antes de aceitar uma configuração repetida.
_HPO_MAX_DRAWS = 32


def _pca_options(dataset, model_type="cnn1d", scales=(0.5, 1.0, 1.5)):
    """Número de componentes de PCA em torno da receita padrão da cena.

    A escala é relativa porque o valor de referência já difere entre cenas
    (Indian Pines usa 30; PaviaU e Salinas, 15).

    O piso vem da **arquitetura**, não da segmentação. A composição RGB usa um PCA
    próprio de 3 componentes; este aqui alimenta as convoluções. A CNN1D encurta a
    sequência em 14 posições, então abaixo de `CNN1D_MIN_COMPONENTS` a saída fica
    vazia. Sem esse piso, cenas com receita 15 (PaviaU, Salinas) gerariam a opção 8,
    e o primeiro trial que a usasse interromperia a busca com o erro da CNN1D.
    Com o piso, a opção 0,5× colapsa em 15 e essas cenas ficam com duas opções
    (15 e 22), contra três em Indian Pines (15, 30 e 45).
    """
    base = PCA_COMPONENTS[dataset]
    piso = CNN1D_MIN_COMPONENTS if model_type == "cnn1d" else 3
    return sorted({max(piso, int(round(base * scale))) for scale in scales})


def segmenter_hpo_overrides(segmenter, dataset):
    """Retorna três opções controladas em torno da receita padrão de cada segmentador.

    A do meio (índice 1) é a própria receita padrão da cena; as outras duas ficam
    abaixo e acima dela no parâmetro de escala. SEEDS varia número de regiões,
    iterações e prior; SLIC varia region_size/ruler; LSC varia region_size/ratio;
    Watershed varia marcadores. O valor efetivamente usado vai para
    `config.segmenter_overrides` e é gravado no `config.json` do run.

    No SLIC, `ruler` não tem efeito: `segment_superpixels` usa o modo SLICO, que
    adapta a compacidade sozinho (conferido nas três cenas: rótulos idênticos com
    ruler 5, 10 e 20). Na prática, as três opções do SLIC diferem só em `region_size`.
    """
    base = {
        "seeds": SEEDS_PARAMS,
        "slic": SLIC_PARAMS,
        "lsc": LSC_PARAMS,
        "watershed": WATERSHED_PARAMS,
    }[segmenter][dataset]
    if segmenter == "seeds":
        return [
            {"num_superpixels": max(40, int(base.num_superpixels * factor)), "num_iterations": iterations, "prior": prior}
            for factor, iterations, prior in ((.70, 120, 1), (1.0, base.num_iterations, base.prior), (1.35, 250, 3))
        ]
    if segmenter == "slic":
        return [
            {"region_size": max(3, base.region_size - 3), "ruler": 5.0},
            {"region_size": base.region_size, "ruler": base.ruler},
            {"region_size": base.region_size + 3, "ruler": 20.0},
        ]
    if segmenter == "lsc":
        return [
            {"region_size": max(3, base.region_size - 3), "ratio": .04},
            {"region_size": base.region_size, "ratio": base.ratio},
            {"region_size": base.region_size + 3, "ratio": .12},
        ]
    return [{"markers": max(20, int(base.markers * factor))} for factor in (.70, 1.0, 1.35)]


def hpo_search_space(base_config, model_type, space=None):
    """Eixos efetivamente buscados para esta CNN e esta cena (o PCA muda por cena).

    Devolver o espaço explicitamente o torna inspecionável: o leitor do TCC vê o
    que foi buscado, e não apenas o que venceu. Nenhuma campanha o grava hoje: os
    artefatos guardam só os valores usados em cada trial.
    """
    axes = {name: tuple(values) for name, values in (space or HPO_SPACE).items()}
    axes["pca_components"] = tuple(_pca_options(base_config.dataset, model_type))
    return axes


def hpo_genes(base_config, model_type, space=None, segmenters=None):
    """O espaço de busca como uma lista ordenada de genes discretos.

    Cada gene é `(nome, valores)` e um candidato é uma tupla de índices — um por
    gene. Essa representação é o denominador comum das três estratégias: o random
    search sorteia índices, o TPE os sugere e o algoritmo genético os cruza e muta.
    Todos os eixos são discretos, o que dispensa qualquer codificação real/binária.
    """
    axes = hpo_search_space(base_config, model_type, space)
    segmenters = tuple(segmenters or SEGMENTERS)
    # `segmenter` precisa ser o gene 0: a cobertura inicial dos buscadores força
    # essa posição. `segmenter_hpo_overrides` devolve sempre três receitas por
    # segmentador, então o índice do override é um gene retangular, independente do
    # segmentador sorteado.
    genes = [("segmenter", segmenters), ("override_index", (0, 1, 2))]
    genes += [(name, values) for name, values in axes.items()]
    return genes


def hpo_decode(genes, individual, dataset):
    """Traduz uma tupla de índices nos campos que `ExperimentConfig` entende."""
    chosen = {name: values[index % len(values)] for (name, values), index in zip(genes, individual)}
    segmenter = chosen.pop("segmenter")
    overrides = segmenter_hpo_overrides(segmenter, dataset)
    position = chosen.pop("override_index") % len(overrides)
    return {"segmenter": segmenter, "segmenter_overrides": dict(overrides[position]), **chosen}
