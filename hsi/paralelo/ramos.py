"""Paralelização inter-etapas da preparação dos dados.

Carregada a cena, os dois ramos do pipeline são independentes até o ponto de fusão:

- **ramo espectral**: PCA para `config.pca_components` componentes sobre os pixels
  válidos;
- **ramo espacial**: PCA de 3 componentes sobre a cena inteira, composição RGB e
  segmentação (SEEDS, SLIC, LSC ou Watershed).

Nenhum precisa do resultado do outro — só a fusão (montar a segunda entrada da rede)
precisa dos dois. Executá-los concorrentemente troca `T_espectral + T_espacial` por
`max(T_espectral, T_espacial) + overhead`. O teto é baixo: o ramo espacial é 78–93%
do tempo serial, então o ganho máximo fica entre 1,08× e 1,29× conforme a cena.

Backends disponíveis:

``sequential``
    Um ramo depois do outro. É a linha de base contra a qual o ganho é medido.
``thread``
    `ThreadPoolExecutor`. Funciona apesar do GIL porque o trabalho pesado está em
    código nativo que o libera: a álgebra linear do PCA (BLAS/LAPACK) e o `iterate`
    do segmentador (OpenCV). Não copia nada — os dois ramos compartilham os arrays
    de entrada.
``process``
    `ProcessPoolExecutor`, que é o que o projeto do TCC descreve. Imune ao GIL, mas
    paga a criação do pool e a serialização: a tabela de pixels inteira vai ao
    worker, e voltam rótulos, componentes e RGB. Medido, perde para o sequencial
    nas três cenas.

Os tempos dos três estão em `RAMOS` e `SERIALIZACAO_MB`, nas medições avulsas
(`hsi/experimentos/medicoes.py`). O resultado é **idêntico** nos três backends: o paralelismo só muda
quando cada etapa roda, não o que ela calcula.
"""

import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from hsi.config import ExperimentConfig
from hsi.dados.cenas import Scene, load_scene, remove_background, to_pixel_table
from hsi.dados.particoes import (
    PreparedData,
    split_indices,
    split_indices_blocks,
    split_indices_region_folds,
    split_indices_regions,
    subsample_train,
)
from hsi.dados.preprocessamento import apply_pca, rgb_composite
from hsi.dados.segmentacao import (
    LSC_PARAMS,
    SEEDS_PARAMS,
    SLIC_PARAMS,
    WATERSHED_PARAMS,
    LscParams,
    SeedsParams,
    SegmentationArtifacts,
    SlicParams,
    segment_superpixels,
)

BACKENDS = ("sequential", "thread", "process")


@dataclass
class StageTimings:
    """Tempo de parede de cada etapa e do bloco paralelo como um todo.

    `stages` guarda o tempo *interno* de cada ramo (o que ele levaria sozinho);
    `wall` é o tempo real do bloco. Num backend paralelo bem-sucedido,
    `wall < soma dos ramos`.
    """

    backend: str
    stages: dict[str, float] = field(default_factory=dict)
    wall: float = 0.0

    @property
    def branch_sum(self) -> float:
        """Soma dos ramos — o que custaria em execução estritamente serial."""
        return self.stages.get("espectral", 0.0) + self.stages.get("espacial", 0.0)

    @property
    def overhead(self) -> float:
        """Sobrecarga de gerenciamento: threads/processos, serialização, sincronização.

        Só tem esse sentido nos backends paralelos: no `sequential`, `wall - max` é
        simplesmente o ramo mais curto.
        """
        return self.wall - max(self.stages.get("espectral", 0.0), self.stages.get("espacial", 0.0))

    def speedup_over(self, baseline: "StageTimings") -> float:
        return baseline.wall / self.wall if self.wall else float("nan")

    def summary(self) -> str:
        linhas = [f"backend: {self.backend} | bloco paralelo: {self.wall:.3f}s"]
        for name, value in self.stages.items():
            linhas.append(f"  {name:12s} {value:7.3f}s")
        linhas.append(f"  {'soma ramos':12s} {self.branch_sum:7.3f}s | overhead: {self.overhead:+.3f}s")
        return "\n".join(linhas)


def spectral_branch(spectra: np.ndarray, n_components: int, seed: int) -> np.ndarray:
    """Ramo espectral: reduz as bandas de cada pixel válido por PCA."""
    reduced, _ = apply_pca(spectra, n_components=n_components, random_state=seed)
    return reduced


def spatial_branch(
    pixels: np.ndarray,
    raw_labels: np.ndarray,
    scene_shape: tuple[int, int],
    params: SeedsParams | SlicParams | LscParams,
    seed: int,
    segmenter: str = "seeds",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ramo espacial: PCA de 3 componentes -> RGB -> superpixels -> id válido.

    A segmentação roda sobre a cena **inteira**, background incluído, porque os
    superpixels precisam da imagem completa; o filtro de background vem depois.

    Função de módulo (e não closure) porque o backend de processos a envia
    ao worker via pickle, que serializa funções pelo nome.
    """
    components, _ = apply_pca(pixels, n_components=3, random_state=seed)
    rgb = rgb_composite(components, scene_shape)
    segmentation = superpixel_labels_for(rgb, segmenter, params)

    ids_valid, _ = remove_background(segmentation.labels.reshape(-1, 1), raw_labels)
    return ids_valid.reshape(-1), components, rgb, segmentation


def superpixel_labels_for(rgb, segmenter, params):
    """Repasse direto para `segment_superpixels`.

    No pacote antigo, era aqui que o módulo de segmentação era importado, já dentro
    do worker. Hoje o `import cv2` fica dentro de `segment_superpixels`, e esta
    indireção não acrescenta nada.
    """

    return segment_superpixels(rgb, segmenter, params)


def _segmenter_params(config: ExperimentConfig):
    """Seleciona os parâmetros da cena sem acoplar o pipeline a um algoritmo.

    `segmenter_overrides` é preenchido pelo HPO; aplicá-lo aqui garante que o
    parâmetro testado seja exatamente o que vai para o `config.json` do run.
    """
    tables = {
        "seeds": SEEDS_PARAMS,
        "slic": SLIC_PARAMS,
        "lsc": LSC_PARAMS,
        "watershed": WATERSHED_PARAMS,
    }
    return replace(tables[config.segmenter][config.dataset], **config.segmenter_overrides)


def _timed(fn, *args):
    """Executa `fn` medindo seu tempo interno. Retorna `(resultado, segundos)`."""
    started = time.perf_counter()
    return fn(*args), time.perf_counter() - started


def prepare_data_parallel(
    config: ExperimentConfig,
    data_dir: Path | None = None,
    scene: Scene | None = None,
    backend: str = "thread",
) -> tuple[PreparedData, SegmentationArtifacts | None, StageTimings]:
    """Da cena aos dados prontos, com os dois ramos executados conforme `backend`.

    É a implementação de fato: `prepare_data` (`hsi/pipeline.py`) só a chama e descarta os
    tempos. Devolve `(dados, artefatos_da_segmentação, tempos)`. Com
    `two_input=False` e `split_mode` diferente de "region", o ramo espacial é
    dispensado por completo.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend inválido: {backend!r}. Use um de {BACKENDS}.")

    scene = scene or load_scene(config.dataset, data_dir)
    pixels, raw_labels = to_pixel_table(scene)
    spectra, labels = remove_background(pixels, raw_labels)
    # `to_pixel_table` achata em row-major, então a posição de cada amostra válida
    # se recupera exatamente do índice achatado. O split por blocos precisa disso.
    valid = np.flatnonzero(raw_labels > 0)
    coords = np.stack(np.divmod(valid, scene.shape[1]), axis=1)

    timings = StageTimings(backend=backend)
    spatial_args = (
        pixels, raw_labels, scene.shape, _segmenter_params(config), config.seed, config.segmenter,
    )
    spectral_args = (spectra, config.pca_components, config.seed)

    started = time.perf_counter()

    # O split por região precisa da segmentação mesmo com `two_input=False`: os
    # grupos do split vêm dela.
    needs_spatial_branch = config.two_input or config.split_mode == "region"

    if not needs_spatial_branch:
        reduced, timings.stages["espectral"] = _timed(spectral_branch, *spectral_args)
        segments, artifacts = None, None

    elif backend == "sequential":
        reduced, timings.stages["espectral"] = _timed(spectral_branch, *spectral_args)
        spatial, timings.stages["espacial"] = _timed(spatial_branch, *spatial_args)
        segments, artifacts = _unpack_spatial(spatial)

    elif backend == "thread":
        # Os dois ramos compartilham os arrays de entrada: nada é copiado.
        with ThreadPoolExecutor(max_workers=2) as pool:
            spatial_future = pool.submit(_timed, spatial_branch, *spatial_args)
            spectral_future = pool.submit(_timed, spectral_branch, *spectral_args)
            reduced, timings.stages["espectral"] = spectral_future.result()
            spatial, timings.stages["espacial"] = spatial_future.result()
        segments, artifacts = _unpack_spatial(spatial)

    else:  # process
        # O ramo espacial vai para outro processo; o espectral roda aqui mesmo,
        # o que evita um segundo round-trip de serialização. O pool usa o contexto
        # padrão — `fork`, no Linux com o Python 3.12 fixado —, e o filho herda o
        # OpenCV de uma thread que o LSC exige. Com `spawn`, o filho voltaria a N
        # threads (medido: 32) e o LSC deixaria de ser determinístico.
        with ProcessPoolExecutor(max_workers=1) as pool:
            spatial_future = pool.submit(_timed, spatial_branch, *spatial_args)
            reduced, timings.stages["espectral"] = _timed(spectral_branch, *spectral_args)
            spatial, timings.stages["espacial"] = spatial_future.result()
        segments, artifacts = _unpack_spatial(spatial)

    timings.wall = time.perf_counter() - started

    # Com `split_mode="region"` o ramo espacial roda mesmo sem fusão. Nesse caso os
    # artefatos visuais existem, mas o id não entra no TensorDataset nem é
    # registrado como segunda entrada do modelo.
    # `segments` é a segunda entrada do modelo; `region_ids` é o agrupamento usado
    # pelo split. Separá-los permite rodar `two_input=False` sob split por região.
    region_ids = segments
    if not config.two_input:
        segments = None

    if config.split_mode == "region":
        if region_ids is None:
            raise ValueError("split_mode='region' exige segmentação disponível.")
        if config.split_folds:
            splits, split_report = split_indices_region_folds(
                labels, region_ids, config.split_folds, config.split_fold,
                # Mesma fração de validação do protocolo por pixel (`test_ratio`
                # separa 15% e `valid_ratio` fica com metade disso), aplicada aqui
                # ao treino do fold, e não à cena inteira.
                config.test_ratio * config.valid_ratio,
                config.seed,
            )
        else:
            splits, split_report = split_indices_regions(
                labels, region_ids, config.test_ratio, config.valid_ratio, config.seed,
            )
    elif config.split_mode == "block":
        splits, split_report = split_indices_blocks(
            labels, coords, scene.shape, config.test_ratio, config.valid_ratio,
            config.split_block, config.seed,
        )
    else:
        splits, split_report = split_indices(
            labels, config.test_ratio, config.valid_ratio, config.seed
        ), {"mode": "pixel"}
    splits, subsample_report = subsample_train(splits, labels, config.train_fraction, config.seed)
    split_report.update(subsample_report)
    if split_report.get("classes_missing_from_train"):
        # Um grupo inteiro pode conter toda uma classe rara. Avisar é melhor que
        # falhar: o run continua, mas a acurácia daquela classe será zero por
        # construção, puxando a AA para baixo.
        print(
            f"Aviso: split por {split_report['mode']} deixou as classes "
            f"{split_report['classes_missing_from_train']} fora do treino."
        )

    data = PreparedData(
        spectra=reduced,
        labels=labels.astype(np.int64),
        segments=segments,
        splits=splits,
        coords=coords,
        scene_shape=scene.shape,
    )
    return data, artifacts, timings


def _unpack_spatial(spatial) -> tuple[np.ndarray, SegmentationArtifacts]:
    ids, components, rgb, segmentation = spatial
    return ids, SegmentationArtifacts(components=components, rgb=rgb, result=segmentation)
