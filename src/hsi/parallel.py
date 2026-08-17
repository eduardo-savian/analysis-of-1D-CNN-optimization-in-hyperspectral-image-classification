"""Paralelização inter-etapas da preparação dos dados.

Depois do PCA, os dois ramos do pipeline são independentes até o ponto de fusão:

- **ramo espectral**: PCA para `config.pca_components` bandas sobre os pixels válidos;
- **ramo espacial**: PCA de 3 componentes sobre a cena inteira, composição RGB e
  segmentação SEEDS, SLIC ou LSC.

Nenhum precisa do resultado do outro — só a fusão (montar a segunda entrada da rede)
precisa dos dois. Executá-los concorrentemente troca `T_espectral + T_espacial` por
`max(T_espectral, T_espacial) + overhead`.

Backends disponíveis:

``sequential``
    Um ramo depois do outro. É a linha de base contra a qual o ganho é medido.
``thread``
    `ThreadPoolExecutor`. Funciona apesar do GIL porque o trabalho pesado está em
    código nativo que o libera: a SVD do PCA (BLAS) e o `iterate` do SEEDS (OpenCV).
    Não copia nada — os dois ramos compartilham os arrays de entrada.
``process``
    `ProcessPoolExecutor`, que é o que o projeto do TCC descreve. Imune ao GIL, mas
    paga serialização: a tabela de pixels é enviada ao worker e o mapa de rótulos
    volta. Em cenas grandes esse custo pode consumir o ganho — por isso o
    `scripts/benchmark_parallel.py` mede os três.

O resultado é **idêntico** nos três backends: o paralelismo só muda quando cada
etapa roda, não o que ela calcula.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .data import Scene, load_scene, remove_background, to_pixel_table
from .dataset import PreparedData, split_indices
from .preprocess import apply_pca, rgb_composite
from .patches import spatial_patches, spectral_spatial_patches
from .segmentation import (
    LSC_PARAMS, SEEDS_PARAMS, SLIC_PARAMS, SegmentationArtifacts, LscParams,
    SeedsParams, SlicParams,
)

__all__ = ["BACKENDS", "StageTimings", "prepare_data_parallel", "spectral_branch", "spatial_branch"]

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
        """Sobrecarga de gerenciamento: threads/processos, serialização, sincronização."""
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

    Função de módulo (e não closure) porque precisa ser serializável para o backend
    de processos.
    """
    components, _ = apply_pca(pixels, n_components=3, random_state=seed)
    rgb = rgb_composite(components, scene_shape)
    segmentation = superpixel_labels_for(rgb, segmenter, params)

    ids_valid, _ = remove_background(segmentation.labels.reshape(-1, 1), raw_labels)
    return ids_valid.reshape(-1), components, rgb, segmentation


def superpixel_labels_for(rgb, segmenter, params):
    """Indireção fina só para manter o import do OpenCV dentro do worker."""
    from .segmentation import segment_superpixels

    return segment_superpixels(rgb, segmenter, params)


def _segmenter_params(config: ExperimentConfig):
    """Seleciona os parâmetros da cena sem acoplar o pipeline a um algoritmo."""
    tables = {"seeds": SEEDS_PARAMS, "slic": SLIC_PARAMS, "lsc": LSC_PARAMS}
    return tables[config.segmenter][config.dataset]


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
    """`prepare_data` com os dois ramos executados concorrentemente.

    Devolve `(dados, artefatos_da_segmentação, tempos)`. Com `two_input=False`,
    CNN1D/CNN3D não precisam do ramo espacial; CNN2D ainda o executa para formar
    seus patches PCA, mas não concatena ids de superpixel.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend inválido: {backend!r}. Use um de {BACKENDS}.")

    scene = scene or load_scene(config.dataset, data_dir)
    pixels, raw_labels = to_pixel_table(scene)
    spectra, labels = remove_background(pixels, raw_labels)

    timings = StageTimings(backend=backend)
    spatial_args = (
        pixels, raw_labels, scene.shape, _segmenter_params(config), config.seed, config.segmenter,
    )
    spectral_args = (spectra, config.pca_components, config.seed)

    started = time.perf_counter()

    # CNN2D depende das componentes PCA espaciais para construir seus patches,
    # mesmo quando a fusão por id de superpixel está desativada.
    needs_spatial_branch = config.two_input or config.model_type == "cnn2d"

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
        # o que evita um segundo round-trip de serialização.
        with ProcessPoolExecutor(max_workers=1) as pool:
            spatial_future = pool.submit(_timed, spatial_branch, *spatial_args)
            reduced, timings.stages["espectral"] = _timed(spectral_branch, *spectral_args)
            spatial, timings.stages["espacial"] = spatial_future.result()
        segments, artifacts = _unpack_spatial(spatial)

    timings.wall = time.perf_counter() - started

    # CNN2D pode precisar do ramo espacial para formar patches mesmo numa ablação
    # sem fusão. Nesse caso os artefatos visuais existem, mas o id não deve entrar
    # no TensorDataset nem ser registrado como segunda entrada do modelo.
    if not config.two_input:
        segments = None

    data = PreparedData(
        spectra=reduced,
        labels=labels.astype(np.int64),
        segments=segments,
        splits=split_indices(labels, config.test_ratio, config.valid_ratio, config.seed),
        scene_shape=scene.shape,
        patches_2d=(
            spatial_patches(artifacts.components, scene.shape, scene.labels, config.patch_size)
            if config.model_type == "cnn2d" and artifacts is not None else None
        ),
        patches_3d=(
            spectral_spatial_patches(scene.image, scene.labels, config.patch_size)
            if config.model_type == "cnn3d" else None
        ),
    )
    return data, artifacts, timings


def _unpack_spatial(spatial) -> tuple[np.ndarray, SegmentationArtifacts]:
    ids, components, rgb, segmentation = spatial
    return ids, SegmentationArtifacts(components=components, rgb=rgb, result=segmentation)
