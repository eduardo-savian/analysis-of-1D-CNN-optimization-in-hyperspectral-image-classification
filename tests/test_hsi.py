#!/usr/bin/env python3
"""Testes do pacote `hsi`.

Rodam em segundos e não precisam de GPU nem dos `.mat` baixados — usam cenas
sintéticas. O que cobrem é justamente o que era fácil de quebrar na refatoração:
o filtro de background, a divisão estratificada, os shapes que atravessam o
Conv1d e a lógica do early stopping.

    python tests/test_hsi.py          # sem dependências extras
    python -m pytest tests/           # se pytest estiver instalado
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch

from hsi import CNN1D, ExperimentConfig, PreparedData, Splits, build_loaders, remove_background, split_indices
from hsi.config import PCA_COMPONENTS
from hsi.data import Scene, to_pixel_table
from hsi.evaluate import binary_readout, _average_accuracy
from hsi.parallel import BACKENDS, StageTimings, spatial_branch, spectral_branch
from hsi.preprocess import mat2gray, rgb_composite
from hsi.segmentation import SEEDS_PARAMS, SegmentationArtifacts, SegmentationResult
from hsi.train import EarlyStopping, History, unpack_batch
from hsi.variants import BASELINE, OPTIMIZED, VARIANTS


def synthetic_scene(height=12, width=10, bands=20, classes=4, seed=0) -> Scene:
    """Cena pequena com background garantido e todas as classes representadas."""
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 5000, size=(height, width, bands), dtype=np.uint16)
    labels = np.zeros((height, width), dtype=np.uint8)
    flat = labels.reshape(-1)
    # 10 pixels por classe; o resto fica como background (rótulo 0).
    for c in range(1, classes + 1):
        flat[(c - 1) * 10 : c * 10] = c
    return Scene(code="TEST", image=image, labels=labels)


def test_to_pixel_table_preserva_bandas():
    scene = synthetic_scene()
    pixels, labels = to_pixel_table(scene)
    assert pixels.shape == (12 * 10, 20)
    assert labels.shape == (12 * 10,)
    # O pixel (0, 0) da tabela é o mesmo do cubo.
    assert np.array_equal(pixels[0], scene.image[0, 0])


def test_remove_background_reindexa_classes():
    scene = synthetic_scene()
    pixels, labels = to_pixel_table(scene)
    x, y = remove_background(pixels, labels)

    assert len(x) == len(y) == 40, "10 pixels por classe, 4 classes"
    assert y.min() == 0 and y.max() == 3, "classes reindexadas para 0..C-1"
    assert sorted(np.unique(y)) == [0, 1, 2, 3]
    # Não mexe no array de entrada — o pipeline reutiliza `labels` depois, para
    # filtrar os ids de superpixel, e um `y -= 1` in-place quebraria isso.
    assert labels.max() == 4


def test_remove_background_aceita_coluna_auxiliar():
    """O id de superpixel é filtrado pelo mesmo vetor de rótulos da cena."""
    labels = np.array([0, 1, 0, 2, 3], dtype=np.uint8)
    ids = np.array([[7], [8], [9], [10], [11]], dtype=np.int32)
    filtered, y = remove_background(ids, labels)
    assert filtered.reshape(-1).tolist() == [8, 10, 11]
    assert y.tolist() == [0, 1, 2]


def test_split_indices_particiona_sem_sobreposicao():
    y = np.repeat(np.arange(5), 40)
    splits = split_indices(y, test_ratio=0.15, valid_ratio=0.5, seed=42)

    todos = np.concatenate([splits.train, splits.val, splits.test])
    assert len(todos) == len(y), "toda amostra cai em exatamente uma partição"
    assert len(np.unique(todos)) == len(y), "nenhum índice repetido entre partições"
    assert len(splits.train) == 170  # 85% de 200
    assert len(splits.val) == len(splits.test) == 15


def test_split_indices_e_deterministico():
    y = np.repeat(np.arange(5), 40)
    a = split_indices(y, 0.15, 0.5, seed=42)
    b = split_indices(y, 0.15, 0.5, seed=42)
    c = split_indices(y, 0.15, 0.5, seed=7)
    assert np.array_equal(a.train, b.train), "mesma semente -> mesma partição"
    assert not np.array_equal(a.train, c.train), "semente diferente -> partição diferente"


def test_split_indices_estratifica():
    """Cada classe mantém sua proporção nas três partições."""
    y = np.repeat(np.arange(4), 100)
    splits = split_indices(y, 0.2, 0.5, seed=42)
    for indices in (splits.train, splits.val, splits.test):
        _, counts = np.unique(y[indices], return_counts=True)
        assert counts.min() == counts.max(), "classes balanceadas em todas as partições"


def test_mat2gray_normaliza_para_0_1():
    a = np.array([-3.0, 0.0, 7.0])
    out = mat2gray(a)
    assert out.min() == 0.0 and out.max() == 1.0


def test_rgb_composite_gera_uint8():
    components = np.random.default_rng(0).normal(size=(120, 3))
    rgb = rgb_composite(components, (12, 10))
    assert rgb.shape == (12, 10, 3)
    assert rgb.dtype == np.uint8


def test_rgb_composite_exige_tres_componentes():
    try:
        rgb_composite(np.zeros((10, 2)), (5, 2))
    except ValueError:
        return
    raise AssertionError("deveria exigir 3 componentes")


def test_cnn_infere_tamanho_do_flatten():
    """Kb=15 encolhe 15 -> 7 -> 5 -> 3 -> 1 pelas convoluções sem padding."""
    modelo = CNN1D(seq_len=15, classes=9)
    assert modelo.n_flat == 64 * 1
    assert CNN1D(seq_len=30, classes=16).n_flat == 64 * 16


def test_cnn_forward_uma_e_duas_entradas():
    x = torch.zeros(4, 1, 15)
    seg = torch.zeros(4, 1)

    uma = CNN1D(seq_len=15, classes=9, two_input=False)
    assert uma(x).shape == (4, 9)

    duas = CNN1D(seq_len=15, classes=9, two_input=True)
    assert duas(x, seg).shape == (4, 9)
    try:
        duas(x)
    except ValueError:
        return
    raise AssertionError("modelo de duas entradas deveria exigir o tensor de segmentação")


def test_cnn_usa_inicializacao_xavier():
    """Xavier/Glorot nos pesos e zero nos biases — não o padrão do PyTorch."""
    modelo = CNN1D(seq_len=15, classes=9)
    for module in modelo.modules():
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)):
            assert torch.all(module.bias == 0), "bias deveria começar zerado"


def test_cnn_reprodutivel_com_mesma_semente():
    torch.manual_seed(42)
    a = CNN1D(seq_len=15, classes=9)
    torch.manual_seed(42)
    b = CNN1D(seq_len=15, classes=9)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_early_stopping_para_apos_patience():
    parada = EarlyStopping(patience=3)
    modelo = CNN1D(seq_len=15, classes=9)

    assert not parada.step(1.0, modelo, 1)
    assert not parada.step(0.5, modelo, 2), "melhorou: zera a espera"
    assert not parada.step(0.6, modelo, 3)
    assert not parada.step(0.7, modelo, 4)
    assert parada.step(0.8, modelo, 5), "3 épocas sem melhora -> para"
    assert parada.best == 0.5
    assert parada.stopped_epoch == 5


def test_early_stopping_restaura_melhores_pesos_quando_pedido():
    modelo = CNN1D(seq_len=15, classes=9)
    parada = EarlyStopping(patience=2, restore_best_weights=True)
    parada.step(0.5, modelo, 1)  # melhor: guarda os pesos

    melhor = modelo.classifier[-1].weight.detach().clone()
    with torch.no_grad():
        modelo.classifier[-1].weight.add_(1.0)  # "treina" e piora

    parada.step(0.9, modelo, 2)
    parada.step(0.9, modelo, 3)  # estoura a paciência -> restaura
    assert torch.equal(modelo.classifier[-1].weight, melhor)


def test_early_stopping_padrao_mantem_ultimos_pesos():
    """restore_best_weights=False é o padrão, e o da implementação de referência."""
    modelo = CNN1D(seq_len=15, classes=9)
    parada = EarlyStopping(patience=1)
    parada.step(0.5, modelo, 1)
    original = modelo.classifier[-1].weight.detach().clone()
    with torch.no_grad():
        modelo.classifier[-1].weight.add_(1.0)
    parada.step(0.9, modelo, 2)
    assert not torch.equal(modelo.classifier[-1].weight, original)


def test_history_acumula_quatro_chaves():
    h = History()
    h.append(1.0, 0.5, 1.1, 0.4)
    h.append(0.9, 0.6, 1.0, 0.5)
    assert len(h) == 2
    assert list(h.epochs) == [1, 2]
    assert set(h.to_dict()) == {"loss", "accuracy", "val_loss", "val_accuracy"}
    assert h.to_dict()["val_accuracy"] == [0.4, 0.5]


def test_unpack_batch_cobre_um_e_dois_tensores():
    device = torch.device("cpu")
    x, seg, y = unpack_batch((torch.zeros(2, 1, 15), torch.zeros(2, 1), torch.zeros(2, dtype=torch.long)), device)
    assert seg is not None and x.shape == (2, 1, 15)

    x, seg, y = unpack_batch((torch.zeros(2, 1, 15), torch.zeros(2, dtype=torch.long)), device)
    assert seg is None


def test_average_accuracy_pesa_classes_igualmente():
    """Classe minoritária ruim derruba a AA sem derrubar a OA."""
    confusion = np.array([[99, 1], [8, 2]])
    per_class, aa = _average_accuracy(confusion)
    assert np.isclose(per_class[0], 0.99)
    assert np.isclose(per_class[1], 0.2)
    assert np.isclose(aa, 0.595)


def test_average_accuracy_tolera_classe_sem_amostras():
    confusion = np.array([[10, 0, 0], [0, 5, 0], [0, 0, 0]])
    _, aa = _average_accuracy(confusion)
    assert not np.isnan(aa), "linha zerada não pode virar NaN"


def test_binary_readout_reproduz_a_conta_do_original():
    confusion = np.array([[50, 10], [5, 35]])
    out = binary_readout(confusion)
    assert np.isclose(out["accuracy"], 85 / 100)
    assert np.isclose(out["precision"], 35 / 45)
    assert np.isclose(out["recall"], 35 / 40)


def test_build_loaders_respeita_two_input_e_shapes():
    config = ExperimentConfig(dataset="PU", batch_size=8)
    n, kb = 40, 15
    labels = np.repeat(np.arange(4), 10).astype(np.int64)
    data = PreparedData(
        spectra=np.random.default_rng(0).normal(size=(n, kb)),
        labels=labels,
        segments=np.arange(n),
        splits=Splits(train=np.arange(24), val=np.arange(24, 32), test=np.arange(32, 40)),
        scene_shape=(8, 5),
    )
    loaders = build_loaders(data, config, torch.device("cpu"))

    x, seg, y = next(iter(loaders["train"]))
    assert x.shape == (8, 1, kb), "espectro em channels-first para o nn.Conv1d"
    assert seg.shape == (8, 1)
    assert y.dtype == torch.int64
    assert x.dtype == torch.float32

    sem_seg = PreparedData(data.spectra, data.labels, None, data.splits, data.scene_shape)
    batch = next(iter(build_loaders(sem_seg, config, torch.device("cpu"))["train"]))
    assert len(batch) == 2


def test_build_loaders_nao_embaralha_val_e_test():
    config = ExperimentConfig(batch_size=4)
    data = PreparedData(
        spectra=np.arange(80, dtype=float).reshape(16, 5),
        labels=np.repeat(np.arange(4), 4).astype(np.int64),
        segments=None,
        splits=Splits(train=np.arange(8), val=np.arange(8, 12), test=np.arange(12, 16)),
        scene_shape=(4, 4),
    )
    loaders = build_loaders(data, config, torch.device("cpu"))
    y = torch.cat([batch[-1] for batch in loaders["test"]]).numpy()
    assert np.array_equal(y, data.labels[data.splits.test]), "ordem do teste deve casar com y_true"


def test_config_valida_dataset_e_deriva_pca():
    for code, expected in PCA_COMPONENTS.items():
        assert ExperimentConfig(dataset=code).pca_components == expected
    assert ExperimentConfig(dataset="PU", pca_components=8).pca_components == 8
    assert ExperimentConfig(dataset="PU").model_name == "PU_CNN"
    try:
        ExperimentConfig(dataset="XX")
    except ValueError:
        return
    raise AssertionError("dataset inválido deveria falhar cedo")


# --- paralelismo ------------------------------------------------------------


def test_spectral_branch_reduz_bandas():
    scene = synthetic_scene(bands=20)
    pixels, labels = to_pixel_table(scene)
    spectra, _ = remove_background(pixels, labels)
    reduced = spectral_branch(spectra, n_components=5, seed=42)
    assert reduced.shape == (40, 5)


def test_spectral_branch_e_deterministico():
    """Sem isso, paralelizar mudaria o resultado — que é o que não pode acontecer."""
    scene = synthetic_scene(bands=20)
    pixels, labels = to_pixel_table(scene)
    spectra, _ = remove_background(pixels, labels)
    a = spectral_branch(spectra, 5, 42)
    b = spectral_branch(spectra, 5, 42)
    assert np.array_equal(a, b)


def test_spatial_branch_devolve_um_id_por_pixel_valido():
    scene = synthetic_scene(height=40, width=40, classes=4)
    pixels, raw_labels = to_pixel_table(scene)
    ids, components, rgb, segmentation = spatial_branch(
        pixels, raw_labels, scene.shape, SEEDS_PARAMS["PU"], 42
    )
    assert len(ids) == 40, "um id de superpixel por pixel não-background"
    assert components.shape == (40 * 40, 3)
    assert rgb.shape == (40, 40, 3) and rgb.dtype == np.uint8
    assert isinstance(segmentation, SegmentationResult)
    assert segmentation.labels.shape == (40, 40)
    assert segmentation.contour_mask.shape == (40, 40)


def test_segmentation_artifacts_expoe_o_resultado():
    labels = np.zeros((4, 4), dtype=np.int32)
    labels[2:, 2:] = 1
    result = SegmentationResult(labels=labels, contour_mask=np.zeros((4, 4), dtype=np.uint8))
    art = SegmentationArtifacts(components=np.zeros((16, 3)), rgb=np.zeros((4, 4, 3), np.uint8), result=result)
    assert art.n_superpixels == 2
    assert np.array_equal(art.labels, labels)
    assert art.contour_mask.shape == (4, 4)


def test_stage_timings_calcula_soma_e_overhead():
    t = StageTimings(backend="thread", stages={"espectral": 1.0, "espacial": 3.0}, wall=3.2)
    assert t.branch_sum == 4.0, "o que custaria em série"
    assert abs(t.overhead - 0.2) < 1e-9, "wall menos o ramo mais longo"

    baseline = StageTimings(backend="sequential", stages={"espectral": 1.0, "espacial": 3.0}, wall=4.0)
    assert abs(t.speedup_over(baseline) - 1.25) < 1e-9


def test_backends_declarados():
    assert BACKENDS == ("sequential", "thread", "process")


def test_variantes_compartilham_hiperparametros():
    """A comparação só é interpretável se a única diferença for a execução."""
    assert BASELINE.backend == "sequential"
    assert OPTIMIZED.backend == "thread"
    assert BASELINE.config(dataset="PU") == OPTIMIZED.config(dataset="PU"), (
        "as duas variantes devem usar exatamente a mesma configuração"
    )
    assert set(VARIANTS) == {"baseline", "optimized"}


def test_prepare_data_rejeita_backend_invalido():
    from hsi.parallel import prepare_data_parallel
    try:
        prepare_data_parallel(ExperimentConfig(), backend="mpi")
    except ValueError as error:
        assert "mpi" in str(error)
        return
    raise AssertionError("backend inválido deveria falhar cedo")


def main() -> int:
    testes = [(name, fn) for name, fn in sorted(globals().items())
              if name.startswith("test_") and callable(fn)]
    falhas = []
    for name, fn in testes:
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as error:  # noqa: BLE001 — runner minimalista
            falhas.append((name, error))
            print(f"  FALHA {name}: {type(error).__name__}: {error}")

    print(f"\n{len(testes) - len(falhas)}/{len(testes)} testes passaram")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
