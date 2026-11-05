"""Uma execução completa: dados, treino e métricas de teste.

É a função que o notebook, a busca de hiperparâmetros e os experimentos chamam.
"""

from hsi.ambiente import configure_determinism, seed_everything, select_device
from hsi.config import ExperimentConfig
from hsi.dados import build_loaders, prepare_data
from hsi.modelo import build_model
from hsi.treino import fit, metrics, predict


def run_experiment(config: ExperimentConfig, verbose: bool = False, device=None) -> dict:
    """Prepara os dados, treina a CNN 1D e avalia no teste.

    Devolve um dicionário com o modelo, os dados, o histórico do treino, as métricas
    de teste e o tempo de treino.
    """
    device = device or select_device(verbose=False)
    configure_determinism()
    dados = prepare_data(config)
    loaders = build_loaders(dados, config, device)

    seed_everything(config.seed)
    modelo, perda, otimizador = build_model(config, dados, device)
    treino = fit(modelo, loaders, perda, otimizador, config, device, verbose=verbose)

    verdade = dados["rotulos"][dados["teste"]]
    previsto = predict(modelo, loaders["teste"], device)
    resultado = {
        "config": config, "dados": dados, "modelo": modelo, "previsto": previsto,
        "historico": treino["historico"], "epocas": len(treino["historico"]["loss"]),
        "tempo_treino": treino["tempo"], "motivo_parada": treino["motivo"],
        "melhor_epoca": treino["melhor_epoca"],
        **metrics(verdade, previsto),
    }
    if verbose:
        print(f"{resultado['epocas']} épocas ({resultado['motivo_parada']}), "
              f"{resultado['tempo_treino']:.1f}s | OA {resultado['OA']:.4f}  "
              f"AA {resultado['AA']:.4f}  Kappa {resultado['Kappa']:.4f}")
    return resultado
