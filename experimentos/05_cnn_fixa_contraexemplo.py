"""Impacto de cada segmentador na CLASSIFICAÇÃO (OA, AA, Kappa).

CRITÉRIO: CNN1D e hiperparâmetros fixos — a única variável é o segmentador.
Sem isso, uma diferença de acurácia poderia vir do learning rate e não do
algoritmo. A condição "sem fusão" é o piso: sem ela, "0,78 de OA" não significa
nada, porque não há com o que comparar.

Protocolo: 10% de treino (estratificado), teto de 200 épocas com patience 10.
Os 200 são teto, não alvo — com poucas amostras o modelo não bate a meta de 0,95
e o early stopping governa.
"""
import sys, time, gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib; matplotlib.use("Agg")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from _bootstrap import carregar, caminho_resultado

ns = carregar()
EC = ns["ExperimentConfig"]

SEEDS = (11, 29, 42, 73, 101)
CENAS = ("IP", "PU", "SA")
CONDICOES = ("sem fusão", "seeds", "slic", "lsc", "watershed")
TABELAS = {"seeds": "SEEDS_PARAMS", "slic": "SLIC_PARAMS",
           "lsc": "LSC_PARAMS", "watershed": "WATERSHED_PARAMS"}


def estrutura(cena: str, segmentador: str) -> dict:
    """Métricas estruturais da segmentação — calculadas uma vez, não por seed."""
    cenario = ns["load_scene"](cena, download=False)
    pixels, _ = ns["to_pixel_table"](cenario)
    componentes, _ = ns["apply_pca"](pixels, n_components=3, random_state=42)
    rgb = ns["rgb_composite"](componentes, cenario.shape)
    parametros = ns[TABELAS[segmentador]][cena]
    resultado = ns["segment_superpixels"](rgb, segmentador, parametros)
    return ns["segmentation_quality"](resultado.labels, cenario.labels)


def main() -> None:
    print("IMPACTO DO SEGMENTADOR NA CLASSIFICAÇÃO — CNN1D, hiperparâmetros fixos")
    print(f"10% de treino | teto de 200 épocas | patience 10 | {len(SEEDS)} seeds", flush=True)
    linhas = []
    for cena in CENAS:
        print(f"\n### {cena} — {ns['DATASETS'][cena]}", flush=True)
        print(f"{'condição':<12}{'OA':>18}{'AA':>18}{'Kappa':>18}"
              f"{'regiões':>9}{'pureza':>9}{'treino':>9}{'épocas':>8}", flush=True)
        piso = None
        for condicao in CONDICOES:
            metricas, tempos, epocas = [], [], []
            for semente in SEEDS:
                config = EC(
                    dataset=cena, model_type="cnn1d",
                    two_input=(condicao != "sem fusão"),
                    segmenter=("seeds" if condicao == "sem fusão" else condicao),
                    fusion="scalar", train_fraction=0.10,
                    epochs=200, patience=10, early_stopping_target=0.95, seed=semente,
                )
                resultado = ns["run_experiment"](config, backend="thread", verbose=False)
                metricas.append((resultado.metrics.overall_accuracy,
                                 resultado.metrics.average_accuracy,
                                 resultado.metrics.kappa))
                tempos.append(resultado.training.total_time)
                epocas.append(len(resultado.training.history))
                linhas.append({
                    "cena": cena, "condição": condicao, "seed": semente,
                    "OA": metricas[-1][0], "AA": metricas[-1][1], "Kappa": metricas[-1][2],
                    "treino_s": tempos[-1], "épocas": epocas[-1],
                })
                ns["release_result"](resultado)
                gc.collect()
            valores = np.array(metricas)
            if piso is None:
                piso = valores[:, 0].mean()
            qualidade = estrutura(cena, condicao) if condicao != "sem fusão" else None
            regioes = qualidade["superpixels"] if qualidade else 0
            pureza = f"{qualidade['pureza rotulada']:.4f}" if qualidade else "—"
            print(f"{condicao:<12}"
                  f"{valores[:,0].mean():>11.4f}±{valores[:,0].std():.4f}"
                  f"{valores[:,1].mean():>11.4f}±{valores[:,1].std():.4f}"
                  f"{valores[:,2].mean():>11.4f}±{valores[:,2].std():.4f}"
                  f"{regioes:>9}{pureza:>9}"
                  f"{np.mean(tempos):>8.1f}s{np.mean(epocas):>8.0f}", flush=True)
    quadro = pd.DataFrame(linhas)
    destino = caminho_resultado("05_impacto_segmentadores.csv")
    quadro.to_csv(destino, index=False)
    print(f"\nGanho sobre a condição sem fusão (OA média):", flush=True)
    for cena in CENAS:
        sub = quadro[quadro["cena"] == cena].groupby("condição")["OA"].mean()
        base = sub["sem fusão"]
        ganhos = ", ".join(f"{k}={100*(v-base):+.2f}pp"
                           for k, v in sub.items() if k != "sem fusão")
        print(f"  {cena}: {ganhos}", flush=True)
    print(f"\nCSV: {destino}", flush=True)


if __name__ == "__main__":
    main()
