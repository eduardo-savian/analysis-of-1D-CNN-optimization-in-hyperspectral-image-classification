"""Arquitetura × segmentador, cada combinação com seus próprios hiperparâmetros.

CRITÉRIO. Mesmo princípio da entrada 9 do diário, estendido a duas dimensões: não
faz sentido comparar arquiteturas com hiperparâmetros fixos, nem comparar
segmentadores com a CNN fixa. Cada célula da matriz (arquitetura × condição)
recebe sua própria busca, com orçamento idêntico, e só então os vencedores são
comparados entre si.

O que muda em relação ao experimento 06:

- `patch_size` entra no espaço de busca para CNN2D e CNN3D (a CNN1D não usa
  contexto espacial, e `hpo_search_space` já o remove de lá automaticamente).
- `release_result` deixa de ser conforto e vira requisito: um trial de CNN3D em
  Indian Pines retém 207,7 MB — 16 trials reteriam 3,3 GB, e em Salinas seriam
  muito mais. Ver entrada 5 do diário.

A estratégia de busca é fixada em TPE em todas as células, para que a diferença
medida seja de arquitetura e segmentador, e não de estratégia.

Uso:  python 07_arquiteturas_x_segmentadores.py [CENA ...]   (padrão: IP)
"""
from __future__ import annotations

import gc
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from _bootstrap import carregar, caminho_resultado, memoria_mb
from _retomada import carregar_feito, salvar

ns = carregar()
EC = ns["ExperimentConfig"]

TRIALS = 16
SEEDS_FINAIS = (11, 29, 42, 73, 101)
EPOCAS = 200
FRACAO = 0.10
ESTRATEGIA = "tpe"
ARQUITETURAS = ("cnn1d", "cnn2d", "cnn3d")
CONDICOES = ("sem fusão", "seeds", "slic", "lsc", "watershed")
CENAS = tuple(a for a in sys.argv[1:] if not a.startswith("-")) or ("IP",)


def buscar(cena: str, arquitetura: str, condicao: str):
    """Etapa 1: HPO restrito a esta arquitetura e a este segmentador."""
    com_fusao = condicao != "sem fusão"
    base = EC(
        dataset=cena, model_type=arquitetura,
        two_input=com_fusao,
        segmenter=("seeds" if not com_fusao else condicao),
        fusion="scalar", train_fraction=FRACAO,
        epochs=EPOCAS, patience=10, early_stopping_target=0.95, seed=42,
    )
    tabela, _ = ns["automated_hpo"](
        base, trials_per_cnn=TRIALS, search_epochs=EPOCAS, backend="thread",
        model_types=(arquitetura,), segmenters=(base.segmenter,),
        strategy=ESTRATEGIA, retain="winner",
    )
    vencedor = tabela.loc[tabela["val_accuracy (objetivo)"].idxmax()]
    return vencedor["config"], float(vencedor["val_accuracy (objetivo)"])


def retreinar(config, sementes):
    """Etapa 2: repete a receita vencedora com várias sementes."""
    metricas, parametros = [], None
    for semente in sementes:
        resultado = ns["run_experiment"](
            replace(config, seed=int(semente)), backend="thread", verbose=False
        )
        metricas.append((resultado.metrics.overall_accuracy,
                         resultado.metrics.average_accuracy,
                         resultado.metrics.kappa))
        if parametros is None:
            parametros = int(resultado.model.n_parameters)
        ns["release_result"](resultado)
        gc.collect()
    return np.array(metricas), parametros


def main() -> None:
    print("ARQUITETURA × SEGMENTADOR — cada célula com seus próprios hiperparâmetros")
    print(f"busca: {ESTRATEGIA}, {TRIALS} trials por célula | final: {len(SEEDS_FINAIS)} "
          f"sementes | {int(FRACAO*100)}% de treino | teto de {EPOCAS} épocas")
    print(f"cenas: {', '.join(CENAS)} | células: "
          f"{len(ARQUITETURAS) * len(CONDICOES) * len(CENAS)}", flush=True)

    destino = caminho_resultado("07_arquiteturas.csv")
    resumo, feitas = carregar_feito(destino, ("cena", "arquitetura", "condição"))
    if feitas:
        print(f"retomando: {len(feitas)} células já medidas em {destino.name}", flush=True)
    for cena in CENAS:
        for arquitetura in ARQUITETURAS:
            print(f"\n{'='*96}\n### {cena} · {arquitetura.upper()}\n{'='*96}", flush=True)
            print(f"{'condição':<12}{'val':>9}{'OA':>18}{'AA':>18}{'Kappa':>18}"
                  f"{'params':>10}{'RSS':>8}{'min':>7}", flush=True)
            piso = next((r["OA"] for r in resumo
                         if r["cena"] == cena and r["arquitetura"] == arquitetura
                         and r["condição"] == "sem fusão"), None)
            for condicao in CONDICOES:
                if (cena, arquitetura, condicao) in feitas:
                    pronto = next(r for r in resumo
                                  if r["cena"] == cena and r["arquitetura"] == arquitetura
                                  and r["condição"] == condicao)
                    print(f"{condicao:<12}{'(já medido)':>9}{pronto['OA']:>18.4f}"
                          f"{pronto['AA']:>18.4f}{pronto['Kappa']:>18.4f}", flush=True)
                    continue
                inicio = time.perf_counter()
                config, melhor_val = buscar(cena, arquitetura, condicao)
                valores, parametros = retreinar(config, SEEDS_FINAIS)
                if piso is None:
                    piso = valores[:, 0].mean()
                minutos = (time.perf_counter() - inicio) / 60
                print(f"{condicao:<12}{melhor_val:>9.4f}"
                      f"{valores[:,0].mean():>11.4f}±{valores[:,0].std():.4f}"
                      f"{valores[:,1].mean():>11.4f}±{valores[:,1].std():.4f}"
                      f"{valores[:,2].mean():>11.4f}±{valores[:,2].std():.4f}"
                      f"{parametros:>10}{memoria_mb():>8.0f}{minutos:>7.1f}", flush=True)
                resumo.append({
                    "cena": cena, "arquitetura": arquitetura, "condição": condicao,
                    "val_busca": melhor_val,
                    "OA": valores[:, 0].mean(), "OA_std": valores[:, 0].std(),
                    "AA": valores[:, 1].mean(), "AA_std": valores[:, 1].std(),
                    "Kappa": valores[:, 2].mean(), "Kappa_std": valores[:, 2].std(),
                    "ganho_OA_pp": 100 * (valores[:, 0].mean() - piso),
                    "parâmetros": parametros,
                    "learning_rate": config.learning_rate, "dropout": config.dropout,
                    "weight_decay": config.weight_decay, "batch_size": config.batch_size,
                    "pca_components": config.pca_components,
                    "patch_size": config.patch_size,
                    "overrides": str(config.segmenter_overrides),
                    "minutos": minutos, "rss_mb": memoria_mb(),
                })
                # Grava a cada célula: uma interrupção perde no máximo uma.
                salvar(resumo, destino)
            sub = [r for r in resumo if r["cena"] == cena and r["arquitetura"] == arquitetura]
            ganhos = ", ".join(f"{r['condição']}={r['ganho_OA_pp']:+.2f}pp"
                               for r in sub if r["condição"] != "sem fusão")
            print(f"\n  ganho sobre 'sem fusão': {ganhos}", flush=True)

    quadro = pd.DataFrame(resumo)
    print(f"\n{'='*96}\nMELHOR COMBINAÇÃO POR CENA\n{'='*96}", flush=True)
    for cena in CENAS:
        sub = quadro[quadro["cena"] == cena]
        melhor = sub.loc[sub["OA"].idxmax()]
        print(f"  {cena}: {melhor['arquitetura']} + {melhor['condição']} → "
              f"OA {melhor['OA']:.4f} ± {melhor['OA_std']:.4f} | "
              f"AA {melhor['AA']:.4f} | Kappa {melhor['Kappa']:.4f} | "
              f"{melhor['parâmetros']} parâmetros", flush=True)
    print(f"\nCSV: {destino}", flush=True)


if __name__ == "__main__":
    main()
