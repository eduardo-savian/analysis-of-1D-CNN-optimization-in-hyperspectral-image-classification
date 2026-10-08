"""O ganho de acurácia das faixas sobrevive quando o atalho posicional é removido?

O QUE JÁ SABEMOS. Particionar a cena em faixas horizontais e treinar um modelo por
faixa rende até +3,47 pp de OA sobre o pipeline monolítico. O controle com partição
aleatória — mesmo custo computacional, mesma quantidade de dados por modelo — derruba
a acurácia em até 28,83 pp. Logo o ganho vem da **posição**, não de partir o problema.

A PERGUNTA QUE FALTA. Posição pode ser duas coisas muito diferentes:

- **Especialização legítima.** A cobertura do solo é espacialmente agrupada; cada
  faixa contém poucas classes, e um modelo por faixa resolve um problema mais fácil.
  Isso é estrutura real do dado, e usá-la é uma decisão de projeto defensável.
- **Atalho de vizinhança.** Sob split aleatório por pixel, cada pixel de teste tem
  vizinhos quase idênticos no treino da mesma faixa. O modelo interpola em vez de
  generalizar — o mesmo mecanismo do vazamento espacial da entrada 3 do diário.

O DESENHO QUE SEPARA OS DOIS. Rodar o mesmo arranjo nos dois protocolos:

- `pixel`   — split aleatório estratificado (vazamento de 100%)
- `region`  — validação cruzada por região (vazamento de 0%)

Se o ganho das faixas se mantiver sob `region`, é especialização: as classes continuam
agrupadas, mas os vizinhos imediatos sumiram. Se evaporar, era interpolação.

A partição aleatória entra como controle nos **dois** protocolos, para que a
comparação não dependa de supor que o custo computacional é igual — ele é medido.

EXECUTOR. `process`, o único que acelera de fato (1,49–1,54× contra 0,26–0,58× das
threads). Ver entrada do diário sobre o GIL.
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

from _bootstrap import carregar, caminho_resultado
from _retomada import carregar_feito, salvar

ns = carregar(globals())      # globals reais: o executor de processos precisa
EC = ns["ExperimentConfig"]
ns["configure_determinism"](True, verbose=False)
torch.set_num_threads(1)

PARTICOES = (2, 4, 8)
FOLDS = 5
SEMENTES_PIXEL = (11, 29, 42, 73, 101)
EPOCAS = 25
CENAS = tuple(a for a in sys.argv[1:] if not a.startswith("-")) or ("IP",)


def config_de(cena: str, protocolo: str, repeticao: int):
    campos = dict(dataset=cena, model_type="cnn1d", two_input=True, segmenter="seeds",
                  fusion="scalar", train_fraction=1.0, epochs=EPOCAS, patience=10,
                  batch_size=64)
    if protocolo == "region":
        campos.update(split_mode="region", split_folds=FOLDS, split_fold=repeticao, seed=42)
    else:
        campos.update(split_mode="pixel", seed=SEMENTES_PIXEL[repeticao])
    return EC(**campos)


def main() -> None:
    print("PARTIÇÃO DA CENA SOB OS DOIS PROTOCOLOS DE SPLIT")
    print(f"cenas: {', '.join(CENAS)} | N: {PARTICOES} | {EPOCAS} épocas | "
          f"executor=process | {FOLDS} folds · {len(SEMENTES_PIXEL)} sementes", flush=True)

    destino = caminho_resultado("12_particao_split.csv")
    linhas, feitas = carregar_feito(destino, ("cena", "protocolo", "arranjo", "n", "rep"))
    if feitas:
        print(f"retomando: {len(feitas)} medições já feitas", flush=True)

    for cena in CENAS:
        for protocolo in ("pixel", "region"):
            print(f"\n{'='*96}\n### {cena} · split {protocolo}\n{'='*96}", flush=True)
            print(f"{'arranjo':<26}{'N':>3}{'OA':>18}{'AA':>18}{'ΔOA vs mono':>13}", flush=True)
            repeticoes = range(FOLDS if protocolo == "region" else len(SEMENTES_PIXEL))

            for rep in repeticoes:
                if (cena, protocolo, "monolítico", 1, rep) in feitas:
                    continue
                resultado = ns["run_experiment"](config_de(cena, protocolo, rep),
                                                 backend="thread", verbose=False)
                linhas.append({
                    "cena": cena, "protocolo": protocolo, "arranjo": "monolítico",
                    "n": 1, "rep": rep, "tempo": float("nan"),
                    "OA": resultado.metrics.overall_accuracy,
                    "AA": resultado.metrics.average_accuracy,
                    "Kappa": resultado.metrics.kappa,
                })
                ns["release_result"](resultado); gc.collect()
                salvar(linhas, destino)

            mono = [l for l in linhas if l["cena"] == cena
                    and l["protocolo"] == protocolo and l["arranjo"] == "monolítico"]
            base = float(np.mean([l["OA"] for l in mono]))
            print(f"{'monolítico':<26}{1:>3}{base:>11.4f}"
                  f"±{np.std([l['OA'] for l in mono]):.4f}"
                  f"{np.mean([l['AA'] for l in mono]):>11.4f}"
                  f"±{np.std([l['AA'] for l in mono]):.4f}{'—':>13}", flush=True)

            for rotulo, particao in (("faixas", "faixas"), ("aleatória", "aleatoria")):
                for n in PARTICOES:
                    novas = [r for r in repeticoes
                             if (cena, protocolo, rotulo, n, r) not in feitas]
                    for rep in novas:
                        inicio = time.perf_counter()
                        try:
                            r = ns["run_experiment_particionado"](
                                config_de(cena, protocolo, rep), n_particoes=n,
                                modo="independente", particao_modo=particao,
                                executor="process", verbose=False)
                        except Exception as erro:
                            print(f"  {rotulo} N={n} rep {rep}: "
                                  f"{type(erro).__name__}: {str(erro)[:70]}", flush=True)
                            break
                        linhas.append({
                            "cena": cena, "protocolo": protocolo, "arranjo": rotulo,
                            "n": n, "rep": rep,
                            "tempo": time.perf_counter() - inicio,
                            "OA": r.metrics.overall_accuracy,
                            "AA": r.metrics.average_accuracy,
                            "Kappa": r.metrics.kappa,
                        })
                        del r; gc.collect()
                        salvar(linhas, destino)
                    grupo = [l for l in linhas if l["cena"] == cena
                             and l["protocolo"] == protocolo
                             and l["arranjo"] == rotulo and l["n"] == n]
                    if not grupo:
                        continue
                    oa = np.array([l["OA"] for l in grupo])
                    aa = np.array([l["AA"] for l in grupo])
                    print(f"{rotulo:<26}{n:>3}{oa.mean():>11.4f}±{oa.std():.4f}"
                          f"{aa.mean():>11.4f}±{aa.std():.4f}"
                          f"{100*(oa.mean()-base):>+12.2f}", flush=True)

    quadro = pd.DataFrame(linhas)
    print(f"\n{'='*96}\nA RESPOSTA: o ganho das faixas sobrevive sem o atalho?\n{'='*96}",
          flush=True)
    for cena in CENAS:
        print(f"\n{cena}:  {'N':>3}{'ganho pixel':>14}{'ganho região':>15}{'quanto sobra':>15}")
        for n in PARTICOES:
            linha = {}
            for protocolo in ("pixel", "region"):
                sub = quadro[(quadro["cena"] == cena) & (quadro["protocolo"] == protocolo)]
                mono = sub[sub["arranjo"] == "monolítico"]["OA"].mean()
                faixas = sub[(sub["arranjo"] == "faixas") & (sub["n"] == n)]["OA"].mean()
                linha[protocolo] = 100 * (faixas - mono)
            sobra = (linha["region"] / linha["pixel"] * 100
                     if abs(linha["pixel"]) > 1e-9 else float("nan"))
            print(f"      {n:>3}{linha['pixel']:>+13.2f}pp{linha['region']:>+14.2f}pp"
                  f"{sobra:>14.0f}%")
        print("\n  Sobra alta  → especialização: as classes seguem agrupadas por faixa.")
        print("  Sobra baixa → era interpolação entre vizinhos, o mesmo vazamento da entrada 3.")
    print(f"\nCSV: {destino}", flush=True)


if __name__ == "__main__":
    main()
