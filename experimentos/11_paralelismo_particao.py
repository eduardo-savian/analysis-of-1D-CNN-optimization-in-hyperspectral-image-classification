"""Paralelismo por partição da cena: até onde escala e o que custa em acurácia.

O PORQUÊ. O paralelismo de dois ramos (`prepare_data_parallel`) tem teto estrutural
de ~1,2×: a segmentação é 78–93% do pré-processamento, e os ramos são desbalanceados.
Pior, o pré-processamento inteiro é 0,9% do tempo de execução. Particionar a cena
ataca o lado certo — as partes têm tamanho parecido, então o teto é ≈ N, e o que fica
paralelo é o treino, que é onde o tempo está.

TRÊS PERGUNTAS, uma por eixo do experimento:

1. **Até onde escala?** N varre 2, 4, 8 e 16. O speedup deve crescer até os núcleos
   acabarem; a acurácia deve cair quando cada partição ficar pequena demais.
2. **Treinar junto ou separado?** `replicado` treina um modelo e paraleliza só a
   inferência (resultado idêntico ao monolítico, ganho puro de velocidade);
   `independente` treina um modelo por partição (paraleliza o caro, mas cada modelo
   vê 1/N das amostras).
3. **O ganho de acurácia das faixas é real?** A partição `aleatoria` tem custo
   computacional idêntico e destrói a contiguidade espacial. Se o ganho sumir nela,
   era localização — o mesmo atalho do vazamento espacial, não especialização.

CONTROLE DE THREADS. `torch.set_num_threads(1)` em tudo. Medido antes: o monolítico
leva 11,67 s com 16 threads e 8,55 s com uma — o modelo é pequeno demais para o
paralelismo intra-operação compensar, e deixá-lo ligado misturaria duas fontes de
paralelismo na mesma medição.

Uso:  python 11_paralelismo_particao.py [CENA ...]   (padrão: IP)
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

ns = carregar()
EC = ns["ExperimentConfig"]
ns["configure_determinism"](True, verbose=False)
torch.set_num_threads(1)          # ver docstring: isola o paralelismo de partição

PARTICOES = (2, 4, 8, 16)
EPOCAS = 25
SEMENTES = (11, 29, 42)
CENAS = tuple(a for a in sys.argv[1:] if not a.startswith("-")) or ("IP",)


def config_base(cena: str, semente: int) -> "EC":
    return EC(dataset=cena, model_type="cnn1d", two_input=True, segmenter="seeds",
              fusion="scalar", train_fraction=1.0, epochs=EPOCAS, patience=10,
              seed=semente, batch_size=64)


def main() -> None:
    print("PARALELISMO POR PARTIÇÃO DA CENA")
    print(f"cenas: {', '.join(CENAS)} | N: {PARTICOES} | {EPOCAS} épocas | "
          f"{len(SEMENTES)} sementes | torch threads=1", flush=True)

    destino = caminho_resultado("11_paralelismo_particao.csv")
    linhas, feitas = carregar_feito(destino, ("cena", "arranjo", "n", "semente"))
    if feitas:
        print(f"retomando: {len(feitas)} medições já feitas", flush=True)

    for cena in CENAS:
        print(f"\n{'='*104}\n### {cena} — {ns['DATASETS'][cena]}\n{'='*104}", flush=True)

        # Linha de base: pipeline monolítico, mesma receita, uma thread.
        for semente in SEMENTES:
            if (cena, "monolítico", 1, semente) in feitas:
                continue
            inicio = time.perf_counter()
            resultado = ns["run_experiment"](config_base(cena, semente),
                                             backend="thread", verbose=False)
            linhas.append({
                "cena": cena, "arranjo": "monolítico", "n": 1, "semente": semente,
                "tempo_total": time.perf_counter() - inicio,
                "bloco_paralelo": float("nan"), "soma_partes": float("nan"),
                "speedup_bloco": float("nan"), "teto": float("nan"),
                "OA": resultado.metrics.overall_accuracy,
                "AA": resultado.metrics.average_accuracy,
                "Kappa": resultado.metrics.kappa,
            })
            ns["release_result"](resultado); gc.collect()
            salvar(linhas, destino)

        base = [l for l in linhas if l["cena"] == cena and l["arranjo"] == "monolítico"]
        tempo_base = float(np.mean([l["tempo_total"] for l in base]))
        oa_base = float(np.mean([l["OA"] for l in base]))
        print(f"{'arranjo':<26}{'N':>3}{'tempo':>9}{'speedup':>9}{'bloco':>9}"
              f"{'teto':>7}{'OA':>18}{'ΔOA':>9}", flush=True)
        print(f"{'monolítico':<26}{1:>3}{tempo_base:>8.2f}s{'—':>9}{'—':>9}{'—':>7}"
              f"{oa_base:>11.4f}±{np.std([l['OA'] for l in base]):.4f}{'—':>9}", flush=True)

        arranjos = [("independente · faixas", "independente", "faixas"),
                    ("independente · aleatória", "independente", "aleatoria"),
                    ("replicado · faixas", "replicado", "faixas")]
        for rotulo, modo, particao in arranjos:
            for n in PARTICOES:
                novas = [s for s in SEMENTES if (cena, rotulo, n, s) not in feitas]
                for semente in novas:
                    inicio = time.perf_counter()
                    try:
                        r = ns["run_experiment_particionado"](
                            config_base(cena, semente), n_particoes=n, modo=modo,
                            particao_modo=particao, backend="thread", verbose=False)
                    except Exception as erro:            # partição pequena demais
                        print(f"  {rotulo} N={n} s={semente}: {type(erro).__name__}: "
                              f"{str(erro)[:70]}", flush=True)
                        break
                    linhas.append({
                        "cena": cena, "arranjo": rotulo, "n": n, "semente": semente,
                        "tempo_total": time.perf_counter() - inicio,
                        "bloco_paralelo": r.tempo_paralelo,
                        "soma_partes": r.soma_das_partes,
                        "speedup_bloco": r.soma_das_partes / max(r.tempo_paralelo, 1e-9),
                        "teto": r.teto_de_amdahl,
                        "OA": r.metrics.overall_accuracy,
                        "AA": r.metrics.average_accuracy,
                        "Kappa": r.metrics.kappa,
                    })
                    del r; gc.collect()
                    salvar(linhas, destino)
                grupo = [l for l in linhas if l["cena"] == cena
                         and l["arranjo"] == rotulo and l["n"] == n]
                if not grupo:
                    continue
                oa = np.array([l["OA"] for l in grupo])
                tempo = float(np.mean([l["tempo_total"] for l in grupo]))
                print(f"{rotulo:<26}{n:>3}{tempo:>8.2f}s{tempo_base/tempo:>8.2f}x"
                      f"{np.mean([l['speedup_bloco'] for l in grupo]):>8.2f}x"
                      f"{np.mean([l['teto'] for l in grupo]):>6.2f}x"
                      f"{oa.mean():>11.4f}±{oa.std():.4f}"
                      f"{100*(oa.mean()-oa_base):>+8.2f}", flush=True)

    quadro = pd.DataFrame(linhas)
    print(f"\n{'='*104}\nFAIXAS CONTRA ALEATÓRIA — o ganho de acurácia é espacial?\n{'='*104}",
          flush=True)
    for cena in CENAS:
        sub = quadro[quadro["cena"] == cena]
        print(f"\n{cena}:  {'N':>3}{'faixas':>12}{'aleatória':>12}{'diferença':>12}")
        for n in PARTICOES:
            f = sub[(sub["arranjo"] == "independente · faixas") & (sub["n"] == n)]["OA"]
            a = sub[(sub["arranjo"] == "independente · aleatória") & (sub["n"] == n)]["OA"]
            if f.empty or a.empty:
                continue
            print(f"      {n:>3}{f.mean():>12.4f}{a.mean():>12.4f}"
                  f"{100*(f.mean()-a.mean()):>+11.2f}pp")
        print("\n  Diferença perto de zero → o ganho vem de partir o problema, não da posição.")
        print("  Faixas bem acima → há atalho espacial, e ele precisa ser reportado.")
    print(f"\nCSV: {destino}", flush=True)


if __name__ == "__main__":
    main()
