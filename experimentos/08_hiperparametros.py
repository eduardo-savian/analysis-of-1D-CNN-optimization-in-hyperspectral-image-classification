"""Análise da automação de hiperparâmetros: estratégias de busca e peso de cada eixo.

Responde três perguntas que o plano de verificação pede e que ainda não tinham
número consolidado no repositório:

1. **Vale um hipertunador?** Random search, TPE (bayesiano) e algoritmo genético
   percorrem o mesmo espaço, com o mesmo orçamento de trials e a mesma cobertura
   inicial dos quatro segmentadores. Só muda como o candidato é escolhido.
2. **O orçamento por trial muda o veredito?** A busca roda sob três tetos de época
   (10, 40, 200). Se o ranking das estratégias mudar entre eles, a triagem barata
   não é um proxy confiável — é o *low-fidelity bias*, e é um resultado
   metodológico por si.
3. **Quais eixos do espaço importam?** Com todos os trials salvos, dá para medir a
   validação média por valor de cada eixo. Um eixo plano é candidato a sair do
   espaço na próxima campanha, liberando orçamento para os que importam.

CRITÉRIO. O que torna a comparação honesta é a simetria: mesmo espaço, mesmo
orçamento, mesma cobertura inicial, mesma semente. A única diferença é a decisão
de qual candidato treinar em seguida.

**Todos os trials são salvos**, não só os vencedores — sem eles não há como medir
o peso dos eixos, e foi exatamente esse dado que se perdeu antes.

Uso:  python 08_hiperparametros.py [CENA ...]   (padrão: IP)
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

from _bootstrap import carregar, caminho_resultado
from _retomada import carregar_feito, salvar

ns = carregar()
EC = ns["ExperimentConfig"]

TRIALS = 20
ORCAMENTOS = (10, 40, 200)
FRACAO = 0.10
CENAS = tuple(a for a in sys.argv[1:] if not a.startswith("-")) or ("IP",)
EIXOS = ("learning rate", "dropout", "weight decay", "batch", "PCA")


def rodar(cena: str, orcamento: int, estrategia: str) -> pd.DataFrame:
    """Uma busca completa; devolve a tabela com TODOS os trials."""
    base = EC(dataset=cena, model_type="cnn1d", two_input=True, segmenter="seeds",
              fusion="scalar", train_fraction=FRACAO,
              epochs=orcamento, patience=10, early_stopping_target=0.95, seed=42)
    tabela, _ = ns["automated_hpo"](
        base, trials_per_cnn=TRIALS, search_epochs=orcamento, backend="thread",
        strategy=estrategia, population=6, retain="winner",
    )
    tabela = tabela.drop(columns=["resultado", "config"], errors="ignore")
    tabela["cena"] = cena
    tabela["orçamento"] = orcamento
    return tabela


def main() -> None:
    print("AUTOMAÇÃO DE HIPERPARÂMETROS — estratégias de busca e peso dos eixos")
    print(f"cenas: {', '.join(CENAS)} | {TRIALS} trials por busca | "
          f"orçamentos: {ORCAMENTOS} | {int(FRACAO*100)}% de treino", flush=True)

    destino = caminho_resultado("08_trials.csv")
    feitos, chaves = carregar_feito(destino, ("cena", "orçamento", "estratégia"))
    if chaves:
        print(f"retomando: {len(chaves)} buscas já feitas", flush=True)
    todos = list(feitos)

    for cena in CENAS:
        for orcamento in ORCAMENTOS:
            print(f"\n--- {cena} · teto de {orcamento} épocas ---", flush=True)
            for estrategia in ns["HPO_STRATEGIES"]:
                if (cena, orcamento, estrategia) in chaves:
                    print(f"  {estrategia:<8} (já medido)", flush=True)
                    continue
                inicio = time.perf_counter()
                tabela = rodar(cena, orcamento, estrategia)
                melhor = tabela.loc[tabela["val_accuracy (objetivo)"].idxmax()]
                print(f"  {estrategia:<8} val={melhor['val_accuracy (objetivo)']:.4f} "
                      f"(trial {int(melhor['trial']):>2})  OA_teste={melhor['OA teste (relato)']:.4f}  "
                      f"seg={melhor['segmentador']:<10} "
                      f"épocas_médias={tabela['épocas executadas'].mean():5.1f}  "
                      f"{time.perf_counter()-inicio:5.1f}s", flush=True)
                todos.extend(tabela.to_dict(orient="records"))
                salvar(todos, destino)
                gc.collect()

    quadro = pd.DataFrame(todos)

    print(f"\n{'='*92}\nCONVERGÊNCIA — melhor validação até o trial n\n{'='*92}", flush=True)
    for cena in CENAS:
        sub = quadro[quadro["cena"] == cena]
        pivo = sub.pivot_table(index="trial", columns=["orçamento", "estratégia"],
                               values="melhor até agora")
        print(f"\n{cena}:"); print(pivo.round(4).to_string(), flush=True)

    print(f"\n{'='*92}\nDISPERSÃO ENTRE ESTRATÉGIAS, POR ORÇAMENTO\n{'='*92}", flush=True)
    melhor_por = (quadro.groupby(["cena", "orçamento", "estratégia"])["val_accuracy (objetivo)"]
                  .max().unstack())
    melhor_por["dispersão (pp)"] = (melhor_por.max(axis=1) - melhor_por.min(axis=1)) * 100
    print(melhor_por.round(4).to_string(), flush=True)

    print(f"\n{'='*92}\nPESO DE CADA EIXO — validação média por valor\n{'='*92}", flush=True)
    for cena in CENAS:
        sub = quadro[quadro["cena"] == cena]
        print(f"\n{cena}:")
        amplitudes = {}
        for eixo in EIXOS:
            if eixo not in sub:
                continue
            medias = sub.groupby(eixo)["val_accuracy (objetivo)"].agg(["mean", "count"])
            amplitudes[eixo] = (medias["mean"].max() - medias["mean"].min()) * 100
            linha = "  ".join(f"{v}→{m:.4f}(n={int(c)})"
                              for v, (m, c) in medias.iterrows())
            print(f"  {eixo:<15} {linha}")
        print("\n  amplitude (pp), do eixo que mais importa ao que menos importa:")
        for eixo, amp in sorted(amplitudes.items(), key=lambda x: -x[1]):
            print(f"    {eixo:<15} {amp:6.2f} pp")

    print(f"\n{'='*92}\nONDE CADA ESTRATÉGIA GASTOU O ORÇAMENTO\n{'='*92}", flush=True)
    print(quadro.groupby(["estratégia", "segmentador"]).size().unstack(fill_value=0).to_string())
    print(f"\nCSV com os {len(quadro)} trials: {destino}", flush=True)


if __name__ == "__main__":
    main()
