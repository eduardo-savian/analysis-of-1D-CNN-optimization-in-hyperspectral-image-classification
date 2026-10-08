"""Quanto do ganho da fusão sobrevive quando o vazamento espacial é removido.

A PERGUNTA. No split aleatório por pixel — o protocolo padrão da literatura de HSI —
**100% das amostras de teste compartilham superpixel com o treino**. A rede pode
aprender "região R → classe C" e aplicar isso a pixels de teste da mesma região, o
que não é generalização. Se for esse o mecanismo, o ganho medido para a fusão está
inflado.

O DESENHO. Cada condição é medida nos **dois protocolos**, com tudo o mais idêntico:

- `pixel`   — split aleatório estratificado (vazamento de 100%), repetido com k sementes
- `region`  — validação cruzada por região, k folds (vazamento de 0%)

Assim a diferença entre as duas colunas é atribuível ao vazamento, e não a mudança
de modelo, de dados ou de orçamento.

CRITÉRIOS que sustentam a comparação:

- **Mesmo número de repetições** nos dois protocolos (k), para que os desvios sejam
  comparáveis.
- **A condição "sem fusão" entra nos dois.** Sem esse piso, não há como separar
  "a fusão perdeu efeito" de "o protocolo ficou mais difícil para todo mundo".
- **Hiperparâmetros fixos e iguais entre protocolos.** Aqui isso é proposital e é a
  exceção à regra da entrada 9 do diário: não queremos o melhor resultado de cada
  condição, queremos a *diferença* entre protocolos para a mesma receita.
- **100% do treino disponível.** O protocolo por região já reduz o treino ao separar
  regiões inteiras; cortar mais confundiria as duas causas.

O QUE ESPERAR NA LEITURA. Sob split por região, algumas classes de Indian Pines ficam
fora do treino — 6 das 16 vivem em uma ou duas regiões apenas. A AA cai por
construção nesses folds. Por isso o script reporta, por fold, quantas classes
ficaram ausentes: a AA precisa ser lida junto desse número.

Uso:  python 09_validacao_cruzada_regiao.py [CENA ...]   (padrão: IP)
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
ns["configure_determinism"](True, verbose=False)

FOLDS = 5
SEMENTES_PIXEL = (11, 29, 42, 73, 101)     # mesmo número de repetições dos folds
EPOCAS = 200
CONDICOES = ("sem fusão", "seeds", "slic", "lsc", "watershed")
CENAS = tuple(a for a in sys.argv[1:] if not a.startswith("-")) or ("IP",)


def base_config(cena: str, condicao: str) -> dict:
    com_fusao = condicao != "sem fusão"
    return dict(
        dataset=cena, model_type="cnn1d",
        two_input=com_fusao,
        segmenter=("seeds" if not com_fusao else condicao),
        fusion="scalar", train_fraction=1.0,
        epochs=EPOCAS, patience=10, early_stopping_target=0.95,
    )


def uma_execucao(cena: str, condicao: str, protocolo: str, repeticao: int) -> dict:
    """Um treino; devolve as métricas e o diagnóstico do split."""
    campos = base_config(cena, condicao)
    if protocolo == "region":
        # `split_mode="region"` exige segmentação; a condição sem fusão também a
        # calcula, para que os dois braços usem exatamente a mesma partição.
        campos.update(split_mode="region", split_folds=FOLDS, split_fold=repeticao, seed=42)
    else:
        campos.update(split_mode="pixel", seed=SEMENTES_PIXEL[repeticao])
    config = EC(**campos)
    resultado = ns["run_experiment"](config, backend="thread", verbose=False)
    dados = resultado.data
    ausentes = sorted(set(np.unique(dados.labels).tolist())
                      - set(np.unique(dados.labels[dados.splits.train]).tolist()))
    vazamento = ns["segment_leakage"](dados)
    linha = {
        "cena": cena, "condição": condicao, "protocolo": protocolo,
        "repetição": repeticao,
        "OA": resultado.metrics.overall_accuracy,
        "AA": resultado.metrics.average_accuracy,
        "Kappa": resultado.metrics.kappa,
        "treino_px": int(len(dados.splits.train)),
        "teste_px": int(len(dados.splits.test)),
        "classes_fora_do_treino": len(ausentes),
        "vazamento": (vazamento["test_samples_sharing_region_with_train"]
                      if vazamento["applicable"] else float("nan")),
        "épocas": len(resultado.training.history),
    }
    ns["release_result"](resultado)
    gc.collect()
    return linha


def main() -> None:
    print("VALIDAÇÃO CRUZADA POR REGIÃO — quanto do ganho sobrevive sem vazamento")
    print(f"cenas: {', '.join(CENAS)} | {FOLDS} folds · {len(SEMENTES_PIXEL)} sementes | "
          f"teto de {EPOCAS} épocas | 100% do treino", flush=True)

    destino = caminho_resultado("09_validacao_regiao.csv")
    linhas, feitas = carregar_feito(destino, ("cena", "condição", "protocolo", "repetição"))
    if feitas:
        print(f"retomando: {len(feitas)} execuções já medidas", flush=True)

    for cena in CENAS:
        print(f"\n{'='*100}\n### {cena} — {ns['DATASETS'][cena]}\n{'='*100}", flush=True)
        print(f"{'condição':<12}{'protocolo':<10}{'OA':>18}{'AA':>18}{'Kappa':>18}"
              f"{'vazam.':>9}{'cls fora':>10}", flush=True)
        for condicao in CONDICOES:
            for protocolo in ("pixel", "region"):
                repeticoes = range(FOLDS if protocolo == "region" else len(SEMENTES_PIXEL))
                novas = [r for r in repeticoes
                         if (cena, condicao, protocolo, r) not in feitas]
                for repeticao in novas:
                    inicio = time.perf_counter()
                    linhas.append(uma_execucao(cena, condicao, protocolo, repeticao))
                    salvar(linhas, destino)
                    print(f"    · {condicao} {protocolo} rep {repeticao} "
                          f"({time.perf_counter()-inicio:.0f}s)", flush=True)
                grupo = [l for l in linhas
                         if l["cena"] == cena and l["condição"] == condicao
                         and l["protocolo"] == protocolo]
                valores = np.array([[g["OA"], g["AA"], g["Kappa"]] for g in grupo])
                print(f"{condicao:<12}{protocolo:<10}"
                      f"{valores[:,0].mean():>11.4f}±{valores[:,0].std():.4f}"
                      f"{valores[:,1].mean():>11.4f}±{valores[:,1].std():.4f}"
                      f"{valores[:,2].mean():>11.4f}±{valores[:,2].std():.4f}"
                      f"{np.nanmean([g['vazamento'] for g in grupo]):>8.0%}"
                      f"{np.mean([g['classes_fora_do_treino'] for g in grupo]):>10.1f}",
                      flush=True)

    quadro = pd.DataFrame(linhas)
    print(f"\n{'='*100}\nO GANHO DA FUSÃO EM CADA PROTOCOLO\n{'='*100}", flush=True)
    for cena in CENAS:
        sub = quadro[quadro["cena"] == cena]
        print(f"\n{cena}:")
        print(f"  {'condição':<12}{'ganho pixel':>14}{'ganho região':>15}{'quanto sobra':>15}")
        pisos = {p: sub[(sub["condição"] == "sem fusão") & (sub["protocolo"] == p)]["OA"].mean()
                 for p in ("pixel", "region")}
        for condicao in CONDICOES[1:]:
            ganhos = {}
            for protocolo in ("pixel", "region"):
                media = sub[(sub["condição"] == condicao)
                            & (sub["protocolo"] == protocolo)]["OA"].mean()
                ganhos[protocolo] = 100 * (media - pisos[protocolo])
            sobra = (ganhos["region"] / ganhos["pixel"] * 100
                     if abs(ganhos["pixel"]) > 1e-9 else float("nan"))
            print(f"  {condicao:<12}{ganhos['pixel']:>+13.2f}pp{ganhos['region']:>+14.2f}pp"
                  f"{sobra:>14.0f}%")
        print(f"\n  piso (sem fusão): pixel {pisos['pixel']:.4f} → região {pisos['region']:.4f} "
              f"({100*(pisos['region']-pisos['pixel']):+.2f} pp)")
    print(f"\nCSV: {destino}", flush=True)


if __name__ == "__main__":
    main()
