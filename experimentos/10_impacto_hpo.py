"""Quanto a otimização de hiperparâmetros acrescenta, em relação à receita herdada.

A PERGUNTA. Os experimentos anteriores mostraram *qual* configuração o HPO encontra
e *quanto cada eixo pesa*. Falta o número que justifica o item inteiro da lista de
atividades: **o que se ganha por rodar a busca, em vez de usar a receita documentada
do trabalho de referência?**

O DESENHO. Duas configurações, tudo o mais igual, 5 sementes cada:

- **padrão** — a receita herdada: learning rate 1e-3, dropout 0,5, weight decay 0,
  batch 128 e o número de componentes de PCA documentado por cena.
- **otimizada** — a vencedora da busca do experimento 06, para aquela mesma cena e
  aquele mesmo segmentador.

A comparação é por condição (sem fusão e os quatro segmentadores), porque o ganho do
HPO não precisa ser igual para todas — um segmentador que produz muitas regiões exige
uma receita mais distante do padrão do que um que produz poucas.

POR QUE ISSO IMPORTA ALÉM DO NÚMERO. A análise dos 540 trials mostrou que a receita
padrão cai no **pior** valor dos dois eixos mais influentes: dropout 0,5 (contra 0,20,
que rende 18,10 pp a mais em média) e batch 32/128 (contra 64). Se a diferença medida
aqui for grande, o HPO não é refinamento — é correção de uma escolha ruim herdada.
"""
from __future__ import annotations

import ast
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

SEMENTES = (11, 29, 42, 73, 101)
EPOCAS = 200
FRACAO = 0.10
CONDICOES = ("sem fusão", "seeds", "slic", "lsc", "watershed")
CENAS = tuple(a for a in sys.argv[1:] if not a.startswith("-")) or ("IP",)

#: A receita herdada do trabalho de referência, como está na seção 2 do notebook.
PADRAO = dict(learning_rate=1e-3, dropout=0.5, weight_decay=0.0, batch_size=128)

VENCEDORES = pd.read_csv(caminho_resultado("06_resumo.csv"))


def config_da(cena: str, condicao: str, receita: str, semente: int):
    com_fusao = condicao != "sem fusão"
    campos = dict(
        dataset=cena, model_type="cnn1d", two_input=com_fusao,
        segmenter=("seeds" if not com_fusao else condicao),
        fusion="scalar", train_fraction=FRACAO,
        epochs=EPOCAS, patience=10, early_stopping_target=0.95, seed=semente,
    )
    if receita == "padrão":
        campos.update(PADRAO, pca_components=ns["PCA_COMPONENTS"][cena])
        return EC(**campos)
    linha = VENCEDORES[(VENCEDORES["cena"] == cena)
                       & (VENCEDORES["condição"] == condicao)]
    if linha.empty:
        raise SystemExit(f"sem vencedor de HPO para {cena}/{condicao} em 06_resumo.csv")
    linha = linha.iloc[0]
    campos.update(
        learning_rate=float(linha["learning_rate"]), dropout=float(linha["dropout"]),
        weight_decay=float(linha["weight_decay"]), batch_size=int(linha["batch_size"]),
        pca_components=int(linha["pca_components"]),
        segmenter_overrides=ast.literal_eval(linha["overrides"]) if com_fusao else {},
    )
    return EC(**campos)


def main() -> None:
    print("IMPACTO DO HPO — receita herdada contra receita otimizada")
    print(f"cenas: {', '.join(CENAS)} | {len(SEMENTES)} sementes | "
          f"{int(FRACAO*100)}% de treino | teto de {EPOCAS} épocas", flush=True)
    print(f"padrão: lr={PADRAO['learning_rate']}, dropout={PADRAO['dropout']}, "
          f"wd={PADRAO['weight_decay']}, batch={PADRAO['batch_size']}", flush=True)

    destino = caminho_resultado("10_impacto_hpo.csv")
    linhas, feitas = carregar_feito(destino, ("cena", "condição", "receita", "semente"))
    if feitas:
        print(f"retomando: {len(feitas)} execuções já medidas", flush=True)

    for cena in CENAS:
        print(f"\n{'='*94}\n### {cena} — {ns['DATASETS'][cena]}\n{'='*94}", flush=True)
        print(f"{'condição':<12}{'receita':<11}{'OA':>18}{'AA':>18}{'Kappa':>18}", flush=True)
        for condicao in CONDICOES:
            medias = {}
            for receita in ("padrão", "otimizada"):
                for semente in SEMENTES:
                    if (cena, condicao, receita, semente) in feitas:
                        continue
                    config = config_da(cena, condicao, receita, semente)
                    resultado = ns["run_experiment"](config, backend="thread", verbose=False)
                    linhas.append({
                        "cena": cena, "condição": condicao, "receita": receita,
                        "semente": semente,
                        "OA": resultado.metrics.overall_accuracy,
                        "AA": resultado.metrics.average_accuracy,
                        "Kappa": resultado.metrics.kappa,
                        "épocas": len(resultado.training.history),
                    })
                    ns["release_result"](resultado)
                    gc.collect()
                    salvar(linhas, destino)
                grupo = [l for l in linhas if l["cena"] == cena
                         and l["condição"] == condicao and l["receita"] == receita]
                valores = np.array([[g["OA"], g["AA"], g["Kappa"]] for g in grupo])
                medias[receita] = valores
                print(f"{condicao:<12}{receita:<11}"
                      f"{valores[:,0].mean():>11.4f}±{valores[:,0].std():.4f}"
                      f"{valores[:,1].mean():>11.4f}±{valores[:,1].std():.4f}"
                      f"{valores[:,2].mean():>11.4f}±{valores[:,2].std():.4f}", flush=True)
            ganho = 100 * (medias["otimizada"][:, 0].mean() - medias["padrão"][:, 0].mean())
            desvio = 100 * np.hypot(medias["otimizada"][:, 0].std(), medias["padrão"][:, 0].std())
            veredito = "AFIRMÁVEL" if abs(ganho) > 2 * desvio else "dentro do ruído"
            print(f"{'':12}{'→ ganho do HPO':<11}{ganho:>+10.2f} pp   "
                  f"(desvio combinado {desvio:.2f} pp) → {veredito}\n", flush=True)

    quadro = pd.DataFrame(linhas)
    print(f"{'='*94}\nRESUMO — ganho do HPO por cena\n{'='*94}", flush=True)
    for cena in CENAS:
        sub = quadro[quadro["cena"] == cena]
        pivo = sub.pivot_table(index="condição", columns="receita", values="OA")
        pivo["ganho (pp)"] = 100 * (pivo["otimizada"] - pivo["padrão"])
        print(f"\n{cena}:"); print(pivo.round(4).to_string(), flush=True)
    print(f"\nCSV: {destino}", flush=True)


if __name__ == "__main__":
    main()
