"""Quanto a otimização de hiperparâmetros acrescenta sobre a receita herdada.

A PERGUNTA. Os experimentos anteriores mostraram *qual* configuração o HPO encontra
e *quanto cada eixo pesa*. Falta o número que justifica o pilar inteiro: **o que se
ganha por rodar a busca, em vez de usar a receita documentada do trabalho de
referência?**

O DESENHO. Duas configurações, tudo o mais igual, 5 sementes cada:

- **padrão** — a receita herdada: learning rate 1e-3, dropout 0,5, weight decay 0,
  batch 128 e o número de componentes de PCA documentado por cena.
- **otimizada** — a vencedora da busca do experimento 06, para aquela mesma cena e
  aquele mesmo segmentador.

A comparação é por segmentador, porque o ganho do HPO não precisa ser igual para
todos — um segmentador que produz muitas regiões exige uma receita mais distante do
padrão do que um que produz poucas. A linha `seeds` é a configuração herdada
inteira: o ganho dela é o ganho efetivo do pilar sobre o trabalho de referência.

POR QUE IMPORTA ALÉM DO NÚMERO. A análise dos 540 trials do experimento 08 mostrou
que a receita padrão cai no **pior** valor do eixo mais influente: dropout 0,5,
contra 0,20, que rende 18,10 pp a mais de validação em média. No segundo eixo, o
learning rate, 1e-3 fica em segundo de três valores; batch e weight decay quase não
pesam. Se a diferença medida aqui for grande, o HPO não é refinamento — é correção
de uma escolha ruim herdada.

DEPENDE DO EXPERIMENTO 06: lê os vencedores de `06_resumo.csv`.
"""

import gc

import numpy as np
import pandas as pd

from hsi.ambiente import configure_determinism
from hsi.config import DATASETS, PCA_COMPONENTS, ExperimentConfig
from hsi.experimentos.infra import caminho_resultado, carregar_feito, salvar
from hsi.pipeline import release_result, run_experiment

E10 = dict(sementes=(11, 29, 42, 73, 101), epocas=200, fracao=0.10,
           condicoes=("seeds", "slic", "lsc", "watershed"),
           #: A receita herdada como o experimento a mediu. Atenção: o batch 128 está
           #: errado — o notebook original de Souza (2023) treina com 32 (Context.md, §1.4).
           padrao=dict(learning_rate=1e-3, dropout=0.5, weight_decay=0.0, batch_size=128))


def _e10_config(vencedores, cena: str, condicao: str, receita: str, semente: int):
    # `ast.literal_eval` lê os overrides do segmentador como estão no CSV, sem
    # `eval`: o arquivo é entrada de dados, não código a executar.
    import ast
    campos = dict(
        dataset=cena, model_type="cnn1d", two_input=True, segmenter=condicao,
        fusion="scalar", train_fraction=E10["fracao"],
        epochs=E10["epocas"], patience=10, early_stopping_target=0.95, seed=semente,
    )
    if receita == "padrão":
        campos.update(E10["padrao"], pca_components=PCA_COMPONENTS[cena])
        return ExperimentConfig(**campos)
    linha = vencedores[(vencedores["cena"] == cena)
                       & (vencedores["condição"] == condicao)]
    if linha.empty:
        raise RuntimeError(f"sem vencedor de HPO para {cena}/{condicao} em 06_resumo.csv")
    linha = linha.iloc[0]
    campos.update(
        learning_rate=float(linha["learning_rate"]), dropout=float(linha["dropout"]),
        weight_decay=float(linha["weight_decay"]), batch_size=int(linha["batch_size"]),
        pca_components=int(linha["pca_components"]),
        segmenter_overrides=ast.literal_eval(linha["overrides"]),
    )
    return ExperimentConfig(**campos)


def experimento_10(*cenas: str) -> "pd.DataFrame":
    """Receita herdada contra receita otimizada, por condição."""
    cenas = cenas or ("IP",)
    configure_determinism(True, verbose=False)
    vencedores = pd.read_csv(caminho_resultado("06_resumo.csv"))

    print("IMPACTO DO HPO — receita herdada contra receita otimizada")
    print(f"cenas: {', '.join(cenas)} | {len(E10['sementes'])} sementes | "
          f"{int(E10['fracao']*100)}% de treino | teto de {E10['epocas']} épocas",
          flush=True)
    print(f"padrão: lr={E10['padrao']['learning_rate']}, "
          f"dropout={E10['padrao']['dropout']}, wd={E10['padrao']['weight_decay']}, "
          f"batch={E10['padrao']['batch_size']}", flush=True)

    destino = caminho_resultado("10_impacto_hpo.csv")
    linhas, feitas = carregar_feito(destino, ("cena", "condição", "receita", "semente"))
    if feitas:
        print(f"retomando: {len(feitas)} execuções já medidas", flush=True)

    for cena in cenas:
        print(f"\n{'='*94}\n### {cena} — {DATASETS[cena]}\n{'='*94}", flush=True)
        print(f"{'condição':<12}{'receita':<11}{'OA':>18}{'AA':>18}{'Kappa':>18}", flush=True)
        for condicao in E10["condicoes"]:
            medias = {}
            for receita in ("padrão", "otimizada"):
                for semente in E10["sementes"]:
                    if (cena, condicao, receita, semente) in feitas:
                        continue
                    config = _e10_config(vencedores, cena, condicao, receita, semente)
                    resultado = run_experiment(config, backend="thread", verbose=False)
                    linhas.append({
                        "cena": cena, "condição": condicao, "receita": receita,
                        "semente": semente,
                        "OA": resultado.metrics.overall_accuracy,
                        "AA": resultado.metrics.average_accuracy,
                        "Kappa": resultado.metrics.kappa,
                        "épocas": len(resultado.training.history),
                    })
                    release_result(resultado)
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
            desvio = 100 * np.hypot(medias["otimizada"][:, 0].std(),
                                    medias["padrão"][:, 0].std())
            veredito = "AFIRMÁVEL" if abs(ganho) > 2 * desvio else "dentro do ruído"
            print(f"{'':12}{'→ ganho do HPO':<11}{ganho:>+10.2f} pp   "
                  f"(desvio combinado {desvio:.2f} pp) → {veredito}\n", flush=True)

    quadro = pd.DataFrame(linhas)
    print(f"{'='*94}\nRESUMO — ganho do HPO por cena\n{'='*94}", flush=True)
    for cena in cenas:
        sub = quadro[quadro["cena"] == cena]
        pivo = sub.pivot_table(index="condição", columns="receita", values="OA")
        pivo["ganho (pp)"] = 100 * (pivo["otimizada"] - pivo["padrão"])
        print(f"\n{cena}:")
        print(pivo.round(4).to_string(), flush=True)
    print(f"\nCSV: {destino}", flush=True)
    return quadro
