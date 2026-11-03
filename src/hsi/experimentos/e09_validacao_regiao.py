"""Quanto dos resultados sobrevive quando o vazamento espacial é removido.

A PERGUNTA. No split aleatório por pixel — o protocolo padrão da literatura de HSI —
**100% das amostras de teste compartilham superpixel com o treino**. A rede pode
aprender "região R → classe C" e aplicar isso a pixels de teste da mesma região, o
que não é generalização. Se for esse o mecanismo, a acurácia da configuração herdada
está inflada, e a comparação entre segmentadores também.

O DESENHO. Cada segmentador é medido nos **dois protocolos**, com tudo o mais idêntico:

- `pixel`   — split aleatório estratificado (vazamento de 100%), com k sementes
- `region`  — validação cruzada por região, k folds (vazamento de 0%)

Assim a diferença entre as duas colunas é atribuível ao vazamento, e não a mudança
de modelo, de dados ou de orçamento.

CRITÉRIOS que sustentam a comparação:

- **Mesmo número de repetições** nos dois protocolos, para que os desvios sejam
  comparáveis.
- **O SEEDS é a referência nos dois protocolos.** É o segmentador do trabalho
  herdado: a queda dele de um protocolo para o outro é o quanto o resultado herdado
  dependia do vazamento, e a diferença de cada segmentador para ele, em cada
  protocolo, é o ganho de trocá-lo.
- **Hiperparâmetros fixos e iguais entre protocolos.** Aqui isso é proposital e é a
  exceção à regra do experimento 06: não queremos o melhor resultado de cada
  condição, queremos a *diferença* entre protocolos para a mesma receita.
- **100% do treino disponível.** O protocolo por região já reduz o treino ao separar
  regiões inteiras; cortar mais confundiria as duas causas.

O QUE ESPERAR NA LEITURA. Sob split por região, algumas classes de Indian Pines
ficam fora do treino — 6 das 16 vivem em uma ou duas regiões apenas. A AA cai por
construção nesses folds. Por isso o experimento reporta, por fold, quantas classes
ficaram ausentes: a AA precisa ser lida junto desse número.

E sob `region` cada segmentador parte a cena em grupos próprios, então a dificuldade
do split muda com o segmentador (`DISTANCIA_TESTE`: o teste do SEEDS fica mais longe
do treino que o dos outros). A diferença para o SEEDS nesse protocolo mistura o
efeito do segmentador com essa dificuldade.
"""

import gc
import time

import numpy as np
import pandas as pd

from hsi.ambiente import configure_determinism
from hsi.config import DATASETS, ExperimentConfig
from hsi.dados.particoes import segment_leakage
from hsi.experimentos.infra import caminho_resultado, carregar_feito, salvar
from hsi.pipeline import release_result, run_experiment

E09 = dict(folds=5, sementes_pixel=(11, 29, 42, 73, 101), epocas=200,
           condicoes=("seeds", "slic", "lsc", "watershed"))


def _e09_uma_execucao(cena: str, condicao: str, protocolo: str, repeticao: int) -> dict:
    """Um treino; devolve as métricas e o diagnóstico do split."""
    campos = dict(
        dataset=cena, model_type="cnn1d", two_input=True, segmenter=condicao,
        fusion="scalar", train_fraction=1.0,
        epochs=E09["epocas"], patience=10, early_stopping_target=0.95,
    )
    if protocolo == "region":
        # Os grupos do split são as regiões do próprio segmentador da condição.
        campos.update(split_mode="region", split_folds=E09["folds"],
                      split_fold=repeticao, seed=42)
    else:
        campos.update(split_mode="pixel", seed=E09["sementes_pixel"][repeticao])
    resultado = run_experiment(ExperimentConfig(**campos), backend="thread", verbose=False)
    dados = resultado.data
    ausentes = sorted(set(np.unique(dados.labels).tolist())
                      - set(np.unique(dados.labels[dados.splits.train]).tolist()))
    vazamento = segment_leakage(dados)
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
    release_result(resultado)
    gc.collect()
    return linha


def experimento_09(*cenas: str) -> "pd.DataFrame":
    """Cada segmentador sob split por pixel contra split por região."""
    cenas = cenas or ("IP",)
    configure_determinism(True, verbose=False)
    print("VALIDAÇÃO CRUZADA POR REGIÃO — quanto do resultado sobrevive sem vazamento")
    print(f"cenas: {', '.join(cenas)} | {E09['folds']} folds · "
          f"{len(E09['sementes_pixel'])} sementes | teto de {E09['epocas']} épocas | "
          f"100% do treino", flush=True)

    destino = caminho_resultado("09_validacao_regiao.csv")
    linhas, feitas = carregar_feito(destino, ("cena", "condição", "protocolo", "repetição"))
    if feitas:
        print(f"retomando: {len(feitas)} execuções já medidas", flush=True)

    for cena in cenas:
        print(f"\n{'='*100}\n### {cena} — {DATASETS[cena]}\n{'='*100}", flush=True)
        print(f"{'condição':<12}{'protocolo':<10}{'OA':>18}{'AA':>18}{'Kappa':>18}"
              f"{'vazam.':>9}{'cls fora':>10}", flush=True)
        for condicao in E09["condicoes"]:
            for protocolo in ("pixel", "region"):
                total = E09["folds"] if protocolo == "region" else len(E09["sementes_pixel"])
                novas = [r for r in range(total)
                         if (cena, condicao, protocolo, r) not in feitas]
                for repeticao in novas:
                    inicio = time.perf_counter()
                    linhas.append(_e09_uma_execucao(cena, condicao, protocolo, repeticao))
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
    print(f"\n{'='*100}\nDIFERENÇA PARA O SEEDS EM CADA PROTOCOLO\n{'='*100}", flush=True)
    for cena in cenas:
        sub = quadro[quadro["cena"] == cena]
        print(f"\n{cena}:")
        print(f"  {'condição':<12}{'Δ pixel':>14}{'Δ região':>15}")
        # O SEEDS é a primeira condição de `E09`: a do trabalho herdado.
        referencia = {p: sub[(sub["condição"] == "seeds") & (sub["protocolo"] == p)]["OA"].mean()
                      for p in ("pixel", "region")}
        for condicao in E09["condicoes"][1:]:
            deltas = {}
            for protocolo in ("pixel", "region"):
                media = sub[(sub["condição"] == condicao)
                            & (sub["protocolo"] == protocolo)]["OA"].mean()
                deltas[protocolo] = 100 * (media - referencia[protocolo])
            print(f"  {condicao:<12}{deltas['pixel']:>+13.2f}pp{deltas['region']:>+14.2f}pp")
        print(f"\n  SEEDS (referência): pixel {referencia['pixel']:.4f} → região "
              f"{referencia['region']:.4f} "
              f"({100*(referencia['region']-referencia['pixel']):+.2f} pp)")
    print(f"\nCSV: {destino}", flush=True)
    return quadro
