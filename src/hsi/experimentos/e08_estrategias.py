"""Vale um hipertunador, ou random search basta?

Responde três perguntas:

1. **Vale um hipertunador?** Random search, TPE (bayesiano) e algoritmo genético
   percorrem o mesmo espaço, com o mesmo orçamento de trials e a mesma cobertura
   inicial dos quatro segmentadores. Só muda como o candidato é escolhido.
2. **O orçamento por trial muda o veredito?** A busca roda sob três tetos de época
   (10, 40, 200). Se o ranking mudar entre eles, a triagem barata não é um proxy
   confiável — é o *low-fidelity bias*, e é um resultado metodológico por si.
3. **Quais eixos do espaço importam?** Com todos os trials salvos, dá para medir a
   validação média por valor de cada eixo. Um eixo plano é candidato a sair do
   espaço na campanha seguinte, liberando orçamento para os que importam.

CRITÉRIO. O que torna a comparação honesta é a simetria: mesmo espaço, mesmo
orçamento, mesma cobertura inicial, mesma semente. A única diferença é a decisão de
qual candidato treinar em seguida.

**Todos os trials são salvos**, não só os vencedores — sem eles não há como medir o
peso dos eixos, e foi exatamente esse dado que se perdeu antes.

Cada busca é repetida com várias sementes. Sem isso, "o TPE ganhou" é uma afirmação
sobre uma execução, não sobre o método — foi o erro das medições anteriores,
agravado pelo LSC não-determinístico (corrigido em 24/09).
"""

import gc
import time

import pandas as pd

from hsi.config import ExperimentConfig
from hsi.experimentos.infra import caminho_resultado, carregar_feito, salvar
from hsi.hpo.busca import automated_hpo
from hsi.hpo.estrategias import HPO_STRATEGIES

E08 = dict(trials=20, orcamentos=(10, 40, 200), sementes=(11, 29, 42), fracao=0.10,
           eixos=("learning rate", "dropout", "weight decay", "batch", "PCA"))


def _e08_rodar(cena: str, orcamento: int, estrategia: str, semente: int):
    """Uma busca completa; devolve a tabela com TODOS os trials.

    A semente governa a busca *e* o treino: é uma réplica completa do experimento,
    não só um reembaralhamento dos pesos iniciais.
    """
    base = ExperimentConfig(
        dataset=cena, model_type="cnn1d", two_input=True, segmenter="seeds",
        fusion="scalar", train_fraction=E08["fracao"],
        epochs=orcamento, patience=10, early_stopping_target=0.95, seed=semente)
    # `population` só vale para o genético: com 6, como em `compare_hpo_strategies`,
    # os 20 trials rendem 4 gerações completas (com elite 2) em vez das 3 do padrão 8.
    tabela, _ = automated_hpo(
        base, trials_per_cnn=E08["trials"], search_epochs=orcamento, backend="thread",
        strategy=estrategia, population=6, retain="winner",
    )
    tabela = tabela.drop(columns=["resultado", "config"], errors="ignore")
    tabela["cena"] = cena
    tabela["orçamento"] = orcamento
    tabela["semente"] = semente
    return tabela


def experimento_08(*cenas: str) -> "pd.DataFrame":
    """Estratégias de busca, orçamento de épocas e peso de cada eixo."""
    cenas = cenas or ("IP",)
    print("AUTOMAÇÃO DE HIPERPARÂMETROS — estratégias de busca e peso dos eixos")
    print(f"cenas: {', '.join(cenas)} | {E08['trials']} trials por busca | "
          f"orçamentos: {E08['orcamentos']} | {int(E08['fracao']*100)}% de treino",
          flush=True)

    destino = caminho_resultado("08_trials.csv")
    feitos, chaves = carregar_feito(destino, ("cena", "orçamento", "estratégia", "semente"))
    if chaves:
        print(f"retomando: {len(chaves)} buscas já feitas", flush=True)
    todos = list(feitos)

    for cena in cenas:
        for orcamento in E08["orcamentos"]:
            print(f"\n--- {cena} · teto de {orcamento} épocas ---", flush=True)
            for estrategia in HPO_STRATEGIES:
                for semente in E08["sementes"]:
                    if (cena, orcamento, estrategia, semente) in chaves:
                        print(f"  {estrategia:<8} s={semente:<4} (já medido)", flush=True)
                        continue
                    inicio = time.perf_counter()
                    tabela = _e08_rodar(cena, orcamento, estrategia, semente)
                    melhor = tabela.loc[tabela["val_accuracy (objetivo)"].idxmax()]
                    print(f"  {estrategia:<8} s={semente:<4} "
                          f"val={melhor['val_accuracy (objetivo)']:.4f} "
                          f"(trial {int(melhor['trial']):>2})  "
                          f"OA_teste={melhor['OA teste (relato)']:.4f}  "
                          f"seg={melhor['segmentador']:<10} "
                          f"{time.perf_counter()-inicio:5.1f}s", flush=True)
                    todos.extend(tabela.to_dict(orient="records"))
                    salvar(todos, destino)
                    gc.collect()

    quadro = pd.DataFrame(todos)

    print(f"\n{'='*92}\nCONVERGÊNCIA — melhor validação até o trial n\n{'='*92}", flush=True)
    for cena in cenas:
        sub = quadro[quadro["cena"] == cena]
        pivo = sub.pivot_table(index="trial", columns=["orçamento", "estratégia"],
                               values="melhor até agora")
        print(f"\n{cena}:")
        print(pivo.round(4).to_string(), flush=True)

    print(f"\n{'='*92}\nESTRATÉGIAS: MÉDIA ± DESVIO ENTRE SEMENTES\n{'='*92}", flush=True)
    por_busca = (quadro.groupby(["cena", "orçamento", "estratégia", "semente"])
                 ["val_accuracy (objetivo)"].max().reset_index())
    resumo = por_busca.groupby(["cena", "orçamento", "estratégia"])["val_accuracy (objetivo)"]
    print(resumo.agg(["mean", "std", "min", "max"]).round(4).to_string(), flush=True)
    print("\nA diferença entre estratégias só é afirmável se exceder o desvio entre sementes.")
    for (cena, orcamento), grupo in por_busca.groupby(["cena", "orçamento"]):
        medias = grupo.groupby("estratégia")["val_accuracy (objetivo)"].mean()
        desvios = grupo.groupby("estratégia")["val_accuracy (objetivo)"].std()
        amplitude = (medias.max() - medias.min()) * 100
        desvio_tipico = desvios.mean() * 100
        veredito = "AFIRMÁVEL" if amplitude > 2 * desvio_tipico else "dentro do ruído"
        print(f"  {cena} · {orcamento:>3} épocas: amplitude {amplitude:5.2f} pp | "
              f"desvio típico {desvio_tipico:5.2f} pp → {veredito} "
              f"(melhor: {medias.idxmax()})")

    print(f"\n{'='*92}\nPESO DE CADA EIXO — validação média por valor\n{'='*92}", flush=True)
    for cena in cenas:
        sub = quadro[quadro["cena"] == cena]
        print(f"\n{cena}:")
        amplitudes = {}
        for eixo in E08["eixos"]:
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
    return quadro
