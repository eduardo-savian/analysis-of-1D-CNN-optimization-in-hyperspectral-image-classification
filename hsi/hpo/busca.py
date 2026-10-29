"""Busca de hiperparâmetros reprodutível, sem vazamento do conjunto de teste.

Esta é a **única** definição de `hpo_candidates` e `automated_hpo` do projeto:
o espaço de busca, a geração de candidatos e a avaliação ficam juntos para que
não exista uma segunda versão sombreando esta em função da ordem de execução.

Três estratégias percorrem o mesmo espaço: *random search* (a linha de base), TPE
via Optuna e um algoritmo genético. A troca é o argumento `strategy` de
`automated_hpo`, resolvido em `make_hpo_searcher`; espaço, orçamento, avaliação e
auditoria são os mesmos para as três. O produto cartesiano completo tem milhares de
combinações (2.916 em Indian Pines), por isso a busca é amostrada. Em todas as
estratégias, os primeiros `len(segmenters)` trials de cada CNN são forçados a
segmentadores distintos, de modo que nenhum segmentador fique de fora por azar.

Medido no experimento 08 (Indian Pines, três sementes, tetos de 10, 40 e 200
épocas): nenhuma estratégia se separa do ruído entre sementes. Uma vantagem vista
numa execução só não basta para preferir uma delas.

A seleção é feita por melhor acurácia de validação; OA/AA/Kappa de teste aparecem
nos relatórios, mas não entram no objetivo de escolha. Para uma publicação final,
reserve o teste para a configuração vencedora e repita-a com várias seeds — é o que
`run_best_hpo_campaign` faz.
"""

import json
from dataclasses import replace

import pandas as pd

from hsi.config import MODEL_TYPES, SEGMENTERS
from hsi.hpo.espaco import hpo_decode, hpo_genes
from hsi.hpo.estrategias import HPO_STRATEGIES, make_hpo_searcher
from hsi.pipeline import release_result, run_experiment


def hpo_candidates(base_config, model_type, segmenters, trials, seed=42, space=None):
    """Pré-gera os candidatos do random search, sem treinar nenhum.

    Com a mesma semente, são os mesmos candidatos, na mesma ordem, que
    `automated_hpo(strategy="random")` treinaria (antes de reduzir épocas e
    paciência). É a fonte da matriz de `build_campaign_plan`, que precisa do plano
    inteiro — e do hash dele — antes do primeiro treino. Só existe para a estratégia
    aleatória: TPE e algoritmo genético são adaptativos, e o candidato `n+1` depende
    do resultado do candidato `n`.
    """
    genes = hpo_genes(base_config, model_type, space, segmenters)
    searcher = make_hpo_searcher("random", genes, segmenters, seed, model_type)
    return [
        replace(base_config, model_type=model_type,
                **hpo_decode(genes, searcher.ask(), base_config.dataset))
        for _ in range(max(trials, len(tuple(segmenters))))
    ]


def automated_hpo(base_config, trials_per_cnn=8, search_epochs=10, backend="thread",
                  model_types=MODEL_TYPES, segmenters=None, space=None,
                  strategy="random", population=8, retain="winner"):
    """Executa o HPO de cada arquitetura e devolve `(tabela, vencedores)`.

    A tabela tem uma linha por trial; `vencedores` mapeia cada CNN ao
    `ExperimentResult` do seu melhor trial. Os trials não abrem runs no tracker: o
    registro deles é esta tabela, que `run_best_hpo_campaign` grava em
    `hpo_screening.json`.

    `strategy` escolhe entre as três buscas — todas percorrem o mesmo espaço, com o
    mesmo orçamento de trials e a mesma cobertura inicial de segmentadores, o que
    torna a comparação entre elas honesta. `population` só vale para o genético: com
    `trials_per_cnn <= population` ele nunca cruza e repete o random search.

    `search_epochs` limita o custo da busca. Os vencedores podem depois ser
    retreinados pelo orçamento completo de `base_config.epochs` antes da comparação
    final. O score de seleção é o maior `val_accuracy` observado no histórico; o
    teste nunca participa da escolha.

    `retain` controla a memória, e não é um detalhe: cada `ExperimentResult` segura
    sua própria cópia dos dados preparados.

    - ``"winner"`` (padrão): mantém só o melhor de cada arquitetura e libera os
      demais assim que a métrica é extraída. É o que torna a campanha viável em
      cenas maiores.
    - ``"all"``: mantém tudo, na coluna ``resultado``. Só use em cenas pequenas, ou
      para inspeção.

    A tabela devolvida carrega sempre a `config` de cada trial — que é barata e
    reconstrói qualquer execução —, então liberar o objeto não perde rastreabilidade.
    """
    if retain not in {"winner", "all"}:
        raise ValueError("retain deve ser 'winner' ou 'all'.")
    segmenters = tuple(segmenters or SEGMENTERS)
    trials_per_cnn = max(trials_per_cnn, len(segmenters))
    rows, winners, winner_trials = [], {}, {}
    for model_type in model_types:
        genes = hpo_genes(base_config, model_type, space, segmenters)
        searcher = make_hpo_searcher(strategy, genes, segmenters, base_config.seed, model_type, population)
        best = None
        for trial in range(1, trials_per_cnn + 1):
            individual = searcher.ask()
            candidate = replace(base_config, model_type=model_type,
                                **hpo_decode(genes, individual, base_config.dataset))
            # A paciência encolhe junto com o orçamento: a da receita (10) nunca
            # dispararia dentro de 10 épocas. Um terço de `search_epochs`, mínimo 2.
            trial_config = replace(
                candidate,
                epochs=search_epochs,
                patience=min(candidate.patience, max(2, search_epochs // 3)),
            )
            result = run_experiment(trial_config, backend=backend, verbose=False)
            validation_score = float(max(result.training.history.val_accuracy))
            searcher.tell(individual, validation_score)
            # As métricas são extraídas ANTES de qualquer liberação: são escalares
            # (mais a config), e é tudo o que os relatórios consomem.
            rows.append({
                "CNN": model_type.upper(), "estratégia": strategy, "trial": trial,
                "segmentador": trial_config.segmenter,
                "parâmetros segmentador": json.dumps(trial_config.segmenter_overrides, sort_keys=True),
                "learning rate": trial_config.learning_rate, "dropout": trial_config.dropout,
                "weight decay": trial_config.weight_decay, "batch": trial_config.batch_size,
                "PCA": trial_config.pca_components,
                "val_accuracy (objetivo)": validation_score,
                "OA teste (relato)": result.metrics.overall_accuracy,
                "AA teste (relato)": result.metrics.average_accuracy,
                "Kappa teste (relato)": result.metrics.kappa,
                "tempo treino (s)": result.training.total_time,
                "épocas executadas": len(result.training.history),
                "parâmetros do modelo": int(result.model.n_parameters),
                "config": trial_config,
            })
            # `>` estrito: no empate fica o trial mais antigo. `select_hpo_winners`
            # desempata por tempo de treino, então os dois podem divergir.
            if best is None or validation_score > best[0]:
                if best is not None and retain != "all":
                    release_result(best[1])          # o campeão anterior perdeu o posto
                best = (validation_score, result, trial)
            elif retain != "all":
                release_result(result)               # não é o melhor: libera agora
            rows[-1]["melhor até agora"] = best[0]
            if retain == "all":
                rows[-1]["resultado"] = result
        winners[model_type] = best[1]
        winner_trials[model_type.upper()] = best[2]
    table = pd.DataFrame(rows)
    # O vencedor é marcado por (CNN, trial), e não pela identidade do objeto: com
    # `retain="winner"` os demais resultados já foram liberados.
    table["vencedor"] = [
        winner_trials.get(cnn) == trial for cnn, trial in zip(table["CNN"], table["trial"])
    ]
    return table, winners


def compare_hpo_strategies(base_config, trials_per_cnn=12, search_epochs=10, backend="thread",
                           model_types=("cnn1d",), segmenters=None, space=None,
                           strategies=HPO_STRATEGIES, population=6):
    """Roda as estratégias pedidas com orçamento idêntico e devolve uma tabela única.

    É o experimento que responde "vale a pena um hipertunador?". Todas partem da
    mesma semente, do mesmo espaço e do mesmo número de treinos; a única diferença
    é como cada trial é escolhido. Compare pela curva de melhor-até-agora, e não só
    pelo valor final: uma busca que chega ao mesmo lugar em metade dos trials é
    melhor mesmo empatando no fim.

    Uma chamada usa uma semente só (`base_config.seed`), então uma diferença vista
    aqui pode ser ruído de semente — repetido com três sementes no experimento 08,
    nenhuma estratégia se separou dele. Com `population=6`, os 6 primeiros trials do
    genético são os mesmos do random search; só os seguintes refletem a evolução.
    """
    frames = []
    for strategy in strategies:
        table, _ = automated_hpo(
            base_config, trials_per_cnn=trials_per_cnn, search_epochs=search_epochs,
            backend=backend, model_types=model_types, segmenters=segmenters,
            space=space, strategy=strategy, population=population,
        )
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def hpo_strategy_summary(comparison):
    """Melhor validação, trial em que ela apareceu e custo total, por estratégia."""
    rows = []
    for (strategy, cnn), group in comparison.groupby(["estratégia", "CNN"]):
        best = group.loc[group["val_accuracy (objetivo)"].idxmax()]
        rows.append({
            "estratégia": strategy, "CNN": cnn,
            "melhor val_accuracy": float(best["val_accuracy (objetivo)"]),
            "encontrado no trial": int(best["trial"]),
            "trials": int(len(group)),
            "tempo total (s)": float(group["tempo treino (s)"].sum()),
            "segmentador vencedor": best["segmentador"],
            "OA teste (relato)": float(best["OA teste (relato)"]),
        })
    return pd.DataFrame(rows).sort_values(["CNN", "melhor val_accuracy"], ascending=[True, False])
