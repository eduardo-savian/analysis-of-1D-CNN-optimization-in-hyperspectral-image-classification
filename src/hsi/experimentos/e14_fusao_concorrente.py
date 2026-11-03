"""O desenho do orientador medido: T0 segmenta enquanto T1…TN extraem e fundem.

O PORQUÊ. O capítulo 3 do TCC promete que "uma [thread] executa a segmentação
espacial enquanto a outra executa a CNN 1D", com sincronização só na fusão. O
paralelismo de dois ramos (`prepare_data_parallel`) entregou isso pela metade: ao
lado do SEEDS rodava o PCA espectral, e a CNN só depois. Em 02/11 o orientador
desenhou o esquema completo — T0 com o SEEDS, um pool T1…TN em que cada thread
recebe uma posição do mapa de segmentação, passa o pixel pelas convoluções e
concatena o id do segmento, com mutex nos dois lados. Este experimento implementa
o desenho (`src/hsi/paralelo/fusao_concorrente.py`) e mede o que ele rende.

O QUE SE COMPARA, sempre com o mesmo modelo treinado e sobre todos os pixels
rotulados da cena:

- **lote** — segmenta e depois infere tudo num lote só. É o que o pipeline faz, e
  aparece com 1 thread do PyTorch e com as 16 padrão (o paralelismo *dentro* das
  operações, que o TCC chama de intra-etapas);
- **sequencial** — segmenta e depois percorre as posições em blocos, numa thread.
  Mesma granularidade das tarefas concorrentes: é o denominador justo do speedup
  das threads, porque só a concorrência muda;
- **concorrente** — o desenho, com N threads, três tamanhos de bloco (1 é o "uma
  posição por thread" literal) e as duas extensões da seção crítica (`desenho`,
  como no papel; `minima`, só a leitura do mapa);
- **desacoplado** — o mesmo, com convolução e fusão em tarefas separadas e todas as
  convoluções enfileiradas antes. No desenho literal a thread que termina as
  convoluções fica parada esperando o mapa e segura a vaga no pool; aqui não. É o
  teto do esquema.

QUATRO CUIDADOS:

1. **PyTorch com uma thread intra-operação** nos arranjos de blocos, para que N
   threads de trabalho sejam N núcleos, e não N × 16. O lote aparece com as duas
   contagens porque é o adversário real.
2. **Pool quente.** As threads são criadas e aquecidas (`aquecer_pool`) fora da
   medição — a lição do experimento 13 vale para threads também.
3. **Uma thread do PyTorch também nas threads do pool.** O número de threads do
   OpenMP é estado de cada thread, e `torch.set_num_threads(1)` na principal não
   chega às do pool. A primeira rodada (02/11, log `14_IP_PU_SA_invalidado_omp.log`)
   mediu as threads de trabalho com 32 threads OpenMP cada: a convolução de um bloco
   levava ~100 ms em vez de 0,2 ms. As linhas concorrentes e desacopladas daquela rodada
   foram apagadas do CSV; as de lote e sequencial, medidas na thread principal, ficaram.
4. **Conferência a cada execução.** Os ids lidos do mapa têm de ser os do
   `PreparedData` com que o modelo treinou, e as classes previstas, as do lote. Uma
   divergência invalida a célula em vez de virar número.

COMO LER `espera_total`. É a soma, sobre as tarefas, do tempo entre o fim das
convoluções e a leitura do mapa. No desenho literal isso é espera por T0 mais fila na
trava. Na variante desacoplada inclui também o tempo em que a convolução pronta ficou
na fila esperando a sua tarefa de fusão ser atendida — cresce com o número de blocos e
não mede bloqueio. Para o desacoplado, a leitura certa é `parede` e
`conv_antes_do_mapa`.
"""

import gc
import time

import numpy as np
import pandas as pd
import torch

from hsi.ambiente import configure_determinism
from hsi.config import DATASETS, ExperimentConfig
from hsi.dados.cenas import load_scene, to_pixel_table
from hsi.experimentos.infra import caminho_resultado, carregar_feito, salvar
from hsi.paralelo.fusao_concorrente import (
    EntradaFusao,
    aquecer_pool,
    executar_concorrente,
    executar_sequencial,
)
from hsi.paralelo.ramos import _segmenter_params
from hsi.pipeline import run_experiment

E14 = dict(threads=(1, 2, 4, 8, 16), blocos=(1, 64, 1024), secoes=("desenho", "minima"),
           repeticoes=5,
           #: Com bloco 1 cada execução é uma tarefa por pixel rotulado, e com 16
           #: threads disputando o GIL leva dezenas de segundos. Três repetições
           #: bastam para um desvio desse tamanho; e só em Indian Pines, porque em
           #: Pavia e Salinas seriam 43 e 54 mil tarefas por execução.
           repeticoes_bloco_1=3,
           cenas_bloco_1=("IP",),
           #: Threads do PyTorch no lote: 1 e as 16 padrão da máquina de medição.
           threads_lote=(1, 16),
           #: A execução em que a linha do tempo de cada tarefa é gravada inteira,
           #: uma por arranjo.
           linha_do_tempo=dict(n=4, bloco=64, secao="minima"))


def _e14_config(cena: str) -> ExperimentConfig:
    """A configuração herdada do Sydney, com fusão: o modelo que se quer servir."""
    return ExperimentConfig(dataset=cena, two_input=True, segmenter="seeds", fusion="scalar")


def _treinar(cena: str):
    """Treina o modelo uma vez e monta a entrada da inferência concorrente."""
    config = _e14_config(cena)
    resultado = run_experiment(config, verbose=False)
    modelo = resultado.model.to(torch.device("cpu")).eval()
    data = resultado.data
    pixels, raw_labels = to_pixel_table(load_scene(cena))
    entrada = EntradaFusao(
        pixels=pixels, raw_labels=raw_labels, scene_shape=data.scene_shape,
        params=_segmenter_params(config), seed=config.seed, segmenter=config.segmenter,
        espectros=np.ascontiguousarray(data.spectra, dtype=np.float32), coords=data.coords,
    )
    return modelo, entrada, data, resultado.metrics.overall_accuracy


def _linha(cena, resultado, rep, torch_threads, data, referencia) -> dict:
    """Uma linha do CSV, com a conferência contra os ids do treino e contra o lote."""
    teste = data.splits.test
    return {
        "cena": cena, "arranjo": resultado.arranjo, "n": resultado.n_threads,
        "bloco": resultado.bloco, "secao": resultado.secao,
        "torch_threads": torch_threads, "rep": rep,
        "parede": resultado.parede,
        "segmentacao": resultado.t0_fim - resultado.t0_inicio,
        "t0_fim": resultado.t0_fim,
        "espera_total": resultado.espera_total,
        "espera_maxima": resultado.espera_maxima,
        "conv_antes_do_mapa": resultado.convolucoes_antes_do_mapa,
        "tarefas": len(resultado.tarefas),
        "OA_teste": float(np.mean(resultado.previsto[teste] == data.labels[teste])),
        "ids_iguais_ao_treino": bool(np.array_equal(resultado.ids, data.segments)),
        "classes_iguais_ao_lote": bool(referencia is None
                                       or np.array_equal(resultado.previsto, referencia)),
    }


def _gravar_linha_do_tempo(cena: str, resultado) -> None:
    quadro = pd.DataFrame([vars(t) for t in resultado.tarefas])
    quadro["t0_inicio"], quadro["t0_fim"] = resultado.t0_inicio, resultado.t0_fim
    nome = f"14_linha_do_tempo_{cena}_{resultado.arranjo}.csv"
    quadro.to_csv(caminho_resultado(nome), index=False)


def experimento_14(*cenas: str) -> "pd.DataFrame":
    """Fusão concorrente (desenho do orientador): T0 segmenta, T1…TN extraem e fundem."""
    cenas = cenas or ("IP",)
    configure_determinism(True, verbose=False)
    print("FUSÃO CONCORRENTE — T0 segmenta enquanto T1…TN extraem características e fundem")
    print(f"cenas: {', '.join(cenas)} | threads: {E14['threads']} | blocos: {E14['blocos']} | "
          f"seções: {E14['secoes']} | {E14['repeticoes']} repetições "
          f"({E14['repeticoes_bloco_1']} com bloco 1)", flush=True)

    destino = caminho_resultado("14_fusao_concorrente.csv")
    chaves = ("cena", "arranjo", "n", "bloco", "secao", "torch_threads", "rep")
    linhas, feitas = carregar_feito(destino, chaves)
    if feitas:
        print(f"retomando: {len(feitas)} medições já feitas", flush=True)

    for cena in cenas:
        print(f"\n{'='*100}\n### {cena} — {DATASETS[cena]}\n{'='*100}", flush=True)
        torch.set_num_threads(16)
        marca = time.perf_counter()
        modelo, entrada, data, oa_treino = _treinar(cena)
        print(f"modelo treinado em {time.perf_counter()-marca:.1f}s | OA de teste {oa_treino:.4f} | "
              f"{len(entrada.espectros)} posições", flush=True)

        # A referência das classes é o lote com uma thread, medido antes de tudo.
        torch.set_num_threads(1)
        referencia = executar_sequencial(modelo, entrada).previsto

        def medir(arranjo, n, bloco, secao, torch_threads, executar):
            reps = E14["repeticoes_bloco_1"] if bloco == 1 else E14["repeticoes"]
            pendentes = [r for r in range(reps)
                         if (cena, arranjo, n, bloco, secao, torch_threads, r) not in feitas]
            if not pendentes:
                return
            novas = []
            for rep in pendentes:
                resultado = executar()
                linha = _linha(cena, resultado, rep, torch_threads, data, referencia)
                if not (linha["ids_iguais_ao_treino"] and linha["classes_iguais_ao_lote"]):
                    print(f"  DIVERGÊNCIA em {arranjo} N={n} bloco={bloco} {secao}: "
                          f"ids={linha['ids_iguais_ao_treino']} "
                          f"classes={linha['classes_iguais_ao_lote']}", flush=True)
                novas.append(linha)
                l2 = E14["linha_do_tempo"]
                if (arranjo != "lote" and arranjo != "sequencial" and rep == 0 and n == l2["n"]
                        and bloco == l2["bloco"] and secao == l2["secao"]):
                    _gravar_linha_do_tempo(cena, resultado)
                del resultado
                gc.collect()
            linhas.extend(novas)
            salvar(linhas, destino)
            parede = np.mean([l["parede"] for l in novas])
            espera = np.mean([l["espera_total"] for l in novas])
            antes = np.mean([l["conv_antes_do_mapa"] for l in novas])
            print(f"  {arranjo:<12}N={n:<3}bloco={bloco:<6}{secao:<9}torch={torch_threads:<3}"
                  f"parede {parede:7.3f}s  espera {espera:7.3f}s  conv antes do mapa "
                  f"{100*antes:5.1f}%", flush=True)

        n_posicoes = len(entrada.espectros)
        blocos = [b for b in E14["blocos"] if b != 1 or cena in E14["cenas_bloco_1"]]
        for torch_threads in E14["threads_lote"]:
            torch.set_num_threads(torch_threads)
            medir("lote", 1, n_posicoes, "-", torch_threads,
                  lambda: executar_sequencial(modelo, entrada))
        torch.set_num_threads(1)
        for bloco in blocos:
            medir("sequencial", 1, bloco, "-", 1,
                  lambda bloco=bloco: executar_sequencial(modelo, entrada, bloco))
        for n in E14["threads"]:
            pool = aquecer_pool(modelo, entrada, n)
            try:
                for arranjo in ("concorrente", "desacoplado"):
                    for bloco in blocos:
                        for secao in E14["secoes"]:
                            medir(arranjo, n, bloco, secao, 1,
                                  lambda n=n, bloco=bloco, secao=secao, arranjo=arranjo:
                                  executar_concorrente(
                                      modelo, entrada, n, bloco, secao,
                                      desacoplado=arranjo == "desacoplado", pool=pool))
            finally:
                pool.shutdown()

    quadro = pd.DataFrame(linhas)
    if quadro.empty:
        print("\nsem medições.")
        return quadro
    resumir_14(quadro)
    print(f"\nCSV: {destino}", flush=True)
    return quadro


def resumir_14(quadro: "pd.DataFrame") -> None:
    """Tabelas do experimento: speedup contra o sequencial no mesmo bloco e contra o lote."""
    for cena, sub in quadro.groupby("cena"):
        lote = sub[sub["arranjo"] == "lote"].groupby("torch_threads")["parede"].mean()
        print(f"\n{'='*100}\n### {cena} — tempo de parede (média) e speedup\n{'='*100}",
              flush=True)
        for threads, tempo in lote.items():
            print(f"  lote, torch={threads:<3} {tempo:7.3f}s", flush=True)
        seq = sub[sub["arranjo"] == "sequencial"].groupby("bloco")["parede"].mean()
        print(f"\n{'arranjo':<13}{'bloco':>6}{'seção':>9}{'N':>4}{'parede':>10}"
              f"{'×seq. mesmo bloco':>20}{'×lote torch=1':>15}{'×lote torch=16':>16}"
              f"{'espera':>9}{'conv antes':>12}", flush=True)
        conc = sub[sub["arranjo"].isin(["concorrente", "desacoplado"])]
        for (arranjo, bloco, secao, n), cel in conc.groupby(["arranjo", "bloco", "secao", "n"]):
            parede = cel["parede"].mean()
            print(f"{arranjo:<13}{bloco:>6}{secao:>9}{n:>4}{parede:>9.3f}s"
                  f"{seq.get(bloco, np.nan) / parede:>19.2f}x"
                  f"{lote.get(1, np.nan) / parede:>14.2f}x"
                  f"{lote.get(16, np.nan) / parede:>15.2f}x"
                  f"{cel['espera_total'].mean():>8.3f}s"
                  f"{100 * cel['conv_antes_do_mapa'].mean():>11.1f}%", flush=True)
        ok = sub["ids_iguais_ao_treino"].all() and sub["classes_iguais_ao_lote"].all()
        print(f"\n  conferência: {'todas as execuções' if ok else 'HÁ EXECUÇÕES QUE NÃO'} "
              f"reproduzem os ids do treino e as classes do lote "
              f"({len(sub)} execuções)", flush=True)
