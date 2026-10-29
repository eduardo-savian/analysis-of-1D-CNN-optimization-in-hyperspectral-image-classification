"""Separar o custo de partida do custo de paralelismo.

O PORQUÊ, e o erro que este experimento corrige. A primeira comparação de executores
(`EXECUTORES_FRIO`) cronometrou o bloco paralelo com o pool sendo criado e destruído
**dentro** da medição, como o experimento 11 ainda faz. Com `spawn`, isso significa
que cada medição do `process` pagou N interpretadores limpos subindo do zero — e,
na época destas medições, cada um ainda recarregava as células do notebook antes de
treinar qualquer coisa (hoje o filho só importa o pacote `hsi`). O número que saiu dali é o pior caso: a primeira e única execução de
um processo recém-nascido.

Não é o caso de uso. Num uso real o pool sobe uma vez e atende muitas execuções, e a
partida se dilui entre elas. Medir só o frio atribui ao paralelismo um custo que é de
inicialização, e foi exatamente a crítica do orientador ao desenho anterior.

O QUE SE MEDE AQUI. O mesmo bloco, nas duas situações, para cada executor:

- **frio** — pool criado e destruído a cada execução, como no experimento 11;
- **quente** — pool criado uma vez, aquecido com `PoolReutilizavel.aquecer()` (que
  sobe todos os workers e paga o custo preguiçoso de cada um) e reusado nas
  repetições seguintes. O tempo de criação e o de aquecimento ficam registrados à
  parte, para que a partida continue visível em vez de sumir.

TRÊS CUIDADOS QUE SUSTENTAM A COMPARAÇÃO:

1. **O denominador do speedup é o executor `sequencial` no mesmo arranjo**, nunca o
   pipeline monolítico. O monolítico treina na GPU (`select_device`), enquanto as
   partições treinam em CPU (`_treinar_e_inferir` fixa `torch.device("cpu")`):
   comparar os dois mede a diferença de dispositivo junto com a de paralelismo. O
   sequencial processa as mesmas partições, no mesmo dispositivo, uma a uma.
2. **Os dois modos aparecem.** `independente` treina N CNNs, uma por partição — é o
   arranjo do experimento 11. `replicado` treina **uma** CNN com a cena inteira e
   paraleliza só a inferência; a OA é idêntica à do monolítico, então ali o
   paralelismo é ganho de velocidade sem custo de acurácia.
3. **A OA é conferida entre as repetições**, porque o pool muda quem executa e não o
   que é executado. A expectativa depende do executor, e a diferença é um resultado:
   `sequencial` e `process` têm de repetir o valor exatamente, enquanto `thread`
   **não repete** no modo `independente`. O motivo é estrutural e foi medido aqui:
   `_treinar_e_inferir` chama `seed_everything`, que semeia o RNG **global** do
   PyTorch; com N threads num só interpretador, as partições disputam esse gerador
   na construção dos pesos e em cada máscara de dropout. Não há semente que conserte
   isso sem dar um gerador próprio a cada partição — que é o que processos dão de
   graça. No modo `replicado` nem o `thread` varia: o bloco paralelo só infere, e
   inferência não consome RNG.
"""

import gc
import time

import numpy as np
import pandas as pd
import torch

from hsi.ambiente import configure_determinism
from hsi.config import DATASETS, ExperimentConfig
from hsi.experimentos.infra import caminho_resultado, carregar_feito, salvar
from hsi.paralelo.particao import PoolReutilizavel, run_experiment_particionado

E13 = dict(particoes=(2, 4, 8, 16, 32), epocas=25, semente=42,
           repeticoes={"frio": 2, "quente": 3},
           #: O `sequencial` vem primeiro porque é o denominador do speedup; o
           #: `process` vem antes do `thread` porque é a medição que interessa e o
           #: `thread` com N alto leva minutos por repetição. `replicado` não aceita
           #: `process`: exigiria serializar o modelo treinado para cada worker.
           arranjos=(("independente", "sequencial"),
                     ("independente", "process"),
                     ("replicado", "sequencial"),
                     ("replicado", "thread"),
                     ("independente", "thread")))


#: Memória proporcional (PSS) de um worker de `process`, medida em 22/10: RSS 714 MB,
#: PSS 481 MB, privado 422 MB. O RSS superestima porque conta as páginas
#: compartilhadas do PyTorch em cada processo; o PSS é o que de fato precisa caber.
PSS_POR_WORKER_MB = 481


#: Folga exigida sobre a projeção. Abaixo disso a medição entraria em swap e passaria
#: a cronometrar paginação de disco em vez de paralelismo.
FOLGA_MEMORIA = 1.25


def memoria_disponivel_mb() -> float:
    """`MemAvailable` do kernel: o que dá para alocar sem empurrar nada para swap."""
    with open("/proc/meminfo") as arquivo:
        for linha in arquivo:
            if linha.startswith("MemAvailable:"):
                return int(linha.split()[1]) / 1024
    return float("inf")


def conferir_memoria(executor: str, n: int) -> None:
    """Recusa uma célula de `process` que não caberia na memória.

    Recusar é melhor que tentar: com N workers grandes demais o kernel começa a
    paginar — e, no limite, o OOM killer escolhe a vítima, que pode ser outra coisa
    da sessão. O limite é um resultado do trabalho, não um acidente: **o paralelismo
    por processos esbarra em memória antes de esbarrar em núcleos**, porque cada
    worker do `spawn` carrega sua própria cópia do PyTorch.
    """
    if executor != "process":
        return
    necessario = n * PSS_POR_WORKER_MB
    disponivel = memoria_disponivel_mb()
    if necessario * FOLGA_MEMORIA > disponivel:
        raise MemoryError(
            f"{n} workers exigem ~{necessario:.0f} MB de PSS "
            f"({PSS_POR_WORKER_MB} MB cada) e há {disponivel:.0f} MB disponíveis; "
            f"com a folga de {FOLGA_MEMORIA:.2f}x isto entraria em swap."
        )


def _e13_config(cena: str):
    return ExperimentConfig(
        dataset=cena, model_type="cnn1d", two_input=True, segmenter="seeds",
        fusion="scalar", train_fraction=1.0, epochs=E13["epocas"], patience=10,
        seed=E13["semente"], batch_size=64)


def _e13_medir(cena: str, modo: str, executor: str, n: int, estado: str) -> list:
    """Mede as repetições de uma célula, devolvendo uma linha por repetição."""
    conferir_memoria(executor, n)
    config = _e13_config(cena)
    linhas = []
    pool = None
    criacao = aquecimento = 0.0
    try:
        if estado == "quente":
            pool = PoolReutilizavel(executor, max_workers=n)
            pool.aquecer()
            criacao, aquecimento = pool.tempo_criacao, pool.tempo_aquecimento
        for rep in range(E13["repeticoes"][estado]):
            resultado = run_experiment_particionado(
                config, n_particoes=n, modo=modo, particao_modo="faixas",
                executor=executor, backend="thread", verbose=False, pool=pool)
            linhas.append({
                "cena": cena, "modo": modo, "executor": executor, "n": n,
                "estado": estado, "rep": rep,
                "bloco_paralelo": resultado.tempo_paralelo,
                "soma_partes": resultado.soma_das_partes,
                "overhead": resultado.overhead,
                "teto": resultado.teto_de_amdahl,
                "tempo_criacao_pool": criacao,
                "tempo_aquecimento": aquecimento,
                "OA": resultado.metrics.overall_accuracy,
            })
            del resultado
            gc.collect()
    finally:
        if pool is not None:
            pool.fechar()
    return linhas


def experimento_13(*cenas: str) -> "pd.DataFrame":
    """Pool frio contra pool quente, por executor e por número de partições."""
    cenas = cenas or ("IP",)
    configure_determinism(True, verbose=False)
    torch.set_num_threads(1)
    print("POOL QUENTE — custo de partida contra custo de paralelismo")
    print(f"cenas: {', '.join(cenas)} | N: {E13['particoes']} | {E13['epocas']} épocas | "
          f"semente {E13['semente']} | frio {E13['repeticoes']['frio']}x, quente "
          f"{E13['repeticoes']['quente']}x | torch threads=1", flush=True)

    destino = caminho_resultado("13_pool_quente.csv")
    chaves = ("cena", "modo", "executor", "n", "estado", "rep")
    linhas, feitas = carregar_feito(destino, chaves)
    if feitas:
        print(f"retomando: {len(feitas)} medições já feitas", flush=True)

    for cena in cenas:
        print(f"\n{'='*100}\n### {cena} — {DATASETS[cena]}\n{'='*100}", flush=True)
        for modo, executor in E13["arranjos"]:
            for n in E13["particoes"]:
                for estado in ("frio", "quente"):
                    pendentes = [r for r in range(E13["repeticoes"][estado])
                                 if (cena, modo, executor, n, estado, r) not in feitas]
                    if not pendentes:
                        continue
                    marca = time.perf_counter()
                    try:
                        novas = _e13_medir(cena, modo, executor, n, estado)
                    except Exception as erro:
                        print(f"  {modo:<13}{executor:<11}N={n:<3}{estado:<8}"
                              f"{type(erro).__name__}: {str(erro)[:160]}", flush=True)
                        continue
                    linhas.extend(novas)
                    salvar(linhas, destino)
                    bloco = np.mean([l["bloco_paralelo"] for l in novas])
                    print(f"  {modo:<13}{executor:<11}N={n:<3}{estado:<8}"
                          f"bloco {bloco:6.2f}s  "
                          f"(parede da célula {time.perf_counter()-marca:5.1f}s)",
                          flush=True)

    quadro = pd.DataFrame(linhas)
    if quadro.empty:
        print("\nsem medições.")
        return quadro

    for cena in cenas:
        for modo in ("independente", "replicado"):
            sub = quadro[(quadro["cena"] == cena) & (quadro["modo"] == modo)]
            if sub.empty:
                continue
            print(f"\n{'='*100}\n### {cena} · modo {modo} — tempo do bloco paralelo"
                  f"\n{'='*100}", flush=True)
            print(f"{'executor':<12}{'N':>3}{'frio':>12}{'quente':>18}"
                  f"{'partida':>11}{'speedup frio':>14}{'speedup quente':>16}", flush=True)
            for executor in ("sequencial", "thread", "process"):
                for n in E13["particoes"]:
                    cel = sub[(sub["executor"] == executor) & (sub["n"] == n)]
                    frio = cel[cel["estado"] == "frio"]["bloco_paralelo"]
                    quente = cel[cel["estado"] == "quente"]["bloco_paralelo"]
                    if frio.empty and quente.empty:
                        continue
                    base = sub[(sub["executor"] == "sequencial") & (sub["n"] == n)]
                    base_frio = base[base["estado"] == "frio"]["bloco_paralelo"]
                    base_quente = base[base["estado"] == "quente"]["bloco_paralelo"]
                    sp_frio = (base_frio.mean() / frio.mean()
                               if not frio.empty and not base_frio.empty else np.nan)
                    sp_quente = (base_quente.mean() / quente.mean()
                                 if not quente.empty and not base_quente.empty else np.nan)
                    partida = (frio.mean() - quente.mean()
                               if not frio.empty and not quente.empty else np.nan)
                    print(f"{executor:<12}{n:>3}{frio.mean():>11.2f}s"
                          f"{quente.mean():>12.2f}s±{quente.std():<4.2f}"
                          f"{partida:>10.2f}s{sp_frio:>13.2f}x{sp_quente:>15.2f}x",
                          flush=True)

    print(f"\n{'='*100}\nCONFERÊNCIA — a OA se repete entre as execuções?\n{'='*100}",
          flush=True)
    print("esperado: 'sequencial' e 'process' repetem o valor; 'thread' no modo "
          "independente\nnão repete, porque as partições disputam o RNG global do "
          "PyTorch (ver docstring).\n", flush=True)
    for (modo, executor), cel in quadro.groupby(["modo", "executor"]):
        # A amplitude é medida dentro de cada N e só depois agregada: número de
        # partições diferente dá OA diferente por construção, e juntar os N aqui
        # mediria isso em vez de repetibilidade.
        amplitude = 100 * cel.groupby("n")["OA"].agg(lambda s: s.max() - s.min()).max()
        esperado = "varia" if (modo == "independente" and executor == "thread") else "repete"
        observado = "varia" if amplitude > 1e-7 else "repete"
        marca = "ok" if esperado == observado else "INESPERADO"
        print(f"  [{marca:^10}] {modo:<13}{executor:<11}amplitude da OA "
              f"{amplitude:5.2f} pp  ({len(cel)} execuções, esperado: {esperado})",
              flush=True)

    print(f"\n{'='*100}\nCUSTO DE PARTIDA, QUE O AQUECIMENTO TIRA DA MEDIÇÃO\n{'='*100}",
          flush=True)
    quentes = quadro[quadro["estado"] == "quente"]
    for executor in ("sequencial", "thread", "process"):
        cel = quentes[quentes["executor"] == executor]
        if cel.empty:
            continue
        print(f"  {executor:<12}criação {cel['tempo_criacao_pool'].mean():6.3f}s | "
              f"aquecimento {cel['tempo_aquecimento'].mean():6.2f}s", flush=True)

    print(f"\nCSV: {destino}", flush=True)
    return quadro
