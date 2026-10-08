"""Figuras do TCC: o que cada pilar de otimização mudou na linha de base do Sydney.

O trabalho de referência (Souza, 2023) entrega uma CNN 1D com fusão de superpixel
SEEDS e uma receita fixa de hiperparâmetros. Este TCC a otimiza em quatro pilares —
paralelismo, organização do código, automação de hiperparâmetros e outros
segmentadores — e estas figuras são a evidência de cada um.

Organização, na ordem em que entram no texto:

  I.   O problema e a linha de base herdada      (01–03)
  II.  Pilar 1 — paralelismo                     (04–10)
  III. Pilar 3 — automação de hiperparâmetros    (11–17)
  IV.  Pilar 4 — segmentadores e fusão           (18–23)
  V.   O que o protocolo de avaliação esconde    (24–26)

O sistema visual está em `estilo.py`; os números que não vivem em CSV de
experimento, em `medicoes.py`. Saída em PNG (tela) e PDF (vetorial, para o Typst),
em `experimentos/figuras/`.

Uso:  python graficos.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import medicoes as M
from estilo import (AGUA, AMARELO, AZUL, COR_CENA, COR_CONDICAO, COR_ESTRATEGIA,
                    COR_PROTOCOLO, FRACO, GANHO, LARANJA, NEUTRO, PERDA, TINTA,
                    TINTA2, VIOLETA, br, creditar, legendar, limpar, num,
                    rotular_barras, rotular_linha, titular, virgula)

RESULTADOS = Path(__file__).resolve().parent / "resultados"
FIGURAS = Path(__file__).resolve().parent / "figuras"
FIGURAS.mkdir(exist_ok=True)

CONDICOES = ["sem fusão", "seeds", "slic", "lsc", "watershed"]
#: A receita herdada do trabalho de referência, eixo a eixo.
RECEITA_SYDNEY = {"learning rate": 0.001, "dropout": 0.5, "weight decay": 0.0,
                  "batch": 128, "PCA": 30}


def salvar(fig, nome: str) -> None:
    for extensao in ("png", "pdf"):
        fig.savefig(FIGURAS / f"{nome}.{extensao}")
    plt.close(fig)
    print(f"  {nome}")


def ler(nome: str) -> pd.DataFrame:
    return pd.read_csv(RESULTADOS / nome)


# ══════════════════════════════════ I. o problema e a linha de base herdada

def f01_escala_das_cenas():
    """Por que Indian Pines é o caso difícil das três cenas."""
    fig, eixos = plt.subplots(1, 3, figsize=(10.5, 3.2))
    chaves = list(M.CENAS)
    paineis = [("pixels rotulados", [M.CENAS[c]["rotulados"] for c in chaves], "{:,.0f}"),
               ("classes a separar", [M.CENAS[c]["classes"] for c in chaves], "{:.0f}"),
               ("pixels rotulados por classe",
                [M.CENAS[c]["rotulados"] / M.CENAS[c]["classes"] for c in chaves],
                "{:,.0f}")]
    for eixo, (titulo, valores, formato) in zip(eixos, paineis):
        barras = eixo.bar(chaves, valores, color=[COR_CENA[c] for c in chaves], width=0.62)
        rotular_barras(eixo, barras, valores, formato, deslocamento=max(valores) * 0.03)
        eixo.set(title=titulo, ylim=(0, max(valores) * 1.24))
        eixo.set_yticks([])
        eixo.spines["left"].set_visible(False)
    titular(fig, "Indian Pines tem cinco vezes menos pixels por classe que Salinas",
            "São 641 pixels rotulados por classe contra 3.383 — com 200 bandas para "
            "separar as 16, é a cena em que o contexto espacial mais importa, e por "
            "isso a principal do trabalho")
    creditar(fig, "Fonte: ground truth de cada cena. Bandas: 200 em Indian Pines, "
                  "103 em Pavia, 204 em Salinas.")
    salvar(fig, "01_escala_das_cenas")


def f02_desbalanceamento():
    """O desbalanceamento de classes que separa OA de AA."""
    ordenadas = sorted(enumerate(M.CLASSES_IP, start=1), key=lambda par: -par[1])
    nomes = [str(indice) for indice, _ in ordenadas]
    valores = [quantidade for _, quantidade in ordenadas]
    fig, eixo = plt.subplots(figsize=(10, 3.4))
    cores = [PERDA if v < 100 else AZUL for v in valores]
    barras = eixo.bar(nomes, valores, color=cores, width=0.68, zorder=3)
    for barra, valor in zip(barras, valores):
        if valor < 100 or valor > 1200:
            eixo.text(barra.get_x() + barra.get_width() / 2, valor + 50,
                      br(f"{valor:,.0f}"), ha="center", fontsize=8.5, weight="bold",
                      color=PERDA if valor < 100 else AZUL)
    eixo.set(ylabel="pixels rotulados", xlabel="classe", ylim=(0, 2800))
    limpar(eixo)
    eixo.annotate("quatro classes com menos de 100 pixels:\ncada uma pesa 1/16 da AA "
                  "e quase nada na OA",
                  xy=(13.4, 160), xytext=(9.6, 1150), fontsize=9, color=PERDA,
                  arrowprops=dict(arrowstyle="->", color=PERDA, lw=1.2))
    titular(fig, "A classe maior tem 123 vezes mais pixels que a menor",
            "Indian Pines, distribuição do ground truth — a razão de reportar OA e AA juntas")
    creditar(fig, "Fonte: ground truth de Indian Pines. Vermelho: classes com menos de "
                  "100 pixels rotulados.")
    salvar(fig, "02_desbalanceamento_classes")


def f03_metricas_divergem():
    """OA, AA e Kappa da mesma execução, mostrando o quanto discordam."""
    d = ler("06_resumo.csv")
    fig, eixo = plt.subplots(figsize=(9.6, 3.8))
    largura, posicoes = 0.26, np.arange(len(CONDICOES))
    sub = d[d["cena"] == "IP"].set_index("condição")
    for deslocamento, (metrica, cor) in zip((-largura, 0, largura),
                                            (("OA", AZUL), ("AA", LARANJA), ("Kappa", AGUA))):
        valores = [sub.loc[c, metrica] for c in CONDICOES]
        desvios = [sub.loc[c, f"{metrica}_std"] for c in CONDICOES]
        eixo.bar(posicoes + deslocamento, valores, largura * 0.92, yerr=desvios, capsize=3,
                 color=cor, label=metrica, zorder=3,
                 error_kw={"ecolor": TINTA2, "alpha": .55, "lw": 1})
    for posicao, condicao in zip(posicoes, CONDICOES):
        distancia = 100 * (sub.loc[condicao, "OA"] - sub.loc[condicao, "AA"])
        eixo.text(posicao, 0.95, f"OA−AA\n{num(distancia, 1)} pp", ha="center",
                  fontsize=8.5, color=FRACO)
    eixo.set(xticks=posicoes, xticklabels=CONDICOES, ylim=(0, 1.05),
             ylabel="pontuação")
    limpar(eixo); virgula(eixo, "y", 1)
    legendar(eixo, 3)
    titular(fig, "A AA fica 12 a 19 pontos abaixo da OA em toda condição",
            "Indian Pines, cada condição com hiperparâmetros próprios, 5 sementes — "
            "a distância é o custo das classes raras")
    creditar(fig, "Fonte: resultados/06_resumo.csv. Barras de erro: ±1 desvio entre sementes.")
    salvar(fig, "03_metricas_divergem")


# ═══════════════════════════════════════════════ II. pilar 1 — paralelismo

def f04_ramos_desbalanceados():
    """Por que o paralelismo de dois ramos tem teto baixo."""
    chaves = list(M.RAMOS)
    fig, eixo = plt.subplots(figsize=(9.2, 3.2))
    espectral = [M.RAMOS[c]["espectral"] for c in chaves]
    espacial = [M.RAMOS[c]["espacial"] for c in chaves]
    posicoes = np.arange(len(chaves))
    eixo.barh(posicoes, espectral, 0.5, color=AZUL, label="ramo espectral (PCA)", zorder=3)
    eixo.barh(posicoes, espacial, 0.5, left=espectral, color=LARANJA,
              label="ramo espacial (segmentação)", zorder=3)
    for posicao, (e, s) in enumerate(zip(espectral, espacial)):
        eixo.text(e + s + 0.03, posicao,
                  f"segmentação = {s / (e + s):.0%}   →   teto {num((e + s) / max(e, s))}×",
                  va="center", fontsize=9, weight="bold", color=TINTA)
    eixo.set(yticks=posicoes, yticklabels=[M.CENAS[c]["nome"] for c in chaves],
             xlabel="segundos", xlim=(0, 1.75))
    eixo.invert_yaxis()
    limpar(eixo, "x"); virgula(eixo, "x", 1)
    legendar(eixo, 2)
    titular(fig, "A segmentação é 78 a 93% do pré-processamento — e isso fixa o teto",
            "Dois ramos desbalanceados: pela lei de Amdahl o speedup máximo é a soma "
            "dividida pelo maior deles")
    creditar(fig, "Fonte: medicoes.RAMOS, 5 repetições por combinação (03/09).")
    salvar(fig, "04_ramos_desbalanceados")


def f05_speedup_dois_ramos():
    """Speedup medido contra o teto teórico, nos dois backends."""
    chaves = list(M.RAMOS)
    fig, eixo = plt.subplots(figsize=(8.6, 3.8))
    largura, posicoes = 0.3, np.arange(len(chaves))
    for deslocamento, (backend, cor) in zip((-largura / 2, largura / 2),
                                            (("thread", AZUL), ("process", LARANJA))):
        valores = [M.RAMOS[c]["sequential"] / M.RAMOS[c][backend] for c in chaves]
        barras = eixo.bar(posicoes + deslocamento, valores, largura, color=cor,
                          label=backend, zorder=3)
        rotular_barras(eixo, barras, valores, "{:.2f}×", cor, 0.02)
    for posicao, cena in enumerate(chaves):
        e, s = M.RAMOS[cena]["espectral"], M.RAMOS[cena]["espacial"]
        teto = (e + s) / max(e, s)
        eixo.plot([posicao - 0.27, posicao + 0.27], [teto, teto], color=VIOLETA, lw=2.2,
                  zorder=4)
        # À esquerda do traço: acima colidiria com o rótulo da barra em Salinas.
        eixo.text(posicao - 0.31, teto, f"teto {num(teto)}×", ha="right", va="center",
                  fontsize=8.5, color=VIOLETA, weight="bold")
    eixo.axhline(1.0, color=TINTA, lw=1.1, ls="--", zorder=2)
    eixo.text(len(chaves) - 0.52, 1.02, "sem ganho", fontsize=8.5, color=FRACO, ha="right")
    eixo.set(xticks=posicoes, xticklabels=[M.CENAS[c]["nome"] for c in chaves],
             ylabel="speedup", ylim=(0, 1.58), xlim=(-0.95, 2.55))
    limpar(eixo); virgula(eixo)
    legendar(eixo, 2)
    titular(fig, "Em Salinas a implementação encosta no teto teórico: 1,22× contra 1,21×",
            "O limite não está no código do paralelismo, está no desequilíbrio entre os ramos")
    creditar(fig, "Fonte: medicoes.RAMOS. Barras abaixo de 1,00× são mais lentas que "
                  "não paralelizar.")
    salvar(fig, "05_speedup_dois_ramos")


def f06_custo_serializacao():
    """Por que o backend de processos perde no pré-processamento."""
    chaves = list(M.SERIALIZACAO_MB)
    fig, eixos = plt.subplots(1, 2, figsize=(9.4, 3.4))
    megas = [M.SERIALIZACAO_MB[c] for c in chaves]
    barras = eixos[0].bar(chaves, megas, color=[COR_CENA[c] for c in chaves], width=0.6,
                          zorder=3)
    rotular_barras(eixos[0], barras, megas, "{:.1f} MB", deslocamento=1.4)
    eixos[0].set(title="arrays enviados ao processo filho", ylim=(0, 58))
    eixos[0].set_yticks([])
    eixos[0].spines["left"].set_visible(False)

    tempos = [M.RAMOS[c]["process"] / M.RAMOS[c]["sequential"] for c in chaves]
    barras = eixos[1].bar(chaves, tempos, color=PERDA, width=0.6, zorder=3)
    rotular_barras(eixos[1], barras, tempos, "{:.2f}×", PERDA, 0.05)
    eixos[1].axhline(1.0, color=TINTA, lw=1.1, ls="--")
    eixos[1].set(title="tempo do backend `process` sobre o sequencial", ylim=(0, 5.6))
    limpar(eixos[1]); virgula(eixos[1])
    titular(fig, "Enviar os dados custa mais do que processá-los",
            "Em Indian Pines o pré-processamento inteiro leva 0,166 s; montar o processo "
            "e serializar 8,4 MB leva 4,5 vezes isso")
    creditar(fig, "Fonte: medicoes.SERIALIZACAO_MB e medicoes.RAMOS.")
    salvar(fig, "06_custo_serializacao")


def f07_threads_pytorch():
    """O paralelismo interno do PyTorch atrapalha em modelos pequenos."""
    fig, eixo = plt.subplots(figsize=(7.8, 3.4))
    threads = [t for t, _ in M.THREADS_TORCH][::-1]
    tempos = [s for _, s in M.THREADS_TORCH][::-1]
    eixo.plot(range(len(threads)), tempos, marker="o", ms=9, lw=2.4, color=AZUL, zorder=3)
    melhor = min(tempos)
    for posicao, segundos in enumerate(tempos):
        pior = segundos > melhor * 1.2
        eixo.annotate(f"{num(segundos)}s", (posicao, segundos), textcoords="offset points",
                      xytext=(0, 12), ha="center", fontsize=9, weight="bold",
                      color=PERDA if pior else TINTA)
    eixo.set(xticks=range(len(threads)), xticklabels=[str(t) for t in threads],
             xlabel="threads internas do PyTorch", ylabel="segundos por execução",
             ylim=(6.6, 12.9))
    limpar(eixo); virgula(eixo, "y", 0)
    titular(fig, "Com 16 threads o treino fica 36% mais lento do que com uma",
            "CNN1D em Indian Pines, 25 épocas — batches de 64 e sequências de 30 são "
            "pequenos demais para pagar a coordenação")
    creditar(fig, "Fonte: medicoes.THREADS_TORCH (08/10). Motiva `torch.set_num_threads(1)` "
                  "nos experimentos de paralelismo.")
    salvar(fig, "07_threads_pytorch")


def f08_gil():
    """A demonstração do GIL: onde threads ajudam e onde não."""
    fig, eixo = plt.subplots(figsize=(8.4, 3.0))
    nomes = [n for n, _, _ in M.GIL]
    posicoes = np.arange(len(nomes))
    largura = 0.32
    sequencial = [s for _, s, _ in M.GIL]
    paralelo = [p for _, _, p in M.GIL]
    eixo.barh(posicoes + largura / 2, sequencial, largura, color=NEUTRO,
              label="sequencial", zorder=3)
    eixo.barh(posicoes - largura / 2, paralelo, largura, color=AZUL,
              label="4 threads", zorder=3)
    for posicao, (s, p) in enumerate(zip(sequencial, paralelo)):
        ganho = s / p
        eixo.text(max(s, p) + 0.03, posicao, f"{num(ganho)}×", va="center",
                  fontsize=12, weight="bold", color=GANHO if ganho > 1.5 else PERDA)
    eixo.set(yticks=posicoes, yticklabels=nomes, xlabel="segundos", xlim=(0, 1.35))
    eixo.invert_yaxis()
    limpar(eixo, "x"); virgula(eixo, "x", 1)
    legendar(eixo, 2)
    titular(fig, "Threads só paralelizam o que sai do interpretador",
            "Mesma máquina, mesmas 4 threads — a diferença é se o trabalho está em "
            "bytecode Python ou em C")
    creditar(fig, "Fonte: medicoes.GIL (08/10). O laço de treino do PyTorch se comporta "
                  "como o primeiro caso.")
    salvar(fig, "08_gil")


def f09_executores_particao():
    """Speedup real da partição da cena, por executor."""
    fig, eixo = plt.subplots(figsize=(8.2, 3.8))
    ns = sorted(M.EXECUTORES)
    largura, posicoes = 0.34, np.arange(len(ns))
    for deslocamento, (executor, cor) in zip((-largura / 2, largura / 2),
                                             (("thread", PERDA), ("process", AZUL))):
        valores = [M.EXECUTORES[n]["sequencial"] / M.EXECUTORES[n][executor] for n in ns]
        barras = eixo.bar(posicoes + deslocamento, valores, largura, color=cor,
                          label=executor, zorder=3)
        rotular_barras(eixo, barras, valores, "{:.2f}×", cor, 0.03)
    eixo.axhline(1.0, color=TINTA, lw=1.2, ls="--", zorder=2)
    eixo.annotate("mesmas partições, executadas em sequência", (0.01, 1.0),
                  xycoords=("axes fraction", "data"), textcoords="offset points",
                  xytext=(0, 5), fontsize=8.5, color=FRACO, ha="left")
    eixo.set(xticks=posicoes, xticklabels=[f"{n} partições" for n in ns],
             ylabel="speedup real", ylim=(0, 1.95))
    limpar(eixo); virgula(eixo)
    legendar(eixo, 2)
    titular(fig, "Threads deixam o treino até 4× mais lento; só processos aceleram",
            "Partição da cena, CNN1D em Indian Pines — o laço de treino é limitado pelo GIL")
    creditar(fig, "Fonte: medicoes.EXECUTORES (08/10). O backend `process` usa spawn, "
                  "exigido pelo PyTorch após autograd em threads.")
    salvar(fig, "09_executores_particao")


def f10_escala_particao():
    """A diferença entre o speedup que a métrica de bloco promete e o que o relógio dá."""
    d = ler("11_paralelismo_particao.csv")
    base = d[d["arranjo"] == "monolítico"]
    ns = [2, 4, 8, 16]
    posicoes = np.arange(len(ns))
    arranjos = [("independente · faixas", "faixas", AZUL, "o"),
                ("independente · aleatória", "aleatória (controle)", LARANJA, "s"),
                ("replicado · faixas", "faixas, um modelo só", AGUA, "^")]
    fig, eixos = plt.subplots(1, 2, figsize=(10.8, 3.8))

    # Painel 1: a métrica otimista contra o relógio, no arranjo que paraleliza o treino.
    sub = d[d["arranjo"] == "independente · faixas"]
    bloco = [sub[sub["n"] == n]["speedup_bloco"].mean() for n in ns]
    parede = [base["tempo_total"].mean() / sub[sub["n"] == n]["tempo_total"].mean() for n in ns]
    eixos[0].plot(posicoes, bloco, marker="o", ms=8, lw=2.4, color=PERDA, zorder=3,
                  label="soma das partes ÷ bloco")
    eixos[0].plot(posicoes, parede, marker="s", ms=8, lw=2.4, color=AZUL, zorder=3,
                  label="tempo de parede")
    for posicao, (b, w) in enumerate(zip(bloco, parede)):
        eixos[0].annotate(f"{num(b, 1)}×", (posicao, b), textcoords="offset points",
                          xytext=(0, 11), ha="center", fontsize=8.5, weight="bold", color=PERDA)
        eixos[0].annotate(f"{num(w)}×", (posicao, w), textcoords="offset points",
                          xytext=(0, -17), ha="center", fontsize=8.5, weight="bold", color=AZUL)
    eixos[0].axhline(1.0, color=TINTA, lw=1.2, ls="--", zorder=2)
    eixos[0].set(xticks=posicoes, xticklabels=[str(n) for n in ns], xlabel="partições (N)",
                 ylabel="speedup", yscale="log", xlim=(-0.35, 3.35))
    eixos[0].set_yticks([0.3, 1, 3, 10])
    eixos[0].set_yticklabels(["0,3", "1", "3", "10"])
    eixos[0].minorticks_off()
    limpar(eixos[0])
    legendar(eixos[0], 2)

    # Painel 2: o que cada arranjo cobra em acurácia.
    for arranjo, rotulo, cor, marcador in arranjos:
        xs, ys, erros = [], [], []
        for posicao, n in enumerate(ns):
            grupo = d[(d["arranjo"] == arranjo) & (d["n"] == n)]
            if grupo.empty:
                continue
            xs.append(posicao); ys.append(100 * (grupo["OA"].mean() - base["OA"].mean()))
            erros.append(100 * grupo["OA"].std())
        eixos[1].errorbar(xs, ys, yerr=erros, marker=marcador, ms=8, lw=2.2, capsize=3,
                          color=cor, label=rotulo, zorder=3)
    eixos[1].axhline(0, color=TINTA, lw=1.2, ls="--", zorder=2)
    eixos[1].annotate("monolítico", (0.02, 0), xycoords=("axes fraction", "data"),
                      textcoords="offset points", xytext=(0, -14), fontsize=8.5, color=FRACO)
    eixos[1].set(xticks=posicoes, xticklabels=[str(n) for n in ns], xlabel="partições (N)",
                 ylabel="ΔOA sobre o monolítico (pp)", xlim=(-0.35, 3.35))
    limpar(eixos[1]); virgula(eixos[1], "y", 0)
    legendar(eixos[1], 3)
    titular(fig, "A métrica de bloco promete 14×; o relógio entrega 0,32×",
            "Indian Pines, executor de threads — a soma das partes é medida com as "
            "partições concorrendo, então cresce com N por construção")
    creditar(fig, "Fonte: resultados/11_paralelismo_particao.csv. O ganho de acurácia das "
                  "faixas sobre a partição aleatória é o atalho espacial da figura 27.")
    salvar(fig, "10_escala_particao")


def f11_memoria_por_arquitetura():
    """Memória retida por trial — o que justifica liberar resultados."""
    fig, eixo = plt.subplots(figsize=(8.6, 3.5))
    cenas = ["IP", "PU", "SA"]
    largura, posicoes = 0.26, np.arange(len(cenas))
    for deslocamento, (arquitetura, cor) in zip((-largura, 0, largura),
                                                (("cnn1d", AZUL), ("cnn2d", LARANJA),
                                                 ("cnn3d", PERDA))):
        valores = [M.MEMORIA_POR_TRIAL[arquitetura][c] for c in cenas]
        barras = eixo.bar(posicoes + deslocamento, valores, largura * 0.92, color=cor,
                          label=arquitetura, zorder=3)
        for barra, valor in zip(barras, valores):
            eixo.text(barra.get_x() + barra.get_width() / 2, valor * 1.15,
                      br(f"{valor:,.0f}") if valor >= 10 else num(valor, 1),
                      ha="center", fontsize=8.5, weight="bold", color=cor)
    eixo.set(xticks=posicoes, xticklabels=[M.CENAS[c]["nome"] for c in cenas],
             ylabel="MB retidos por trial", yscale="log", ylim=(1, 12000))
    limpar(eixo)
    legendar(eixo, 3)
    titular(fig, "Um trial de CNN3D em Salinas retém 2,2 GB — vinte trials seriam 44 GB",
            "Escala logarítmica: três ordens de grandeza entre a CNN1D e a CNN3D, "
            "o que motivou liberar o resultado após cada trial")
    creditar(fig, "Fonte: medicoes.MEMORIA_POR_TRIAL (10/09), antes de o escopo "
                  "se restringir à CNN1D.")
    salvar(fig, "11_memoria_por_arquitetura")


# ════════════════════════════ III. pilar 3 — automação de hiperparâmetros

def f12_convergencia_estrategias():
    """Melhor validação até o trial n, por orçamento de épocas."""
    d = ler("08_trials.csv")
    orcamentos = sorted(d["orçamento"].unique())
    fig, eixos = plt.subplots(1, len(orcamentos), figsize=(11.8, 3.5), sharey=True)
    for eixo, orcamento in zip(eixos, orcamentos):
        sub = d[d["orçamento"] == orcamento]
        for estrategia, cor in COR_ESTRATEGIA.items():
            curva = (sub[sub["estratégia"] == estrategia]
                     .groupby("trial")["melhor até agora"].mean())
            eixo.plot(curva.index, curva.values, lw=2.2, color=cor, zorder=3,
                      label=estrategia)
        eixo.set(title=f"{orcamento} épocas por trial", xlabel="trial", xlim=(1, 20.5))
        limpar(eixo); virgula(eixo, "y", 2)
    eixos[0].set_ylabel("melhor validação até o trial")
    # As curvas convergem: rótulo colado à linha colidiria. Legenda no canto vazio.
    eixos[0].legend(loc="lower right", ncol=1)
    titular(fig, "Com orçamento maior as três estratégias convergem para o mesmo lugar",
            "Indian Pines, 20 trials por busca, média de 3 sementes — curva do recorde "
            "acumulado, que é o que uma busca realmente entrega")
    creditar(fig, "Fonte: resultados/08_trials.csv (540 trials).")
    salvar(fig, "12_convergencia_estrategias")


def f13_estrategias_vs_ruido():
    """A diferença entre estratégias contra o desvio entre sementes."""
    d = ler("08_trials.csv")
    melhores = (d.groupby(["orçamento", "estratégia", "semente"])["val_accuracy (objetivo)"]
                .max().reset_index())
    orcamentos = sorted(melhores["orçamento"].unique())
    fig, eixo = plt.subplots(figsize=(8.8, 3.8))
    largura, posicoes = 0.26, np.arange(len(orcamentos))
    for deslocamento, (estrategia, cor) in zip((-largura, 0, largura), COR_ESTRATEGIA.items()):
        medias, desvios = [], []
        for orcamento in orcamentos:
            valores = melhores[(melhores["orçamento"] == orcamento)
                               & (melhores["estratégia"] == estrategia)]["val_accuracy (objetivo)"]
            medias.append(valores.mean()); desvios.append(valores.std())
        eixo.bar(posicoes + deslocamento, medias, largura * 0.9, yerr=desvios, capsize=4,
                 color=cor, label=estrategia, zorder=3,
                 error_kw={"ecolor": TINTA, "alpha": .6, "lw": 1.3})
    topo = melhores["val_accuracy (objetivo)"].max()
    for posicao, orcamento in zip(posicoes, orcamentos):
        grupo = melhores[melhores["orçamento"] == orcamento]
        medias = grupo.groupby("estratégia")["val_accuracy (objetivo)"].mean()
        amplitude = 100 * (medias.max() - medias.min())
        ruido = 100 * grupo.groupby("estratégia")["val_accuracy (objetivo)"].std().mean()
        eixo.text(posicao, topo * 1.04,
                  f"entre estratégias {num(amplitude, 1)} pp\n"
                  f"entre sementes {num(ruido, 1)} pp",
                  ha="center", fontsize=8.5, color=FRACO)
    eixo.set(xticks=posicoes, xticklabels=[f"{o} épocas por trial" for o in orcamentos],
             ylabel="melhor validação da busca", ylim=(0.55, topo * 1.17))
    limpar(eixo); virgula(eixo, "y", 1)
    legendar(eixo, 3)
    titular(fig, "Nenhuma estratégia de busca se separa do ruído entre sementes",
            "Trocar busca aleatória por TPE ou algoritmo genético não paga; o orçamento "
            "de épocas, sim")
    creditar(fig, "Fonte: resultados/08_trials.csv. Barras de erro: ±1 desvio sobre 3 sementes.")
    salvar(fig, "13_estrategias_vs_ruido")


def f14_orcamento_de_epocas():
    """O eixo que realmente move o resultado: épocas por trial."""
    d = ler("08_trials.csv")
    melhores = (d.groupby(["orçamento", "estratégia", "semente"])["val_accuracy (objetivo)"]
                .max().reset_index())
    orcamentos = sorted(melhores["orçamento"].unique())
    medias = [melhores[melhores["orçamento"] == o]["val_accuracy (objetivo)"].mean()
              for o in orcamentos]
    custo = [d[d["orçamento"] == o]["tempo treino (s)"].mean() for o in orcamentos]
    fig, eixo = plt.subplots(figsize=(8.4, 3.6))
    eixo.plot(range(len(orcamentos)), medias, marker="o", ms=10, lw=2.6, color=AZUL, zorder=4)
    for posicao, (valor, segundos) in enumerate(zip(medias, custo)):
        eixo.annotate(num(valor, 4), (posicao, valor), textcoords="offset points",
                      xytext=(0, 13), ha="center", fontsize=10, weight="bold", color=AZUL)
        eixo.annotate(f"{num(segundos, 1)} s por trial", (posicao, valor),
                      textcoords="offset points", xytext=(0, -20), ha="center",
                      fontsize=8.5, color=FRACO)
    for inicio in range(len(orcamentos) - 1):
        salto = 100 * (medias[inicio + 1] - medias[inicio])
        fator = custo[inicio + 1] / custo[inicio]
        eixo.annotate(f"{num(salto, 1)} pp\npor {num(fator, 1)}× o custo",
                      ((inicio + inicio + 1) / 2, (medias[inicio] + medias[inicio + 1]) / 2),
                      textcoords="offset points", xytext=(0, 24), ha="center",
                      fontsize=9, weight="bold", color=GANHO if salto > 3 else FRACO)
    eixo.set(xticks=range(len(orcamentos)),
             xticklabels=[f"{o} épocas" for o in orcamentos],
             xlabel="orçamento por trial", ylabel="melhor validação da busca",
             xlim=(-0.35, len(orcamentos) - 0.65), ylim=(0.68, 0.93))
    limpar(eixo); virgula(eixo, "y", 2)
    titular(fig, "Quadruplicar as épocas rende 6,8 pontos; multiplicar por cinco de novo, 2,8",
            "Média sobre as três estratégias e três sementes — o custo cresce menos que o "
            "orçamento porque a parada antecipada corta os trials ruins antes do fim")
    creditar(fig, "Fonte: resultados/08_trials.csv.")
    salvar(fig, "14_orcamento_de_epocas")


def f15_eixos_amplitude():
    """Quanto cada eixo do espaço de busca separa os resultados."""
    d = ler("08_trials.csv")
    dados = []
    for nome, valor_sydney in RECEITA_SYDNEY.items():
        medias = d.groupby(nome)["val_accuracy (objetivo)"].mean().sort_values(ascending=False)
        dados.append((nome, 100 * (medias.max() - medias.min()),
                      list(medias.index).index(valor_sydney) + 1, len(medias)))
    dados.sort(key=lambda linha: linha[1])
    fig, eixo = plt.subplots(figsize=(9.2, 3.4))
    posicoes = np.arange(len(dados))
    cores = [PERDA if lugar == total else AZUL for _, _, lugar, total in dados]
    eixo.barh(posicoes, [linha[1] for linha in dados], 0.56, color=cores, zorder=3)
    for posicao, (_, amplitude, lugar, total) in zip(posicoes, dados):
        eixo.text(amplitude + 0.4, posicao, f"{num(amplitude, 1)} pp", va="center",
                  fontsize=9.5, weight="bold", color=TINTA)
        eixo.text(amplitude + 3.6, posicao, f"receita herdada: {lugar}º de {total}",
                  va="center", fontsize=8.5, color=PERDA if lugar == total else FRACO)
    eixo.set(yticks=posicoes, yticklabels=[linha[0] for linha in dados],
             xlabel="amplitude da validação média entre o melhor e o pior valor (pp)",
             xlim=(0, 28))
    limpar(eixo, "x"); virgula(eixo, "x", 0)
    titular(fig, "O dropout é o eixo que mais separa — e é onde a receita herdada "
                 "escolhe o pior valor",
            "540 trials em Indian Pines; nos outros eixos o valor herdado é o melhor dos "
            "três, com exceção da taxa de aprendizado, em que fica em segundo")
    creditar(fig, "Fonte: resultados/08_trials.csv, já com o LSC determinístico.")
    salvar(fig, "15_eixos_amplitude")


def f16_efeito_dos_valores():
    """Validação média por valor, eixo a eixo, marcando a receita herdada."""
    d = ler("08_trials.csv")
    eixos_busca = ["dropout", "learning rate", "PCA", "weight decay", "batch"]
    fig, eixos = plt.subplots(1, len(eixos_busca), figsize=(12.4, 3.0), sharey=True)
    for eixo, nome in zip(eixos, eixos_busca):
        medias = d.groupby(nome)["val_accuracy (objetivo)"].mean()
        rotulos = [br(f"{v:g}") for v in medias.index]
        cores = [PERDA if v == RECEITA_SYDNEY[nome] else AZUL for v in medias.index]
        barras = eixo.bar(rotulos, medias.values, 0.6, color=cores, zorder=3)
        for barra, valor in zip(barras, medias.values):
            eixo.text(barra.get_x() + barra.get_width() / 2, valor + 0.012,
                      num(valor, 3), ha="center", fontsize=8.5, weight="bold", color=TINTA2)
        eixo.set(title=nome, ylim=(0.45, 0.80))
        limpar(eixo); virgula(eixo, "y", 2)
    eixos[0].set_ylabel("validação média")
    titular(fig, "O dropout herdado é o pior dos três valores do eixo mais sensível",
            "Média sobre todos os trials que usaram cada valor — vermelho marca a receita "
            "do trabalho de referência, que acerta o melhor valor em PCA, weight decay e batch")
    creditar(fig, "Fonte: resultados/08_trials.csv (540 trials em Indian Pines).")
    salvar(fig, "16_efeito_dos_valores")


def f17_impacto_hpo():
    """Receita herdada contra receita otimizada, condição a condição."""
    d = ler("10_impacto_hpo.csv")
    fig, eixo = plt.subplots(figsize=(9.6, 4.0))
    largura, posicoes = 0.34, np.arange(len(CONDICOES))
    for deslocamento, (receita, cor, rotulo) in zip(
            (-largura / 2, largura / 2),
            (("padrão", NEUTRO, "receita herdada"), ("otimizada", AZUL, "receita otimizada"))):
        medias = [d[(d["condição"] == c) & (d["receita"] == receita)]["OA"].mean()
                  for c in CONDICOES]
        desvios = [d[(d["condição"] == c) & (d["receita"] == receita)]["OA"].std()
                   for c in CONDICOES]
        eixo.bar(posicoes + deslocamento, medias, largura, yerr=desvios, capsize=4,
                 color=cor, label=rotulo, zorder=3,
                 error_kw={"ecolor": TINTA, "alpha": .55, "lw": 1.2})
    for posicao, condicao in zip(posicoes, CONDICOES):
        padrao = d[(d["condição"] == condicao) & (d["receita"] == "padrão")]["OA"]
        otimizada = d[(d["condição"] == condicao) & (d["receita"] == "otimizada")]["OA"]
        ganho = 100 * (otimizada.mean() - padrao.mean())
        # Ruído de referência: desvios das duas condições somados em quadratura.
        ruido = 100 * float(np.hypot(padrao.std(), otimizada.std()))
        afirmavel = abs(ganho) > 2 * ruido
        eixo.text(posicao, 1.05, f"{num(ganho, 1, '{:+.1f}')} pp", ha="center", fontsize=10.5,
                  weight="bold", color=(GANHO if ganho > 0 else PERDA) if afirmavel else FRACO)
        eixo.text(posicao, 1.005, "afirmável" if afirmavel else "dentro do ruído",
                  ha="center", fontsize=8, color=GANHO if afirmavel else FRACO)
    eixo.set(xticks=posicoes, xticklabels=CONDICOES, ylabel="acurácia global (OA)",
             ylim=(0, 1.15))
    limpar(eixo); virgula(eixo, "y", 1)
    legendar(eixo, 2)
    titular(fig, "Em quatro das cinco condições, o ganho da otimização some dentro do ruído",
            "Indian Pines, 10% de treino, 5 sementes — sem fusão são +0,44 pp; só com "
            "SEEDS o ganho supera o dobro do desvio entre sementes")
    creditar(fig, "Fonte: resultados/10_impacto_hpo.csv. Ganho afirmável quando excede "
                  "o dobro do desvio combinado.")
    salvar(fig, "17_impacto_hpo")


def f18_maldicao_vencedor():
    """Watershed: a receita vencedora da busca é pior e muito mais instável."""
    d = ler("10_impacto_hpo.csv")
    sub = d[d["condição"] == "watershed"]
    fig, eixo = plt.subplots(figsize=(7.6, 3.8))
    for posicao, (receita, cor, rotulo) in enumerate(
            (("padrão", NEUTRO, "receita herdada"), ("otimizada", LARANJA, "receita “ótima”"))):
        valores = sub[sub["receita"] == receita]["OA"].values
        eixo.add_patch(plt.Rectangle((posicao - 0.17, valores.mean() - valores.std()),
                                     0.34, 2 * valores.std(), color=cor, alpha=.14, zorder=2))
        eixo.plot([posicao - 0.22, posicao + 0.22], [valores.mean()] * 2, color=cor,
                  lw=3, zorder=3)
        eixo.scatter([posicao] * len(valores), valores, s=110, color=cor, zorder=4,
                     edgecolor="white", linewidth=1.6)
        eixo.text(posicao, 0.16,
                  f"média {num(valores.mean(), 4)}\ndesvio ±{num(valores.std(), 4)}",
                  ha="center", fontsize=9, weight="bold",
                  color=PERDA if valores.std() > 0.1 else FRACO)
    eixo.set(xticks=[0, 1], xticklabels=["receita herdada", "receita escolhida pela busca"],
             ylabel="acurácia global (OA)", ylim=(0.08, 0.95), xlim=(-0.5, 1.5))
    limpar(eixo); virgula(eixo, "y", 1)
    titular(fig, "A maldição do vencedor: duas sementes colapsam com a receita “ótima”",
            "Watershed em Indian Pines — a busca escolhe o máximo de 16 trials, e o "
            "máximo de um objetivo ruidoso é enviesado para cima")
    creditar(fig, "Fonte: resultados/10_impacto_hpo.csv. Cada ponto é uma semente; "
                  "a faixa é ±1 desvio.")
    salvar(fig, "18_maldicao_vencedor")


def f19_fracao_treino():
    """O custo de treinar com menos amostras, em qualidade e em tempo."""
    fracoes = [linha[0] for linha in M.FRACAO_TREINO]
    fig, eixos = plt.subplots(1, 2, figsize=(10.4, 3.6))
    posicoes = np.arange(len(fracoes))
    # O eixo é invertido, então a ponta livre para rotular é a de 5% (à esquerda).
    for metrica, cor, coluna in (("OA", AZUL, 2), ("AA", LARANJA, 4), ("Kappa", AGUA, 5)):
        valores = [linha[coluna] for linha in M.FRACAO_TREINO]
        eixos[0].plot(posicoes, valores, marker="o", ms=8, lw=2.2, color=cor, zorder=3)
        rotular_linha(eixos[0], posicoes[-1], valores[-1], metrica, cor, -12)
        queda = 100 * (valores[3] - valores[0])
        eixos[0].annotate(f"{num(queda, 1)} pp", (3, valores[3]),
                          textcoords="offset points", xytext=(11, -1), fontsize=8.5,
                          color=cor, weight="bold")
    eixos[0].axvline(3, color=NEUTRO, lw=1.1, ls="--", zorder=2)
    eixos[0].annotate("regime usado nos experimentos", (3, 0.475),
                      textcoords="offset points", xytext=(-9, 0), ha="right",
                      fontsize=8.5, color=FRACO)
    eixos[0].set(xticks=posicoes, xticklabels=[f"{f:.0%}" for f in fracoes],
                 xlabel="fração do conjunto de treino", ylabel="pontuação", ylim=(0.45, 0.97))
    eixos[0].set_xlim(4.65, -0.3)        # margem à esquerda para os nomes das séries
    limpar(eixos[0]); virgula(eixos[0], "y", 1)

    tempos = [linha[6] for linha in M.FRACAO_TREINO]
    barras = eixos[1].bar(posicoes, tempos, 0.6, color=NEUTRO, zorder=3)
    rotular_barras(eixos[1], barras, tempos, "{:.1f} s", TINTA2, 0.7)
    eixos[1].set(xticks=posicoes, xticklabels=[f"{f:.0%}" for f in fracoes],
                 xlabel="fração do conjunto de treino", ylabel="segundos por execução",
                 ylim=(0, 42))
    eixos[1].invert_xaxis()
    limpar(eixos[1])
    titular(fig, "Treinar com 10% custa 14,3 pontos de OA e 23,0 de AA, e devolve 5,7× "
                 "de tempo",
            "Indian Pines, amostragem estratificada por classe, 3 sementes — a AA cai o "
            "dobro porque as classes raras somem primeiro")
    creditar(fig, "Fonte: medicoes.FRACAO_TREINO (10/09).")
    salvar(fig, "19_fracao_treino")


# ══════════════════════════ IV. pilar 4 — segmentadores e fusão

def f20_estrutura_segmentadores():
    """Regiões contra pureza: a métrica estrutural não prevê a acurácia."""
    # SLIC e LSC produzem exatamente o mesmo número de regiões: os pontos se empilham,
    # então cada rótulo sai por um lado.
    LADOS = {"seeds": (0, 15, "center"), "watershed": (0, 15, "center"),
             "lsc": (0, 15, "center"), "slic": (-15, -4, "right")}
    fig, eixos = plt.subplots(1, 2, figsize=(10.4, 3.8))
    for eixo, cena in zip(eixos, ("IP", "PU")):
        for nome, dados in M.SEGMENTADORES[cena].items():
            dx, dy, ha = LADOS[nome]
            eixo.scatter(dados["regioes"], dados["pureza"], s=190,
                         color=COR_CONDICAO[nome], zorder=4, edgecolor="white", linewidth=2)
            eixo.annotate(nome, (dados["regioes"], dados["pureza"]),
                          textcoords="offset points", xytext=(dx, dy), ha=ha,
                          fontsize=9.5, weight="bold", color=COR_CONDICAO[nome])
        eixo.set(title=M.CENAS[cena]["nome"], xscale="log",
                 xlabel="regiões produzidas", ylim=(0.82, 1.03), xlim=(32, 3200))
        virgula(eixo, "y", 2)
        eixo.set_xticks([50, 100, 300, 1000])
        eixo.set_xticklabels(["50", "100", "300", "1000"])
        eixo.minorticks_off()
        limpar(eixo)
    eixos[0].set_ylabel("pureza das regiões")
    titular(fig, "O LSC é o mais puro nas duas cenas — e não é o que classifica melhor",
            "Pureza mede a homogeneidade da região no ground truth; é uma propriedade "
            "do segmentador, não do classificador")
    creditar(fig, "Fonte: medicoes.SEGMENTADORES (10/09). Escala logarítmica: o SEEDS faz "
                  "45 regiões em Pavia, o LSC faz 1.428.")
    salvar(fig, "20_estrutura_segmentadores")


def f21_amostras_por_regiao():
    """Amostras de treino por região contra o ganho — e o quanto isso explica.

    A hipótese era que a razão amostras/região prevê o sinal do ganho. Os dados só
    a sustentam em parte: ela separa o SEEDS dos demais, mas entre os três
    segmentadores densos, todos em torno de três amostras por região, o ganho varia
    de +6,7 a −11,3 pp. Fica registrada como condição necessária, não suficiente.
    """
    d = ler("06_resumo.csv")
    sub = d[d["cena"] == "IP"].set_index("condição")
    piso = sub.loc["sem fusão", "OA"]
    treino = 871          # 10% de treino em Indian Pines
    # Três pontos quase empilhados no mesmo x: cada rótulo sai por um lado.
    LADOS = {"seeds": (0, 17, "center"), "lsc": (0, 17, "center"),
             "slic": (16, 0, "left"), "watershed": (16, 0, "left")}
    fig, eixo = plt.subplots(figsize=(8.6, 3.8))
    for nome in ("seeds", "slic", "lsc", "watershed"):
        regioes = M.SEGMENTADORES["IP"][nome]["regioes"]
        razao = treino / regioes
        ganho = 100 * (sub.loc[nome, "OA"] - piso)
        dx, dy, ha = LADOS[nome]
        eixo.scatter(razao, ganho, s=210, color=COR_CONDICAO[nome], zorder=4,
                     edgecolor="white", linewidth=2)
        eixo.annotate(f"{nome}\n{regioes} regiões", (razao, ganho),
                      textcoords="offset points", xytext=(dx, dy), ha=ha,
                      va="center" if dx else "bottom" if dy > 0 else "top",
                      fontsize=9, weight="bold", color=COR_CONDICAO[nome])
    eixo.axhline(0, color=TINTA, lw=1.2, ls="--", zorder=2)
    eixo.text(11.8, 0.8, "piso: sem fusão", fontsize=8.5, color=FRACO, ha="right")
    eixo.axvspan(2.3, 3.6, color=NEUTRO, alpha=.10, zorder=1)
    eixo.annotate("os três segmentadores densos têm a mesma razão\n"
                  "e ganhos que vão de +6,7 a −11,3 pp", (8.6, -15), ha="center",
                  fontsize=9, color=FRACO)
    eixo.annotate("", xy=(3.1, -13.5), xytext=(6.1, -15.2),
                  arrowprops=dict(arrowstyle="->", color=FRACO, lw=1.1,
                                  connectionstyle="arc3,rad=-0.25"))
    eixo.set(xlabel="amostras de treino por região", ylim=(-19, 22),
             ylabel="ganho de OA sobre “sem fusão” (pp)", xlim=(1.6, 12))
    limpar(eixo); virgula(eixo, "x", 0); virgula(eixo, "y", 0)
    titular(fig, "Só o SEEDS tem amostras de sobra por região — e é o único com ganho grande",
            "Indian Pines com 871 amostras de treino: a razão separa o SEEDS dos demais, "
            "mas não ordena os três que se amontoam em torno de três amostras por região")
    creditar(fig, "Fonte: resultados/06_resumo.csv e medicoes.SEGMENTADORES. A razão é "
                  "condição necessária para a fusão ajudar, não suficiente.")
    salvar(fig, "21_amostras_por_regiao")


def f22_segmentadores_por_cena():
    """O vencedor muda de cena para cena."""
    d = ler("06_resumo.csv")
    cenas = ["IP", "PU", "SA"]
    condicoes = ["seeds", "slic", "lsc", "watershed"]
    fig, eixos = plt.subplots(1, 3, figsize=(11.8, 3.6))
    for eixo, cena in zip(eixos, cenas):
        sub = d[d["cena"] == cena].set_index("condição")
        piso = sub.loc["sem fusão", "OA"]
        ganhos = [100 * (sub.loc[c, "OA"] - piso) for c in condicoes]
        melhor = condicoes[int(np.argmax(ganhos))]
        barras = eixo.bar(condicoes, ganhos, 0.62,
                          color=[COR_CONDICAO[c] for c in condicoes], zorder=3)
        for barra, valor, nome in zip(barras, ganhos, condicoes):
            eixo.text(barra.get_x() + barra.get_width() / 2,
                      valor + (0.6 if valor >= 0 else -1.8), num(valor, 1, "{:+.1f}"),
                      ha="center", fontsize=9, weight="bold",
                      color=GANHO if valor > 0 else PERDA)
            if nome == melhor:
                eixo.text(barra.get_x() + barra.get_width() / 2, valor + 2.6, "★",
                          ha="center", fontsize=13, color=COR_CONDICAO[nome])
        eixo.axhline(0, color=TINTA, lw=1.2)
        eixo.set(title=f"{M.CENAS[cena]['nome']}\npiso sem fusão: {num(piso, 4)}",
                 ylim=(-15, 20))
        eixo.tick_params(axis="x", rotation=18)
        limpar(eixo)
    eixos[0].set_ylabel("ganho de OA sobre “sem fusão” (pp)")
    titular(fig, "Não existe um melhor segmentador: o vencedor muda a cada cena",
            "Cada condição otimizada em conjunto com seus hiperparâmetros, 5 sementes — "
            "★ marca o melhor da cena")
    creditar(fig, "Fonte: resultados/06_resumo.csv. O SEEDS, herdado do trabalho de "
                  "referência, vence em duas das três cenas.")
    salvar(fig, "22_segmentadores_por_cena")


def f23_ablacao_fusao():
    """O que cada propriedade do id do superpixel contribui."""
    fig, eixo = plt.subplots(figsize=(9.6, 3.4))
    nomes = [linha[0] for linha in M.ABLACAO_FUSAO]
    valores = [linha[2] for linha in M.ABLACAO_FUSAO]
    piso = dict((n, v) for n, _, v in M.ABLACAO_FUSAO)["sem fusão"]
    cores = [NEUTRO if n == "sem fusão" else (GANHO if v > piso else PERDA)
             for n, v in zip(nomes, valores)]
    posicoes = np.arange(len(nomes))
    eixo.barh(posicoes, valores, 0.58, color=cores, zorder=3)
    for posicao, (nome, valor) in enumerate(zip(nomes, valores)):
        delta = 100 * (valor - piso)
        texto = num(valor, 4) + ("" if nome == "sem fusão"
                                 else f"   ({num(delta, 2, '{:+.2f}')} pp)")
        # Começa depois do piso, para o texto não ser cortado pela linha tracejada.
        eixo.text(max(valor, piso) + 0.005, posicao, texto, va="center", fontsize=9,
                  weight="bold", color=TINTA)
    eixo.axvline(piso, color=TINTA, lw=1.3, ls="--", zorder=2)
    eixo.set(yticks=posicoes, yticklabels=nomes, xlabel="acurácia global (OA)",
             xlim=(0.70, 0.97))
    eixo.invert_yaxis()
    limpar(eixo, "x"); virgula(eixo, "x", 2)
    titular(fig, "Metade do ganho da fusão vem da ordem dos ids, não da identidade da região",
            "Indian Pines, split por pixel — cada linha altera uma propriedade do id e "
            "mede o que resta do ganho")
    creditar(fig, "Fonte: medicoes.ABLACAO_FUSAO (03/09). Linha tracejada: piso sem fusão.")
    salvar(fig, "23_ablacao_fusao")


def f24_modo_de_fusao():
    """Escalar contra embedding: a codificação importa tanto quanto a informação."""
    fig, eixo = plt.subplots(figsize=(8.4, 3.6))
    cenas = list(M.MODOS_FUSAO)
    largura, posicoes = 0.26, np.arange(len(cenas))
    for deslocamento, (modo, cor) in zip((-largura, 0, largura),
                                         (("sem fusão", NEUTRO), ("escalar", LARANJA),
                                          ("embedding", AZUL))):
        valores = [M.MODOS_FUSAO[c][modo] for c in cenas]
        barras = eixo.bar(posicoes + deslocamento, valores, largura * 0.9, color=cor,
                          label=modo, zorder=3)
        rotular_barras(eixo, barras, valores, "{:.4f}", cor, 0.006, 8.5)
    for posicao, cena in enumerate(cenas):
        salto = 100 * (M.MODOS_FUSAO[cena]["embedding"] - M.MODOS_FUSAO[cena]["escalar"])
        eixo.annotate(f"escalar → embedding: {num(salto, 1, '{:+.1f}')} pp",
                      xy=(posicao + largura, M.MODOS_FUSAO[cena]["embedding"] + 0.012),
                      xytext=(posicao + largura, 1.055), ha="center", fontsize=9,
                      weight="bold", color=GANHO,
                      arrowprops=dict(arrowstyle="-|>", color=GANHO, lw=1.3))
    eixo.set(xticks=posicoes, xticklabels=[M.CENAS[c]["nome"] for c in cenas],
             ylabel="acurácia global (OA)", ylim=(0.70, 1.13))
    limpar(eixo); virgula(eixo, "y", 2)
    legendar(eixo, 3)
    titular(fig, "O mesmo id rende mais quando entra como vetor aprendido, não como número",
            "+4,9 pp em Indian Pines e +0,8 pp em Pavia sem nenhuma informação nova — "
            "o id do superpixel é nominal, e usá-lo como número impõe uma ordem que não existe")
    creditar(fig, "Fonte: medicoes.MODOS_FUSAO (03/09), split por pixel, 3 sementes.")
    salvar(fig, "24_modo_de_fusao")


# ══════════════════════ V. o que o protocolo de avaliação esconde

def f25_vazamento_no_piso():
    """O efeito do vazamento espacial no modelo sem fusão nenhuma."""
    d = ler("09_validacao_regiao.csv")
    sub = d[d["condição"] == "sem fusão"]
    fig, eixo = plt.subplots(figsize=(7.8, 3.6))
    medias = {}
    for posicao, protocolo in enumerate(("pixel", "region")):
        valores = sub[sub["protocolo"] == protocolo]["OA"].values
        medias[protocolo] = valores.mean()
        cor = COR_PROTOCOLO[protocolo]
        eixo.scatter([posicao] * len(valores), valores, s=110, color=cor, zorder=4,
                     edgecolor="white", linewidth=1.6)
        eixo.plot([posicao - 0.2, posicao + 0.2], [valores.mean()] * 2, color=cor,
                  lw=3.5, zorder=3)
        eixo.text(posicao - 0.26, valores.mean(), num(valores.mean(), 4), ha="right",
                  va="center", fontsize=11, weight="bold", color=cor)
    # A seta fica à direita da nuvem de pontos, não por cima dela.
    eixo.annotate("", xy=(1.32, medias["region"]), xytext=(1.32, medias["pixel"]),
                  arrowprops=dict(arrowstyle="-|>", color=PERDA, lw=2))
    eixo.text(1.40, (medias["pixel"] + medias["region"]) / 2,
              f"{num(100 * (medias['region'] - medias['pixel']), 1)} pontos",
              fontsize=12, weight="bold", color=PERDA, va="center")
    eixo.set(xticks=[0, 1], xlim=(-0.6, 2.05), ylim=(0.40, 0.95),
             xticklabels=["split aleatório por pixel\n(100% do teste vaza)",
                          "validação cruzada por região\n(0% vaza)"],
             ylabel="acurácia global (OA)")
    limpar(eixo); virgula(eixo, "y", 1)
    titular(fig, "O protocolo padrão infla o resultado em 26,6 pontos, mesmo sem fusão",
            "CNN1D pura em Indian Pines — pixels vizinhos são quase-duplicatas, e o split "
            "aleatório os reparte entre treino e teste")
    creditar(fig, "Fonte: resultados/09_validacao_regiao.csv. Cada ponto é uma repetição.")
    salvar(fig, "25_vazamento_no_piso")


def f26_fusao_sob_os_dois_protocolos():
    """Por que a comparação entre segmentadores quebra no split por região.

    O resultado é contraintuitivo: ao remover o vazamento, o ganho da fusão **cresce**
    em três dos quatro segmentadores. A explicação está no painel da direita — sob
    split por região a partição de teste é definida pelo próprio segmentador avaliado,
    e cada um entrega um teste de dificuldade diferente.
    """
    d = ler("09_validacao_regiao.csv")
    condicoes = ["seeds", "lsc", "slic", "watershed"]
    fig, eixos = plt.subplots(1, 2, figsize=(11.2, 3.8),
                              gridspec_kw={"width_ratios": [1.25, 1]})
    largura, posicoes = 0.36, np.arange(len(condicoes))
    for deslocamento, (protocolo, rotulo) in zip(
            (-largura / 2, largura / 2),
            (("pixel", "split por pixel (100% vaza)"),
             ("region", "split por região (0% vaza)"))):
        sub = d[d["protocolo"] == protocolo]
        piso = sub[sub["condição"] == "sem fusão"]["OA"].mean()
        ganhos = [100 * (sub[sub["condição"] == c]["OA"].mean() - piso) for c in condicoes]
        barras = eixos[0].bar(posicoes + deslocamento, ganhos, largura,
                              color=COR_PROTOCOLO[protocolo], label=rotulo, zorder=3)
        for barra, valor in zip(barras, ganhos):
            eixos[0].text(barra.get_x() + barra.get_width() / 2,
                          valor + (0.8 if valor >= 0 else -2.4), num(valor, 1, "{:+.1f}"),
                          ha="center", fontsize=8.5, weight="bold",
                          color=COR_PROTOCOLO[protocolo])
    eixos[0].axhline(0, color=TINTA, lw=1.2)
    eixos[0].set(xticks=posicoes, xticklabels=condicoes,
                 ylabel="ganho de OA sobre “sem fusão” (pp)", ylim=(-8, 28))
    limpar(eixos[0]); virgula(eixos[0], "y", 0)
    legendar(eixos[0], 2)

    nomes = list(M.DISTANCIA_TESTE)
    distancias = [M.DISTANCIA_TESTE[n]["media"] for n in nomes]
    barras = eixos[1].bar(nomes, distancias, 0.6,
                          color=[COR_CONDICAO[n] for n in nomes], zorder=3)
    rotular_barras(eixos[1], barras, distancias, "{:.2f} px", deslocamento=0.12)
    eixos[1].set(title="dificuldade do teste que cada segmentador cria",
                 ylabel="distância média ao treino (pixels)", ylim=(0, 5.6))
    limpar(eixos[1]); virgula(eixos[1], "y", 0)
    titular(fig, "Sem vazamento o ganho da fusão cresce — menos justamente no SEEDS",
            "Porque sob split por região a partição de teste é definida pelo próprio "
            "segmentador avaliado: o SEEDS faz as maiores regiões e produz o teste mais "
            "distante do treino")
    creditar(fig, "Fonte: resultados/09_validacao_regiao.csv e medicoes.DISTANCIA_TESTE. "
                  "É por isso que este protocolo mede a fusão, mas não compara "
                  "segmentadores entre si.")
    salvar(fig, "26_fusao_sob_os_dois_protocolos")


def f27_atalho_da_particao():
    """A figura decisiva do paralelismo por partição."""
    d = ler("12_particao_split.csv")
    fig, eixos = plt.subplots(1, 2, figsize=(10.8, 3.9), sharey=True)
    titulos = {"pixel": "split aleatório por pixel  (100% vaza)",
               "region": "validação cruzada por região  (0% vaza)"}
    ordem = [2, 4, 8]
    for eixo, protocolo in zip(eixos, ("pixel", "region")):
        sub = d[d["protocolo"] == protocolo]
        base = sub[sub["arranjo"] == "monolítico"]["OA"].mean()
        eixo.axhline(base, color=TINTA, lw=1.3, ls="--", zorder=2)
        # O vão livre fica abaixo da linha num painel e acima no outro.
        eixo.annotate(f"monolítico  {num(base, 3)}", xy=(0.97, base),
                      xycoords=("axes fraction", "data"), color=TINTA2, fontsize=8.5,
                      va="top" if protocolo == "pixel" else "bottom", ha="right")
        for arranjo, rotulo, cor, marcador in (("faixas", "faixas", AZUL, "o"),
                                               ("aleatória", "aleatória (controle)",
                                                LARANJA, "s")):
            xs, medias, desvios = [], [], []
            for posicao, n in enumerate(ordem):
                valores = sub[(sub["arranjo"] == arranjo) & (sub["n"] == n)]["OA"]
                if valores.empty:
                    continue
                xs.append(posicao); medias.append(valores.mean()); desvios.append(valores.std())
            if xs:
                eixo.errorbar(xs, medias, yerr=desvios, marker=marcador, ms=8, lw=2.2,
                              capsize=4, color=cor, label=rotulo, zorder=3)
        vantagem = [100 * (sub[(sub["arranjo"] == "faixas") & (sub["n"] == n)]["OA"].mean()
                           - sub[(sub["arranjo"] == "aleatória") & (sub["n"] == n)]["OA"].mean())
                    for n in ordem]
        media = float(np.nanmean(vantagem))
        eixo.set(title=f"{titulos[protocolo]}\nfaixas − aleatória: "
                       f"{num(media, 2, '{:+.2f}')} pp em média",
                 xlabel="partições (N)", xticks=range(len(ordem)),
                 xlim=(-0.25, len(ordem) - 0.75))
        eixo.set_xticklabels([str(n) for n in ordem])
        eixo.title.set_color(GANHO if media > 3 else FRACO)
        limpar(eixo); virgula(eixo, "y", 1)
    eixos[0].set_ylabel("acurácia global (OA)")
    # O painel da esquerda tem um vão grande embaixo (as curvas ficam acima de 0,70).
    eixos[0].legend(loc="lower left", ncol=2)
    titular(fig, "Particionar só ajuda quando o protocolo deixa o modelo interpolar "
                 "entre vizinhos",
            "A vantagem das faixas sobre a partição aleatória — ambas com o mesmo custo "
            "computacional — desaparece ao remover o vazamento")
    creditar(fig, "Fonte: resultados/12_particao_split.csv. Barras de erro: ±1 desvio "
                  "sobre 5 repetições.")
    salvar(fig, "27_atalho_da_particao")


def f28_sintese_dos_atalhos():
    """As técnicas espaciais do trabalho, medidas com e sem vazamento."""
    achados = [("fusão por id de\nsuperpixel (SEEDS)", 8.84, 4.49),
               ("partição em faixas\n(N = 4)", 2.86, -5.07)]
    fig, eixo = plt.subplots(figsize=(8.8, 3.6))
    largura, posicoes = 0.34, np.arange(len(achados))
    com = [linha[1] for linha in achados]
    sem = [linha[2] for linha in achados]
    b1 = eixo.bar(posicoes - largura / 2, com, largura, color=COR_PROTOCOLO["pixel"],
                  label="medido com vazamento (split por pixel)", zorder=3)
    b2 = eixo.bar(posicoes + largura / 2, sem, largura, color=COR_PROTOCOLO["region"],
                  label="medido sem vazamento (split por região)", zorder=3)
    for barras, valores in ((b1, com), (b2, sem)):
        for barra, valor in zip(barras, valores):
            eixo.text(barra.get_x() + barra.get_width() / 2,
                      valor + (0.4 if valor >= 0 else -1.3), num(valor, 2, "{:+.2f}"),
                      ha="center", fontsize=9.5, weight="bold",
                      color=GANHO if valor > 0 else PERDA)
    for posicao, linha in zip(posicoes, achados):
        sobra = linha[2] / linha[1] * 100
        # Percentual negativo não comunica nada: ali o efeito trocou de sinal.
        texto = f"sobra {sobra:.0f}%" if sobra > 0 else "o ganho vira perda"
        eixo.text(posicao, 11.4, texto, ha="center", fontsize=10.5,
                  weight="bold", color=PERDA if sobra < 60 else GANHO)
    eixo.axhline(0, color=TINTA, lw=1.2)
    eixo.set(xticks=posicoes, xticklabels=[linha[0] for linha in achados],
             ylabel="ganho de OA (pontos percentuais)", ylim=(-8, 14))
    limpar(eixo); virgula(eixo, "y", 0)
    legendar(eixo, 2)
    titular(fig, "Duas técnicas espaciais, o mesmo veredito: metade sobrevive, ou nada",
            "Indian Pines — sob split aleatório por pixel, qualquer técnica que use "
            "posição parece funcionar")
    creditar(fig, "Fonte: resultados/09_validacao_regiao.csv e 12_particao_split.csv. "
                  "Nos outros três segmentadores o ganho vai na direção oposta, pelo "
                  "motivo da figura 26.")
    salvar(fig, "28_sintese_dos_atalhos")


FIGURAS_TODAS = [
    f01_escala_das_cenas, f02_desbalanceamento, f03_metricas_divergem,
    f04_ramos_desbalanceados, f05_speedup_dois_ramos, f06_custo_serializacao,
    f07_threads_pytorch, f08_gil, f09_executores_particao, f10_escala_particao,
    f11_memoria_por_arquitetura,
    f12_convergencia_estrategias, f13_estrategias_vs_ruido, f14_orcamento_de_epocas,
    f15_eixos_amplitude, f16_efeito_dos_valores, f17_impacto_hpo, f18_maldicao_vencedor,
    f19_fracao_treino,
    f20_estrutura_segmentadores, f21_amostras_por_regiao, f22_segmentadores_por_cena,
    f23_ablacao_fusao, f24_modo_de_fusao,
    f25_vazamento_no_piso, f26_fusao_sob_os_dois_protocolos, f27_atalho_da_particao,
    f28_sintese_dos_atalhos,
]

if __name__ == "__main__":
    print(f"Gerando {len(FIGURAS_TODAS)} figuras em experimentos/figuras/\n")
    falhas = 0
    for funcao in FIGURAS_TODAS:
        try:
            funcao()
        except Exception as erro:
            falhas += 1
            print(f"  FALHOU {funcao.__name__}: {type(erro).__name__}: {erro}")
    print(f"\n{len(FIGURAS_TODAS) - falhas} de {len(FIGURAS_TODAS)} geradas.")
