"""Gera as figuras do TCC a partir dos CSVs de resultado.

Uma figura por achado, em PNG (para ler na tela) e PDF (vetorial, para o Typst).
Saída em `experimentos/figuras/`.

DECISÕES DE VISUALIZAÇÃO, para que as figuras sejam lidas e não só vistas:

- **Paleta categórica fixa**, atribuída por entidade e nunca reciclada: a mesma
  condição tem a mesma cor em todas as figuras. Azul, laranja, verde-água e amarelo
  são distinguíveis por quem tem daltonismo (verificados por separação em OKLab).
- **Rótulo direto no ponto**, não só na legenda — quem lê a figura impressa não deve
  precisar cruzar cores com um quadrinho no canto.
- **Barra de erro sempre que há repetição.** Uma diferença sem o desvio ao lado é
  uma afirmação sem evidência, e vários resultados deste trabalho só se entendem
  olhando o desvio (o Watershed otimizado, por exemplo).
- **Nenhum eixo duplo.** Quando duas grandezas têm escalas diferentes, viram dois
  painéis — eixo duplo deixa a comparação visual arbitrária.
- **Linha de referência explícita** onde existe um valor neutro: o 1,00× do
  sequencial, o zero do ganho, o piso sem fusão.

Uso:  python graficos.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTADOS = Path(__file__).resolve().parent / "resultados"
FIGURAS = Path(__file__).resolve().parent / "figuras"
FIGURAS.mkdir(exist_ok=True)

AZUL, LARANJA, VERDE, AMARELO = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
CINZA, TINTA = "#9aa0a6", "#1b1d1f"

COR_CONDICAO = {"sem fusão": CINZA, "seeds": AZUL, "slic": LARANJA,
                "lsc": VERDE, "watershed": AMARELO}

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c7ccd1", "axes.labelcolor": TINTA, "text.color": TINTA,
    "xtick.color": "#5f6368", "ytick.color": "#5f6368",
    "grid.color": "#e6e9ec", "grid.linewidth": 0.8, "legend.frameon": False,
})


def salvar(fig, nome: str) -> None:
    for extensao in ("png", "pdf"):
        fig.savefig(FIGURAS / f"{nome}.{extensao}")
    plt.close(fig)
    print(f"  figuras/{nome}.png e .pdf")


def ler(nome: str) -> pd.DataFrame:
    return pd.read_csv(RESULTADOS / nome)


# --------------------------------------------------------------------------- 1
def fig_atalho_espacial() -> None:
    """A figura decisiva: o ganho da partição espacial some sem o vazamento."""
    d = ler("12_particao_split.csv")
    fig, eixos = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    titulos = {"pixel": "Split aleatório por pixel\n(100% de vazamento)",
               "region": "Validação cruzada por região\n(0% de vazamento)"}
    ordem = [2, 4, 8]
    for eixo, protocolo in zip(eixos, ("pixel", "region")):
        sub = d[d["protocolo"] == protocolo]
        base = sub[sub["arranjo"] == "monolítico"]["OA"].mean()
        eixo.axhline(base, color=CINZA, lw=1.4, ls="--", zorder=1)
        # Posição em fração dos eixos: o rótulo nunca escapa da área de desenho.
        eixo.annotate(f"monolítico  {base:.3f}", xy=(0.03, base),
                      xycoords=("axes fraction", "data"), color="#5f6368",
                      fontsize=9, va="bottom")
        for arranjo, cor, marcador in (("faixas", AZUL, "o"), ("aleatória", LARANJA, "s")):
            xs, medias, desvios = [], [], []
            for posicao, n in enumerate(ordem):
                valores = sub[(sub["arranjo"] == arranjo) & (sub["n"] == n)]["OA"]
                if valores.empty:
                    continue
                xs.append(posicao); medias.append(valores.mean()); desvios.append(valores.std())
            if not xs:
                continue
            eixo.errorbar(xs, medias, yerr=desvios, marker=marcador, ms=7, lw=2,
                          capsize=4, color=cor, label=arranjo, zorder=3)
            eixo.annotate(arranjo, (xs[-1], medias[-1]), textcoords="offset points",
                          xytext=(9, 0), color=cor, fontsize=9.5, weight="bold",
                          va="center")
        eixo.set(title=titulos[protocolo], xlabel="partições (N)",
                 xticks=range(len(ordem)), xlim=(-0.3, len(ordem) - 0.45))
        eixo.set_xticklabels([str(n) for n in ordem])
        eixo.grid(axis="y")
    eixos[0].set_ylabel("Acurácia global (OA)")
    fig.suptitle("Particionar a cena só ajuda quando o protocolo permite interpolar entre vizinhos",
                 fontsize=12.5, weight="bold", y=1.03)
    salvar(fig, "01_atalho_espacial")


# --------------------------------------------------------------------------- 2
def fig_paralelismo() -> None:
    """Speedup por executor, com a linha do sequencial como referência."""
    medido = {"thread": {4: 0.58, 8: 0.26}, "process": {4: 1.49, 8: 1.54}}
    fig, eixo = plt.subplots(figsize=(7.2, 4.2))
    largura, posicoes = 0.34, np.arange(2)
    for deslocamento, (nome, cor) in zip((-largura / 2, largura / 2),
                                         (("thread", LARANJA), ("process", AZUL))):
        valores = [medido[nome][n] for n in (4, 8)]
        barras = eixo.bar(posicoes + deslocamento, valores, largura, color=cor, label=nome)
        for barra, valor in zip(barras, valores):
            eixo.text(barra.get_x() + barra.get_width() / 2, valor + 0.04,
                      f"{valor:.2f}×", ha="center", fontsize=10, weight="bold", color=cor)
    eixo.axhline(1.0, color=TINTA, lw=1.4, ls="--")
    eixo.text(1.46, 1.03, "execução sequencial das mesmas partições", fontsize=8.5,
              color="#5f6368", ha="right")
    eixo.set(xticks=posicoes, xticklabels=["4 partições", "8 partições"],
             ylabel="Speedup real", ylim=(0, 1.85),
             title="Threads tornam o treino mais lento; só processos aceleram")
    eixo.legend(loc="upper left")
    eixo.grid(axis="y")
    eixo.text(0.5, 0.06, "abaixo da linha = mais lento que não paralelizar",
              transform=eixo.transAxes, ha="center", fontsize=8.5, color="#5f6368")
    salvar(fig, "02_paralelismo_executores")


# --------------------------------------------------------------------------- 3
def fig_vazamento_fusao() -> None:
    """Ganho da fusão em cada protocolo, por segmentador."""
    d = ler("09_validacao_regiao.csv")
    condicoes = ["seeds", "lsc", "slic", "watershed"]
    fig, eixo = plt.subplots(figsize=(8.4, 4.4))
    largura, posicoes = 0.36, np.arange(len(condicoes))
    for deslocamento, (protocolo, cor, rotulo) in zip(
            (-largura / 2, largura / 2),
            (("pixel", LARANJA, "split por pixel (vazamento 100%)"),
             ("region", AZUL, "split por região (vazamento 0%)"))):
        sub = d[d["protocolo"] == protocolo]
        piso = sub[sub["condição"] == "sem fusão"]["OA"].mean()
        ganhos = [100 * (sub[sub["condição"] == c]["OA"].mean() - piso) for c in condicoes]
        barras = eixo.bar(posicoes + deslocamento, ganhos, largura, color=cor, label=rotulo)
        for barra, valor in zip(barras, ganhos):
            eixo.text(barra.get_x() + barra.get_width() / 2,
                      valor + (0.6 if valor >= 0 else -1.6),
                      f"{valor:+.1f}", ha="center", fontsize=9, weight="bold", color=cor)
    eixo.axhline(0, color=TINTA, lw=1.2)
    eixo.set(xticks=posicoes, xticklabels=condicoes,
             ylabel="Ganho de OA sobre “sem fusão” (pontos percentuais)",
             title="O ganho da fusão sob split por região não é comparável entre segmentadores")
    eixo.legend(loc="upper left")
    eixo.grid(axis="y")
    fig.text(0.5, -0.06,
             "Atenção: sob split por região a partição depende do segmentador avaliado — as barras "
             "azuis medem\ntestes de dificuldades diferentes (distância média ao treino: seeds "
             "4,24 px, watershed 2,06 px).",
             ha="center", fontsize=8.5, color="#5f6368")
    salvar(fig, "03_vazamento_fusao")


# --------------------------------------------------------------------------- 4
def fig_impacto_hpo() -> None:
    """Receita herdada contra receita otimizada, por condição."""
    d = ler("10_impacto_hpo.csv")
    condicoes = ["sem fusão", "seeds", "lsc", "slic", "watershed"]
    fig, eixo = plt.subplots(figsize=(8.8, 4.4))
    largura, posicoes = 0.36, np.arange(len(condicoes))
    for deslocamento, (receita, cor) in zip((-largura / 2, largura / 2),
                                            (("padrão", CINZA), ("otimizada", AZUL))):
        medias = [d[(d["condição"] == c) & (d["receita"] == receita)]["OA"].mean()
                  for c in condicoes]
        desvios = [d[(d["condição"] == c) & (d["receita"] == receita)]["OA"].std()
                   for c in condicoes]
        eixo.bar(posicoes + deslocamento, medias, largura, yerr=desvios, capsize=4,
                 color=cor, label=f"receita {receita}",
                 error_kw={"ecolor": TINTA, "alpha": 0.6, "lw": 1.2})
    for posicao, condicao in zip(posicoes, condicoes):
        padrao = d[(d["condição"] == condicao) & (d["receita"] == "padrão")]["OA"].mean()
        otimizada = d[(d["condição"] == condicao) & (d["receita"] == "otimizada")]["OA"].mean()
        ganho = 100 * (otimizada - padrao)
        eixo.text(posicao, max(padrao, otimizada) + 0.16, f"{ganho:+.1f} pp", ha="center",
                  fontsize=9.5, weight="bold",
                  color=VERDE if ganho > 0 else LARANJA)
    eixo.set(xticks=posicoes, xticklabels=condicoes, ylabel="Acurácia global (OA)",
             ylim=(0, 1.12),
             title="A otimização de hiperparâmetros só compensa quando há fusão")
    eixo.legend(loc="lower right")
    eixo.grid(axis="y")
    fig.text(0.5, -0.04,
             "Barras de erro: ±1 desvio sobre 5 sementes. Watershed e SLIC têm desvio grande "
             "demais\npara que a diferença seja afirmável — ver a maldição do vencedor.",
             ha="center", fontsize=8.5, color="#5f6368")
    salvar(fig, "04_impacto_hpo")


# --------------------------------------------------------------------------- 5
def fig_eixos_hpo() -> None:
    """Quanto cada eixo do espaço de busca separa os resultados."""
    d = ler("08_trials.csv")
    eixos_busca = ["dropout", "learning rate", "PCA", "weight decay", "batch"]
    padrao = {"learning rate": 0.001, "dropout": 0.5, "weight decay": 0.0,
              "batch": 128, "PCA": 30}
    dados = []
    for nome in eixos_busca:
        medias = d.groupby(nome)["val_accuracy (objetivo)"].mean()
        dados.append((nome, 100 * (medias.max() - medias.min()),
                      list(medias.sort_values(ascending=False).index).index(padrao[nome]) + 1,
                      len(medias)))
    dados.sort(key=lambda linha: linha[1])
    fig, eixo = plt.subplots(figsize=(8.2, 3.8))
    posicoes = np.arange(len(dados))
    cores = [LARANJA if posicao == total else AZUL for _, _, posicao, total in dados]
    eixo.barh(posicoes, [linha[1] for linha in dados], color=cores, height=0.6)
    for posicao, (nome, amplitude, lugar, total) in zip(posicoes, dados):
        eixo.text(amplitude + 0.4, posicao, f"{amplitude:.1f} pp", va="center",
                  fontsize=9.5, weight="bold")
        eixo.text(amplitude + 3.2, posicao, f"padrão {lugar}º/{total}", va="center",
                  fontsize=8.5, color=LARANJA if lugar == total else "#5f6368")
    eixo.set(yticks=posicoes, yticklabels=[linha[0] for linha in dados],
             xlabel="Amplitude da validação média entre o melhor e o pior valor (pp)",
             xlim=(0, 26),
             title="Dropout domina — e é o único eixo em que a receita herdada usa o pior valor")
    eixo.grid(axis="x")
    fig.text(0.5, -0.05, "540 trials em Indian Pines, com o LSC já determinístico. Laranja = a "
             "receita herdada está no pior valor daquele eixo.\nNos demais ela acerta: batch 128, "
             "weight decay 0 e PCA 30 são os melhores valores de seus eixos.",
             ha="center", fontsize=8.5, color="#5f6368")
    salvar(fig, "05_eixos_hiperparametros")


# --------------------------------------------------------------------------- 6
def fig_segmentadores() -> None:
    """Desempenho de cada segmentador nas três cenas, sob otimização conjunta."""
    d = ler("06_resumo.csv")
    cenas = ["IP", "PU", "SA"]
    nomes = {"IP": "Indian Pines", "PU": "Pavia University", "SA": "Salinas"}
    condicoes = ["seeds", "slic", "lsc", "watershed"]
    fig, eixos = plt.subplots(1, 3, figsize=(12.4, 4), sharey=False)
    for eixo, cena in zip(eixos, cenas):
        sub = d[d["cena"] == cena]
        piso = sub[sub["condição"] == "sem fusão"]["OA"].iloc[0]
        ganhos = [100 * (sub[sub["condição"] == c]["OA"].iloc[0] - piso) for c in condicoes]
        barras = eixo.bar(condicoes, ganhos, color=[COR_CONDICAO[c] for c in condicoes])
        for barra, valor in zip(barras, ganhos):
            eixo.text(barra.get_x() + barra.get_width() / 2,
                      valor + (0.3 if valor >= 0 else -1.0),
                      f"{valor:+.1f}", ha="center", fontsize=9.5, weight="bold")
        eixo.axhline(0, color=TINTA, lw=1.2)
        eixo.set(title=f"{nomes[cena]}\npiso sem fusão: {piso:.4f}")
        eixo.tick_params(axis="x", rotation=20)
        eixo.grid(axis="y")
    eixos[0].set_ylabel("Ganho de OA sobre “sem fusão” (pp)")
    fig.suptitle("Não existe um melhor segmentador: o vencedor muda por cena",
                 fontsize=12.5, weight="bold", y=1.02)
    salvar(fig, "06_segmentadores_por_cena")


if __name__ == "__main__":
    print("Gerando figuras em experimentos/figuras/")
    for funcao in (fig_atalho_espacial, fig_paralelismo, fig_vazamento_fusao,
                   fig_impacto_hpo, fig_eixos_hpo, fig_segmentadores):
        try:
            funcao()
        except Exception as erro:
            print(f"  FALHOU {funcao.__name__}: {type(erro).__name__}: {erro}")
