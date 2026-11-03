"""Figuras da monografia, geradas dos CSVs de `experimentos/resultados/`.

Uso:  .venv/bin/python -m hsi.visualizacao.monografia

Saem em PDF (vetorial, para o LaTeX) e PNG em
`tcc/TCC III - EDUARDO SAVIAN/src/assets/resultados/`, ao lado das figuras do
projeto, e o `main.tex` as inclui como `assets/resultados/...`.

DIFERENÇAS PARA AS FIGURAS DO NOTEBOOK. Na monografia o título e a fonte vão na
legenda do LaTeX, como pede a ABNT, e não dentro da imagem; por isso nada aqui
escreve título. O resto do sistema visual das 28 figuras de 08/10 (commit `614b31d`)
continua: paleta validada para daltonismo, uma cor fixa por entidade, cores
semânticas reservadas, grade recessiva, rótulo direto no dado e vírgula decimal.

TODA COMPARAÇÃO É CONTRA A CONFIGURAÇÃO DO SYDNEY (CNN 1D com fusão SEEDS e a
receita herdada), como no texto. Nenhuma figura usa um braço sem fusão como piso.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from hsi.config import repo_root  # noqa: E402
from hsi.experimentos import medicoes as M  # noqa: E402

RESULTADOS = repo_root() / "experimentos" / "resultados"
DESTINO = repo_root() / "tcc" / "TCC III - EDUARDO SAVIAN" / "src" / "assets" / "resultados"

# Paleta categórica validada (separação ≥ 8 ΔE em OKLab para deutan/protan/tritan).
AZUL, LARANJA, AGUA, AMARELO, VIOLETA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
# Semânticas — reservadas, nunca usadas como categoria.
GANHO, PERDA, NEUTRO = "#1baf7a", "#d1495b", "#9aa0a6"
TINTA, TINTA2, FRACO, GRADE, LINHA = "#15181c", "#454b52", "#6b7280", "#e8eaed", "#c9ced4"

COR_CONDICAO = {"seeds": AZUL, "slic": LARANJA, "lsc": AGUA, "watershed": AMARELO}
COR_ESTRATEGIA = {"random": NEUTRO, "tpe": AZUL, "genetic": LARANJA}
COR_CENA = {"IP": AZUL, "PU": LARANJA, "SA": AGUA}
NOME_CENA = {"IP": "Indian Pines", "PU": "University of Pavia", "SA": "Salinas Valley"}

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 220, "savefig.bbox": "tight",
    # Fontes embutidas como Type 42: o PDF da figura não depende das fontes da
    # máquina que compila a monografia.
    "pdf.fonttype": 42,
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 10.5, "axes.labelsize": 9.5, "axes.labelcolor": TINTA2,
    "axes.edgecolor": LINHA, "axes.linewidth": 0.9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "grid.color": GRADE, "grid.linewidth": 0.9,
    "xtick.color": FRACO, "ytick.color": FRACO,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "xtick.major.size": 0, "ytick.major.size": 0,
    "legend.frameon": False, "legend.fontsize": 9,
    "text.color": TINTA, "lines.solid_capstyle": "round",
})


def num(valor: float, casas: int = 2) -> str:
    """Número em português, com sinal de menos tipográfico: 0.8981 → "0,8981"."""
    texto = f"{valor:,.{casas}f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")
    return texto.replace("-", "−")


def virgula(eixo, qual: str = "y", casas: int = 1) -> None:
    formatador = FuncFormatter(lambda valor, _: num(valor, casas))
    (eixo.yaxis if qual == "y" else eixo.xaxis).set_major_formatter(formatador)


def rotulo_valor(valor, potencia: bool = False) -> str:
    """Valor de hiperparâmetro como rótulo: 0,0003 em vez de 0.0003; com `potencia`, 10⁻⁵."""
    if potencia and isinstance(valor, float) and valor != 0:
        expoente = int(np.floor(np.log10(abs(valor))))
        mantissa = valor / 10 ** expoente
        potencia = f"10$^{{{expoente}}}$"
        return potencia if np.isclose(mantissa, 1) else f"{num(mantissa, 0)}×{potencia}"
    if isinstance(valor, float):
        return num(valor, 4).rstrip("0").rstrip(",") or "0"
    return str(valor)


def limpar(eixo, grade: str = "y") -> None:
    if grade:
        eixo.grid(axis=grade, zorder=0)
        eixo.set_axisbelow(True)


def legendar(eixo, ncol: int = 3, y: float = 1.02) -> None:
    """Legenda acima do painel, horizontal — nunca sobre o dado."""
    eixo.legend(ncol=ncol, loc="lower left", bbox_to_anchor=(0.0, y), frameon=False,
                borderaxespad=0.0, handlelength=1.4, columnspacing=1.6)


def rotular_barras(eixo, barras, valores, casas=2, sufixo="", cor=None, deslocamento=0.0,
                   fonte=8.5, erros=None, fundo=False) -> None:
    """Valor acima de cada barra — e acima da barra de erro, quando `erros` é dado."""
    erros = erros if erros is not None else [0.0] * len(valores)
    for barra, valor, erro in zip(barras, valores, erros):
        altura = barra.get_height() + (erro if barra.get_height() >= 0 else -erro)
        eixo.annotate(f"{num(valor, casas)}{sufixo}",
                      (barra.get_x() + barra.get_width() / 2, altura),
                      textcoords="offset points",
                      xytext=(0, 3 + deslocamento if altura >= 0 else -11 - deslocamento),
                      ha="center", fontsize=fonte, weight="bold", color=cor or TINTA, zorder=5,
                      bbox=dict(fc="white", ec="none", pad=0.6) if fundo else None)


def salvar(fig, nome: str) -> None:
    DESTINO.mkdir(parents=True, exist_ok=True)
    for extensao in ("pdf", "png"):
        fig.savefig(DESTINO / f"{nome}.{extensao}")
    plt.close(fig)
    print(f"  {nome}")


def ler(nome: str) -> pd.DataFrame:
    return pd.read_csv(RESULTADOS / nome)


# ═════════════════════════════════════════════════════════ pilar 2 — paralelismo

def f_ramos():
    """Tempo de cada ramo do pré-processamento e o teto do paralelismo entre eles."""
    cenas = ["IP", "PU", "SA"]
    fig, eixos = plt.subplots(1, 2, figsize=(7.6, 3.2), gridspec_kw={"width_ratios": [1.1, 1]})
    posicoes = np.arange(len(cenas))
    largura = 0.36
    for deslocamento, ramo, cor in ((-largura / 2, "espectral", LARANJA),
                                    (largura / 2, "espacial", AZUL)):
        valores = [M.RAMOS[c][ramo] for c in cenas]
        barras = eixos[0].bar(posicoes + deslocamento, valores, largura, color=cor,
                              label=f"ramo {ramo}" + (" (PCA)" if ramo == "espectral"
                                                       else " (PCA(3) + SEEDS)"), zorder=3)
        rotular_barras(eixos[0], barras, valores, 3, "", cor, fonte=8.5)
    em_duas_linhas = [NOME_CENA[c].replace(" ", "\n", 1) if c != "PU" else "University\nof Pavia"
                      for c in cenas]
    eixos[0].set(xticks=posicoes, xticklabels=em_duas_linhas,
                 ylabel="tempo (s)", ylim=(0, 1.02))
    limpar(eixos[0]); virgula(eixos[0], casas=1); legendar(eixos[0], 1)

    teto = [(M.RAMOS[c]["espectral"] + M.RAMOS[c]["espacial"])
            / max(M.RAMOS[c]["espectral"], M.RAMOS[c]["espacial"]) for c in cenas]
    medido = {b: [M.RAMOS[c]["sequential"] / M.RAMOS[c][b] for c in cenas]
              for b in ("thread", "process")}
    for deslocamento, (rotulo, valores, cor) in zip(
            (-largura / 2, largura / 2),
            (("threads", medido["thread"], AZUL), ("processos", medido["process"], PERDA))):
        barras = eixos[1].bar(posicoes + deslocamento, valores, largura, color=cor,
                              label=rotulo, zorder=3)
        rotular_barras(eixos[1], barras, valores, 2, "×", cor, fonte=8.5)
    for posicao, valor in zip(posicoes, teto):
        eixos[1].plot([posicao - largura, posicao + largura], [valor, valor], color=TINTA,
                      lw=1.6, ls=(0, (3, 2)), zorder=4)
        eixos[1].annotate(f"teto\n{num(valor, 2)}×", (posicao + largura, valor),
                          textcoords="offset points", xytext=(3, 0), ha="left",
                          va="center", fontsize=8, color=TINTA2, linespacing=0.95)
    eixos[1].axhline(1.0, color=LINHA, lw=1.0, zorder=2)
    eixos[1].set(xticks=posicoes, xticklabels=em_duas_linhas,
                 ylabel="speedup sobre o sequencial", ylim=(0, 1.45))
    limpar(eixos[1]); virgula(eixos[1], casas=1); legendar(eixos[1], 1)
    fig.tight_layout(w_pad=1.5)
    salvar(fig, "ramos_preprocessamento")


def f_pool_quente():
    """Speedup da partição da cena: pool frio, pool quente e threads."""
    d = ler("13_pool_quente.csv")
    d = d[(d["cena"] == "IP") & (d["modo"] == "independente")]
    ns = sorted(d["n"].unique())
    fig, eixo = plt.subplots(figsize=(6.2, 3.2))
    series = (("process", "quente", "processos, pool quente", AZUL, "o"),
              ("process", "frio", "processos, pool frio", LARANJA, "s"),
              ("thread", "quente", "threads", PERDA, "^"))
    for executor, estado, rotulo, cor, marcador in series:
        xs, ys = [], []
        for n in ns:
            base = d[(d["executor"] == "sequencial") & (d["n"] == n) & (d["estado"] == estado)]
            cel = d[(d["executor"] == executor) & (d["n"] == n) & (d["estado"] == estado)]
            if base.empty or cel.empty:
                continue
            xs.append(n)
            ys.append(base["bloco_paralelo"].mean() / cel["bloco_paralelo"].mean())
        eixo.plot(xs, ys, marker=marcador, ms=7, lw=2.2, color=cor, label=rotulo, zorder=3)
        eixo.annotate(f"{num(ys[-1], 2)}×", (xs[-1], ys[-1]), textcoords="offset points",
                      xytext=(7, 4), va="bottom", fontsize=9, weight="bold", color=cor)
    eixo.plot(ns, ns, color=LINHA, lw=1.2, ls=(0, (3, 2)), zorder=2)
    eixo.annotate("linear", (ns[2], ns[2]), textcoords="offset points", xytext=(-34, 6),
                  fontsize=9, color=FRACO, rotation=0)
    eixo.axhline(1.0, color=TINTA, lw=1.0, ls="--", zorder=2)
    eixo.set(xscale="log", yscale="log", xticks=ns, xticklabels=[str(n) for n in ns],
             xlabel="partições (N)", ylabel="speedup sobre o sequencial", ylim=(0.15, 20))
    eixo.set_yticks([0.25, 0.5, 1, 2, 4, 8, 16])
    eixo.set_yticklabels(["0,25", "0,5", "1", "2", "4", "8", "16"])
    eixo.minorticks_off()
    limpar(eixo); legendar(eixo, 3)
    salvar(fig, "pool_quente")


def f_fusao_linha_do_tempo(cena: str = "PU"):
    """O desenho do orientador em ação: o que cada thread faz enquanto T0 segmenta.

    University of Pavia por padrão: é onde a segmentação é longa (0,87 s) e a
    sobreposição — ou a falta dela — fica visível.
    """
    arranjos = [("concorrente", "como no esquema (convolução e fusão na mesma tarefa)"),
                ("desacoplado", "variante desacoplada (todas as convoluções primeiro)")]
    quadros = {a: ler(f"14_linha_do_tempo_{cena}_{a}.csv") for a, _ in arranjos}
    fim = max(q["fim"].max() for q in quadros.values())
    fig, eixos = plt.subplots(2, 1, figsize=(7.4, 4.4), sharex=True)
    for eixo, (arranjo, titulo) in zip(eixos, arranjos):
        q = quadros[arranjo]
        threads = sorted(q["thread"].unique(), key=lambda t: int(t.rsplit("_", 1)[-1]))
        linhas = {"T0": 0} | {t: i + 1 for i, t in enumerate(threads)}
        t0_inicio, t0_fim = q["t0_inicio"].iloc[0], q["t0_fim"].iloc[0]
        eixo.barh(0, t0_fim - t0_inicio, left=t0_inicio, height=0.6, color=VIOLETA, zorder=3)
        eixo.annotate("SEEDS", (t0_inicio + (t0_fim - t0_inicio) / 2, 0), ha="center",
                      va="center", fontsize=8, color="white", weight="bold")
        for linha in q.itertuples():
            y = linhas[linha.thread]
            if arranjo == "concorrente":
                eixo.barh(y, linha.fim_convolucoes - linha.inicio, left=linha.inicio,
                          height=0.6, color=AZUL, zorder=3, lw=0)
            eixo.barh(y, linha.fim - linha.pegou_mapa, left=linha.pegou_mapa, height=0.6,
                      color=LARANJA, zorder=3, lw=0)
        if arranjo == "desacoplado":
            # As convoluções da variante desacoplada não ficam numa thread fixa: a
            # linha do tempo grava a thread da fusão. O que importa é *quando* elas
            # terminaram — a faixa azul acima da T0 mostra isso.
            eixo.barh(len(linhas), q["fim_convolucoes"].max() - q["inicio"].min(),
                      left=q["inicio"].min(), height=0.6, color=AZUL, zorder=3)
            eixo.annotate("todas as convoluções",
                          (q["fim_convolucoes"].max(), len(linhas)),
                          textcoords="offset points", xytext=(6, 0), va="center",
                          fontsize=8, color=AZUL, weight="bold")
            linhas = linhas | {"conv.": len(linhas)}
        eixo.axvline(t0_fim, color=TINTA, lw=1.1, ls="--", zorder=4)
        # As threads do pool se chamam T_0, T_1…; no esquema do orientador elas são
        # T1…TN, e T0 é a da segmentação.
        eixo.set_yticks(list(linhas.values()))
        eixo.set_yticklabels([f"T{int(n.rsplit('_', 1)[-1]) + 1}" if n.startswith("T_") else n
                              for n in linhas])
        eixo.invert_yaxis()
        eixo.set_title(titulo, loc="left", color=TINTA2)
        limpar(eixo, "x")
    eixos[0].annotate("mapa publicado", (quadros["concorrente"]["t0_fim"].iloc[0], 0),
                      textcoords="offset points", xytext=(4, 0), va="center",
                      fontsize=8, color=TINTA)
    eixos[-1].set(xlabel="tempo desde o início (s)", xlim=(0, fim * 1.02))
    virgula(eixos[-1], "x", 2)
    from matplotlib.patches import Patch
    eixos[0].legend(handles=[Patch(color=VIOLETA, label="T0: PCA(3) + SEEDS"),
                             Patch(color=AZUL, label="convoluções"),
                             Patch(color=LARANJA, label="leitura do mapa, concatenação e densas")],
                    ncol=3, loc="lower left", bbox_to_anchor=(0, 1.18), frameon=False)
    fig.tight_layout(h_pad=1.2)
    salvar(fig, "fusao_linha_do_tempo")


def f_fusao_speedup():
    """Speedup do desenho contra o lote, por número de threads, nas três cenas."""
    d = ler("14_fusao_concorrente.csv")
    cenas = [c for c in ("IP", "PU", "SA") if c in set(d["cena"])]
    fig, eixos = plt.subplots(1, len(cenas), figsize=(2.55 * len(cenas), 3.2), sharey=True,
                              squeeze=False)
    # Seção crítica "como no esquema" (a do desenho do orientador), a mesma do quadro.
    series = (("concorrente", 1024, "como no esquema, bloco 1.024", AZUL, "o", "-"),
              ("desacoplado", 1024, "desacoplada, bloco 1.024", AGUA, "s", "-"),
              ("concorrente", 64, "como no esquema, bloco 64", LARANJA, "^", "-"),
              ("concorrente", 1, "como no esquema, bloco 1", PERDA, "v", "-"))
    for eixo, cena in zip(eixos[0], cenas):
        sub = d[d["cena"] == cena]
        lote = sub[(sub["arranjo"] == "lote") & (sub["torch_threads"] == 1)]["parede"].mean()
        lote16 = sub[(sub["arranjo"] == "lote") & (sub["torch_threads"] == 16)]["parede"].mean()
        for arranjo, bloco, rotulo, cor, marcador, estilo in series:
            cel = sub[(sub["arranjo"] == arranjo) & (sub["bloco"] == bloco)
                      & (sub["secao"] == "desenho")]
            if cel.empty:
                continue
            serie = cel.groupby("n")["parede"].mean()
            eixo.plot(serie.index, lote / serie.values, marker=marcador, ms=6, lw=2,
                      color=cor, ls=estilo, label=rotulo, zorder=3)
        segmentacao = sub[sub["arranjo"] == "sequencial"]["segmentacao"].mean()
        teto = lote / segmentacao
        eixo.axhline(1.0, color=TINTA, lw=1.1, ls="--", zorder=2)
        eixo.axhline(teto, color=NEUTRO, lw=1.4, ls=(0, (4, 2)), zorder=2,
                     label="teto (lote ÷ segmentação)")
        eixo.axhline(lote / lote16, color=VIOLETA, lw=1.4, ls=(0, (1, 2)), zorder=2,
                     label="lote, 16 threads do PyTorch")
        eixo.set(xscale="log", xticks=[1, 2, 4, 8, 16], xticklabels=["1", "2", "4", "8", "16"],
                 xlabel="threads de trabalho (N)", title=NOME_CENA[cena], yscale="log")
        eixo.minorticks_off()
        limpar(eixo)
    eixos[0][0].set_ylabel("speedup sobre o lote (1 thread)")
    eixos[0][0].set_yticks([0.02, 0.05, 0.1, 0.25, 0.5, 1, 2])
    eixos[0][0].set_yticklabels(["0,02", "0,05", "0,1", "0,25", "0,5", "1", "2"])
    eixos[0][0].legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.12), frameon=False,
                       fontsize=8.5, handlelength=2.2, columnspacing=1.2)
    fig.tight_layout(w_pad=1.2)
    salvar(fig, "fusao_speedup")


# ═════════════════════════════════════════ pilar 3 — automação de hiperparâmetros

def f_estrategias():
    """Melhor validação por estratégia e orçamento de épocas, com o desvio entre sementes."""
    t = ler("08_trials.csv")
    melhor = t.groupby(["orçamento", "estratégia", "semente"])["val_accuracy (objetivo)"].max()
    # Desvio populacional, como os experimentos imprimem e os CSVs de resumo gravam.
    resumo = melhor.groupby(["orçamento", "estratégia"]).agg(
        mean="mean", std=lambda x: np.std(x, ddof=0))
    orcamentos = sorted(t["orçamento"].unique())
    fig, eixo = plt.subplots(figsize=(6.2, 3.1))
    posicoes = np.arange(len(orcamentos))
    largura = 0.26
    for k, estrategia in enumerate(("random", "tpe", "genetic")):
        medias = [resumo.loc[(o, estrategia), "mean"] for o in orcamentos]
        desvios = [resumo.loc[(o, estrategia), "std"] for o in orcamentos]
        barras = eixo.bar(posicoes + (k - 1) * largura, medias, largura,
                          yerr=desvios, capsize=3, error_kw={"lw": 1, "color": TINTA2},
                          color=COR_ESTRATEGIA[estrategia],
                          label={"random": "aleatória", "tpe": "TPE (bayesiana)",
                                 "genetic": "genética"}[estrategia], zorder=3)
        rotular_barras(eixo, barras, medias, 3, fonte=8.5, erros=desvios)
    eixo.set(xticks=posicoes, xticklabels=[f"{o} épocas por trial" for o in orcamentos],
             ylabel="melhor acurácia de validação", ylim=(0.6, 0.92))
    limpar(eixo); virgula(eixo, casas=2); legendar(eixo, 3)
    salvar(fig, "estrategias_hpo")


def f_eixos():
    """Validação média por valor de cada eixo, com a receita herdada marcada."""
    t = ler("08_trials.csv")
    # Taxa de aprendizado e weight decay em unidades de 10⁻³: rótulos curtos e uma
    # notação só por eixo.
    eixos_hpo = [("dropout", "dropout", 0.5, 1),
                 ("learning rate", "taxa de aprendizado\n(×10$^{-3}$)", 0.001, 1e3),
                 ("PCA", "componentes\ndo PCA", 30, 1),
                 ("weight decay", "weight decay\n(×10$^{-3}$)", 0.0, 1e3),
                 ("batch", "tamanho\ndo lote", 32, 1)]
    fig, eixos = plt.subplots(1, len(eixos_hpo), figsize=(7.8, 3.4), sharey=True)
    for eixo, (coluna, rotulo, herdado, fator) in zip(eixos, eixos_hpo):
        medias = t.groupby(coluna)["val_accuracy (objetivo)"].mean()
        cores = [PERDA if np.isclose(v, herdado) else NEUTRO for v in medias.index]
        rotulos = [rotulo_valor(v * fator) if isinstance(v, float) else str(v)
                   for v in medias.index]
        barras = eixo.bar(range(len(medias)), medias.values, 0.6, color=cores, zorder=3)
        rotular_barras(eixo, barras, medias.values, 2, fonte=8.5)
        amplitude = 100 * (medias.max() - medias.min())
        eixo.set(xticks=range(len(medias)), xticklabels=rotulos, ylim=(0.4, 0.75))
        eixo.set_title(f"{rotulo}\n({num(amplitude, 1)} pp)", color=TINTA2, fontsize=9.5)
        limpar(eixo); virgula(eixo, casas=2)
    eixos[0].set_ylabel("validação média dos trials")
    from matplotlib.patches import Patch
    eixos[0].legend(handles=[Patch(color=PERDA, label="valor da receita herdada"),
                             Patch(color=NEUTRO, label="outros valores do espaço")],
                    ncol=2, loc="lower left", bbox_to_anchor=(0, 1.34), frameon=False)
    fig.tight_layout(w_pad=0.8)
    salvar(fig, "eixos_hpo")


def f_impacto_hpo():
    """Receita herdada contra receita otimizada, por segmentador, em Indian Pines."""
    d = ler("10_impacto_hpo.csv")
    d = d[d["cena"] == "IP"]
    condicoes = ["seeds", "slic", "lsc", "watershed"]
    receitas = [("herdada", "receita herdada (lote 32)", NEUTRO),
                ("otimizada", "receita otimizada pela busca", AZUL)]
    fig, eixo = plt.subplots(figsize=(6.2, 3.1))
    posicoes = np.arange(len(condicoes))
    largura = 0.36
    for k, (receita, rotulo, cor) in enumerate(receitas):
        grupo = d[d["receita"] == receita].groupby("condição")["OA"]
        medias = [grupo.mean().get(c, np.nan) for c in condicoes]
        desvios = [grupo.std(ddof=0).get(c, np.nan) for c in condicoes]
        barras = eixo.bar(posicoes + (k - 0.5) * largura, medias, largura, yerr=desvios,
                          capsize=3, error_kw={"lw": 1, "color": TINTA2}, color=cor,
                          label=rotulo, zorder=3)
        rotular_barras(eixo, barras, medias, 3, fonte=8.5, erros=desvios)
    eixo.set(xticks=posicoes, xticklabels=["SEEDS (herdado)", "SLIC", "LSC", "Watershed"],
             ylabel="OA de teste", ylim=(0, 1.05))
    limpar(eixo); virgula(eixo, casas=1); legendar(eixo, 2)
    salvar(fig, "impacto_hpo")


# ═══════════════════════════════════════════════════════ pilar 4 — segmentadores

def f_segmentadores():
    """OA de cada segmentador otimizado para si, por cena, com a referência do SEEDS."""
    r = ler("06_resumo.csv")
    cenas = ["IP", "PU", "SA"]
    condicoes = ["seeds", "slic", "lsc", "watershed"]
    fig, eixos = plt.subplots(1, 3, figsize=(7.8, 3.1))
    for eixo, cena in zip(eixos, cenas):
        sub = r[r["cena"] == cena].set_index("condição")
        medias = [sub.loc[c, "OA"] for c in condicoes]
        desvios = [sub.loc[c, "OA_std"] for c in condicoes]
        barras = eixo.bar(range(4), medias, 0.62, yerr=desvios, capsize=3,
                          error_kw={"lw": 1, "color": TINTA2},
                          color=[COR_CONDICAO[c] for c in condicoes], zorder=3)
        rotular_barras(eixo, barras, medias, 3, fonte=8, erros=desvios, fundo=True)
        eixo.axhline(sub.loc["seeds", "OA"], color=AZUL, lw=1, ls="--", zorder=2)
        # IP tem desvios de até 0,23; PU e SA cabem numa faixa de 10 pontos.
        limites, marcas = (((0.0, 1.1), np.arange(0, 1.01, 0.2)) if cena == "IP"
                           else ((0.90, 1.0), np.arange(0.90, 1.001, 0.02)))
        eixo.set(xticks=range(4), xticklabels=["SEEDS", "SLIC", "LSC*", "WS"],
                 ylim=limites, yticks=marcas, title=NOME_CENA[cena])
        limpar(eixo); virgula(eixo, casas=2)
    eixos[0].set_ylabel("OA de teste (5 sementes)")
    fig.tight_layout(w_pad=1.5)
    salvar(fig, "segmentadores")


FIGURAS = (f_ramos, f_pool_quente, f_fusao_linha_do_tempo, f_fusao_speedup, f_estrategias,
           f_eixos, f_impacto_hpo, f_segmentadores)


def gerar(*nomes: str) -> None:
    """Gera todas as figuras, ou só as pedidas pelo nome da função (`f_ramos`, ...)."""
    for figura in FIGURAS:
        if not nomes or figura.__name__ in nomes:
            figura()


if __name__ == "__main__":
    import sys

    print(f"figuras em {DESTINO}")
    gerar(*sys.argv[1:])
