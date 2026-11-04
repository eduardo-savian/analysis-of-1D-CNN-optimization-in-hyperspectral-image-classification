"""Figuras da monografia, a partir dos CSVs de experimentos/resultados/.

    .venv/bin/python -m hsi.figuras             todas
    .venv/bin/python -m hsi.figuras f_eixos     só uma

Cada figura sai em PDF, que o LaTeX inclui, e em PNG, na pasta assets/resultados/ da
monografia. Título e fonte ficam na legenda do LaTeX, como pede a ABNT, então as
figuras não têm título.
"""

from functools import cache

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from hsi.ambiente import configure_determinism  # noqa: E402
from hsi.config import RAIZ, RESULTADOS, SEGMENTERS, ExperimentConfig  # noqa: E402
from hsi.dados import (SEGMENTER_PARAMS, apply_pca, load_scene, pixel_table,  # noqa: E402
                       prepare_data, rgb_composite, segment)
from hsi.particao import partition_scene  # noqa: E402
from hsi.pipeline import run_experiment  # noqa: E402

DESTINO = RAIZ / "tcc" / "TCC III - EDUARDO SAVIAN" / "src" / "assets" / "resultados"

# Paleta que continua distinguível para daltônicos; cada entidade tem uma cor fixa.
AZUL, LARANJA, AGUA, AMARELO, VIOLETA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
# Cores com significado, fora da paleta de categorias.
PERDA, NEUTRO = "#d1495b", "#9aa0a6"
TINTA, TINTA2, FRACO, GRADE, LINHA = "#15181c", "#454b52", "#6b7280", "#e8eaed", "#c9ced4"

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
    """Legenda em uma linha acima do painel, para não cobrir os dados."""
    eixo.legend(ncol=ncol, loc="lower left", bbox_to_anchor=(0.0, y), frameon=False,
                borderaxespad=0.0, handlelength=1.4, columnspacing=1.6)


def rotular_barras(eixo, barras, valores, casas=2, sufixo="", cor=None, deslocamento=0.0,
                   fonte=8.5, erros=None, fundo=False) -> None:
    """Escreve o valor acima de cada barra, ou acima da barra de erro, se houver."""
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


# ------------------------------------------------------------------ paralelismo

def f_fusao_linha_do_tempo(cena: str = "PU"):
    """O que cada thread faz enquanto T0 segmenta.

    University of Pavia por padrão: a segmentação ali leva 0,87 s, o bastante para
    a sobreposição aparecer.
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
            # Na variante desacoplada a linha do tempo grava a thread da fusão, não a da
            # convolução; a faixa azul mostra só quando as convoluções terminaram.
            eixo.barh(len(linhas), q["fim_convolucoes"].max() - q["inicio"].min(),
                      left=q["inicio"].min(), height=0.6, color=AZUL, zorder=3)
            eixo.annotate("todas as convoluções",
                          (q["fim_convolucoes"].max(), len(linhas)),
                          textcoords="offset points", xytext=(6, 0), va="center",
                          fontsize=8, color=AZUL, weight="bold")
            linhas = linhas | {"conv.": len(linhas)}
        eixo.axvline(t0_fim, color=TINTA, lw=1.1, ls="--", zorder=4)
        # O pool nomeia as threads T_0, T_1...; no esquema elas são T1..TN e T0 é a
        # da segmentação.
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
    eixos[0].legend(handles=[Patch(color=VIOLETA, label="T0: PCA(3) + SEEDS"),
                             Patch(color=AZUL, label="convoluções"),
                             Patch(color=LARANJA, label="leitura do mapa, concatenação e densas")],
                    ncol=3, loc="lower left", bbox_to_anchor=(0, 1.18), frameon=False)
    fig.tight_layout(h_pad=1.2)
    salvar(fig, "fusao_linha_do_tempo")


# ------------------------------------------------------ busca de hiperparâmetros

def f_eixos():
    """Validação média por valor de cada eixo, com a configuração original marcada."""
    t = ler("08_trials.csv")
    # Em ordem de amplitude, como na tabela do apêndice. Taxa de aprendizado e weight
    # decay em unidades de 10⁻³: rótulos curtos e uma notação só por eixo.
    eixos_hpo = [("dropout", "dropout", 0.5, 1),
                 ("learning rate", "taxa de aprendizado (×10$^{-3}$)", 0.001, 1e3),
                 ("segmentador", "segmentador", "seeds", None),
                 ("PCA", "componentes do PCA", 30, 1),
                 ("weight decay", "weight decay (×10$^{-3}$)", 0.0, 1e3),
                 ("batch", "tamanho do lote", 32, 1)]
    fig, eixos = plt.subplots(2, 3, figsize=(7.4, 5.0), sharey=True)
    for eixo, (coluna, rotulo, herdado, fator) in zip(eixos.flat, eixos_hpo):
        medias = t.groupby(coluna)["val_accuracy (objetivo)"].mean()
        if coluna == "segmentador":
            medias = medias.reindex(SEGMENTERS)
            cores = [PERDA if v == herdado else NEUTRO for v in medias.index]
            rotulos = [NOME_SEG[v] for v in medias.index]
            eixo.tick_params(axis="x", labelsize=8)
        else:
            cores = [PERDA if np.isclose(v, herdado) else NEUTRO for v in medias.index]
            rotulos = [rotulo_valor(v * fator) if isinstance(v, float) else str(v)
                       for v in medias.index]
        barras = eixo.bar(range(len(medias)), medias.values, 0.6, color=cores, zorder=3)
        rotular_barras(eixo, barras, medias.values, 4, fonte=7.5)
        amplitude = 100 * (medias.max() - medias.min())
        eixo.set(xticks=range(len(medias)), xticklabels=rotulos, ylim=(0.4, 0.75))
        eixo.set_title(f"{rotulo}\n({num(amplitude, 2)} pp)", color=TINTA2, fontsize=9.5)
        limpar(eixo); virgula(eixo, casas=2)
    for eixo in eixos[:, 0]:
        eixo.set_ylabel("validação média dos trials")
    # Legenda da figura, e não de um painel: ancorada num painel, ela encolhe as colunas.
    fig.legend(handles=[Patch(color=PERDA, label="valor da configuração original"),
                        Patch(color=NEUTRO, label="outros valores do espaço")],
               ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=0.8, h_pad=1.6)
    salvar(fig, "eixos_hpo")


# ------------------------------------------------------- cenas, segmentação e partição

NOME_SEG = {"seeds": "SEEDS", "slic": "SLIC", "lsc": "LSC", "watershed": "Watershed"}
# 20 cores distinguíveis; Indian Pines usa as 16 primeiras, uma por classe.
CORES_CLASSES = plt.get_cmap("tab20").colors


def _rgb_e_gt(cena):
    """Composição RGB que os segmentadores recebem e a verdade terrestre."""
    cubo, gt = load_scene(cena)
    return rgb_composite(apply_pca(pixel_table(cubo, gt)[0], 3, 42), gt.shape), gt


def _com_contornos(rgb, rotulos):
    borda = np.zeros(rotulos.shape, dtype=bool)
    borda[:, 1:] |= rotulos[:, 1:] != rotulos[:, :-1]
    borda[1:, :] |= rotulos[1:, :] != rotulos[:-1, :]
    imagem = rgb.copy()
    imagem[borda] = 255
    return imagem


def _sem_eixos(*eixos):
    for eixo in eixos:
        eixo.set_xticks([]); eixo.set_yticks([])
        for lado in eixo.spines.values():
            lado.set_visible(False)


def f_segmentacoes_ip():
    """A composição RGB de Indian Pines e as regiões de cada segmentador."""
    configure_determinism()
    rgb, _ = _rgb_e_gt("IP")
    fig, eixos = plt.subplots(1, 5, figsize=(7.8, 2.1))
    eixos[0].imshow(rgb)
    eixos[0].set_title("composição RGB\n(PCA de 3 componentes)", fontsize=8.5, color=TINTA2)
    for eixo, seg in zip(eixos[1:], SEGMENTERS):
        rotulos = segment(rgb, seg, SEGMENTER_PARAMS[seg]["IP"])
        eixo.imshow(_com_contornos(rgb, rotulos))
        eixo.set_title(f"{NOME_SEG[seg]}\n{len(np.unique(rotulos))} regiões", fontsize=8.5, color=TINTA2)
    _sem_eixos(*eixos)
    fig.tight_layout(w_pad=0.4)
    salvar(fig, "segmentacoes_ip")


def f_particao_faixas():
    """Os pixels rotulados de Indian Pines divididos em 4 e em 16 faixas."""
    dados, _ = prepare_data(ExperimentConfig(dataset="IP"))
    fig, eixos = plt.subplots(1, 2, figsize=(5.0, 2.7))
    for eixo, n in zip(eixos, (4, 16)):
        grupo = partition_scene(dados["coords"], n)
        imagem = np.ones((*dados["forma"], 3))
        cores = [AZUL, LARANJA, AGUA, AMARELO] if n == 4 else CORES_CLASSES
        for k in range(n):
            linhas, colunas = dados["coords"][grupo == k].T
            imagem[linhas, colunas] = matplotlib.colors.to_rgb(cores[k % len(cores)])
        eixo.imshow(imagem)
        eixo.set_title(f"$N$ = {n}", fontsize=9, color=TINTA2)
    _sem_eixos(*eixos)
    fig.tight_layout(w_pad=1.0)
    salvar(fig, "particao_faixas")


# ------------------------------------------------------------- linha de base

@cache
def _linha_de_base_ip():
    """A configuração original em Indian Pines e a classe prevista de cada pixel rotulado."""
    configure_determinism()
    r = run_experiment(ExperimentConfig(dataset="IP"))
    d, modelo = r["dados"], r["modelo"].eval()
    device = next(modelo.parameters()).device
    with torch.no_grad():
        x = torch.from_numpy(d["espectros"]).float().unsqueeze(1).to(device)
        seg = torch.from_numpy(d["segmentos"]).float().reshape(-1, 1).to(device)
        todos = modelo(x, seg).argmax(dim=1).cpu().numpy()
    return r, todos


def f_curvas_treino():
    """Perda e acurácia por época na linha de base de Indian Pines."""
    r, _ = _linha_de_base_ip()
    h = r["historico"]
    epocas = np.arange(1, len(h["loss"]) + 1)
    fig, (perda, acuracia) = plt.subplots(1, 2, figsize=(7.4, 2.9))
    for eixo, chave, rotulo in ((perda, "loss", "perda"), (acuracia, "accuracy", "acurácia")):
        eixo.plot(epocas, h[chave], color=LARANJA, lw=1.8, label="treino")
        eixo.plot(epocas, h["val_" + chave], color=AZUL, lw=1.8, label="validação")
        eixo.axvline(r["melhor_epoca"], color=TINTA, lw=1, ls="--")
        eixo.set(xlabel="época", ylabel=rotulo)
        limpar(eixo); virgula(eixo, casas=1 if chave == "loss" else 2)
    acuracia.annotate(f"melhor época ({r['melhor_epoca']}),\npesos restaurados",
                      (r["melhor_epoca"], 0.45), textcoords="offset points", xytext=(-6, 0),
                      ha="right", fontsize=8, color=TINTA2)
    legendar(perda, 2)
    fig.tight_layout(w_pad=1.5)
    salvar(fig, "curvas_treino")


def f_mapa_classificacao():
    """Verdade terrestre, classes previstas e acertos no teste, em Indian Pines."""
    r, todos = _linha_de_base_ip()
    d = r["dados"]
    cores = np.array(CORES_CLASSES[:d["n_classes"]])
    linhas, colunas = d["coords"].T
    gt = np.ones((*d["forma"], 3)); gt[linhas, colunas] = cores[d["rotulos"]]
    previsto = np.ones((*d["forma"], 3)); previsto[linhas, colunas] = cores[todos]
    teste = np.ones((*d["forma"], 3))
    acerto = r["previsto"] == d["rotulos"][d["teste"]]
    lt, ct = d["coords"][d["teste"]].T
    teste[lt, ct] = np.where(acerto[:, None], matplotlib.colors.to_rgb(NEUTRO), matplotlib.colors.to_rgb(PERDA))
    fig, eixos = plt.subplots(1, 3, figsize=(7.4, 2.9))
    titulos = ("verdade terrestre", "classes previstas", f"teste: {(~acerto).sum()} erros em {len(acerto)}")
    for eixo, imagem, titulo in zip(eixos, (gt, previsto, teste), titulos):
        eixo.imshow(imagem); eixo.set_title(titulo, fontsize=9, color=TINTA2)
    _sem_eixos(*eixos)
    fig.legend(handles=[Patch(color=cores[k], label=str(k + 1)) for k in range(d["n_classes"])],
               ncol=16, loc="lower center", bbox_to_anchor=(0.36, -0.06), frameon=False,
               handlelength=0.9, columnspacing=0.6, fontsize=7.5, title="classe", title_fontsize=7.5)
    eixos[2].legend(handles=[Patch(color=NEUTRO, label="acerto"), Patch(color=PERDA, label="erro")],
                    ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=7.5)
    fig.tight_layout(w_pad=0.6)
    salvar(fig, "mapa_classificacao")


def f_matriz_confusao():
    """Matriz de confusão do teste da linha de base em Indian Pines."""
    r, _ = _linha_de_base_ip()
    confusao = r["confusao"]
    proporcao = confusao / confusao.sum(axis=1, keepdims=True)
    fig, eixo = plt.subplots(figsize=(5.0, 4.4))
    mapa = matplotlib.colors.LinearSegmentedColormap.from_list("azul", ["#ffffff", AZUL])
    eixo.imshow(proporcao, cmap=mapa, vmin=0, vmax=1)
    for i, j in zip(*np.nonzero(confusao)):
        eixo.text(j, i, str(confusao[i, j]), ha="center", va="center", fontsize=6.5,
                  color="white" if proporcao[i, j] > 0.6 else TINTA)
    marcas = np.arange(len(confusao))
    eixo.set(xticks=marcas, yticks=marcas, xticklabels=marcas + 1, yticklabels=marcas + 1,
             xlabel="classe prevista", ylabel="classe real")
    eixo.tick_params(labelsize=7.5)
    for lado in eixo.spines.values():
        lado.set_visible(False)
    salvar(fig, "matriz_confusao")


FIGURAS = (f_segmentacoes_ip, f_particao_faixas, f_curvas_treino, f_mapa_classificacao,
           f_matriz_confusao, f_fusao_linha_do_tempo, f_eixos)


def gerar(*nomes: str) -> None:
    """Gera todas as figuras, ou só as pedidas pelo nome da função (`f_eixos`, ...)."""
    for figura in FIGURAS:
        if not nomes or figura.__name__ in nomes:
            figura()


if __name__ == "__main__":
    import sys

    print(f"figuras em {DESTINO}")
    gerar(*sys.argv[1:])
