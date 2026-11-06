from functools import cache

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402

from hsi.ambiente import configure_determinism  # noqa: E402
from hsi.config import FIGURAS, RESULTADOS, SEGMENTERS, ExperimentConfig  # noqa: E402
from hsi.dados import apply_pca, load_scene, pixel_table, prepare_data  # noqa: E402
from hsi.paralelismo.particao import partition_scene  # noqa: E402
from hsi.pipeline import run_experiment  # noqa: E402
from hsi.segmentadores import SEGMENTER_PARAMS, rgb_composite, segment  # noqa: E402

DESTINO = FIGURAS

# Paleta distinguível para daltônicos; cada entidade tem uma cor fixa.
AZUL, LARANJA, AGUA, AMARELO, VIOLETA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
# Cores com significado, fora da paleta de categorias.
PERDA, NEUTRO = "#d1495b", "#9aa0a6"
TINTA, TINTA2, FRACO, GRADE, LINHA = "#15181c", "#454b52", "#6b7280", "#e8eaed", "#c9ced4"

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 220, "savefig.bbox": "tight",
    # Type 42 embute as fontes: o PDF não depende das fontes de quem compila.
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


def salvar_figura(fig, nome: str) -> None:
    DESTINO.mkdir(parents=True, exist_ok=True)
    for extensao in ("pdf", "png"):
        fig.savefig(DESTINO / f"{nome}.{extensao}")
    plt.close(fig)
    print(f"  {nome}")


def ler(nome: str) -> pd.DataFrame:
    return pd.read_csv(RESULTADOS / nome)


def f_fusao_linha_do_tempo(cena: str = "PU"):
    """Linha do tempo de cada thread enquanto T0 segmenta.

    University of Pavia por padrão: a segmentação é longa o bastante para a sobreposição aparecer.
    """
    arranjos = [("concorrente", "implementação proposta: cada thread faz as convoluções de um bloco e espera o mapa"),
                ("desacoplado", "convoluções primeiro: todas entram na fila antes das fusões")]
    quadros = {a: ler(f"08_linha_do_tempo_{cena}_{a}.csv") for a, _ in arranjos}
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
            # Aqui a linha do tempo grava a thread da fusão; a faixa azul só marca quando
            # as convoluções terminaram.
            eixo.barh(len(linhas), q["fim_convolucoes"].max() - q["inicio"].min(),
                      left=q["inicio"].min(), height=0.6, color=AZUL, zorder=3)
            eixo.annotate("todas as convoluções",
                          (q["fim_convolucoes"].max(), len(linhas)),
                          textcoords="offset points", xytext=(6, 0), va="center",
                          fontsize=8, color=AZUL, weight="bold")
            linhas = linhas | {"conv.": len(linhas)}
        eixo.axvline(t0_fim, color=TINTA, lw=1.1, ls="--", zorder=4)
        # O pool nomeia as threads T_0, T_1...; no rótulo viram T1..TN, e T0 é a da segmentação.
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
    salvar_figura(fig, "fusao_linha_do_tempo")


def f_eixos():
    """Validação média por valor de cada eixo, com a configuração original marcada."""
    t = ler("04_trials.csv")
    # Em ordem de amplitude. Taxa e weight decay em 10⁻³, para rótulos curtos.
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
    # Legenda na figura: ancorada num painel, encolheria as colunas.
    fig.legend(handles=[Patch(color=PERDA, label="valor da configuração original"),
                        Patch(color=NEUTRO, label="outros valores do espaço")],
               ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=0.8, h_pad=1.6)
    salvar_figura(fig, "eixos_hpo")


NOME_SEG = {"seeds": "SEEDS", "lsc": "LSC", "watershed": "Watershed"}
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
    fig, eixos = plt.subplots(1, 1 + len(SEGMENTERS), figsize=(6.4, 2.1))
    eixos[0].imshow(rgb)
    eixos[0].set_title("composição RGB\n(PCA de 3 componentes)", fontsize=8.5, color=TINTA2)
    for eixo, seg in zip(eixos[1:], SEGMENTERS):
        rotulos = segment(rgb, seg, SEGMENTER_PARAMS[seg]["IP"])
        eixo.imshow(_com_contornos(rgb, rotulos))
        eixo.set_title(f"{NOME_SEG[seg]}\n{len(np.unique(rotulos))} regiões", fontsize=8.5, color=TINTA2)
    _sem_eixos(*eixos)
    fig.tight_layout(w_pad=0.4)
    salvar_figura(fig, "segmentacoes_ip")


def f_particao_faixas():
    """Os pixels rotulados de Indian Pines divididos em 4 e em 16 faixas."""
    dados = prepare_data(ExperimentConfig(dataset="IP"))
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
    salvar_figura(fig, "particao_faixas")


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
    salvar_figura(fig, "curvas_treino")


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
    salvar_figura(fig, "mapa_classificacao")


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
    salvar_figura(fig, "matriz_confusao")



# Valores reportados por Souza (2023) e tempo de cada segmentador (medicoes_avulsas.md).
OA_SOUZA = {"IP": 0.9142, "PU": 0.9928, "SA": 0.9965}
TEMPO_SEG_MS = {"IP": {"seeds": 33.3, "lsc": 37.7, "watershed": 1.2},
                "PU": {"seeds": 599.1, "lsc": 280.2, "watershed": 10.6},
                "SA": {"seeds": 343.1, "lsc": 151.0, "watershed": 5.5}}
CENAS = {"IP": "Indian Pines", "PU": "University of Pavia", "SA": "Salinas Valley"}


def f_linha_base():
    """OA da linha de base por cena: CNN de Souza, CNN do projeto e o valor reportado."""
    t = ler("01_linha_base.csv").set_index(["cena", "cnn"])["OA"]
    cenas = [c for c in CENAS if (c, "projeto") in t.index]
    fig, eixo = plt.subplots(figsize=(6.4, 3.2))
    x = np.arange(len(cenas))
    for desloc, (cnn, rotulo, cor) in zip((-0.2, 0.2), (("original", "CNN de Souza (2023)", NEUTRO),
                                                      ("projeto", "CNN do projeto", AZUL))):
        valores = [t[(c, cnn)] for c in cenas]
        eixo.bar(x + desloc, valores, 0.36, color=cor, label=rotulo, zorder=3)
        for xi, v in zip(x + desloc, valores):
            eixo.text(xi, v - 0.003, num(v, 4), ha="center", va="top", fontsize=8,
                      color="white", weight="bold", zorder=5)
    eixo.scatter(x, [OA_SOUZA[c] for c in cenas], marker="D", s=36, color=TINTA, zorder=6,
                 label="reportado por Souza (2023)")
    eixo.set(xticks=x, xticklabels=[CENAS[c] for c in cenas], ylim=(0.85, 1.01), ylabel="OA")
    virgula(eixo, casas=2)
    limpar(eixo)
    legendar(eixo, ncol=3)
    salvar_figura(fig, "linha_base")


def f_hpo():
    """Ganho da busca (experimento 05) e maior validação a cada trial (experimento 04)."""
    d = ler("05_impacto_hpo.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [1.1, 1]})
    segs = [s for s in SEGMENTERS if s in set(d["condição"])]
    x = np.arange(len(segs))
    for desloc, (conf, rotulo, cor) in zip((-0.2, 0.2), (("herdada", "configuração original", NEUTRO),
                                                       ("otimizada", "otimizada (experimento 03)", AZUL))):
        g = d[d["configuração"] == conf].groupby("condição")["OA"]
        medias, desvios = g.mean().reindex(segs), g.std(ddof=0).reindex(segs)
        barras = a.bar(x + desloc, medias, 0.36, yerr=desvios, capsize=3, color=cor, label=rotulo,
                       zorder=3, error_kw={"ecolor": TINTA2, "lw": 1})
        rotular_barras(a, barras, medias, casas=4, fonte=7.5, erros=desvios)
    a.set(xticks=x, xticklabels=[NOME_SEG[s] for s in segs], ylim=(0.85, 0.96), ylabel="OA de teste")
    a.set_title("Configuração original e otimizada", loc="left")
    virgula(a, casas=2)
    limpar(a)
    legendar(a, ncol=1, y=1.12)

    t = ler("04_trials.csv")
    t = t[t["orçamento"] == 200]
    for estrategia, rotulo, cor in (("grid", "busca em grade", LARANJA), ("bayes", "bayesiana", AZUL)):
        curvas = t[t["estratégia"] == estrategia].pivot(index="trial", columns="semente",
                                                       values="melhor até agora")
        media, desvio = curvas.mean(axis=1), curvas.std(axis=1, ddof=0)
        b.plot(curvas.index, media, lw=2, color=cor, label=rotulo)
        b.fill_between(curvas.index, media - desvio, media + desvio, color=cor, alpha=0.15, lw=0)
    b.set(xlim=(1, 27), xticks=[1, 5, 10, 15, 20, 25, 27], xlabel="trial",
          ylabel="maior validação até o trial")
    b.set_title("Grade e bayesiana, até 200 épocas por trial", loc="left")
    virgula(b, casas=2)
    limpar(b)
    legendar(b, ncol=1, y=1.12)
    fig.tight_layout(w_pad=3)
    salvar_figura(fig, "hpo")


def f_segmentadores():
    """OA otimizada (experimento 03) e o tempo de uma execução com cada segmentador.

    No tempo, a segmentação vem de medicoes_avulsas.md e o treino é a média dos retreinos
    do vencedor de cada segmentador, que tem os próprios hiperparâmetros e épocas.
    """
    r = ler("03_resumo.csv")
    cenas = [c for c in CENAS if c in set(r["cena"])]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.2), gridspec_kw={"width_ratios": [1, 1.15]})
    largura = 0.8 / len(SEGMENTERS)
    cores = {"seeds": AZUL, "lsc": LARANJA, "watershed": AGUA}
    for k, seg in enumerate(SEGMENTERS):
        linhas = r[r["condição"] == seg].set_index("cena").reindex(cenas)
        x = np.arange(len(cenas)) + (k - 1) * largura
        a.bar(x, linhas["OA"], largura * 0.92, yerr=linhas["OA_std"], capsize=2, color=cores[seg],
              label=NOME_SEG[seg], zorder=3, error_kw={"ecolor": TINTA2, "lw": 1})
    a.set(xticks=np.arange(len(cenas)), xticklabels=[CENAS[c].replace(" ", "\n", 1) for c in cenas], ylim=(0.85, 1.0),
          ylabel="OA de teste")
    a.set_title("OA com a CNN otimizada para cada um", loc="left")
    virgula(a, casas=2)
    legendar(a, ncol=3, y=1.12)

    ip = r[r["cena"] == "IP"].set_index("condição")
    segs = [s for s in SEGMENTERS if s in ip.index]
    y = np.arange(len(segs))
    segmentacao = np.array([TEMPO_SEG_MS["IP"][s] / 1000 for s in segs])
    treino = ip.loc[segs, "treino_s"].to_numpy()
    b.barh(y, segmentacao, color=TINTA, zorder=3, label="segmentação")
    b.barh(y, treino, left=segmentacao, color=[cores[s] for s in segs], zorder=3, label="treino")
    for yi, seg, t_seg, t_tr, ep in zip(y, segs, segmentacao, treino, ip.loc[segs, "épocas"]):
        b.text(t_seg + t_tr + 0.6, yi, f"{num(t_seg + t_tr, 1)} s ({num(ep, 0)} épocas; "
               f"segmentação {num(1000 * t_seg, 1)} ms)", va="center", fontsize=8, color=TINTA2)
    b.set(yticks=y, yticklabels=[NOME_SEG[s] for s in segs], xlabel="segundos por execução",
          xlim=(0, 1.9 * treino.max()))
    b.invert_yaxis()
    b.set_title("Tempo de uma execução em Indian Pines", loc="left")
    virgula(b, "x", casas=0)
    for eixo in (a, b):
        limpar(eixo, "y" if eixo is a else "x")
    fig.tight_layout(w_pad=3)
    salvar_figura(fig, "segmentadores")


def f_fusao_speedup():
    """Fusão concorrente contra a execução em série, por número de threads (experimento 08).

    Em cima, o speedup do tempo total; embaixo, o tempo que sobra depois de a segmentação
    publicar o mapa, que não depende da variação do próprio SEEDS. Bloco 1.024, seção mínima.
    """
    q = ler("08_fusao_concorrente.csv")
    q["resto"] = q["parede"] - q["t0_fim"]
    g = q.groupby(["cena", "arranjo", "n", "bloco", "secao", "torch_threads"])[["parede", "resto", "segmentacao"]].median()
    cenas = [c for c in CENAS if c in set(q["cena"])]
    fig, eixos = plt.subplots(2, len(cenas), figsize=(9, 4.4), sharex=True)
    for col, cena in enumerate(cenas):
        serie_1024 = g.loc[(cena, "sequencial", 1, 1024, "-", 1)]
        lote16 = g.loc[(cena, "lote")].xs(16, level="torch_threads")["parede"].iloc[0]
        a, b = eixos[0, col], eixos[1, col]
        a.axhline(1, color=LINHA, lw=1)
        a.axhline(serie_1024["parede"] / serie_1024["segmentacao"], color=TINTA2, lw=1.2, ls=":",
                  label="teto (série ÷ segmentação)")
        a.axhline(serie_1024["parede"] / lote16, color=NEUTRO, lw=1.5, ls="--",
                  label="inferência normal com 16 threads")
        b.axhline(1000 * serie_1024["resto"], color=TINTA, lw=1.2, ls="-.", label="em série (embaixo)")
        for arranjo, rotulo, cor in (("concorrente", "implementação proposta", AZUL),
                                     ("desacoplado", "convoluções primeiro", LARANJA)):
            serie = g.loc[(cena, arranjo)].xs((1024, "minima", 1), level=("bloco", "secao", "torch_threads"))
            a.plot(serie.index, serie_1024["parede"] / serie["parede"], marker="o", ms=4, lw=2, color=cor, label=rotulo)
            b.plot(serie.index, 1000 * serie["resto"], marker="o", ms=4, lw=2, color=cor)
        a.set_title(CENAS[cena], loc="left")
        a.set(ylim=(0.6, 2.2), yticks=[0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
        b.set_ylim(0, None)
        b.set(xscale="log", xticks=[1, 2, 4, 8, 16], xticklabels=["1", "2", "4", "8", "16"], xlabel="threads")
        b.xaxis.set_minor_locator(plt.NullLocator())
        virgula(a, casas=1)
        virgula(b, casas=0)
        limpar(a)
        limpar(b)
    eixos[0, 0].set_ylabel("vezes mais rápido\nque em série")
    eixos[1, 0].set_ylabel("ms depois da\nsegmentação")
    alcas, rotulos = eixos[0, 0].get_legend_handles_labels()
    alca_serie, rotulo_serie = eixos[1, 0].get_legend_handles_labels()
    alcas, rotulos = alcas + alca_serie[:1], rotulos + rotulo_serie[:1]
    fig.tight_layout(w_pad=1.5, h_pad=1.2, rect=(0, 0, 1, 0.88))
    fig.legend(alcas, rotulos, ncol=3, loc="upper left", bbox_to_anchor=(0.06, 1.0), frameon=False)
    salvar_figura(fig, "fusao_speedup")


def f_particao_resultados():
    """Speedup da partição sobre o sequencial (experimento 07) e OA por N (experimento 06)."""
    q = ler("07_pool_quente.csv")
    q = q[q["modo"] == "independente"]
    tempo = q.groupby(["executor", "estado", "n"])["bloco_paralelo"].median()
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4))
    sequencial = tempo.xs("sequencial", level="executor").groupby("n").median()
    curvas = (("process", "quente", "processos já abertos", AZUL, "-"),
              ("process", "frio", "processos abertos na hora", AZUL, "--"),
              ("thread", "quente", "threads já abertas", LARANJA, "-"),
              ("thread", "frio", "threads abertas na hora", LARANJA, "--"))
    for executor, estado, rotulo, cor, estilo in curvas:
        if (executor, estado) not in tempo.index.droplevel("n"):
            continue
        serie = tempo[(executor, estado)]
        ns = [n for n in serie.index if n in sequencial.index]
        a.plot(ns, [sequencial[n] / serie[n] for n in ns], marker="o", ms=5, lw=2, color=cor,
               ls=estilo, label=rotulo)
    a.axhline(1, color=LINHA, lw=1)
    a.set(xscale="log", xticks=[2, 4, 8, 16, 32], xticklabels=["2", "4", "8", "16", "32"],
          xlabel="faixas", ylabel="vezes mais rápido que\numa faixa por vez")
    a.set_title("Cada faixa treina o seu modelo", loc="left")
    virgula(a, casas=0)
    limpar(a)
    legendar(a, ncol=2, y=1.12)

    d = ler("06_paralelismo_particao.csv")
    mono = d[d["arranjo"] == "monolítico"]["OA"]
    ind = d[d["arranjo"] == "independente"].groupby("n")["OA"]
    b.axhspan(mono.mean() - mono.std(ddof=0), mono.mean() + mono.std(ddof=0), color=GRADE, zorder=0)
    b.axhline(mono.mean(), color=NEUTRO, lw=1.5, ls="--", label="um modelo para a cena inteira")
    b.errorbar(ind.mean().index, ind.mean(), yerr=ind.std(ddof=0), marker="o", ms=5, lw=2, capsize=3,
               color=AZUL, label="um modelo por faixa")
    b.set(xscale="log", xticks=[2, 4, 8, 16], xticklabels=["2", "4", "8", "16"],
          xlabel="faixas", ylabel="OA de teste")
    b.set_title("Acurácia com um modelo por faixa", loc="left")
    b.yaxis.set_major_locator(MultipleLocator(0.01))
    for eixo in (a, b):
        eixo.minorticks_off()
    virgula(b, casas=2)
    limpar(b)
    legendar(b, ncol=1, y=1.12)
    fig.tight_layout(w_pad=3)
    salvar_figura(fig, "particao_resultados")


FIGURAS = (f_segmentacoes_ip, f_particao_faixas, f_curvas_treino, f_mapa_classificacao,
           f_matriz_confusao, f_fusao_linha_do_tempo, f_eixos, f_linha_base, f_hpo,
           f_segmentadores, f_fusao_speedup, f_particao_resultados)


def gerar(*nomes: str) -> None:
    """Gera todas as figuras, ou só as pedidas pelo nome da função (`f_eixos`, ...)."""
    for figura in FIGURAS:
        if not nomes or figura.__name__ in nomes:
            figura()


if __name__ == "__main__":
    import sys

    print(f"figuras em {DESTINO}")
    gerar(*sys.argv[1:])
