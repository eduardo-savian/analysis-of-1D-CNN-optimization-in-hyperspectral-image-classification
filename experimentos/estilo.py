"""Sistema visual das figuras: paleta, tipografia e as peças repetidas.

Existe para que as 19 figuras pareçam uma coleção, e não dezenove gráficos soltos.
Centralizar aqui também evita o pior defeito de figura gerada em lote: a mesma
entidade aparecer de cores diferentes em painéis diferentes.

DECISÕES, e o porquê de cada uma:

- **Cor por entidade, fixa.** `seeds` é sempre azul, `watershed` sempre amarelo, em
  toda figura. A paleta é a validada para daltonismo (separação em OKLab), atribuída
  em ordem fixa e nunca reciclada.
- **Cor semântica separada da categórica.** Verde para ganho, vermelho-terra para
  perda, cinza para referência. Nunca reaproveitadas como "série 5".
- **Rótulo direto no dado.** Legenda só quando o gráfico tem séries que se cruzam;
  do contrário o nome vai colado à linha ou à barra.
- **Grade recessiva e sem molduras.** Só eixo esquerdo e inferior, em cinza claro.
  A tinta deve estar no dado, não na moldura.
- **Título que afirma, subtítulo que explica.** O título diz o achado; o subtítulo
  diz o protocolo. Quem folheia o TCC lê só os títulos e entende a história.
- **Nota de rodapé com a procedência.** Toda figura diz de onde vieram os números.
- **Título e crédito fora da área de desenho.** Ficam acima e abaixo da figura, com
  afastamento em pontos tipográficos, para nunca colidirem com título de painel.
- **Vírgula decimal.** O texto é em português; `br()` converte qualquer número
  formatado, inclusive o separador de milhar.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter  # noqa: F401  (reexportado por conveniência)

# Paleta categórica validada (separação ≥ 8 ΔE em OKLab para deutan/protan/tritan).
AZUL, LARANJA, AGUA, AMARELO = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
VIOLETA, ROSA = "#4a3aa7", "#e87ba4"

# Semânticas — reservadas, nunca usadas como categoria.
GANHO, PERDA, NEUTRO = "#1baf7a", "#d1495b", "#9aa0a6"

# Neutros com leve viés frio, para não parecerem "cinza de sistema".
TINTA, TINTA2, FRACO, GRADE, LINHA = "#15181c", "#454b52", "#6b7280", "#e8eaed", "#c9ced4"

#: Uma cor por condição experimental, usada em todas as figuras.
COR_CONDICAO = {
    "sem fusão": NEUTRO, "seeds": AZUL, "slic": LARANJA,
    "lsc": AGUA, "watershed": AMARELO,
}
COR_ESTRATEGIA = {"random": NEUTRO, "tpe": AZUL, "genetic": LARANJA}
COR_PROTOCOLO = {"pixel": LARANJA, "region": AZUL}
COR_CENA = {"IP": AZUL, "PU": LARANJA, "SA": AGUA}

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 220, "savefig.bbox": "tight",
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 9.5, "axes.labelcolor": TINTA2,
    "axes.edgecolor": LINHA, "axes.linewidth": 0.9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "grid.color": GRADE, "grid.linewidth": 0.9,
    "xtick.color": FRACO, "ytick.color": FRACO,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "xtick.major.size": 0, "ytick.major.size": 0,
    "legend.frameon": False, "legend.fontsize": 9,
    "text.color": TINTA, "lines.solid_capstyle": "round",
})


def br(texto: str) -> str:
    """Troca o ponto decimal por vírgula e o milhar por ponto, no padrão brasileiro."""
    return texto.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def num(valor: float, casas: int = 2, formato: str | None = None) -> str:
    """Número já em português, com sinal de menos tipográfico.

    `num(0.8981, 4)` → "0,8981";  `num(-9.4, 1, "{:+.1f}")` → "−9,4".
    """
    return br((formato or f"{{:.{casas}f}}").format(valor)).replace("-", "\u2212")


def virgula(eixo, qual: str = "y", casas: int = 1) -> None:
    """Vírgula decimal nos ticks de um eixo numérico."""
    from matplotlib.ticker import FuncFormatter
    formatador = FuncFormatter(lambda valor, _: num(valor, casas))
    (eixo.yaxis if qual == "y" else eixo.xaxis).set_major_formatter(formatador)


def titular(fig, titulo: str, subtitulo: str = "", x: float = 0.0) -> None:
    """Título que afirma o achado, subtítulo que dá o protocolo.

    Ambos ficam **acima** da figura, com afastamento medido em pontos — assim não
    disputam espaço com um `set_title` de painel, que foi o defeito da primeira versão.
    """
    altura = fig.get_figheight() * 72
    y = 1.0 + 9 / altura
    if subtitulo:
        fig.text(x, y, subtitulo, fontsize=9.5, color=FRACO, va="bottom")
        y += 16 / altura
    fig.text(x, y, titulo, fontsize=13, weight="bold", color=TINTA, va="bottom")


def creditar(fig, texto: str) -> None:
    """Procedência dos números, abaixo da figura."""
    fig.text(0.0, -10 / (fig.get_figheight() * 72), texto, fontsize=8, color=FRACO,
             va="top")


def legendar(eixo, ncol: int = 3, y: float = 1.0) -> None:
    """Legenda acima do painel, horizontal — nunca sobre o dado.

    Legenda dentro da área de desenho foi o que mais estragou a primeira versão:
    com cinco condições não sobra canto vazio.
    """
    eixo.legend(ncol=ncol, loc="lower left", bbox_to_anchor=(0.0, y), frameon=False,
                borderaxespad=0.0, handlelength=1.4, columnspacing=1.6)


def limpar(eixo, grade: str = "y") -> None:
    """Grade recessiva num eixo só, barras na frente dela."""
    if grade:
        eixo.grid(axis=grade, zorder=0)
        eixo.set_axisbelow(True)


def rotular_barras(eixo, barras, valores, formato="{:.3f}", cor=None,
                   deslocamento=0.004, fonte=9) -> None:
    """Valor acima de cada barra — evita a ida e volta até o eixo."""
    for barra, valor in zip(barras, valores):
        altura = barra.get_height()
        eixo.text(barra.get_x() + barra.get_width() / 2,
                  altura + deslocamento if altura >= 0 else altura - deslocamento * 3,
                  br(formato.format(valor)), ha="center",
                  va="bottom" if altura >= 0 else "top",
                  fontsize=fonte, weight="bold", color=cor or TINTA)


def rotular_linha(eixo, x, y, texto, cor, dx=9, dy=0, fonte=9.5, ha=None) -> None:
    """Nome da série colado a um ponto da curva, dispensando legenda."""
    eixo.annotate(texto, (x, y), textcoords="offset points", xytext=(dx, dy),
                  color=cor, fontsize=fonte, weight="bold", va="center",
                  ha=ha or ("right" if dx < 0 else "left"), annotation_clip=False)
