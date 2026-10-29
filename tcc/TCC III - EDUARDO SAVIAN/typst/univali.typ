// =============================================================================
// Template ABNT — UNIVALI / Escola Politécnica / Ciência da Computação
//
// Porte para Typst de `src/styles/abntex2.cls` + `src/styles/ttc_univali.sty`.
// Reproduz: capa, folha de rosto, epígrafe, resumo/abstract, listas de
// ilustrações / quadros / equações / siglas / símbolos, sumário, numeração de
// páginas no canto superior direito e títulos de capítulo/seção em caixa alta.
// =============================================================================

// ---------------------------------------------------------------- constantes

// Times New Roman (\usepackage{times}); as demais são alternativas livres
// métricamente equivalentes, usadas quando a primeira não está instalada.
#let serif = (
  "Times New Roman",
  "Nimbus Roman",
  "TeX Gyre Termes",
  "Liberation Serif",
  "Libertinus Serif",
)

#let corpo-tamanho = 12pt
// \ABNTEXfontereduzida (\small): legendas, fontes das figuras e paginação.
#let reduzida = 11pt
// Espaçamento entre linhas equivalente ao 1,5 do abntex2.
#let entrelinhas = 0.95em
// \setlength{\parindent}{1.27cm} e \setlength{\parskip}{16pt}
#let recuo = 1.27cm
#let entre-paragrafos = 16pt

#let cinza-forte = rgb("#C0C0C0") // \rowcolor[HTML]{C0C0C0}
#let cinza-fraco = rgb("#EFEFEF") // \rowcolor[HTML]{EFEFEF}

// ------------------------------------------------------------------- citações

// \citeonline{chave} — "Souza (2023)"
#let citeonline(chave) = cite(chave, form: "prose")

// \citefigure{chave} — usada na linha "Fonte:" das figuras e quadros
#let fonte-cite(chave) = [Fonte: #cite(chave, form: "prose")]

#let fonte-autor = [Fonte: Elaborada pelo autor]

// \noindent — parágrafo de continuação (após equações e ilustrações)
#let noindent(corpo) = par(first-line-indent: 0pt, corpo)

// -------------------------------------------------------- figuras e quadros

// Linha "Fonte: ..." logo abaixo da ilustração, alinhada à esquerda.
#let _fonte-linha(fonte) = {
  if fonte != none {
    block(width: 100%, above: 0.6em, below: 0pt, text(size: reduzida, align(left, fonte)))
  }
}

// \begin{figure}[H] ... \caption ... \includegraphics ... \caption*{Fonte:}
#let figura(legenda, conteudo, fonte: none) = figure(
  caption: legenda,
  kind: image,
  supplement: [Figura],
  {
    conteudo
    _fonte-linha(fonte)
  },
)

// \begin{quadro}[H] — float próprio do ttc_univali, com lista independente
#let quadro(legenda, conteudo, fonte: none) = figure(
  caption: legenda,
  kind: "quadro",
  supplement: [Quadro],
  {
    conteudo
    _fonte-linha(fonte)
  },
)

// Célula de cabeçalho de quadro (\rowcolor + \textbf)
#let ch(corpo, fill: cinza-forte) = table.cell(fill: fill, strong(corpo))

// ---------------------------------------------------------------- equações

// \addEquacao{Nome}{n} — registra a equação na "Lista de Equações".
// O número é lido do contador de equações no ponto de inserção, portanto
// acompanha automaticamente a ordem do documento.
#let equacao(nome, corpo) = {
  corpo
  // marcador invisível: não ocupa espaço, mas preserva o espaçamento de parágrafo
  block(height: 0pt, above: 0pt, below: entre-paragrafos, [#metadata(nome)<equacao-entrada>])
}

// ------------------------------------------------- títulos pré-textuais

#let titulo-pretextual(titulo) = heading(
  level: 1,
  numbering: none,
  outlined: false,
  bookmarked: true,
  titulo,
)

// Linha de sumário/lista com pontilhado e número de página.
#let _linha-lista(corpo, loc, recuo-nivel: 0pt, acima: 0.55em) = link(
  loc,
  block(
    above: acima,
    below: 0.55em,
    inset: (left: recuo-nivel),
    grid(
      columns: (1fr, auto),
      column-gutter: 0.4em,
      // o número de página acompanha a última linha da entrada
      align: (left + bottom, right + bottom),
      {
        set par(justify: false, first-line-indent: 0pt, leading: 0.6em)
        [#corpo #box(width: 1fr, repeat(gap: 2.5pt)[.])]
      },
      [#counter(page).at(loc).first()],
    ),
  ),
)

// Lista de Equações (\listofequacao)
#let lista-equacoes() = {
  titulo-pretextual[Lista de Equações]
  context {
    for entrada in query(<equacao-entrada>) {
      let n = counter(math.equation).at(entrada.location()).first()
      _linha-lista([Equação #n. #entrada.value], entrada.location())
    }
  }
}

// \begin{siglas} / \begin{simbolos}
#let lista-descritiva(titulo, itens) = {
  titulo-pretextual(titulo)
  block(
    grid(
      columns: (2.4cm, 1fr),
      row-gutter: 0.75em,
      column-gutter: 0.4em,
      ..itens.map(((sigla, significado)) => (sigla, significado)).flatten(),
    ),
  )
}

// ------------------------------------------------------------------ template

#let tcc(
  universidade: "",
  escola: "",
  curso: "",
  titulo: "",
  autor: "",
  local-data: "",
  area: "",
  orientador: "",
  coorientador: none,
  epigrafe: none,
  dedicatoria: none,
  agradecimentos: none,
  resumo: none,
  palavras-chave: none,
  abstract: none,
  keywords: none,
  siglas: (),
  simbolos: (),
  bibliografia: none,
  corpo,
) = {
  // ------------------------------------------------------------ configuração
  set document(title: titulo)
  set page(
    paper: "a4",
    // \setlrmarginsandblock{3cm}{2cm} / \setulmarginsandblock{3cm}{2cm}
    margin: (left: 3cm, right: 2cm, top: 3cm, bottom: 2cm),
  )
  set text(font: serif, size: corpo-tamanho, lang: "pt", region: "br")
  set par(
    justify: true,
    leading: entrelinhas,
    spacing: entre-paragrafos,
    // \usepackage{indentfirst} recua também o 1º parágrafo da seção
    first-line-indent: (amount: recuo, all: true),
  )
  set heading(numbering: "1.1")
  set math.equation(numbering: "(1)")
  // \setlist{topsep=0pt} + itemsep=0pt do enumitem: itens pouco espaçados
  set enum(indent: recuo, spacing: entrelinhas)
  set list(indent: recuo, spacing: entrelinhas)

  // Legendas: acima da ilustração, "Figura 1 – Texto", fonte reduzida.
  set figure.caption(position: top, separator: [ – ])
  show figure.caption: set text(size: reduzida)
  // As figuras usam \captionsetup{justification=centering}; os quadros seguem
  // o padrão global raggedright.
  show figure.caption: set align(center)
  show figure.where(kind: "quadro"): it => {
    show figure.caption: set align(left)
    it
  }
  // Ilustrações não se separam da legenda; quadros longos podem quebrar.
  show figure.where(kind: image): set block(breakable: false)
  show figure.where(kind: "quadro"): set block(breakable: true)
  show figure: set par(first-line-indent: 0pt)
  set figure(gap: 0.8em)

  // Referências: espaçamento simples dentro da entrada, linha em branco entre
  // entradas (NBR 6023).
  show bibliography: set par(leading: 0.65em, spacing: entre-paragrafos)

  // Quadros são fechados por linhas em todos os lados (ABNT).
  set table(stroke: 0.5pt, inset: 5pt)
  show table: set text(size: reduzida)
  show table: set par(first-line-indent: 0pt, justify: false, leading: 0.6em)

  // O Typst hifeniza imediatamente antes de um hífen já existente, produzindo
  // "baseando-" / "-se". Palavras compostas ficam em box para evitar o artefato
  // (o TeX também não hifeniza palavras que contenham hífen explícito).
  show regex("[\p{L}\p{N}]+(-[\p{L}\p{N}]+)+"): it => box(it)

  // \autoref{cap:...} → "Capítulo 3"; seções continuam como "Seção 3.1"
  show ref: it => {
    let el = it.element
    if el != none and el.func() == heading and el.level == 1 and el.numbering != none {
      link(
        el.location(),
        [Capítulo #numbering(el.numbering, ..counter(heading).at(el.location()))],
      )
    } else {
      it
    }
  }

  // Títulos: Times, negrito, caixa alta; capítulos em nova página.
  show heading: it => {
    let num(espaco) = if it.numbering != none {
      context [#counter(heading).display(it.numbering)#h(espaco)]
    }
    set text(font: serif, weight: "bold")
    set par(first-line-indent: 0pt, justify: false, leading: 0.65em)

    if it.level == 1 {
      pagebreak(weak: true)
      set text(size: 16pt)
      // títulos sem numeração (Resumo, Sumário, Referências…) são centralizados
      block(width: 100%, above: 0pt, below: 1.2em, {
        if it.numbering == none {
          align(center, upper(it.body))
        } else {
          [#num(0.5em)#upper(it.body)]
        }
      })
    } else if it.level == 2 {
      // section=TITLE: seções também em caixa alta
      block(above: 1.6em, below: 1em, text(size: 14pt, [#num(1.2em)#upper(it.body)]))
    } else if it.level == 3 {
      block(above: 1.6em, below: 1em, text(size: 14pt, [#num(1.2em)#it.body]))
    } else {
      block(above: 1.4em, below: 0.9em, text(size: 12pt, [#num(1.2em)#it.body]))
    }
  }

  // --------------------------------------------------------------- capa
  set align(center)
  {
    set text(size: 14pt, weight: "bold")
    set par(first-line-indent: 0pt, leading: 0.8em, spacing: 0pt)
    [#universidade \ #escola \ #curso]
  }
  v(0.37 * 100%)
  text(size: 14pt, weight: "bold", hyphenate: false, upper(titulo))
  v(48pt)
  text(size: 14pt)[por]
  v(24pt)
  text(size: 14pt)[#autor]
  v(1fr)
  text(size: 14pt)[#local-data]

  // --------------------------------------------------------- folha de rosto
  pagebreak()
  {
    set text(size: 14pt, weight: "bold")
    set par(first-line-indent: 0pt, leading: 0.8em, spacing: 0pt)
    [#universidade \ #escola \ #curso]
  }
  v(0.27 * 100%)
  text(size: 14pt, weight: "bold", hyphenate: false, upper(titulo))
  v(24pt)
  text(size: 14pt)[Área de #area]
  v(12pt)
  text(size: 14pt)[por]
  v(12pt)
  text(size: 14pt)[#autor]
  v(36pt)
  align(
    right,
    block(width: 50% + 15pt, {
      set align(left)
      set par(justify: true, first-line-indent: 0pt, leading: entrelinhas)
      set text(size: 12pt, weight: "regular")
      [Relatório apresentado à Banca Examinadora do Trabalho Técnico-científico de
        Conclusão de Curso de Ciência de Computação para análise e aprovação.]
      v(entre-paragrafos)
      [Orientador(a): #orientador]
      if coorientador != none {
        v(entre-paragrafos)
        [Coorientador(a): #coorientador]
      }
    }),
  )
  v(1fr)
  text(size: 14pt)[#local-data]
  set align(left)

  // ------------------------------------------------- dedicatória / epígrafe
  if dedicatoria != none {
    pagebreak()
    v(1fr)
    align(right, block(width: 60%, {
      set par(justify: false, first-line-indent: 0pt)
      emph(dedicatoria)
    }))
  }

  if agradecimentos != none {
    titulo-pretextual[Agradecimentos]
    agradecimentos
  }

  if epigrafe != none {
    pagebreak()
    v(1fr)
    align(right, block(width: 60%, {
      set par(justify: false, first-line-indent: 0pt)
      emph(epigrafe)
    }))
  }

  // ----------------------------------------------------- resumo e abstract
  if resumo != none {
    titulo-pretextual[Resumo]
    set par(first-line-indent: 0pt)
    resumo
    if palavras-chave != none {
      parbreak()
      [*Palavras-Chave*: #palavras-chave]
    }
  }

  if abstract != none {
    titulo-pretextual[Abstract]
    set par(first-line-indent: 0pt)
    set text(lang: "en")
    emph(abstract)
    if keywords != none {
      parbreak()
      emph[*Keywords*: #keywords]
    }
  }

  // ---------------------------------------------------------------- listas
  show outline.entry: it => {
    let el = it.element
    if el == none {
      return it
    }
    if el.func() == figure {
      // "Figura 1 – Legenda .......... 12"
      _linha-lista([#it.prefix() – #it.body()], el.location())
    } else if el.func() == heading and el.level == 1 {
      _linha-lista(strong([#it.prefix() #upper(el.body)]), el.location(), acima: 1.3em)
    } else {
      // ttc_univali só remove o negrito de subsection e subsubsection:
      // as seções (nível 2) permanecem em negrito no sumário
      let corpo = [#it.prefix() #it.body()]
      _linha-lista(
        if el.level == 2 { strong(corpo) } else { corpo },
        el.location(),
        recuo-nivel: (el.level - 1) * 0.8cm,
      )
    }
  }

  titulo-pretextual[Lista de ilustrações]
  outline(title: none, target: figure.where(kind: image))

  titulo-pretextual[Lista de quadros]
  outline(title: none, target: figure.where(kind: "quadro"))

  lista-equacoes()

  if siglas.len() > 0 {
    lista-descritiva("Lista de abreviaturas e siglas", siglas)
  }
  if simbolos.len() > 0 {
    lista-descritiva("Lista de símbolos", simbolos)
  }

  titulo-pretextual[Sumário]
  outline(title: none, depth: 4)

  // --------------------------------------------------- elementos textuais
  // A paginação aparece a partir da primeira folha textual, no canto
  // superior direito, contando a partir da capa (NBR 14724).
  set page(header: context align(right, text(size: reduzida, [#counter(page).get().first()])))

  corpo

  // ----------------------------------------------- elementos pós-textuais
  if bibliografia != none {
    bibliografia
  }
}
