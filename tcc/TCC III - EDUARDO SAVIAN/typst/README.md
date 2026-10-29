# Versão Typst do TCC

Porte da monografia de LaTeX (`src/`) para [Typst](https://typst.app). As duas
versões coexistem: nada em `src/` foi alterado e as imagens são compartilhadas.

## Como compilar

```bash
typst compile main.typ      # gera main.pdf
typst watch main.typ        # recompila a cada salvamento
./build.sh                  # atalho; ./build.sh --watch para o modo watch
```

O arquivo de entrada é `main.typ`, na **raiz** do diretório do TCC (não dentro de
`typst/`). Isso faz com que a raiz do projeto Typst seja o próprio diretório do
TCC, permitindo que as figuras sejam referenciadas como `/src/assets/...` —
os mesmos caminhos usados pelo LaTeX, sem duplicar os 5,5 MB de imagens.

Testado com Typst 0.15.

### Fontes

O template pede Times New Roman e cai, nesta ordem, para Nimbus Roman, TeX Gyre
Termes, Liberation Serif e Libertinus Serif (esta última é embutida no Typst).
Para o resultado idêntico ao do LaTeX (`\usepackage{times}`), instale uma delas:

```bash
sudo apt install fonts-texgyre        # TeX Gyre Termes (métricas de Times)
```

## Estrutura

| Arquivo                        | Equivalente em LaTeX                     |
| ------------------------------ | ---------------------------------------- |
| `../main.typ`                  | `src/main.tex` + `src/_info.tex` + `src/0_pre_textual.tex` |
| `univali.typ`                  | `src/styles/abntex2.cls` + `src/styles/ttc_univali.sty` |
| `1-introducao.typ`             | `src/1_introducao.tex`                   |
| `2-fundamentacao.typ`          | `src/2_fundamentacao.tex`                |
| `3-projeto.typ`                | `src/3_projeto.tex`                      |
| `4-consideracoes-finais.typ`   | `src/4_consideracoes_finais.tex`         |
| `refs.bib`                     | `src/refs.bib` (ver "Bibliografia")      |

Metadados (título, autor, orientador, resumo, abstract, siglas e símbolos) ficam
todos nos argumentos de `tcc(...)` em `main.typ`.

## Equivalências do template

| LaTeX                                    | Typst                                             |
| ---------------------------------------- | ------------------------------------------------- |
| `\begin{Info}{...}`                      | argumentos `universidade`, `escola`, `curso`, `titulo`, `autor`, `local-data`, `area`, `orientador` |
| `\chapter` / `\section` / `\subsection`  | `=` / `==` / `===` (numeração e caixa alta automáticas) |
| `\begin{figure}[H]` + `\caption` + `\caption*{Fonte:}` | `#figura(legenda, image(...), fonte: ...)` |
| `\begin{quadro}[H]`                      | `#quadro(legenda, table(...), fonte: ...)`        |
| `\rowcolor` + `\textbf` no cabeçalho     | `#ch[...]` (célula de cabeçalho cinza e em negrito) |
| `\addEquacao{Nome}{n}`                   | `#equacao(nome, $ ... $)`                         |
| `\cite{chave}`                           | `@chave` (várias adjacentes viram um só grupo)    |
| `\citeonline{chave}`                     | `#citeonline(<chave>)`                            |
| `\citefigure{chave}`                     | `#fonte-cite(<chave>)`                            |
| `Fonte: Elaborada pelo autor`            | `#fonte-autor`                                    |
| `\noindent`                              | `#noindent[...]`                                  |
| `\autoref{cap:x}` / `\ref{fig:x}`        | `@cap-x` / `@fig-x` (rótulos usam `-`, não `:`)   |
| `\listoffigures*` / `\listofquadros*` / `\listofequacao*` / `\tableofcontents*` | gerados pelo template |

Layout preservado: A4, 12 pt, margens 3/2/3/2 cm, entrelinhas 1,5, recuo de
1,27 cm com `indentfirst`, espaçamento de 16 pt entre parágrafos, legendas acima
da ilustração no formato `Figura 1 – Título` com a fonte abaixo, numeração
contínua de figuras/quadros/equações e paginação em arábico no canto superior
direito a partir da primeira folha textual (contando desde a capa).

## Bibliografia

`refs.bib` é uma cópia de `src/refs.bib` **sem as 12 chaves duplicadas**
(`cuConv`, `CUDA_Programming`, `Jorda_cuDNN`, `IMANI202059` e outras 8 apareciam
duas vezes, com conteúdo idêntico). O BibTeX ignora repetições silenciosamente; o
Typst as trata como erro. Ao adicionar referências, mantenha os dois arquivos em
sincronia ou passe a usar apenas este.

O estilo é o CSL `associacao-brasileira-de-normas-tecnicas`, embutido no Typst,
em lugar de `abntex2-alf.bst`.

## Diferenças conhecidas em relação ao PDF do LaTeX

- **Citações entre parênteses não ficam em caixa alta**: o CSL da ABNT gera
  `(Souza; Viel, 2023)`, enquanto o `abntex2cite` gera `(SOUZA; VIEL, 2023)`.
  O sobrenome em caixa alta é mantido na lista de referências.
- **Citações agrupadas são reordenadas alfabeticamente** pelo CSL: `@Li2019DeepLF
  @Ahmad_2025` sai como `(Ahmad et al., 2025; Li et al., 2019)`.
- **A lista de equações é numerada automaticamente**. No LaTeX os números eram
  passados à mão em `\addEquacao{...}{n}` e estavam fora de ordem (a "Projeção
  PCA", 6ª entrada da lista, era a 1ª equação do texto).
- Os quadros não têm linha `Fonte:`, como no original.
