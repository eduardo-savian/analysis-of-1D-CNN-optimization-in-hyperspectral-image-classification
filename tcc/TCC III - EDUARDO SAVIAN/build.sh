#!/usr/bin/env bash
# Compila a versão Typst do TCC (gera main.pdf na raiz deste diretório).
#
#   ./build.sh            # compila uma vez
#   ./build.sh --watch    # recompila a cada alteração
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v typst > /dev/null; then
    echo "typst não encontrado. Instale com uma das opções:" >&2
    echo "  cargo install --locked typst-cli" >&2
    echo "  ou baixe o binário em https://github.com/typst/typst/releases" >&2
    exit 1
fi

if [[ "${1:-}" == "--watch" ]]; then
    exec typst watch main.typ
fi

typst compile main.typ
echo "gerado: main.pdf"
