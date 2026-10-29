"""Geração portátil de PDFs somente com Matplotlib, já usado pelo notebook.

O PDF reúne as curvas/métricas disponíveis e todos os documentos textuais ou
estruturados do diretório auditado. Checkpoints e outras saídas binárias entram no
inventário, preservando a ligação entre o relatório e cada artefato gerado.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from textwrap import wrap
from uuid import uuid4

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from hsi.config import output_dir
from hsi.mlops.rastreio import result_metrics


def _pdf_text(pdf, title, content, lines_per_page=52):
    """Acrescenta texto longo ao PDF em páginas legíveis, sem truncar documentos."""
    lines = []
    for raw_line in str(content).splitlines() or [""]:
        lines.extend(wrap(raw_line, width=115, replace_whitespace=False) or [""])
    for offset in range(0, len(lines), lines_per_page):
        figure = plt.figure(figsize=(11.69, 8.27))  # A4 horizontal
        figure.text(.04, .95, title, fontsize=14, fontweight="bold", va="top")
        figure.text(.04, .90, "\n".join(lines[offset:offset + lines_per_page]), family="monospace", fontsize=7.5, va="top")
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)


def _pdf_inventory(directory):
    """Lista todos os artefatos e separa documentos transcritíveis de binários."""
    root = Path(directory)
    documents, binaries = [], []
    for item in sorted(root.rglob("*")):
        if not item.is_file() or item.name == "report.pdf":
            continue
        relative = item.relative_to(root)
        entry = f"{relative} ({item.stat().st_size:,} bytes)"
        if item.suffix.lower() in {".json", ".csv", ".md", ".txt"}:
            documents.append((item, entry))
        else:
            binaries.append(entry)
    return documents, binaries


def generate_pdf_report(directory, title, result=None):
    """Cria ``report.pdf`` a partir de todos os documentos de uma execução/campanha.

    ``result`` é opcional. Quando existe, o relatório ganha curvas de loss e
    acurácia, além de OA, AA e Kappa. Sem resultado (caso de campanha), os
    documentos agregados continuam sendo incluídos integralmente.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "report.pdf"
    documents, binaries = _pdf_inventory(directory)
    with PdfPages(destination) as pdf:
        cover = plt.figure(figsize=(11.69, 8.27))
        cover.text(.06, .88, title, fontsize=22, fontweight="bold")
        cover.text(.06, .78, "Relatório MLOps gerado automaticamente\n"
                   f"Diretório: {directory.resolve()}\n"
                   f"Documentos transcritos: {len(documents)}\n"
                   f"Artefatos binários inventariados: {len(binaries)}", fontsize=12)
        pdf.savefig(cover, bbox_inches="tight")
        plt.close(cover)

        if result is not None:
            history = result.training.history
            figure, axes = plt.subplots(1, 2, figsize=(11.69, 5.2), constrained_layout=True)
            axes[0].plot(history.epochs, history.loss, label="treino")
            axes[0].plot(history.epochs, history.val_loss, label="validação")
            axes[0].set(title="Loss", xlabel="Época", ylabel="Cross-entropy")
            axes[1].plot(history.epochs, history.accuracy, label="treino")
            axes[1].plot(history.epochs, history.val_accuracy, label="validação")
            axes[1].set(title="Acurácia", xlabel="Época", ylabel="Pontuação", ylim=(0, 1))
            for axis in axes:
                axis.legend(); axis.grid(alpha=.25)
            pdf.savefig(figure)
            plt.close(figure)
            _pdf_text(pdf, "Métricas e decisão de treino", json.dumps(result_metrics(result), ensure_ascii=False, indent=2))

        _pdf_text(pdf, "Inventário de artefatos", "Documentos:\n" + "\n".join(entry for _, entry in documents) + "\n\nBinários:\n" + "\n".join(binaries))
        for document, entry in documents:
            try:
                _pdf_text(pdf, f"Documento: {entry}", document.read_text(encoding="utf-8"))
            except UnicodeDecodeError:
                _pdf_text(pdf, f"Documento não textual: {entry}", "Não foi possível decodificar como UTF-8.")
    return destination


def generate_standalone_pdf_report(result):
    """Audita a execução simples da seção 8 em um diretório de relatório próprio.

    Não passa pelo tracker: o diretório fica em `outputs/reports/`, fora de
    `outputs/runs/`, então esta execução não aparece na observabilidade da seção 9.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = output_dir() / "reports" / f"run_{stamp}_{uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.json").write_text(json.dumps(result.config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (directory / "metrics.json").write_text(json.dumps(result_metrics(result), ensure_ascii=False, indent=2), encoding="utf-8")
    (directory / "report.md").write_text("# Execução individual\n\nRelatório automático da seção 8.\n", encoding="utf-8")
    return generate_pdf_report(directory, f"Execução individual {stamp}", result=result)
