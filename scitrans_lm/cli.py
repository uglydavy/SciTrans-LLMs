from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Optional

import typer
from rich import print as rprint

from . import __version__
from .bootstrap import ensure_default_glossary, ensure_layout_model, run_all
from .ingest.analyzer import analyze_document
from .keys import set_key as store_key
from .pipeline import translate_document
from .utils import parse_page_range

app = typer.Typer(add_completion=False, help="SciTrans-LM – EN↔FR scientific PDF translator (GUI + CLI)")


@app.command()
def gui():
    """Launch modern web GUI (Gradio)."""
    from .gui import launch

    launch()


@app.command()
def setup(
    all: bool = typer.Option(False, "--all", help="Run all setup steps"),
    yolo: bool = typer.Option(False, "--yolo", help="Ensure/download YOLO layout model"),
    glossary: bool = typer.Option(False, "--glossary", help="Create default glossary"),
):
    if all or yolo:
        ensure_layout_model()
        rprint("[green]✔ Layout model placeholder ensured. Run training/downloader to replace with real weights.[/green]")
    if all or glossary:
        ensure_default_glossary()
        rprint("[green]✔ Default glossary created (data/glossary/default_en_fr.csv).[/green]")
    if not (all or yolo or glossary):
        run_all()


@app.command()
def set_key(
    service: str = typer.Argument(..., help="Service name: openai|deepl|google|deepseek|perplexity"),
    key: Optional[str] = typer.Option(None, "--key", help="API key (omit to be prompted)"),
):
    if key is None:
        key = typer.prompt(f"Enter API key for {service}", hide_input=True)
    store_key(service, key)
    rprint(f"[green]✔ Stored key for {service}[/green]")


@app.command()
def translate(
    i: Path = typer.Option(..., "--input", "-i", exists=True, file_okay=True, dir_okay=False, readable=True, help="Input PDF"),
    o: Path = typer.Option(..., "--output", "-o", help="Output PDF path"),
    engine: str = typer.Option("dictionary", "--engine", help="Backend engine (openai|deepl|google|deepseek|perplexity|dictionary)"),
    direction: str = typer.Option("en-fr", "--direction", help="en-fr or fr-en"),
    pages: str = typer.Option("all", "--pages", help="Page range, e.g., all or 1-5"),
    preserve_figures: bool = typer.Option(True, "--preserve-figures", help="Preserve figures & formulas"),
    quality_loops: int = typer.Option(3, "--quality-loops", min=1, max=6, help="Number of refinement loops"),
    rerank: bool = typer.Option(True, "--rerank/--no-rerank", help="Enable glossary-aware reranking"),
):
    outp = translate_document(
        str(i),
        str(o),
        engine=engine,
        direction=direction,
        pages=pages,
        preserve_figures=preserve_figures,
        quality_loops=quality_loops,
        enable_rerank=rerank,
    )
    rprint(f"[green]✔ Wrote {outp}[/green]")


@app.command()
def inspect(
    i: Path = typer.Option(..., "--input", "-i", exists=True, file_okay=True, dir_okay=False, readable=True, help="Input PDF"),
    pages: str = typer.Option("all", "--pages", help="Page range, e.g., all or 1-5"),
    json_output: Optional[Path] = typer.Option(None, "--json", help="Optional path to save analysis JSON"),
):
    import fitz

    doc = fitz.open(i)
    total = doc.page_count
    doc.close()
    s, e = parse_page_range(pages, total)
    page_indices = list(range(s, e + 1))
    summaries = analyze_document(str(i), page_indices)
    for s in summaries[:20]:
        rprint(f"p{s.page_index+1}: [{s.kind}] {s.text_preview}")
    rprint(f"[bold]Total blocks analyzed:[/bold] {len(summaries)}")
    if json_output:
        json_output.write_text(json.dumps([s.__dict__ for s in summaries], indent=2), encoding="utf-8")
        rprint(f"[green]✔ Saved analysis to {json_output}[/green]")


@app.command()
def evaluate(
    ref: Path = typer.Option(..., "--ref", exists=True, help="Reference translation file or directory"),
    hyp: Path = typer.Option(..., "--hyp", exists=True, help="Hypothesis translation file or directory"),
):
    """Compute SacreBLEU between reference(s) and hypothesis."""
    from .refine.scoring import bleu

    def _collect(p: Path) -> Dict[str, str]:
        if p.is_file():
            return {"__single__": p.read_text(encoding="utf-8")}
        data = {}
        for child in sorted(p.glob("*")):
            if child.is_file():
                data[child.name] = child.read_text(encoding="utf-8")
        if not data:
            raise typer.BadParameter(f"No files found under {p}")
        return data

    refs = _collect(ref)
    hyps = _collect(hyp)
    missing = set(refs) - set(hyps)
    if missing:
        raise typer.BadParameter(f"Missing hypothesis files for: {', '.join(sorted(missing))}")
    scores = []
    for key in sorted(refs):
        score = bleu(hyps[key], refs[key])
        scores.append(score)
        rprint(f"{key}: {score:.2f} BLEU")
    avg = sum(scores) / len(scores)
    rprint(f"[bold]Average BLEU: {avg:.2f}[/bold]")


@app.callback()
def main():
    rprint(f"SciTrans-LM v{__version__}")


if __name__ == "__main__":
    app()
