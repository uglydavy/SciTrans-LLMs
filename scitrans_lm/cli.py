
from __future__ import annotations
import typer
from pathlib import Path
from typing import Dict, Optional
from .pipeline import translate_document
from .bootstrap import ensure_default_glossary, ensure_layout_model, run_all
from .keys import set_key as store_key
from . import __version__
from rich import print as rprint

app = typer.Typer(add_completion=False, help="SciTrans-LM – EN↔FR scientific PDF translator (GUI + CLI)")

@app.command()
def gui():
    """Launch modern web GUI (Gradio)."""
    from .gui import launch
    launch()

@app.command()
def setup(all: bool = typer.Option(False, "--all", help="Run all setup steps"),
          yolo: bool = typer.Option(False, "--yolo", help="Ensure/download YOLO layout model"),
          glossary: bool = typer.Option(False, "--glossary", help="Create default glossary"),
          refresh_remote: bool = typer.Option(False, "--refresh-remote", help="Fetch/update remote bilingual glossary (FreeDict by default)"),
          glossary_url: Optional[str] = typer.Option(None, "--glossary-url", help="Optional custom glossary URL (CSV or TSV)")):
    if all or yolo:
        ensure_layout_model()
        rprint("[green]✔ Layout model placeholder ensured. Run training/downloader to replace with real weights.[/green]")
    if all or glossary:
        ensure_default_glossary(refresh_remote=refresh_remote, remote_url=glossary_url)
        rprint("[green]✔ Default glossary created (data/glossary/default_en_fr.csv).[/green]")
        if refresh_remote:
            rprint("[green]✔ Remote glossary downloaded into data/glossary/remote_en_fr.csv (if available).[/green]")
    if not (all or yolo or glossary):
        run_all()

@app.command()
def set_key(service: str = typer.Argument(..., help="Service name: openai|deepl|google|deepseek|perplexity"),
            key: Optional[str] = typer.Option(None, "--key", help="API key (omit to be prompted)")):
    if key is None:
        key = typer.prompt(f"Enter API key for {service}", hide_input=True)
    store_key(service, key)
    rprint(f"[green]✔ Stored key for {service}[/green]")

@app.command()
def translate(i: Path = typer.Option(..., '--input', '-i', exists=True, file_okay=True, dir_okay=False, readable=True, help='Input PDF'),
              o: Path = typer.Option(..., '--output', '-o', help='Output PDF path'),
              engine: str = typer.Option('dictionary', '--engine', help='Backend engine (openai|deepl|google|deepseek|perplexity|dictionary)'),
              direction: str = typer.Option('en-fr', '--direction', help='en-fr or fr-en'),
              pages: str = typer.Option('all', '--pages', help='Page range, e.g., all or 1-5'),
              preserve_figures: bool = typer.Option(True, '--preserve-figures', help='Preserve figures & formulas')):
    outp = translate_document(str(i), str(o), engine=engine, direction=direction, pages=pages, preserve_figures=preserve_figures)
    rprint(f"[green]✔ Wrote {outp}[/green]")

@app.command()
def evaluate(ref: Path = typer.Option(..., '--ref', exists=True, help='Reference translation file or directory'),
             hyp: Path = typer.Option(..., '--hyp', exists=True, help='Hypothesis translation file or directory')):
    """Compute SacreBLEU between reference(s) and hypothesis."""
    from .refine.scoring import bleu

    def _collect(p: Path) -> Dict[str, str]:
        if p.is_file():
            return {"__single__": p.read_text(encoding="utf-8")}
        data = {}
        for child in sorted(p.glob('*')):
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
