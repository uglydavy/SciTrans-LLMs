
from __future__ import annotations
import typer, sys
from pathlib import Path
from typing import Optional
from .pipeline import translate_document
from .bootstrap import run_all, ensure_layout_model, ensure_default_glossary
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
          glossary: bool = typer.Option(False, "--glossary", help="Create default glossary")):
    if all or yolo:
        ensure_layout_model()
        rprint("[green]✔ Layout model placeholder ensured. Run training/downloader to replace with real weights.[/green]")
    if all or glossary:
        ensure_default_glossary()
        rprint("[green]✔ Default glossary created (data/glossary/default_en_fr.csv).[/green]")
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

@app.callback()
def main():
    rprint(f"SciTrans-LM v{__version__}")

if __name__ == "__main__":
    app()
