from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from . import __version__
from .bootstrap import ensure_default_glossary, ensure_layout_model, run_all
from .ingest.analyzer import analyze_document
from .keys import list_keys as stored_keys
from .keys import set_key as store_key
from .pipeline import translate_document
from .diagnostics import collect_diagnostics, summarize_checks
from .overview import get_component_map
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
def doctor(json_output: bool = typer.Option(False, "--json", help="Emit diagnostics as JSON instead of a table")):
    """Inspect dependencies, models, and keys to catch issues early."""

    checks = collect_diagnostics()
    if json_output:
        print(json.dumps([c.__dict__ for c in checks], indent=2))
        return

    table = Table(title="SciTrans-LM environment health", show_lines=True)
    table.add_column("Status", justify="center")
    table.add_column("Item")
    table.add_column("Detail")

    icons = {"ok": "✅", "warn": "⚠️", "error": "❌"}
    for c in checks:
        table.add_row(icons.get(c.status, "•"), c.name, c.detail)

    console = Console()
    console.print(table)
    summary = summarize_checks(checks)
    rprint(
        f"[bold]{summary['ok']} OK[/bold], "
        f"[yellow]{summary['warn']} warning(s)[/yellow], "
        f"[red]{summary['error']} error(s)[/red]."
    )


@app.command()
def map(json_output: bool = typer.Option(False, "--json", help="Return the component map as JSON")):
    """Show a concise architecture map so newcomers can find the right modules."""

    components = get_component_map()
    if json_output:
        print(json.dumps([c.to_dict() for c in components], indent=2))
        return

    table = Table(title="SciTrans-LM component map", show_lines=True)
    table.add_column("Component")
    table.add_column("Responsibility")
    table.add_column("Key files")
    table.add_column("Notes")

    for comp in components:
        table.add_row(comp.name, comp.responsibility, "\n".join(comp.key_files), comp.notes)

    console = Console()
    console.print(table)


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
    engine: str = typer.Option("dictionary", "--engine", help="Backend engine (openai|deepl|google|google-free|deepseek|perplexity|dictionary)"),
    direction: str = typer.Option("en-fr", "--direction", help="en-fr or fr-en"),
    pages: str = typer.Option("all", "--pages", help="Page range, e.g., all or 1-5"),
    preserve_figures: bool = typer.Option(True, "--preserve-figures", help="Preserve figures & formulas"),
    quality_loops: int = typer.Option(3, "--quality-loops", min=1, max=6, help="Number of refinement loops"),
    rerank: bool = typer.Option(True, "--rerank/--no-rerank", help="Enable glossary-aware reranking"),
    preview: bool = typer.Option(False, "--preview/--no-preview", help="Print a short preview of the translated PDF"),
    preview_chars: int = typer.Option(600, "--preview-chars", min=120, max=3200, help="Maximum characters to show when previewing"),
    quiet: bool = typer.Option(False, "--quiet", help="Reduce console noise; still prints final status"),
):
    events: list[str] = []
    progress_bar = Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("{task.description}"),
        BarColumn(bar_width=None),
        TimeElapsedColumn(),
        expand=True,
        transient=quiet,
    )

    with progress_bar:
        task = progress_bar.add_task("Preparing…", total=None)

        def log(msg: str):
            events.append(msg)
            progress_bar.update(task, description=msg)
            if not quiet:
                progress_bar.log(f"[cyan]{msg}[/cyan]")

        outp = translate_document(
            str(i),
            str(o),
            engine=engine,
            direction=direction,
            pages=pages,
            preserve_figures=preserve_figures,
            quality_loops=quality_loops,
            enable_rerank=rerank,
            progress=log,
        )
        progress_bar.update(task, description="Translation complete")

    rprint(f"[green]✔ Wrote {outp}[/green]")
    if preview:
        snippet = _preview_pdf_text(outp, max_chars=preview_chars)
        if snippet:
            rprint("[bold]Preview (first pages):[/bold]")
            rprint(snippet)
        else:
            rprint("[yellow]Preview unavailable (no extractable text).[/yellow]")


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
    notes = []
    summaries = analyze_document(str(i), page_indices, notes=notes)
    for s in summaries[:20]:
        rprint(f"p{s.page_index+1}: [{s.kind}] {s.text_preview}")
    rprint(f"[bold]Total blocks analyzed:[/bold] {len(summaries)}")
    for note in notes:
        rprint(f"[yellow]- {note}[/yellow]")
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


@app.command("engines")
def list_engines():
    """List available translation engines and key requirements."""

    engines = {
        "dictionary": "Offline glossary/dictionary (no key)",
        "google-free": "Keyless Google backend (deep-translator community endpoint)",
        "openai": "Requires OPENAI_API_KEY or stored key",
        "deepl": "Requires DEEPL_API_KEY or stored key",
        "google": "Requires GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_API_KEY",
        "deepseek": "Requires DEEPSEEK_API_KEY",
        "perplexity": "Requires PERPLEXITY_API_KEY",
    }
    for name, desc in engines.items():
        rprint(f"[bold]{name}[/bold]: {desc}")


@app.command("keys")
def show_keys():
    """Show which API keys are already stored (values are not printed)."""

    found = stored_keys()
    if not found:
        rprint("No keys found. Run 'python3 -m scitrans_lm set-key <service>' to store one.")
        return
    for svc in sorted(found):
        state = "present" if found[svc] else "missing"
        rprint(f"{svc}: {state}")


@app.callback()
def main():
    rprint(f"SciTrans-LM v{__version__}")


def _preview_pdf_text(pdf_path: str, pages: int = 2, max_chars: int = 600) -> str:
    """Extract a short preview from the translated PDF for quick inspection."""

    try:
        import fitz
    except Exception:
        return ""
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return ""
    snippets = []
    try:
        for idx in range(min(pages, doc.page_count)):
            page = doc.load_page(idx)
            snippets.append(page.get_text("text"))
    finally:
        doc.close()
    preview = "\n".join(snippets).strip()
    return preview[:max_chars]


if __name__ == "__main__":
    app()
