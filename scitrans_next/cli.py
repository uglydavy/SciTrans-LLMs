"""
Command-line interface for SciTrans-Next.

Provides commands for:
- Translating documents (text and PDF)
- Managing glossaries
- Evaluating translations
- Running ablation studies
- System diagnostics

Usage:
    scitrans translate --input doc.pdf --output translated.pdf
    scitrans glossary --list
    scitrans evaluate --hyp output.txt --ref reference.txt
    scitrans ablation --input docs/ --refs refs/
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from scitrans_next import __version__
from scitrans_next.models import Document
from scitrans_next.pipeline import TranslationPipeline, PipelineConfig
from scitrans_next.translate.glossary import get_default_glossary, load_glossary_csv

app = typer.Typer(
    name="scitrans",
    help="SciTrans-Next: Adaptive Document Translation Enhanced by LLMs",
    add_completion=False,
)
console = Console()


def version_callback(value: bool):
    if value:
        console.print(f"SciTrans-Next v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
):
    """SciTrans-Next: Research-grade document translation."""
    pass


@app.command()
def translate(
    input_text: Optional[str] = typer.Option(
        None, "--text", "-t",
        help="Text to translate (for quick tests)",
    ),
    input_file: Optional[Path] = typer.Option(
        None, "--input", "-i",
        help="Input file (PDF or TXT)",
    ),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output file path",
    ),
    source_lang: str = typer.Option(
        "en", "--source", "-s",
        help="Source language code",
    ),
    target_lang: str = typer.Option(
        "fr", "--target", "-l",
        help="Target language code",
    ),
    backend: str = typer.Option(
        "dummy", "--backend", "-b",
        help="Translation backend (dummy, dictionary, openai, deepseek, anthropic)",
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="Model name for LLM backends",
    ),
    glossary_file: Optional[Path] = typer.Option(
        None, "--glossary", "-g",
        help="Custom glossary CSV file",
    ),
    no_glossary: bool = typer.Option(
        False, "--no-glossary",
        help="Disable glossary (for ablation)",
    ),
    no_refinement: bool = typer.Option(
        False, "--no-refinement",
        help="Disable refinement pass (for ablation)",
    ),
    no_masking: bool = typer.Option(
        False, "--no-masking",
        help="Disable masking (for ablation)",
    ),
    no_context: bool = typer.Option(
        False, "--no-context",
        help="Disable document context (for ablation)",
    ),
):
    """Translate text or documents."""
    
    # Validate input
    if not input_text and not input_file:
        console.print("[red]Error:[/] Provide either --text or --input", style="bold")
        raise typer.Exit(1)
    
    # Load glossary
    glossary = None
    if not no_glossary:
        if glossary_file and glossary_file.exists():
            glossary = load_glossary_csv(glossary_file)
            console.print(f"[green]Loaded glossary:[/] {len(glossary)} terms from {glossary_file}")
        else:
            glossary = get_default_glossary()
            console.print(f"[green]Using default glossary:[/] {len(glossary)} terms")
    
    # Configure pipeline
    translator_kwargs = {}
    if model:
        translator_kwargs["model"] = model
    
    config = PipelineConfig(
        source_lang=source_lang,
        target_lang=target_lang,
        translator_backend=backend,
        translator_kwargs=translator_kwargs,
        enable_masking=not no_masking,
        enable_glossary=not no_glossary,
        glossary=glossary,
        enable_context=not no_context,
        enable_refinement=not no_refinement,
    )
    
    # Create document
    if input_text:
        doc = Document.from_text(input_text, source_lang, target_lang)
        console.print(f"[dim]Translating text ({len(input_text)} chars)...[/]")
    elif input_file:
        if input_file.suffix.lower() == ".pdf":
            try:
                from scitrans_next.ingest import parse_pdf
                doc = parse_pdf(input_file, source_lang=source_lang, target_lang=target_lang)
                console.print(f"[green]Parsed PDF:[/] {len(doc.all_blocks)} blocks from {input_file}")
            except ImportError:
                console.print("[yellow]PDF parsing requires PyMuPDF. Falling back to text mode.[/]")
                text = input_file.read_text(encoding="utf-8")
                doc = Document.from_text(text, source_lang, target_lang)
        else:
            text = input_file.read_text(encoding="utf-8")
            doc = Document.from_text(text, source_lang, target_lang)
    
    # Translate with progress
    pipeline = TranslationPipeline(config)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Translating...", total=100)
        
        def update_progress(msg: str, pct: float):
            progress.update(task, description=msg, completed=int(pct * 100))
        
        pipeline.progress_callback = update_progress
        result = pipeline.translate(doc)
        progress.update(task, description="[green]Complete!", completed=100)
    
    # Output
    if result.success:
        console.print("\n[bold green]Translation complete![/]\n")
        
        # Show stats
        table = Table(title="Translation Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        for key, value in result.stats.items():
            table.add_row(key.replace("_", " ").title(), str(value))
        console.print(table)
        
        # Show result
        if not output_file:
            console.print("\n[bold]Translated text:[/]\n")
            console.print(result.translated_text)
        
        # Save if output specified
        if output_file:
            if output_file.suffix.lower() == ".pdf" and input_file and input_file.suffix.lower() == ".pdf":
                try:
                    from scitrans_next.render import render_pdf
                    render_pdf(doc, input_file, output_file)
                    console.print(f"\n[green]Saved PDF to:[/] {output_file}")
                except ImportError:
                    # Fallback to text
                    output_file = output_file.with_suffix(".txt")
                    output_file.write_text(result.translated_text, encoding="utf-8")
                    console.print(f"\n[yellow]PDF rendering unavailable. Saved text to:[/] {output_file}")
            else:
                output_file.write_text(result.translated_text, encoding="utf-8")
                console.print(f"\n[green]Saved to:[/] {output_file}")
    else:
        console.print(f"[red]Translation failed with {len(result.errors)} errors:[/]")
        for error in result.errors:
            console.print(f"  - {error}")


@app.command()
def glossary(
    list_terms: bool = typer.Option(
        False, "--list", "-l",
        help="List all terms in default glossary",
    ),
    domain: Optional[str] = typer.Option(
        None, "--domain", "-d",
        help="Filter by domain (e.g., 'ml', 'math')",
    ),
    search: Optional[str] = typer.Option(
        None, "--search", "-s",
        help="Search for a term",
    ),
):
    """Manage and view glossaries."""
    gloss = get_default_glossary()
    
    if domain:
        gloss = gloss.filter_by_domain(domain)
    
    if search:
        target = gloss.get_target(search)
        if target:
            console.print(f"[green]{search}[/] → [cyan]{target}[/]")
        else:
            console.print(f"[yellow]Term not found:[/] {search}")
        return
    
    if list_terms or (not search and not domain):
        table = Table(title=f"Glossary: {gloss.name} ({len(gloss)} terms)")
        table.add_column("Source (EN)", style="cyan")
        table.add_column("Target (FR)", style="green")
        table.add_column("Domain", style="dim")
        
        for entry in sorted(gloss.entries, key=lambda e: e.source.lower()):
            table.add_row(entry.source, entry.target, entry.domain)
        
        console.print(table)


@app.command()
def evaluate(
    hypothesis: Path = typer.Option(
        ..., "--hyp", "-h",
        help="System output file",
    ),
    reference: Path = typer.Option(
        ..., "--ref", "-r",
        help="Reference translation file",
    ),
    source: Optional[Path] = typer.Option(
        None, "--source", "-s",
        help="Source file (for glossary/numeric checks)",
    ),
    glossary_file: Optional[Path] = typer.Option(
        None, "--glossary", "-g",
        help="Glossary file for adherence check",
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output file for results (json, csv, or tex)",
    ),
):
    """Evaluate translation quality."""
    from scitrans_next.eval.runner import EvaluationRunner
    
    gloss = None
    if glossary_file:
        gloss = load_glossary_csv(glossary_file)
    elif source:
        gloss = get_default_glossary()
    
    runner = EvaluationRunner(glossary=gloss)
    report = runner.evaluate_files(hypothesis, reference, source)
    
    console.print(report.summary())
    
    if output:
        runner.save_report(report, output)
        console.print(f"\n[green]Saved report to:[/] {output}")


@app.command()
def ablation(
    input_dir: Path = typer.Option(
        ..., "--input", "-i",
        help="Directory with source documents",
    ),
    reference_dir: Path = typer.Option(
        ..., "--refs", "-r",
        help="Directory with reference translations",
    ),
    output: Path = typer.Option(
        "ablation_results.json", "--output", "-o",
        help="Output file for results",
    ),
    backends: str = typer.Option(
        "dummy", "--backends", "-b",
        help="Comma-separated list of backends to test",
    ),
):
    """Run ablation study comparing configurations."""
    from scitrans_next.eval.ablation import AblationStudy, AblationConfig
    
    # Load documents
    docs = []
    refs = []
    sources = []
    
    for src_file in sorted(input_dir.glob("*.txt")):
        text = src_file.read_text(encoding="utf-8")
        doc = Document.from_text(text)
        docs.append(doc)
        sources.append([b.source_text for b in doc.all_blocks if b.is_translatable])
        
        ref_file = reference_dir / src_file.name
        if ref_file.exists():
            ref_text = ref_file.read_text(encoding="utf-8")
            ref_paras = [p.strip() for p in ref_text.split("\n\n") if p.strip()]
            refs.append(ref_paras)
        else:
            refs.append(sources[-1])  # Use source as fallback
    
    if not docs:
        console.print("[red]No documents found in input directory[/]")
        raise typer.Exit(1)
    
    config = AblationConfig(
        name="cli_ablation",
        backends=backends.split(","),
    )
    
    study = AblationStudy(config=config)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running ablation...", total=100)
        
        def update(msg: str, pct: float):
            progress.update(task, description=msg, completed=int(pct * 100))
        
        study.run(docs, refs, sources, get_default_glossary(), update)
    
    console.print("\n" + study.summary())
    study.save(output)
    console.print(f"\n[green]Saved results to:[/] {output}")


@app.command()
def demo():
    """Run a quick demo of the translation pipeline."""
    console.print("[bold]SciTrans-Next Demo[/]\n")
    
    # Sample scientific text
    sample_text = """
Machine learning has revolutionized natural language processing.

The transformer architecture, introduced in the paper "Attention is All You Need",
uses attention mechanisms to process sequential data. The model computes
attention weights using the formula $Q K^T / \\sqrt{d_k}$.

Recent advances in deep learning have enabled large language models to achieve
state of the art performance on machine translation tasks.

For more information, see https://arxiv.org/abs/1706.03762.
"""
    
    console.print("[dim]Source text:[/]")
    console.print(sample_text)
    console.print()
    
    # Create document
    doc = Document.from_text(sample_text, "en", "fr")
    console.print(doc.summary())
    console.print()
    
    # Configure and run pipeline
    config = PipelineConfig(
        translator_backend="dummy",
        enable_glossary=True,
        enable_refinement=True,
    )
    pipeline = TranslationPipeline(config)
    
    result = pipeline.translate(doc)
    
    console.print("[bold green]Translation result:[/]")
    console.print(result.translated_text)
    console.print()
    
    # Show stats
    console.print("[dim]Stats:[/]", result.stats)


@app.command()
def info():
    """Show system information and available backends."""
    console.print(f"[bold]SciTrans-Next v{__version__}[/]\n")
    
    # Check available backends
    table = Table(title="Available Translation Backends")
    table.add_column("Backend", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Notes")
    
    # Always available
    table.add_row("dummy", "✓ Available", "For testing")
    table.add_row("dictionary", "✓ Available", "Glossary-only translation")
    
    # Check OpenAI
    try:
        import openai
        import os
        has_key = bool(os.getenv("OPENAI_API_KEY"))
        table.add_row("openai", "✓ Available" if has_key else "⚠ No API key", "GPT-4, GPT-4o")
    except ImportError:
        table.add_row("openai", "✗ Not installed", "pip install openai")
    
    # Check DeepSeek
    import os
    has_ds_key = bool(os.getenv("DEEPSEEK_API_KEY"))
    table.add_row("deepseek", "✓ Available" if has_ds_key else "⚠ No API key", "Uses OpenAI client")
    
    # Check Anthropic
    try:
        import anthropic
        has_key = bool(os.getenv("ANTHROPIC_API_KEY"))
        table.add_row("anthropic", "✓ Available" if has_key else "⚠ No API key", "Claude 3")
    except ImportError:
        table.add_row("anthropic", "✗ Not installed", "pip install anthropic")
    
    console.print(table)
    
    # Check PDF support
    console.print("\n[bold]PDF Support:[/]")
    try:
        import fitz
        console.print("  [green]✓[/] PyMuPDF installed")
    except ImportError:
        console.print("  [yellow]✗[/] PyMuPDF not installed (pip install PyMuPDF)")
    
    # Check evaluation
    console.print("\n[bold]Evaluation:[/]")
    try:
        import sacrebleu
        console.print("  [green]✓[/] SacreBLEU installed")
    except ImportError:
        console.print("  [yellow]✗[/] SacreBLEU not installed (pip install sacrebleu)")


if __name__ == "__main__":
    app()
