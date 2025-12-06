"""
Command-line interface for SciTrans-LLMs.

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

from scitrans_llms import __version__
from scitrans_llms.models import Document
from scitrans_llms.pipeline import TranslationPipeline, PipelineConfig
from scitrans_llms.translate.glossary import get_default_glossary, load_glossary_csv

app = typer.Typer(
    name="scitrans",
    help="SciTrans-LLMs: Adaptive Document Translation Enhanced by Technology based on LLMs",
    add_completion=False,
)
console = Console()


def version_callback(value: bool):
    if value:
        console.print(f"SciTrans-LLMs v{__version__}")
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
    """SciTrans-LLMs: Research-grade document translation."""
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
        "dictionary", "--backend", "-b",
        help="Translation backend (dictionary ⭐ default, free, improved-offline, openai, gpt-5.1, deepseek, anthropic, huggingface, ollama)",
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
    # NEW: Advanced options
    passes: int = typer.Option(
        1, "--passes", "-p",
        help="Number of translation/refinement passes (1-5)",
    ),
    rerank: bool = typer.Option(
        False, "--rerank", "-r",
        help="Enable candidate reranking for better quality",
    ),
    candidates: int = typer.Option(
        3, "--candidates",
        help="Number of candidates to generate when reranking",
    ),
    preview_prompt: bool = typer.Option(
        False, "--preview-prompt",
        help="Show the translation prompt before running",
    ),
    interactive: bool = typer.Option(
        False, "--interactive",
        help="Interactive mode: review translations before finalizing",
    ),
    preserve_structure: bool = typer.Option(
        True, "--preserve-structure/--no-preserve-structure",
        help="Preserve section numbers and bullet points",
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
    
    # Configure masking with structure preservation
    from scitrans_llms.masking import MaskConfig
    mask_config = MaskConfig(
        preserve_section_numbers=preserve_structure,
        preserve_bullets=preserve_structure,
        preserve_indentation=preserve_structure,
    )
    
    config = PipelineConfig(
        source_lang=source_lang,
        target_lang=target_lang,
        translator_backend=backend,
        translator_kwargs=translator_kwargs,
        enable_masking=not no_masking,
        mask_config=mask_config,
        enable_glossary=not no_glossary,
        glossary=glossary,
        enable_context=not no_context,
        enable_refinement=not no_refinement,
        num_candidates=candidates if rerank else 1,
    )
    
    # Preview prompt if requested
    if preview_prompt:
        from scitrans_llms.refine.prompting import build_prompt
        prompt = build_prompt(source_lang, target_lang, glossary.to_dict() if glossary else {})
        console.print("\n[bold cyan]Translation Prompt:[/]\n")
        console.print(prompt)
        console.print("\n" + "="*50 + "\n")
        if not typer.confirm("Continue with translation?"):
            raise typer.Exit(0)
    
    # Create document
    if input_text:
        doc = Document.from_text(input_text, source_lang, target_lang)
        console.print(f"[dim]Translating text ({len(input_text)} chars)...[/]")
    elif input_file:
        if input_file.suffix.lower() == ".pdf":
            try:
                from scitrans_llms.ingest import parse_pdf
                doc = parse_pdf(
                    input_file,
                    source_lang=source_lang,
                    target_lang=target_lang,
                )
                console.print(f"[green]Parsed PDF (high-fidelity):[/] {len(doc.all_blocks)} blocks from {input_file}")
            except ImportError:
                console.print("[yellow]PDF parsing requires PyMuPDF. Falling back to text mode.[/]")
                text = input_file.read_text(encoding="utf-8")
                doc = Document.from_text(text, source_lang, target_lang)
        else:
            text = input_file.read_text(encoding="utf-8")
            doc = Document.from_text(text, source_lang, target_lang)
    
    # Translate with progress (multiple passes if requested)
    pipeline = TranslationPipeline(config)
    
    # Clamp passes to valid range
    num_passes = max(1, min(passes, 5))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Translating...", total=100)
        
        def update_progress(msg: str, pct: float):
            # Adjust progress for multiple passes
            adjusted_pct = pct / num_passes
            progress.update(task, description=msg, completed=int(adjusted_pct * 100))
        
        pipeline.progress_callback = update_progress
        
        # Run multiple passes if requested
        result = None
        for pass_num in range(num_passes):
            if pass_num > 0:
                progress.update(task, description=f"Pass {pass_num + 1}/{num_passes}...")
                # For subsequent passes, use previous output as input context
                pipeline.progress_callback = lambda msg, pct: progress.update(
                    task,
                    description=f"[Pass {pass_num + 1}] {msg}",
                    completed=int(((pass_num + pct) / num_passes) * 100)
                )
            
            result = pipeline.translate(doc)
            
            # Interactive review mode
            if interactive and pass_num < num_passes - 1:
                console.print(f"\n[bold]Pass {pass_num + 1} complete. Preview:[/]")
                console.print(result.translated_text[:500] + "..." if len(result.translated_text) > 500 else result.translated_text)
                if not typer.confirm("\nContinue with next pass?"):
                    break
        
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
                    from scitrans_llms.render import render_pdf
                    # BUG FIX: Use result.document (translated) not doc (original)
                    render_pdf(result.document, input_file, output_file)
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
    from scitrans_llms.eval.runner import EvaluationRunner
    
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
        "dictionary", "--backends", "-b",
        help="Comma-separated list of backends to test",
    ),
):
    """Run ablation study comparing configurations."""
    from scitrans_llms.eval.ablation import AblationStudy, AblationConfig
    
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
    console.print("[bold]SciTrans-LLMs Demo[/]\n")
    
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
        translator_backend="dictionary",
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
    from scitrans_llms.keys import KeyManager
    
    console.print(f"[bold]SciTrans-LLMs v{__version__}[/]\n")
    
    # Initialize key manager to check keys from all sources (env, keyring, config)
    km = KeyManager()
    
    # Check available backends
    table = Table(title="Available Translation Backends")
    table.add_column("Backend", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Notes")
    
    # Always available
    table.add_row("dictionary", "✓ Available", "Offline glossary/dictionary translation")
    
    # Check OpenAI
    try:
        import openai
        has_key = km.get_key("openai") is not None
        table.add_row("openai", "✓ Available" if has_key else "⚠ No API key", "GPT-4, GPT-4o, GPT-5.1")
    except ImportError:
        table.add_row("openai", "✗ Not installed", "pip install openai")
    
    # Check Free Cascading Translator
    table.add_row("free", "✓ Available", "⭐ Smart cascade: Lingva→LibreTranslate→MyMemory")
    
    # Check improved offline
    table.add_row("improved-offline", "✓ Available", "Enhanced offline translation")
    
    # Check Hugging Face
    try:
        import requests
        has_hf_key = km.get_key("huggingface") is not None
        status = "✓ Available (with key)" if has_hf_key else "✓ Available (free tier)"
        table.add_row("huggingface", status, "Free API, 1000 req/month")
    except ImportError:
        table.add_row("huggingface", "✗ Not installed", "pip install requests")
    
    # Check Ollama
    try:
        import requests
        ollama_resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        if ollama_resp.status_code == 200:
            models = ollama_resp.json().get("models", [])
            model_names = [m.get("name", "?") for m in models[:3]]
            model_info = ", ".join(model_names) if model_names else "no models"
            table.add_row("ollama", f"✓ Running ({model_info})", "Local LLM (free, offline)")
        else:
            table.add_row("ollama", "⚠ Running (no models)", "Pull with: ollama pull llama3")
    except Exception:
        table.add_row("ollama", "✗ Not running", "Start with: ollama serve")
    
    # Check Google Free (uses deep-translator)
    try:
        from deep_translator import GoogleTranslator as _GT
        table.add_row("googlefree", "✓ Available", "Uses deep-translator (no API key)")
    except ImportError:
        table.add_row("googlefree", "✗ Not installed", "pip install deep-translator")
    
    # Check DeepSeek
    has_ds_key = km.get_key("deepseek") is not None
    table.add_row("deepseek", "✓ Available" if has_ds_key else "⚠ No API key", "Uses OpenAI client")
    
    # Check Anthropic
    try:
        import anthropic
        has_key = km.get_key("anthropic") is not None
        table.add_row("anthropic", "✓ Available" if has_key else "⚠ No API key", "Claude 3")
    except ImportError:
        table.add_row("anthropic", "✗ Not installed", "pip install anthropic")
    
    # Check DeepL
    has_deepl_key = km.get_key("deepl") is not None
    table.add_row("deepl", "✓ Available" if has_deepl_key else "⚠ No API key", "DeepL API (paid)")
    
    # Check Google Cloud
    has_google_key = km.get_key("google") is not None
    table.add_row("google", "✓ Available" if has_google_key else "⚠ No API key", "Google Cloud Translation")
    
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


@app.command()
def keys(
    action: str = typer.Argument(..., help="Action: list, set, get, delete, status, export"),
    service: Optional[str] = typer.Argument(None, help="Service name (openai, deepseek, anthropic, etc.)"),
):
    """Manage API keys securely.
    
    Examples:
        scitrans keys list              # List all keys
        scitrans keys set openai        # Set OpenAI key
        scitrans keys status openai     # Check OpenAI key status
        scitrans keys delete openai     # Delete OpenAI key
        scitrans keys export            # Export keys as env vars
    """
    from scitrans_llms.keys import KeyManager, SERVICES
    
    km = KeyManager()
    
    if action == "list":
        keys_info = km.list_keys()
        
        if not keys_info:
            console.print("[yellow]No API keys configured.[/]")
            console.print("\nTo set a key, run: [cyan]scitrans keys set <service>[/]")
            return
        
        table = Table(title="API Keys Status")
        table.add_column("Service", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Source", style="yellow")
        table.add_column("Value", style="dim")
        
        for key_info in keys_info:
            status = "✓ Set" if key_info.is_set else "✗ Not set"
            status_color = "green" if key_info.is_set else "red"
            table.add_row(
                key_info.service,
                f"[{status_color}]{status}[/]",
                key_info.source,
                key_info.masked_value if key_info.is_set else "-",
            )
        
        console.print(table)
        console.print("\n[dim]Priority: env > keychain > config file[/]")
    
    elif action == "set":
        if not service:
            console.print("[red]Error:[/] Service name required")
            console.print("Usage: [cyan]scitrans keys set <service>[/]")
            console.print(f"Available services: {', '.join(SERVICES.keys())}")
            raise typer.Exit(1)
        
        # Prompt for key
        from getpass import getpass
        key = getpass(f"Enter API key for {service}: ")
        
        if not key:
            console.print("[red]Error:[/] Key cannot be empty")
            raise typer.Exit(1)
        
        storage = km.set_key(service, key)
        console.print(f"[green]✓[/] API key for {service} saved to {storage}")
        
        if storage == "config":
            console.print("[yellow]Note:[/] Key stored in local file (~/.scitrans/keys.json)")
            console.print("       For better security, use environment variables")
    
    elif action == "get":
        if not service:
            console.print("[red]Error:[/] Service name required")
            raise typer.Exit(1)
        
        key = km.get_key(service)
        if key:
            console.print(f"[green]✓[/] Key found: {km._mask_key(key)}")
        else:
            console.print(f"[red]✗[/] No key found for {service}")
            console.print(f"Set with: [cyan]scitrans keys set {service}[/]")
    
    elif action == "status":
        if not service:
            console.print("[red]Error:[/] Service name required")
            raise typer.Exit(1)
        
        key_info = km.get_key_info(service)
        
        if key_info.is_set:
            console.print(f"[green]✓[/] API key for {service} is set")
            console.print(f"    Source: {key_info.source}")
            console.print(f"    Value: {key_info.masked_value}")
        else:
            console.print(f"[red]✗[/] No API key found for {service}")
            env_var = SERVICES.get(service, f"{service.upper()}_API_KEY")
            console.print(f"\nTo set the key:")
            console.print(f"  Option 1: [cyan]scitrans keys set {service}[/]")
            console.print(f"  Option 2: [cyan]export {env_var}='your-key-here'[/]")
    
    elif action == "delete":
        if not service:
            console.print("[red]Error:[/] Service name required")
            raise typer.Exit(1)
        
        deleted = km.delete_key(service)
        if deleted:
            console.print(f"[green]✓[/] API key for {service} deleted")
        else:
            console.print(f"[yellow]⚠[/] No key found to delete for {service}")
    
    elif action == "export":
        env_vars = km.export_to_env()
        
        if not env_vars:
            console.print("[yellow]No keys to export.[/]")
            return
        
        console.print("[bold]Export these to your environment:[/]\n")
        for var, value in env_vars.items():
            console.print(f"export {var}='{value}'")
        
        console.print("\n[dim]Or add to your ~/.bashrc or ~/.zshrc[/]")
    
    else:
        console.print(f"[red]Error:[/] Unknown action '{action}'")
        console.print("Available actions: list, set, get, delete, status, export")
        raise typer.Exit(1)


@app.command()
def corpus(
    action: str = typer.Argument(..., help="Action: list, download, build-dict, status"),
    name: Optional[str] = typer.Argument(None, help="Corpus name (europarl, opus-euconst, tatoeba)"),
    source_lang: str = typer.Option("en", "--source", "-s", help="Source language"),
    target_lang: str = typer.Option("fr", "--target", "-t", help="Target language"),
    limit: int = typer.Option(10000, "--limit", "-l", help="Max entries for dictionary"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for dictionary"),
):
    """Manage downloadable translation corpora.
    
    Available corpora:
    - europarl: European Parliament parallel corpus (~200MB)
    - opus-euconst: EU Constitution corpus (~5MB)
    - tatoeba: Crowdsourced sentences (~50MB)
    
    Examples:
        scitrans corpus list                    # List available corpora
        scitrans corpus download europarl       # Download Europarl EN-FR
        scitrans corpus build-dict europarl     # Build dictionary from Europarl
        scitrans corpus status                  # Show downloaded corpora
    """
    from scitrans_llms.translate.corpus_manager import (
        CorpusManager, list_corpora, download_corpus, get_corpus_dictionary
    )
    
    if action == "list":
        corpora = list_corpora()
        
        table = Table(title="Available Translation Corpora")
        table.add_column("Name", style="cyan")
        table.add_column("Description")
        table.add_column("Languages", style="green")
        table.add_column("Size", style="yellow")
        table.add_column("License", style="dim")
        
        for c in corpora:
            langs = ", ".join(c["languages"][:5])
            if len(c["languages"]) > 5:
                langs += f" (+{len(c['languages']) - 5} more)"
            table.add_row(
                c["key"],
                c["description"],
                langs,
                f"{c['size_mb']}MB",
                c["license"],
            )
        
        console.print(table)
        console.print("\n[dim]Download with: scitrans corpus download <name>[/]")
    
    elif action == "status":
        manager = CorpusManager()
        downloaded = manager.list_downloaded()
        
        if not downloaded:
            console.print("[yellow]No corpora downloaded yet.[/]")
            console.print("Download with: [cyan]scitrans corpus download europarl[/]")
        else:
            console.print("[green]Downloaded corpora:[/]")
            for name in downloaded:
                console.print(f"  • {name}")
    
    elif action == "download":
        if not name:
            console.print("[red]Error:[/] Corpus name required")
            console.print("Usage: [cyan]scitrans corpus download <name>[/]")
            raise typer.Exit(1)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Downloading...", total=100)
            
            def update(msg: str, pct: float):
                progress.update(task, description=msg, completed=int(pct * 100))
            
            try:
                path = download_corpus(name, source_lang, target_lang, update)
                progress.update(task, description="[green]Complete!", completed=100)
                console.print(f"\n[green]✓[/] Downloaded to: {path}")
            except Exception as e:
                console.print(f"\n[red]Error:[/] {e}")
                raise typer.Exit(1)
    
    elif action == "build-dict":
        if not name:
            console.print("[red]Error:[/] Corpus name required")
            raise typer.Exit(1)
        
        console.print(f"[dim]Building dictionary from {name} ({source_lang}-{target_lang})...[/]")
        
        try:
            dictionary = get_corpus_dictionary(name, source_lang, target_lang, limit)
            
            console.print(f"[green]✓[/] Built dictionary with {len(dictionary)} entries")
            
            # Show sample
            console.print("\n[bold]Sample entries:[/]")
            for src, tgt in list(dictionary.items())[:5]:
                console.print(f"  {src[:50]} → {tgt[:50]}")
            
            # Save dictionary
            import csv, os
            if output is None:
                # Default location discovered automatically by DictionaryTranslator
                default_root = Path(os.path.expanduser("~")) / ".scitrans" / "dictionaries"
                default_root.mkdir(parents=True, exist_ok=True)
                output_path = default_root / f"{name}_{source_lang}_{target_lang}.tsv"
            else:
                output_path = output
            
            # Write as TSV: source<TAB>target (preferred by DictionaryTranslator)
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter='\t')
                for src, tgt in dictionary.items():
                    writer.writerow([src, tgt])
            console.print(f"\n[green]Saved to:[/] {output_path}")
            console.print("[dim]Dictionary will be auto-loaded by the 'dictionary' backend when matching languages are used.[/]")
            
        except Exception as e:
            console.print(f"[red]Error:[/] {e}")
            raise typer.Exit(1)
    
    else:
        console.print(f"[red]Unknown action:[/] {action}")
        console.print("Available: list, download, build-dict, status")
        raise typer.Exit(1)


@app.command()
def gui(
    port: int = typer.Option(7860, "--port", "-p", help="Port to run GUI on"),
    share: bool = typer.Option(False, "--share", "-s", help="Share publicly (if supported)"),
    experimental: bool = typer.Option(False, "--experimental", help="Use experimental Gradio interface (UNSTABLE)"),
):
    """Launch the web GUI for scientific document translation.
    
    Opens a browser window with the NiceGUI interface.
    
    Features:
    - PDF document support with layout preservation
    - Drag-and-drop file upload
    - URL fetching for online PDFs
    - French ↔ English bilingual translation
    - Page selection and quality options
    - Custom glossary support (CSV/TXT/JSON)
    - Multiple translation engines (Free, OpenAI, DeepSeek, etc.)
    - Corpus training for dictionary backend
    - Real-time translation progress
    
    Use --experimental flag to test the unstable Gradio interface (NOT RECOMMENDED).
    """
    # Check PDF support
    try:
        import fitz
    except ImportError:
        console.print("[yellow]Warning:[/] PyMuPDF not installed - PDF features limited")
        console.print("  [cyan]pip install PyMuPDF[/]")
    
    try:
        if experimental:
            # Launch experimental Gradio GUI
            try:
                import gradio
            except ImportError:
                console.print("[red]Error:[/] Gradio not installed")
                console.print("  [cyan]pip install 'gradio>=4.0.0'[/]")
                raise typer.Exit(1)
            
            from scitrans_llms.gui_gradio import launch
            
            console.print("[bold yellow]⚠️  Starting EXPERIMENTAL Gradio GUI (UNSTABLE)...[/bold yellow]\n")
            console.print(f"[dim]Opening browser at http://127.0.0.1:{port}[/]")
            console.print("[dim]Press Ctrl+C to stop the server[/]\n")
            console.print("[yellow]Warning: This interface has known bugs. Use at your own risk.[/]")
            console.print("[dim]For stable interface, run: scitrans gui[/]\n")
            
            launch(port=port, share=share)
        else:
            # Launch stable NiceGUI (default)
            try:
                import nicegui
            except ImportError:
                console.print("[red]Error:[/] NiceGUI not installed")
                console.print("  [cyan]pip install 'nicegui>=1.4.0'[/]")
                raise typer.Exit(1)
            
            from scitrans_llms.gui import launch
            
            console.print("[bold]🚀 Starting SciTrans-LLMs GUI (NiceGUI)...[/]\n")
            console.print(f"[dim]Opening browser at http://127.0.0.1:{port}[/]")
            console.print("[dim]Press Ctrl+C to stop the server[/]\n")
            
            launch(port=port, share=share)
    except Exception as e:
        console.print(f"[red]Error launching GUI:[/] {e}")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
