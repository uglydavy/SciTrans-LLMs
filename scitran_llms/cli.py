"""
Command-line interface for SciTrans-LLMs.
"""

import sys
import os
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import track

from .pipeline import TranslationPipeline, PipelineConfig
from .models import Document, Block, BlockType
from .config import BACKENDS, LANGUAGES, DEFAULT_BACKEND, MODEL_DESCRIPTIONS
from . import __version__

app = typer.Typer(help="SciTrans-LLMs: Scientific Document Translator")
console = Console()


@app.command()
def translate(
    input_file: str = typer.Option(None, "--input", "-i", help="Input file (PDF or text)"),
    output_file: str = typer.Option(None, "--output", "-o", help="Output file"),
    text: str = typer.Option(None, "--text", "-t", help="Text to translate directly"),
    backend: str = typer.Option(DEFAULT_BACKEND, "--backend", "-b", help=f"Translation backend"),
    source_lang: str = typer.Option("en", "--source", "-s", help="Source language"),
    target_lang: str = typer.Option("fr", "--target", "-T", help="Target language"),
    enable_masking: bool = typer.Option(True, "--masking/--no-masking", help="Enable masking"),
    enable_glossary: bool = typer.Option(True, "--glossary/--no-glossary", help="Use glossary"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Translate a document or text."""
    
    try:
        # Create pipeline config
        config = PipelineConfig(
            backend=backend,
            source_lang=source_lang,
            target_lang=target_lang,
            enable_masking=enable_masking,
            enable_glossary=enable_glossary
        )
        
        # Create pipeline
        pipeline = TranslationPipeline(config)
        
        # Translate
        if text:
            # Direct text translation
            result = pipeline.translate_text(text)
            console.print(f"[green]Translation:[/green] {result}")
            
        elif input_file:
            # File translation
            input_path = Path(input_file)
            if not input_path.exists():
                console.print(f"[red]Error:[/red] File not found: {input_file}")
                raise typer.Exit(1)
            
            console.print(f"[blue]Translating {input_file}...[/blue]")
            
            if input_path.suffix.lower() == ".pdf":
                # PDF translation
                from .ingest.pdf import parse_pdf
                from .render.pdf import render_pdf
                
                # Parse PDF
                document = parse_pdf(input_path)
                
                # Translate
                result = pipeline.translate_document(document)
                
                # Render output
                if output_file:
                    output_path = Path(output_file)
                    if output_path.suffix.lower() == ".pdf":
                        render_pdf(result.document, output_path)
                    else:
                        # Save as text
                        with open(output_path, "w", encoding="utf-8") as f:
                            for block in result.document.blocks:
                                f.write(block.translation or block.text)
                                f.write("\n\n")
                    console.print(f"[green]✓ Saved to {output_file}[/green]")
                else:
                    # Print to console
                    for block in result.document.blocks:
                        console.print(block.translation or block.text)
                        console.print()
            
            else:
                # Text file translation
                text_content = input_path.read_text(encoding="utf-8")
                result = pipeline.translate_text(text_content)
                
                if output_file:
                    Path(output_file).write_text(result, encoding="utf-8")
                    console.print(f"[green]✓ Saved to {output_file}[/green]")
                else:
                    console.print(result)
        else:
            console.print("[red]Error:[/red] Provide --text or --input")
            raise typer.Exit(1)
            
    except Exception as e:
        if verbose:
            console.print_exception()
        else:
            console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def gui(
    port: int = typer.Option(7860, "--port", "-p", help="Port to run GUI on"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
):
    """Launch the GUI interface."""
    console.print("🚀 [bold blue]Starting SciTrans-LLMs GUI (NiceGUI)...[/bold blue]\n")
    console.print(f"Opening browser at [link]http://127.0.0.1:{port}[/link]")
    console.print("[dim]Press Ctrl+C to stop the server[/dim]\n")
    
    try:
        from .gui import launch
        launch(port=port, reload=reload)
    except ImportError as e:
        console.print(f"[red]Error:[/red] GUI dependencies not installed: {e}")
        console.print("Install with: pip install scitran-llms[gui]")
        raise typer.Exit(1)


@app.command()
def info():
    """Show system information."""
    console.print(f"\n[bold]SciTrans-LLMs v{__version__}[/bold]")
    console.print("Scientific Document Translation System\n")
    
    # Backends table
    table = Table(title="Available Backends", show_header=True)
    table.add_column("Backend", style="cyan", width=15)
    table.add_column("Description", style="white")
    
    for key, desc in BACKENDS.items():
        default_marker = " [green]★[/green]" if key == DEFAULT_BACKEND else ""
        table.add_row(f"{key}{default_marker}", desc)
    
    console.print(table)
    console.print(f"\n[dim]★ = Default backend[/dim]")
    
    # Languages table
    lang_table = Table(title="\nSupported Languages", show_header=True)
    lang_table.add_column("Code", style="cyan", width=10)
    lang_table.add_column("Language", style="white")
    
    for code, name in list(LANGUAGES.items())[:6]:  # Show first 6
        lang_table.add_row(code, name)
    lang_table.add_row("...", f"and {len(LANGUAGES)-6} more")
    
    console.print(lang_table)
    
    # API Keys status
    console.print("\n[bold]API Keys Status:[/bold]")
    keys = {
        "OPENAI_API_KEY": "OpenAI",
        "DEEPSEEK_API_KEY": "DeepSeek", 
        "ANTHROPIC_API_KEY": "Anthropic",
    }
    
    for env_var, service in keys.items():
        if os.getenv(env_var):
            console.print(f"  ✓ {service}: [green]Set[/green]")
        else:
            console.print(f"  ✗ {service}: [yellow]Not set[/yellow]")


@app.command()
def demo():
    """Run a quick demo."""
    console.print("\n[bold]SciTrans-LLMs Demo[/bold]\n")
    
    sample_text = "The quantum mechanical wave function describes the probability amplitude of finding a particle in a particular state."
    
    console.print(f"[blue]Original text:[/blue] {sample_text}\n")
    console.print(f"[blue]Using backend:[/blue] {DEFAULT_BACKEND}\n")
    
    try:
        config = PipelineConfig(
            backend=DEFAULT_BACKEND,
            source_lang="en",
            target_lang="fr"
        )
        pipeline = TranslationPipeline(config)
        
        with console.status("[bold green]Translating..."):
            result = pipeline.translate_text(sample_text)
        
        console.print(f"[green]Translation:[/green] {result}\n")
        console.print("[dim]✓ Demo completed successfully[/dim]")
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        console.print("\n[yellow]Note:[/yellow] Make sure the backend is properly configured")


@app.command()
def models():
    """Show detailed information about translation models."""
    console.print("\n[bold]Translation Models[/bold]\n")
    
    for backend, description in MODEL_DESCRIPTIONS.items():
        marker = " [green](default)[/green]" if backend == DEFAULT_BACKEND else ""
        console.print(f"[cyan]{backend}[/cyan]{marker}")
        console.print(f"  {description}\n")


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
