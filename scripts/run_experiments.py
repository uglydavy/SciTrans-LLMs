#!/usr/bin/env python3
"""
Master experiment runner for SciTrans-Next thesis research.

This script runs the complete experimental pipeline:
1. Setup and validation
2. Corpus loading
3. Translation experiments (ablations, baselines)
4. Evaluation
5. Export thesis-ready outputs

Usage:
    # Full experiment suite
    python scripts/run_experiments.py
    
    # Quick test with sample data
    python scripts/run_experiments.py --quick
    
    # Use real LLM backend
    python scripts/run_experiments.py --backend openai
    
    # Custom corpus
    python scripts/run_experiments.py --corpus path/to/corpus
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel

console = Console()


def check_setup():
    """Verify system is properly configured."""
    console.print("\n[bold]1. Checking Setup...[/]")
    
    issues = []
    
    # Check imports
    try:
        from scitrans_next import Document, TranslationPipeline
        console.print("  [green]✓[/] Core modules")
    except ImportError as e:
        issues.append(f"Import error: {e}")
    
    # Check API keys for LLM backends
    from scitrans_next.keys import KeyManager
    km = KeyManager()
    
    for service in ["openai", "deepseek", "anthropic"]:
        info = km.get_key_info(service)
        if info.is_set:
            console.print(f"  [green]✓[/] {service}: {info.masked_value} ({info.source})")
        else:
            console.print(f"  [yellow]○[/] {service}: not configured")
    
    # Check evaluation dependencies
    try:
        import sacrebleu
        console.print("  [green]✓[/] SacreBLEU")
    except ImportError:
        console.print("  [yellow]○[/] SacreBLEU not installed")
    
    return len(issues) == 0


def load_corpus(corpus_path: Path, quick: bool = False):
    """Load the test corpus."""
    console.print("\n[bold]2. Loading Corpus...[/]")
    
    from scitrans_next.experiments.corpus import load_corpus, create_synthetic_corpus
    
    if not corpus_path.exists():
        console.print(f"  [yellow]Corpus not found at {corpus_path}[/]")
        console.print("  [dim]Creating synthetic corpus for testing...[/]")
        corpus = create_synthetic_corpus(corpus_path, num_documents=5 if quick else 10)
    else:
        corpus = load_corpus(corpus_path)
    
    console.print(f"  [green]✓[/] Loaded: {len(corpus)} documents, {corpus.total_segments} segments")
    return corpus


def run_ablation_experiments(corpus, backend: str = "dummy"):
    """Run ablation study experiments."""
    console.print("\n[bold]3. Running Ablation Study...[/]")
    
    from scitrans_next.experiments.runner import ExperimentRunner
    
    runner = ExperimentRunner(corpus, output_dir=project_root / "results" / "ablation")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running ablations...", total=100)
        
        def update(msg, pct):
            progress.update(task, description=msg, completed=int(pct * 100))
        
        results = runner.run_ablation(base_backend=backend, progress_callback=update)
    
    # Show results table
    table = Table(title="Ablation Results")
    table.add_column("Configuration", style="cyan")
    table.add_column("BLEU", justify="right")
    table.add_column("chrF++", justify="right")
    table.add_column("Glossary", justify="right")
    
    for r in results:
        bleu = f"{r.evaluation.bleu:.2f}" if r.evaluation.bleu else "N/A"
        chrf = f"{r.evaluation.chrf:.2f}" if r.evaluation.chrf else "N/A"
        gloss = f"{r.evaluation.glossary_adherence:.1%}" if r.evaluation.glossary_adherence else "N/A"
        table.add_row(r.config.name, bleu, chrf, gloss)
    
    console.print(table)
    
    return runner, results


def run_backend_comparison(corpus, backends: list[str]):
    """Compare different translation backends."""
    console.print("\n[bold]4. Comparing Backends...[/]")
    
    from scitrans_next.experiments.runner import ExperimentRunner
    
    runner = ExperimentRunner(corpus, output_dir=project_root / "results" / "backends")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Testing backends...", total=100)
        
        def update(msg, pct):
            progress.update(task, description=msg, completed=int(pct * 100))
        
        results = runner.run_backend_comparison(backends, progress_callback=update)
    
    return runner, results


def export_thesis_outputs(results):
    """Export thesis-ready tables and figures."""
    console.print("\n[bold]5. Exporting Thesis Outputs...[/]")
    
    from scitrans_next.experiments.thesis import ThesisExporter
    
    output_dir = project_root / "results" / "thesis"
    exporter = ThesisExporter(results, output_dir=output_dir)
    
    outputs = exporter.export_all()
    
    for name, path in outputs.items():
        console.print(f"  [green]✓[/] {name}: {path}")
    
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Run SciTrans-Next experiments")
    parser.add_argument("--corpus", type=Path, default=project_root / "corpus",
                       help="Path to corpus directory")
    parser.add_argument("--backend", type=str, default="dummy",
                       help="Translation backend (dummy, dictionary, openai, deepseek)")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test with minimal data")
    parser.add_argument("--skip-ablation", action="store_true",
                       help="Skip ablation study")
    parser.add_argument("--compare-backends", type=str, nargs="+",
                       default=["dummy", "dictionary"],
                       help="Backends to compare")
    
    args = parser.parse_args()
    
    console.print(Panel.fit(
        "[bold blue]SciTrans-Next Experiment Runner[/]\n"
        "Systematic evaluation for thesis research",
        border_style="blue"
    ))
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    console.print(f"[dim]Started: {timestamp}[/]\n")
    
    # 1. Check setup
    if not check_setup():
        console.print("[red]Setup issues found. Please fix and retry.[/]")
        sys.exit(1)
    
    # 2. Load corpus
    corpus = load_corpus(args.corpus, quick=args.quick)
    
    all_results = []
    
    # 3. Run ablation study
    if not args.skip_ablation:
        runner, ablation_results = run_ablation_experiments(corpus, args.backend)
        runner.export_results(prefix="ablation", formats=["json", "latex", "summary"])
        all_results.extend(ablation_results)
    
    # 4. Compare backends
    if len(args.compare_backends) > 1:
        runner, backend_results = run_backend_comparison(corpus, args.compare_backends)
        runner.export_results(prefix="backends", formats=["json", "latex", "summary"])
        all_results.extend(backend_results)
    
    # 5. Export thesis outputs
    if all_results:
        outputs = export_thesis_outputs(all_results)
        
        console.print("\n" + "="*60)
        console.print("[bold green]Experiments Complete![/]")
        console.print("="*60)
        console.print("\n[bold]Output files:[/]")
        console.print(f"  Results: {project_root / 'results'}")
        console.print(f"  Thesis:  {project_root / 'results' / 'thesis'}")
    else:
        console.print("\n[yellow]No results to export.[/]")
    
    console.print("\n[dim]Finished: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "[/]")


if __name__ == "__main__":
    main()

