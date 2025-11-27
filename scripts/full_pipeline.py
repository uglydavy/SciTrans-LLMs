#!/usr/bin/env python3
"""
Full thesis experiment pipeline.

This is the master script that runs all experiments needed for the thesis:
1. Validates setup (keys, dependencies)
2. Prepares/validates corpus
3. Runs main translation experiments
4. Runs ablation studies
5. Runs baseline comparisons
6. Generates all thesis outputs

Usage:
    # Full pipeline
    python scripts/full_pipeline.py
    
    # With specific backend
    python scripts/full_pipeline.py --backend openai
    
    # Skip certain steps
    python scripts/full_pipeline.py --skip-baselines
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.prompt import Confirm, Prompt

console = Console()


def print_header():
    """Print welcome header."""
    console.print("\n")
    console.print(Panel.fit(
        "[bold blue]SciTrans-Next Full Experiment Pipeline[/]\n"
        "[dim]Complete thesis experiment automation[/]",
        border_style="blue"
    ))
    console.print(f"[dim]Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/]\n")


def validate_setup():
    """Step 1: Validate all dependencies and keys."""
    console.print("[bold]Step 1: Validating Setup[/]")
    console.print("─" * 50)
    
    issues = []
    
    # Check core imports
    try:
        from scitrans_next import Document, TranslationPipeline
        from scitrans_next.pipeline import PipelineConfig
        console.print("  [green]✓[/] Core modules")
    except ImportError as e:
        issues.append(f"Core import: {e}")
        console.print(f"  [red]✗[/] Core modules: {e}")
    
    # Check evaluation
    try:
        import sacrebleu
        console.print("  [green]✓[/] SacreBLEU evaluation")
    except ImportError:
        issues.append("sacrebleu not installed")
        console.print("  [yellow]○[/] SacreBLEU not installed (pip install sacrebleu)")
    
    # Check PDF support
    try:
        import fitz
        console.print("  [green]✓[/] PyMuPDF (PDF support)")
    except ImportError:
        console.print("  [dim]○[/] PyMuPDF not installed (optional for PDF)")
    
    # Check API keys
    from scitrans_next.keys import KeyManager
    km = KeyManager()
    
    console.print("\n  [bold]API Keys:[/]")
    available_backends = ["dummy", "dictionary"]
    
    for service in ["openai", "deepseek", "anthropic"]:
        info = km.get_key_info(service)
        if info.is_set:
            console.print(f"    [green]✓[/] {service}: {info.masked_value}")
            available_backends.append(service)
        else:
            console.print(f"    [dim]○[/] {service}: not configured")
    
    if issues:
        console.print(f"\n  [red]Found {len(issues)} issue(s)[/]")
        for issue in issues:
            console.print(f"    - {issue}")
        if not Confirm.ask("\n  Continue anyway?", default=False):
            return None
    else:
        console.print("\n  [green]✓ All checks passed[/]")
    
    return available_backends


def validate_corpus(corpus_path: Path, min_docs: int = 20):
    """Step 2: Validate or create corpus."""
    console.print("\n[bold]Step 2: Validating Corpus[/]")
    console.print("─" * 50)
    
    source_dir = corpus_path / "source" / "abstracts"
    ref_dir = corpus_path / "reference" / "abstracts"
    
    if not source_dir.exists() or not ref_dir.exists():
        console.print(f"  [yellow]Corpus not found at {corpus_path}[/]")
        
        if Confirm.ask("  Create sample corpus?", default=True):
            from scitrans_next.experiments.corpus import create_synthetic_corpus
            corpus = create_synthetic_corpus(corpus_path, num_documents=min_docs)
            console.print(f"  [green]✓[/] Created {len(corpus)} documents")
        else:
            console.print("  [red]Cannot continue without corpus[/]")
            return None
    
    # Count documents
    source_files = list(source_dir.glob("*.txt"))
    ref_files = list(ref_dir.glob("*.txt"))
    
    matched = len(set(f.name for f in source_files) & set(f.name for f in ref_files))
    
    console.print(f"  Source files: {len(source_files)}")
    console.print(f"  Reference files: {len(ref_files)}")
    console.print(f"  Matched pairs: {matched}")
    
    if matched < min_docs:
        console.print(f"  [yellow]Warning: Only {matched} documents (recommend {min_docs}+)[/]")
    else:
        console.print(f"  [green]✓[/] Corpus ready")
    
    # Load corpus
    from scitrans_next.experiments.corpus import load_corpus
    corpus = load_corpus(corpus_path)
    
    return corpus


def run_main_experiments(corpus, backend: str, output_dir: Path):
    """Step 3: Run main translation experiments."""
    console.print("\n[bold]Step 3: Running Main Experiments[/]")
    console.print("─" * 50)
    console.print(f"  Backend: {backend}")
    console.print(f"  Documents: {len(corpus)}")
    
    from scitrans_next.experiments.runner import ExperimentRunner, ExperimentConfig
    
    runner = ExperimentRunner(corpus, output_dir=output_dir / "main")
    
    # Full system configuration
    config = ExperimentConfig(
        name="scitrans_full",
        description="Full SciTrans-Next system",
        backend=backend,
        enable_glossary=True,
        enable_context=True,
        enable_refinement=True,
        enable_masking=True,
    )
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Translating...", total=100)
        
        def update(msg, pct):
            progress.update(task, description=f"  {msg}", completed=int(pct * 100))
        
        result = runner.run_experiment(config, progress_callback=update)
    
    console.print(f"  [green]✓[/] Translation complete")
    console.print(f"    BLEU: {result.evaluation.bleu:.2f}" if result.evaluation.bleu else "")
    console.print(f"    chrF++: {result.evaluation.chrf:.2f}" if result.evaluation.chrf else "")
    
    runner.export_results(prefix="main", formats=["json", "latex", "summary"])
    
    return runner, [result]


def run_ablation_study(corpus, backend: str, output_dir: Path):
    """Step 4: Run ablation study."""
    console.print("\n[bold]Step 4: Running Ablation Study[/]")
    console.print("─" * 50)
    
    from scitrans_next.experiments.runner import ExperimentRunner
    
    runner = ExperimentRunner(corpus, output_dir=output_dir / "ablation")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running ablations...", total=100)
        
        def update(msg, pct):
            progress.update(task, description=f"  {msg}", completed=int(pct * 100))
        
        results = runner.run_ablation(base_backend=backend, progress_callback=update)
    
    # Show results table
    table = Table(title="Ablation Results")
    table.add_column("Configuration", style="cyan")
    table.add_column("BLEU", justify="right")
    table.add_column("ΔBLEU", justify="right")
    table.add_column("Glossary", justify="right")
    
    full_bleu = next((r.evaluation.bleu for r in results if r.config.name == "full"), 0) or 0
    
    for r in results:
        bleu = r.evaluation.bleu or 0
        delta = bleu - full_bleu if r.config.name != "full" else 0
        delta_str = f"{delta:+.2f}" if r.config.name != "full" else "---"
        gloss = f"{r.evaluation.glossary_adherence:.1%}" if r.evaluation.glossary_adherence else "N/A"
        table.add_row(r.config.name, f"{bleu:.2f}", delta_str, gloss)
    
    console.print(table)
    
    runner.export_results(prefix="ablation", formats=["json", "latex", "summary"])
    
    return runner, results


def run_baseline_comparison(corpus, our_results: list, output_dir: Path):
    """Step 5: Run baseline comparisons."""
    console.print("\n[bold]Step 5: Running Baseline Comparisons[/]")
    console.print("─" * 50)
    
    from scitrans_next.eval.baselines import (
        BaselineComparison,
        GoogleTranslateBaseline,
        NaiveLLMBaseline,
    )
    
    comparison = BaselineComparison()
    
    # Add available baselines
    try:
        comparison.add_baseline(GoogleTranslateBaseline())
        console.print("  [green]✓[/] Google Translate baseline")
    except Exception as e:
        console.print(f"  [dim]○[/] Google Translate: {e}")
    
    # Get our system outputs
    our_hypotheses = []
    for result in our_results:
        if hasattr(result, 'hypotheses'):
            our_hypotheses.extend(result.hypotheses)
    
    if not our_hypotheses:
        console.print("  [yellow]No system outputs to compare[/]")
        return None
    
    # Prepare data
    sources = corpus.get_all_sources()[:len(our_hypotheses)]
    references = corpus.get_all_references()[:len(our_hypotheses)]
    
    # Run comparison
    try:
        results = comparison.run_comparison(
            documents=list(corpus),
            references=[[ref] for ref in references],
            sources=[[src] for src in sources],
            scitrans_hypotheses=[[hyp] for hyp in our_hypotheses],
        )
        
        console.print("\n" + comparison.summary())
        
        # Save LaTeX table
        latex_path = output_dir / "baselines" / "comparison.tex"
        latex_path.parent.mkdir(parents=True, exist_ok=True)
        latex_path.write_text(comparison.to_latex_table())
        console.print(f"\n  [green]✓[/] Saved to {latex_path}")
        
    except Exception as e:
        console.print(f"  [yellow]Baseline comparison failed: {e}[/]")
    
    return comparison


def generate_thesis_outputs(all_results: list, output_dir: Path):
    """Step 6: Generate thesis-ready outputs."""
    console.print("\n[bold]Step 6: Generating Thesis Outputs[/]")
    console.print("─" * 50)
    
    from scitrans_next.experiments.thesis import ThesisExporter
    
    thesis_dir = output_dir / "thesis"
    exporter = ThesisExporter(all_results, output_dir=thesis_dir)
    
    outputs = exporter.export_all()
    
    for name, path in outputs.items():
        console.print(f"  [green]✓[/] {name}: {path.name}")
    
    # Generate summary statistics
    stats = {
        "total_experiments": len(all_results),
        "best_bleu": max((r.evaluation.bleu or 0 for r in all_results), default=0),
        "best_config": max(all_results, key=lambda r: r.evaluation.bleu or 0).config.name if all_results else "N/A",
        "timestamp": datetime.now().isoformat(),
    }
    
    (thesis_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    
    return outputs


def print_summary(output_dir: Path, all_results: list):
    """Print final summary."""
    console.print("\n")
    console.print("═" * 60)
    console.print("[bold green]  EXPERIMENT PIPELINE COMPLETE[/]")
    console.print("═" * 60)
    
    console.print("\n[bold]Results Summary:[/]")
    
    if all_results:
        best = max(all_results, key=lambda r: r.evaluation.bleu or 0)
        console.print(f"  Best configuration: {best.config.name}")
        console.print(f"  Best BLEU: {best.evaluation.bleu:.2f}" if best.evaluation.bleu else "")
    
    console.print(f"\n[bold]Output Location:[/]")
    console.print(f"  {output_dir}")
    console.print(f"\n[bold]Thesis Files:[/]")
    console.print(f"  {output_dir / 'thesis'}")
    
    console.print("\n[bold]Next Steps:[/]")
    console.print("  1. Review results in results/thesis/")
    console.print("  2. Copy LaTeX tables to your thesis")
    console.print("  3. Include figures in your document")
    
    console.print(f"\n[dim]Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/]")


def main():
    parser = argparse.ArgumentParser(description="Full thesis experiment pipeline")
    parser.add_argument("--backend", type=str, default="dummy",
                       help="Translation backend (dummy, dictionary, openai, deepseek)")
    parser.add_argument("--corpus", type=Path, default=project_root / "corpus",
                       help="Path to corpus directory")
    parser.add_argument("--output", type=Path, default=project_root / "results",
                       help="Output directory")
    parser.add_argument("--min-docs", type=int, default=20,
                       help="Minimum documents required")
    parser.add_argument("--skip-ablation", action="store_true",
                       help="Skip ablation study")
    parser.add_argument("--skip-baselines", action="store_true",
                       help="Skip baseline comparisons")
    
    args = parser.parse_args()
    
    print_header()
    
    # Step 1: Validate setup
    available_backends = validate_setup()
    if available_backends is None:
        sys.exit(1)
    
    # Check if requested backend is available
    if args.backend not in available_backends:
        console.print(f"\n[yellow]Backend '{args.backend}' not available.[/]")
        console.print(f"Available: {', '.join(available_backends)}")
        args.backend = Prompt.ask("Select backend", choices=available_backends, default="dummy")
    
    # Step 2: Validate corpus
    corpus = validate_corpus(args.corpus, args.min_docs)
    if corpus is None:
        sys.exit(1)
    
    all_results = []
    
    # Step 3: Main experiments
    _, main_results = run_main_experiments(corpus, args.backend, args.output)
    all_results.extend(main_results)
    
    # Step 4: Ablation study
    if not args.skip_ablation:
        _, ablation_results = run_ablation_study(corpus, args.backend, args.output)
        all_results.extend(ablation_results)
    
    # Step 5: Baseline comparisons
    if not args.skip_baselines:
        run_baseline_comparison(corpus, main_results, args.output)
    
    # Step 6: Generate thesis outputs
    generate_thesis_outputs(all_results, args.output)
    
    # Summary
    print_summary(args.output, all_results)


if __name__ == "__main__":
    main()

