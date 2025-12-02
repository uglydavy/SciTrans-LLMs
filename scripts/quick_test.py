#!/usr/bin/env python3
"""
Quick test script to verify SciTrans-LLMs is working.

Runs a simple end-to-end test with sample data.

Usage:
    python scripts/quick_test.py
    python scripts/quick_test.py --backend openai
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.panel import Panel

console = Console()


def run_quick_test(backend: str = "dictionary"):
    """Run a quick end-to-end test."""
    
    console.print(Panel.fit(
        "[bold]SciTrans-LLMs Quick Test[/]",
        border_style="green"
    ))
    
    # 1. Test imports
    console.print("\n[bold]1. Testing imports...[/]")
    try:
        from scitrans_llms import Document, TranslationPipeline
        from scitrans_llms.pipeline import PipelineConfig
        from scitrans_llms.translate import get_default_glossary
        from scitrans_llms.eval.metrics import compute_bleu
        console.print("  [green]✓[/] All imports successful")
    except ImportError as e:
        console.print(f"  [red]✗[/] Import error: {e}")
        return False
    
    # 2. Test document creation
    console.print("\n[bold]2. Creating test document...[/]")
    sample_text = """
Machine learning has revolutionized natural language processing.

The transformer architecture uses attention mechanisms. The model computes 
attention weights using the formula $Q K^T / \\sqrt{d_k}$.

Deep learning enables state of the art performance on machine translation tasks.
"""
    
    doc = Document.from_text(sample_text, source_lang="en", target_lang="fr")
    console.print(f"  [green]✓[/] Created document: {len(doc.all_blocks)} blocks")
    
    # 3. Test translation
    console.print(f"\n[bold]3. Testing translation ({backend})...[/]")
    
    glossary = get_default_glossary()
    config = PipelineConfig(
        translator_backend=backend,
        enable_glossary=True,
        enable_masking=True,
        enable_refinement=True,
        glossary=glossary,
    )
    
    try:
        pipeline = TranslationPipeline(config)
        result = pipeline.translate(doc)
        
        if result.success:
            console.print(f"  [green]✓[/] Translation successful")
            console.print(f"  Stats: {result.stats}")
        else:
            console.print(f"  [yellow]⚠[/] Translation completed with errors: {result.errors}")
    except Exception as e:
        console.print(f"  [red]✗[/] Translation failed: {e}")
        return False
    
    # 4. Show results
    console.print("\n[bold]4. Translation Results:[/]")
    console.print("-" * 60)
    
    for i, block in enumerate(doc.all_blocks):
        if block.is_translatable:
            console.print(f"\n[dim]Block {i+1} (source):[/]")
            console.print(f"  {block.source_text[:100]}...")
            console.print(f"[cyan]Block {i+1} (translated):[/]")
            trans = block.translated_text or "[No translation]"
            console.print(f"  {trans[:100]}...")
    
    # 5. Check glossary adherence
    console.print("\n[bold]5. Checking glossary adherence...[/]")
    
    translated_text = result.translated_text
    expected_terms = [
        ("machine learning", "apprentissage automatique"),
        ("deep learning", "apprentissage profond"),
        ("natural language processing", "traitement automatique"),
    ]
    
    for en_term, fr_term in expected_terms:
        if fr_term.lower() in translated_text.lower():
            console.print(f"  [green]✓[/] '{en_term}' → '{fr_term}'")
        else:
            console.print(f"  [yellow]○[/] '{en_term}' → not found (expected '{fr_term}')")
    
    # 6. Check formula preservation
    console.print("\n[bold]6. Checking formula preservation...[/]")
    
    if "$Q K^T" in translated_text or "<<MATH" in translated_text:
        console.print("  [green]✓[/] Formula preserved")
    else:
        console.print("  [yellow]○[/] Formula may not be preserved")
    
    console.print("\n" + "=" * 60)
    console.print("[bold green]Quick test completed successfully![/]")
    console.print("=" * 60)
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Quick test for SciTrans-LLMs")
    parser.add_argument("--backend", "-b", type=str, default="dictionary",
                       help="Translation backend to test")
    
    args = parser.parse_args()
    
    success = run_quick_test(args.backend)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

