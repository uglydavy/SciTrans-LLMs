#!/usr/bin/env python3
"""
Corpus collection script for SciTrans-LLMs.

Downloads and prepares parallel EN/FR scientific data from multiple sources:
1. OPUS corpora (Scielo, EMEA, JRC-Acquis)
2. Europarl (EU parliamentary proceedings)
3. WMT News (machine translation benchmarks)
4. ArXiv abstracts (if available bilingually)

Usage:
    python scripts/collect_corpus.py --target 100
    python scripts/collect_corpus.py --source opus --target 50
    python scripts/collect_corpus.py --source europarl --domain scientific
"""

import argparse
import json
import os
import re
import sys
import urllib.request
import zipfile
import gzip
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn
from rich.prompt import Prompt, Confirm
from rich.table import Table

console = Console()

# ============================================================================
# Data Sources Configuration
# ============================================================================

SOURCES = {
    "opus_scielo": {
        "name": "OPUS Scielo",
        "description": "Scientific articles from Latin American journals",
        "url": "https://opus.nlpl.eu/download.php?f=Scielo/v1/moses/en-fr.txt.zip",
        "domain": "scientific",
        "estimated_pairs": 50000,
    },
    "opus_emea": {
        "name": "OPUS EMEA",
        "description": "European Medicines Agency documents",
        "url": "https://opus.nlpl.eu/download.php?f=EMEA/v3/moses/en-fr.txt.zip",
        "domain": "medical",
        "estimated_pairs": 360000,
    },
    "opus_kde4": {
        "name": "OPUS KDE4",
        "description": "KDE software documentation (technical)",
        "url": "https://opus.nlpl.eu/download.php?f=KDE4/v2/moses/en-fr.txt.zip",
        "domain": "technical",
        "estimated_pairs": 120000,
    },
    "europarl": {
        "name": "Europarl",
        "description": "European Parliament proceedings",
        "url": "https://www.statmt.org/europarl/v7/fr-en.tgz",
        "domain": "political",
        "estimated_pairs": 2000000,
    },
    "tatoeba": {
        "name": "Tatoeba",
        "description": "Collaborative sentence translations",
        "url": "https://downloads.tatoeba.org/exports/sentences.tar.bz2",
        "domain": "general",
        "estimated_pairs": 100000,
    },
}

# Scientific vocabulary for filtering
SCIENTIFIC_KEYWORDS = {
    "en": [
        "research", "study", "method", "results", "analysis", "data",
        "experiment", "hypothesis", "conclusion", "figure", "table",
        "algorithm", "model", "neural", "learning", "network",
        "equation", "theorem", "proof", "formula", "parameter",
        "optimization", "performance", "evaluation", "benchmark",
    ],
    "fr": [
        "recherche", "étude", "méthode", "résultats", "analyse", "données",
        "expérience", "hypothèse", "conclusion", "figure", "tableau",
        "algorithme", "modèle", "neuronal", "apprentissage", "réseau",
        "équation", "théorème", "preuve", "formule", "paramètre",
        "optimisation", "performance", "évaluation", "référence",
    ],
}


@dataclass
class ParallelPair:
    """A parallel sentence pair."""
    source: str
    target: str
    source_lang: str = "en"
    target_lang: str = "fr"
    domain: str = "general"
    source_file: str = ""


def download_file(url: str, dest: Path, desc: str = "Downloading") -> bool:
    """Download a file with progress bar."""
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(desc, total=None)
            
            def reporthook(block_num, block_size, total_size):
                if total_size > 0:
                    progress.update(task, total=total_size, completed=block_num * block_size)
            
            urllib.request.urlretrieve(url, dest, reporthook)
        return True
    except Exception as e:
        console.print(f"[red]Download failed: {e}[/]")
        return False


def extract_archive(archive: Path, dest: Path) -> bool:
    """Extract zip/tgz/gz archive."""
    try:
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive, 'r') as zf:
                zf.extractall(dest)
        elif archive.suffix in (".tgz", ".gz"):
            import tarfile
            if archive.suffix == ".tgz" or str(archive).endswith(".tar.gz"):
                with tarfile.open(archive, "r:gz") as tf:
                    tf.extractall(dest)
            else:
                with gzip.open(archive, 'rb') as f_in:
                    with open(dest / archive.stem, 'wb') as f_out:
                        f_out.write(f_in.read())
        return True
    except Exception as e:
        console.print(f"[red]Extraction failed: {e}[/]")
        return False


def load_moses_format(en_file: Path, fr_file: Path, max_pairs: int = 10000) -> list[ParallelPair]:
    """Load parallel corpus in Moses format (separate .en and .fr files)."""
    pairs = []
    
    try:
        with open(en_file, 'r', encoding='utf-8') as f_en, \
             open(fr_file, 'r', encoding='utf-8') as f_fr:
            
            for i, (en_line, fr_line) in enumerate(zip(f_en, f_fr)):
                if i >= max_pairs:
                    break
                
                en_text = en_line.strip()
                fr_text = fr_line.strip()
                
                # Skip empty or very short pairs
                if len(en_text) < 20 or len(fr_text) < 20:
                    continue
                
                # Skip very long pairs (likely parsing errors)
                if len(en_text) > 1000 or len(fr_text) > 1000:
                    continue
                
                pairs.append(ParallelPair(
                    source=en_text,
                    target=fr_text,
                    source_file=str(en_file),
                ))
    except Exception as e:
        console.print(f"[red]Error loading {en_file}: {e}[/]")
    
    return pairs


def is_scientific(text: str, lang: str = "en") -> bool:
    """Check if text appears to be scientific content."""
    text_lower = text.lower()
    keywords = SCIENTIFIC_KEYWORDS.get(lang, SCIENTIFIC_KEYWORDS["en"])
    
    # Count keyword matches
    matches = sum(1 for kw in keywords if kw in text_lower)
    
    # Consider scientific if 2+ keyword matches or contains numbers/formulas
    has_numbers = bool(re.search(r'\d+\.?\d*', text))
    has_formula_hints = any(c in text for c in ['=', '<', '>', '∑', '∫', '$'])
    
    return matches >= 2 or (matches >= 1 and (has_numbers or has_formula_hints))


def filter_scientific(pairs: list[ParallelPair]) -> list[ParallelPair]:
    """Filter pairs to keep only scientific content."""
    return [p for p in pairs if is_scientific(p.source, "en") or is_scientific(p.target, "fr")]


def group_into_documents(pairs: list[ParallelPair], sentences_per_doc: int = 5) -> list[dict]:
    """Group sentence pairs into document-like units."""
    documents = []
    
    for i in range(0, len(pairs), sentences_per_doc):
        chunk = pairs[i:i + sentences_per_doc]
        if len(chunk) < 2:
            continue
        
        doc = {
            "id": f"doc_{len(documents)+1:04d}",
            "source": "\n\n".join(p.source for p in chunk),
            "target": "\n\n".join(p.target for p in chunk),
            "num_sentences": len(chunk),
            "domain": chunk[0].domain,
        }
        documents.append(doc)
    
    return documents


def save_corpus(documents: list[dict], output_dir: Path):
    """Save corpus in the expected format."""
    source_dir = output_dir / "source" / "abstracts"
    ref_dir = output_dir / "reference" / "abstracts"
    source_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)
    
    for doc in documents:
        doc_id = doc["id"]
        
        source_file = source_dir / f"{doc_id}.txt"
        ref_file = ref_dir / f"{doc_id}.txt"
        
        source_file.write_text(doc["source"], encoding="utf-8")
        ref_file.write_text(doc["target"], encoding="utf-8")
    
    # Save metadata
    metadata = {
        "num_documents": len(documents),
        "total_sentences": sum(d["num_sentences"] for d in documents),
        "domains": list(set(d["domain"] for d in documents)),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    
    return source_dir, ref_dir


def collect_from_opus(source_key: str, target_docs: int, output_dir: Path, cache_dir: Path) -> list[dict]:
    """Download and process OPUS corpus."""
    source = SOURCES[source_key]
    console.print(f"\n[bold]Collecting from {source['name']}...[/]")
    console.print(f"  [dim]{source['description']}[/]")
    
    # Download
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_name = source["url"].split("/")[-1].split("?")[-1]
    if "=" in archive_name:
        archive_name = source["url"].split("=")[-1]
    archive_path = cache_dir / archive_name
    
    if not archive_path.exists():
        console.print(f"  Downloading {archive_name}...")
        if not download_file(source["url"], archive_path, f"Downloading {source['name']}"):
            return []
    else:
        console.print(f"  [dim]Using cached {archive_name}[/]")
    
    # Extract
    extract_dir = cache_dir / source_key
    if not extract_dir.exists():
        console.print("  Extracting...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        if not extract_archive(archive_path, extract_dir):
            return []
    
    # Find Moses format files
    en_files = list(extract_dir.rglob("*.en"))
    fr_files = list(extract_dir.rglob("*.fr"))
    
    if not en_files or not fr_files:
        console.print("  [yellow]No Moses format files found[/]")
        return []
    
    # Load pairs
    console.print(f"  Loading parallel pairs...")
    all_pairs = []
    max_pairs_needed = target_docs * 10  # Assume ~5 sentences per doc, with buffer
    
    for en_file in en_files:
        fr_file = en_file.with_suffix(".fr")
        if fr_file.exists():
            pairs = load_moses_format(en_file, fr_file, max_pairs=max_pairs_needed // len(en_files))
            for p in pairs:
                p.domain = source["domain"]
            all_pairs.extend(pairs)
    
    console.print(f"  Loaded {len(all_pairs)} pairs")
    
    # Filter for scientific content
    if source["domain"] != "scientific":
        console.print("  Filtering for scientific content...")
        all_pairs = filter_scientific(all_pairs)
        console.print(f"  {len(all_pairs)} scientific pairs remaining")
    
    # Group into documents
    documents = group_into_documents(all_pairs, sentences_per_doc=5)
    console.print(f"  Created {len(documents)} documents")
    
    return documents[:target_docs]


def create_sample_corpus(output_dir: Path, num_docs: int = 20) -> list[dict]:
    """Create a sample corpus with synthetic scientific content."""
    console.print("\n[bold]Creating sample scientific corpus...[/]")
    
    # Templates for scientific abstracts
    templates = [
        {
            "en": """This paper presents a novel approach to {topic} using {method}. We demonstrate that our method achieves state of the art performance on standard benchmarks.

Our experiments show an improvement of {num}% over baseline approaches. The proposed algorithm has time complexity O(n log n) and space complexity O(n).

We validate our approach on the {dataset} dataset, achieving a BLEU score of {bleu}. Future work will explore applications to {future}.""",
            "fr": """Cet article présente une nouvelle approche de {topic_fr} utilisant {method_fr}. Nous démontrons que notre méthode atteint des performances de pointe sur les benchmarks standards.

Nos expériences montrent une amélioration de {num}% par rapport aux approches de référence. L'algorithme proposé a une complexité temporelle O(n log n) et une complexité spatiale O(n).

Nous validons notre approche sur le jeu de données {dataset}, atteignant un score BLEU de {bleu}. Les travaux futurs exploreront les applications à {future_fr}.""",
        },
        {
            "en": """We investigate the problem of {topic} in the context of {context}. Our analysis reveals important insights about the relationship between {var1} and {var2}.

The experimental results demonstrate that {finding}. Statistical analysis confirms significance with p < 0.{pval}.

These findings have implications for {implication}. We discuss limitations and propose directions for future research.""",
            "fr": """Nous étudions le problème de {topic_fr} dans le contexte de {context_fr}. Notre analyse révèle des informations importantes sur la relation entre {var1_fr} et {var2_fr}.

Les résultats expérimentaux démontrent que {finding_fr}. L'analyse statistique confirme la significativité avec p < 0,{pval}.

Ces résultats ont des implications pour {implication_fr}. Nous discutons des limitations et proposons des directions pour les recherches futures.""",
        },
        {
            "en": """Recent advances in {field} have enabled significant progress in {application}. This work proposes a framework for {goal}.

We introduce a novel architecture consisting of {components}. The model is trained on {data} using {training_method}.

Evaluation on {benchmark} shows that our approach outperforms existing methods by {margin}%. Code is available at {url}.""",
            "fr": """Les avancées récentes en {field_fr} ont permis des progrès significatifs dans {application_fr}. Ce travail propose un cadre pour {goal_fr}.

Nous introduisons une nouvelle architecture composée de {components_fr}. Le modèle est entraîné sur {data_fr} en utilisant {training_method_fr}.

L'évaluation sur {benchmark} montre que notre approche surpasse les méthodes existantes de {margin}%. Le code est disponible à {url}.""",
        },
    ]
    
    import random
    
    # Vocabulary for template filling
    vocab = {
        "topic": ["machine translation", "text classification", "named entity recognition", "sentiment analysis", "question answering"],
        "topic_fr": ["la traduction automatique", "la classification de texte", "la reconnaissance d'entités nommées", "l'analyse de sentiment", "la réponse aux questions"],
        "method": ["transformer-based models", "attention mechanisms", "graph neural networks", "contrastive learning", "reinforcement learning"],
        "method_fr": ["des modèles basés sur les transformeurs", "des mécanismes d'attention", "des réseaux de neurones graphiques", "l'apprentissage contrastif", "l'apprentissage par renforcement"],
        "dataset": ["WMT", "GLUE", "SQuAD", "CoNLL", "MNLI"],
        "field": ["natural language processing", "computer vision", "deep learning", "machine learning", "artificial intelligence"],
        "field_fr": ["traitement du langage naturel", "vision par ordinateur", "apprentissage profond", "apprentissage automatique", "intelligence artificielle"],
        "context": ["low-resource scenarios", "multilingual settings", "domain adaptation", "few-shot learning", "real-time applications"],
        "context_fr": ["des scénarios à faibles ressources", "des contextes multilingues", "l'adaptation de domaine", "l'apprentissage à quelques exemples", "les applications en temps réel"],
    }
    
    documents = []
    
    for i in range(num_docs):
        template = random.choice(templates)
        
        # Fill template
        doc_en = template["en"]
        doc_fr = template["fr"]
        
        # Replace placeholders with random values
        for key in ["topic", "method", "dataset", "field", "context"]:
            if key in vocab:
                val = random.choice(vocab[key])
                doc_en = doc_en.replace("{" + key + "}", val)
            key_fr = key + "_fr"
            if key_fr in vocab:
                val_fr = random.choice(vocab[key_fr])
                doc_fr = doc_fr.replace("{" + key_fr + "}", val_fr)
        
        # Fill numeric placeholders
        doc_en = doc_en.replace("{num}", str(random.randint(5, 25)))
        doc_en = doc_en.replace("{bleu}", str(random.randint(30, 45)))
        doc_en = doc_en.replace("{margin}", str(random.randint(2, 15)))
        doc_en = doc_en.replace("{pval}", f"0{random.randint(1, 5)}")
        
        doc_fr = doc_fr.replace("{num}", str(random.randint(5, 25)))
        doc_fr = doc_fr.replace("{bleu}", str(random.randint(30, 45)))
        doc_fr = doc_fr.replace("{margin}", str(random.randint(2, 15)))
        doc_fr = doc_fr.replace("{pval}", f"0{random.randint(1, 5)}")
        
        # Clean remaining placeholders
        doc_en = re.sub(r'\{[^}]+\}', 'X', doc_en)
        doc_fr = re.sub(r'\{[^}]+\}', 'X', doc_fr)
        
        documents.append({
            "id": f"sample_{i+1:04d}",
            "source": doc_en,
            "target": doc_fr,
            "num_sentences": 3,
            "domain": "scientific",
        })
    
    return documents


def main():
    parser = argparse.ArgumentParser(
        description="Collect parallel EN/FR corpus for thesis experiments"
    )
    parser.add_argument("--target", type=int, default=50,
                       help="Target number of documents")
    parser.add_argument("--source", type=str, default="auto",
                       choices=["auto", "opus_scielo", "opus_emea", "opus_kde4", "sample"],
                       help="Data source")
    parser.add_argument("--output", type=Path, default=project_root / "corpus",
                       help="Output directory")
    parser.add_argument("--cache", type=Path, default=project_root / ".cache",
                       help="Cache directory for downloads")
    parser.add_argument("--scientific-only", action="store_true",
                       help="Filter for scientific content only")
    
    args = parser.parse_args()
    
    console.print("\n[bold blue]═══════════════════════════════════════════[/]")
    console.print("[bold blue]  SciTrans-LLMs Corpus Collection Tool[/]")
    console.print("[bold blue]═══════════════════════════════════════════[/]\n")
    
    console.print(f"Target: {args.target} documents")
    console.print(f"Output: {args.output}")
    
    all_documents = []
    
    if args.source == "sample":
        # Create sample corpus
        all_documents = create_sample_corpus(args.output, args.target)
    
    elif args.source == "auto":
        # Try multiple sources
        console.print("\n[bold]Available sources:[/]")
        
        table = Table()
        table.add_column("Source", style="cyan")
        table.add_column("Description")
        table.add_column("Domain")
        table.add_column("Est. Pairs")
        
        for key, src in SOURCES.items():
            table.add_row(key, src["description"], src["domain"], f"{src['estimated_pairs']:,}")
        
        console.print(table)
        
        # Ask user which sources to use
        selected = Prompt.ask(
            "\nSelect sources (comma-separated, or 'sample' for synthetic)",
            default="sample"
        )
        
        if selected.lower() == "sample":
            all_documents = create_sample_corpus(args.output, args.target)
        else:
            sources = [s.strip() for s in selected.split(",")]
            docs_per_source = args.target // len(sources) + 1
            
            for source in sources:
                if source in SOURCES:
                    docs = collect_from_opus(source, docs_per_source, args.output, args.cache)
                    all_documents.extend(docs)
    
    else:
        # Specific source
        if args.source in SOURCES:
            all_documents = collect_from_opus(args.source, args.target, args.output, args.cache)
        else:
            all_documents = create_sample_corpus(args.output, args.target)
    
    if not all_documents:
        console.print("\n[yellow]No documents collected. Creating sample corpus instead...[/]")
        all_documents = create_sample_corpus(args.output, args.target)
    
    # Limit to target
    all_documents = all_documents[:args.target]
    
    # Save corpus
    console.print(f"\n[bold]Saving {len(all_documents)} documents...[/]")
    source_dir, ref_dir = save_corpus(all_documents, args.output)
    
    console.print("\n[bold green]✓ Corpus collection complete![/]")
    console.print(f"  Documents: {len(all_documents)}")
    console.print(f"  Source files: {source_dir}")
    console.print(f"  Reference files: {ref_dir}")
    
    # Show sample
    if all_documents:
        console.print("\n[bold]Sample document:[/]")
        sample = all_documents[0]
        console.print(f"  [cyan]EN:[/] {sample['source'][:200]}...")
        console.print(f"  [green]FR:[/] {sample['target'][:200]}...")


if __name__ == "__main__":
    main()

