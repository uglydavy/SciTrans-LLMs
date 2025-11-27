"""
Corpus loading and management for experiments.

Handles loading parallel corpora from disk with:
- Source texts (English)
- Reference translations (French)
- Optional glossaries
- Metadata

Usage:
    corpus = load_corpus("corpus/")
    for doc in corpus.documents:
        print(doc.source_text)
        print(doc.reference_text)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Iterator

from scitrans_llms.models import Document
from scitrans_llms.translate.glossary import Glossary, load_glossary_csv


@dataclass
class CorpusDocument:
    """A document in the corpus with source and reference."""
    doc_id: str
    source_text: str
    reference_text: str
    source_file: Path
    reference_file: Path
    domain: str = "general"
    metadata: dict = field(default_factory=dict)
    
    @property
    def source_paragraphs(self) -> list[str]:
        """Split source into paragraphs."""
        return [p.strip() for p in self.source_text.split("\n\n") if p.strip()]
    
    @property
    def reference_paragraphs(self) -> list[str]:
        """Split reference into paragraphs."""
        return [p.strip() for p in self.reference_text.split("\n\n") if p.strip()]
    
    def to_document(self) -> Document:
        """Convert to a Document for translation."""
        return Document.from_text(
            self.source_text,
            source_lang="en",
            target_lang="fr",
        )


@dataclass
class Corpus:
    """A collection of parallel documents for evaluation."""
    name: str
    documents: list[CorpusDocument] = field(default_factory=list)
    glossary: Optional[Glossary] = None
    source_lang: str = "en"
    target_lang: str = "fr"
    metadata: dict = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.documents)
    
    def __iter__(self) -> Iterator[CorpusDocument]:
        return iter(self.documents)
    
    def __getitem__(self, idx: int) -> CorpusDocument:
        return self.documents[idx]
    
    @property
    def total_segments(self) -> int:
        """Total number of segments (paragraphs) across all documents."""
        return sum(len(doc.source_paragraphs) for doc in self.documents)
    
    def get_all_sources(self) -> list[str]:
        """Get all source texts as flat list."""
        sources = []
        for doc in self.documents:
            sources.extend(doc.source_paragraphs)
        return sources
    
    def get_all_references(self) -> list[str]:
        """Get all reference texts as flat list."""
        refs = []
        for doc in self.documents:
            refs.extend(doc.reference_paragraphs)
        return refs
    
    def filter_by_domain(self, domain: str) -> Corpus:
        """Create a new corpus with only documents from a domain."""
        filtered = [d for d in self.documents if d.domain == domain]
        return Corpus(
            name=f"{self.name}_{domain}",
            documents=filtered,
            glossary=self.glossary,
            source_lang=self.source_lang,
            target_lang=self.target_lang,
        )
    
    def split(self, train_ratio: float = 0.8) -> tuple[Corpus, Corpus]:
        """Split corpus into train/test sets."""
        split_idx = int(len(self.documents) * train_ratio)
        
        train = Corpus(
            name=f"{self.name}_train",
            documents=self.documents[:split_idx],
            glossary=self.glossary,
        )
        test = Corpus(
            name=f"{self.name}_test",
            documents=self.documents[split_idx:],
            glossary=self.glossary,
        )
        return train, test
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Corpus: {self.name}",
            f"  Documents: {len(self.documents)}",
            f"  Segments: {self.total_segments}",
            f"  Languages: {self.source_lang} → {self.target_lang}",
        ]
        if self.glossary:
            lines.append(f"  Glossary: {len(self.glossary)} terms")
        return "\n".join(lines)


def load_corpus(
    corpus_dir: str | Path,
    name: Optional[str] = None,
    source_subdir: str = "source/abstracts",
    reference_subdir: str = "reference/abstracts",
    glossary_file: Optional[str] = None,
) -> Corpus:
    """Load a corpus from directory structure.
    
    Expected structure:
        corpus_dir/
            source/abstracts/*.txt
            reference/abstracts/*.txt
            glossary/*.csv (optional)
    
    Args:
        corpus_dir: Root directory of corpus
        name: Corpus name (default: directory name)
        source_subdir: Subdirectory for source files
        reference_subdir: Subdirectory for reference files
        glossary_file: Optional path to glossary CSV
        
    Returns:
        Loaded Corpus
    """
    corpus_dir = Path(corpus_dir)
    name = name or corpus_dir.name
    
    source_dir = corpus_dir / source_subdir
    reference_dir = corpus_dir / reference_subdir
    
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    documents = []
    
    # Find all source files
    for source_file in sorted(source_dir.glob("*.txt")):
        doc_id = source_file.stem
        reference_file = reference_dir / source_file.name
        
        if not reference_file.exists():
            print(f"Warning: No reference for {source_file.name}, skipping")
            continue
        
        source_text = source_file.read_text(encoding="utf-8").strip()
        reference_text = reference_file.read_text(encoding="utf-8").strip()
        
        # Try to detect domain from filename or directory
        domain = "general"
        if "_ml_" in doc_id or "machine" in doc_id.lower():
            domain = "ml"
        elif "_phys_" in doc_id or "physics" in doc_id.lower():
            domain = "physics"
        
        documents.append(CorpusDocument(
            doc_id=doc_id,
            source_text=source_text,
            reference_text=reference_text,
            source_file=source_file,
            reference_file=reference_file,
            domain=domain,
        ))
    
    # Load glossary
    glossary = None
    if glossary_file:
        glossary_path = corpus_dir / glossary_file
        if glossary_path.exists():
            glossary = load_glossary_csv(glossary_path)
    else:
        # Try to find glossary in standard location
        for gloss_file in (corpus_dir / "glossary").glob("*.csv"):
            if glossary is None:
                glossary = load_glossary_csv(gloss_file)
            else:
                glossary = glossary.merge(load_glossary_csv(gloss_file))
    
    # If no glossary found, use default
    if glossary is None:
        from scitrans_llms.translate.glossary import get_default_glossary
        glossary = get_default_glossary()
    
    return Corpus(
        name=name,
        documents=documents,
        glossary=glossary,
    )


def create_synthetic_corpus(
    output_dir: str | Path,
    num_documents: int = 10,
    paragraphs_per_doc: int = 3,
) -> Corpus:
    """Create a synthetic corpus for testing.
    
    Uses templates and the default glossary to generate
    source texts, then uses dictionary translation for
    pseudo-references.
    """
    from scitrans_llms.translate import DictionaryTranslator, get_default_glossary
    
    output_dir = Path(output_dir)
    source_dir = output_dir / "source" / "abstracts"
    ref_dir = output_dir / "reference" / "abstracts"
    source_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)
    
    glossary = get_default_glossary()
    translator = DictionaryTranslator(glossary)
    
    # Templates with glossary terms
    templates = [
        "This paper presents a novel approach to machine learning using deep learning techniques.",
        "We propose a new method for natural language processing based on the transformer architecture.",
        "Our experiments demonstrate state of the art performance on the benchmark dataset.",
        "The model uses attention mechanisms to process sequential data efficiently.",
        "We evaluate our approach using standard metrics including BLEU and COMET scores.",
        "The training procedure follows the standard optimization approach with gradient descent.",
        "Our neural network architecture consists of multiple layers with attention mechanisms.",
        "The algorithm achieves significant improvements over baseline methods.",
    ]
    
    documents = []
    
    for i in range(num_documents):
        doc_id = f"synthetic_{i+1:03d}"
        
        # Generate source paragraphs
        import random
        source_paras = random.sample(templates, min(paragraphs_per_doc, len(templates)))
        source_text = "\n\n".join(source_paras)
        
        # Generate pseudo-reference using dictionary translator
        ref_paras = []
        for para in source_paras:
            result = translator.translate(para)
            ref_paras.append(result.text)
        reference_text = "\n\n".join(ref_paras)
        
        # Save files
        source_file = source_dir / f"{doc_id}.txt"
        ref_file = ref_dir / f"{doc_id}.txt"
        
        source_file.write_text(source_text, encoding="utf-8")
        ref_file.write_text(reference_text, encoding="utf-8")
        
        documents.append(CorpusDocument(
            doc_id=doc_id,
            source_text=source_text,
            reference_text=reference_text,
            source_file=source_file,
            reference_file=ref_file,
            domain="ml",
        ))
    
    return Corpus(
        name="synthetic",
        documents=documents,
        glossary=glossary,
    )

