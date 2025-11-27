"""
Glossary module for terminology control.

This module handles:
- Loading bilingual glossaries from CSV/TXT files
- Matching terms in source text (case-aware)
- Enforcing glossary terms in translated text
- Measuring glossary adherence for evaluation

Thesis Contribution #1: Terminology-constrained translation
via glossary prompting and post-processing enforcement.

Design Philosophy:
- Glossaries are immutable after loading
- Matching is case-aware but configurable
- Enforcement can be done as post-processing or prompting
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class GlossaryEntry:
    """A single glossary entry mapping source term to target term.
    
    Attributes:
        source: Source language term
        target: Target language term
        domain: Optional domain/category (e.g., "physics", "computer science")
        notes: Optional usage notes
    """
    source: str
    target: str
    domain: str = ""
    notes: str = ""
    
    @property
    def source_lower(self) -> str:
        return self.source.lower()
    
    @property
    def target_lower(self) -> str:
        return self.target.lower()


@dataclass
class GlossaryMatch:
    """A matched glossary term in text.
    
    Attributes:
        entry: The matching glossary entry
        start: Start position in text
        end: End position in text
        matched_text: The actual text that matched (may differ in case)
    """
    entry: GlossaryEntry
    start: int
    end: int
    matched_text: str


@dataclass
class Glossary:
    """A bilingual glossary for terminology control.
    
    The glossary supports:
    - Case-sensitive and case-insensitive matching
    - Whole-word matching to avoid partial matches
    - Domain filtering for specialized terminology
    - Bidirectional lookup (source→target and target→source)
    """
    entries: list[GlossaryEntry] = field(default_factory=list)
    name: str = "default"
    source_lang: str = "en"
    target_lang: str = "fr"
    
    def __len__(self) -> int:
        return len(self.entries)
    
    def __iter__(self) -> Iterator[GlossaryEntry]:
        return iter(self.entries)
    
    def add_entry(self, source: str, target: str, domain: str = "", notes: str = "") -> None:
        """Add a new entry to the glossary."""
        self.entries.append(GlossaryEntry(source, target, domain, notes))
    
    def get_target(self, source: str, case_sensitive: bool = False) -> str | None:
        """Look up target term for a source term."""
        if case_sensitive:
            for entry in self.entries:
                if entry.source == source:
                    return entry.target
        else:
            source_lower = source.lower()
            for entry in self.entries:
                if entry.source_lower == source_lower:
                    return entry.target
        return None
    
    def get_source(self, target: str, case_sensitive: bool = False) -> str | None:
        """Reverse lookup: get source term for a target term."""
        if case_sensitive:
            for entry in self.entries:
                if entry.target == target:
                    return entry.source
        else:
            target_lower = target.lower()
            for entry in self.entries:
                if entry.target_lower == target_lower:
                    return entry.source
        return None
    
    def find_matches(
        self,
        text: str,
        case_sensitive: bool = False,
        whole_word: bool = True,
    ) -> list[GlossaryMatch]:
        """Find all glossary term occurrences in text.
        
        Args:
            text: Text to search
            case_sensitive: Whether matching is case-sensitive
            whole_word: Whether to match whole words only
            
        Returns:
            List of GlossaryMatch objects, sorted by position
        """
        matches = []
        flags = 0 if case_sensitive else re.IGNORECASE
        
        for entry in self.entries:
            pattern_str = re.escape(entry.source)
            if whole_word:
                pattern_str = rf'\b{pattern_str}\b'
            pattern = re.compile(pattern_str, flags)
            
            for match in pattern.finditer(text):
                matches.append(GlossaryMatch(
                    entry=entry,
                    start=match.start(),
                    end=match.end(),
                    matched_text=match.group(0),
                ))
        
        # Sort by position
        matches.sort(key=lambda m: m.start)
        return matches
    
    def filter_by_domain(self, domain: str) -> Glossary:
        """Create a new glossary containing only entries from a specific domain."""
        filtered = [e for e in self.entries if e.domain.lower() == domain.lower()]
        return Glossary(
            entries=filtered,
            name=f"{self.name}_{domain}",
            source_lang=self.source_lang,
            target_lang=self.target_lang,
        )
    
    def to_dict(self) -> dict[str, str]:
        """Convert to simple dict (source → target)."""
        return {e.source: e.target for e in self.entries}
    
    def to_prompt_string(self, max_entries: int = 50) -> str:
        """Format glossary for inclusion in LLM prompts.
        
        Returns a formatted string suitable for injection into translation prompts.
        """
        lines = ["Terminology glossary (use these exact translations):"]
        for entry in self.entries[:max_entries]:
            lines.append(f"  • {entry.source} → {entry.target}")
        if len(self.entries) > max_entries:
            lines.append(f"  ... and {len(self.entries) - max_entries} more terms")
        return "\n".join(lines)
    
    def merge(self, other: Glossary) -> Glossary:
        """Merge with another glossary (other takes precedence on conflicts)."""
        combined = {e.source.lower(): e for e in self.entries}
        for entry in other.entries:
            combined[entry.source.lower()] = entry
        return Glossary(
            entries=list(combined.values()),
            name=f"{self.name}+{other.name}",
            source_lang=self.source_lang,
            target_lang=self.target_lang,
        )


# ============================================================================
# Loading Functions
# ============================================================================

def load_glossary_csv(
    path: str | Path,
    source_lang: str = "en",
    target_lang: str = "fr",
    source_col: int = 0,
    target_col: int = 1,
    domain_col: int | None = None,
    has_header: bool = True,
) -> Glossary:
    """Load a glossary from a CSV file.
    
    Expected format (default):
        source_term,target_term[,domain][,notes]
        
    Args:
        path: Path to CSV file
        source_lang: Source language code
        target_lang: Target language code
        source_col: Column index for source terms (0-based)
        target_col: Column index for target terms (0-based)
        domain_col: Optional column index for domain
        has_header: Whether file has a header row to skip
        
    Returns:
        Loaded Glossary
    """
    path = Path(path)
    entries = []
    
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        if has_header:
            next(reader, None)  # Skip header
            
        for row in reader:
            if len(row) < 2:
                continue
            source = row[source_col].strip()
            target = row[target_col].strip()
            if not source or not target:
                continue
                
            domain = ""
            if domain_col is not None and len(row) > domain_col:
                domain = row[domain_col].strip()
                
            entries.append(GlossaryEntry(source, target, domain))
    
    return Glossary(
        entries=entries,
        name=path.stem,
        source_lang=source_lang,
        target_lang=target_lang,
    )


def load_glossary_txt(
    path: str | Path,
    source_lang: str = "en",
    target_lang: str = "fr",
    separator: str = "\t",
) -> Glossary:
    """Load a glossary from a tab-separated text file.
    
    Format: source_term<separator>target_term
    """
    path = Path(path)
    entries = []
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(separator, 1)
            if len(parts) == 2:
                source, target = parts[0].strip(), parts[1].strip()
                if source and target:
                    entries.append(GlossaryEntry(source, target))
    
    return Glossary(
        entries=entries,
        name=path.stem,
        source_lang=source_lang,
        target_lang=target_lang,
    )


# ============================================================================
# Enforcement Functions
# ============================================================================

def enforce_glossary(
    translated_text: str,
    source_text: str,
    glossary: Glossary,
    case_sensitive: bool = False,
) -> str:
    """Post-process translated text to enforce glossary terms.
    
    This finds glossary terms in the source text and ensures
    the corresponding target terms appear in the translation.
    
    Strategy:
    1. Find all source terms that appear in the source text
    2. For each, check if the target term appears in translation
    3. If not, try to find and replace common mistranslations
    
    Args:
        translated_text: The translated text to fix
        source_text: Original source text
        glossary: Glossary to enforce
        case_sensitive: Whether matching is case-sensitive
        
    Returns:
        Corrected translated text
    """
    # Find which glossary terms appear in source
    source_matches = glossary.find_matches(source_text, case_sensitive=case_sensitive)
    
    result = translated_text
    
    for match in source_matches:
        target_term = match.entry.target
        
        # Check if target term already appears (case-insensitive check)
        if case_sensitive:
            already_present = target_term in result
        else:
            already_present = target_term.lower() in result.lower()
        
        if already_present:
            continue
        
        # Try to find the source term left untranslated and replace it
        # This handles cases where the translator left the term in source language
        source_term = match.entry.source
        pattern = re.compile(re.escape(source_term), re.IGNORECASE if not case_sensitive else 0)
        
        # Replace first occurrence only to be conservative
        result, n = pattern.subn(target_term, result, count=1)
        if n > 0:
            continue
        
        # More aggressive: look for partial translations or common errors
        # This is a simple heuristic and could be extended
        # For now, we just log the miss and continue
    
    return result


def check_glossary_adherence(
    translated_text: str,
    source_text: str,
    glossary: Glossary,
    case_sensitive: bool = False,
) -> dict:
    """Measure how well a translation adheres to the glossary.
    
    Returns:
        Dictionary with:
        - total_terms: Number of glossary terms in source
        - found_terms: Number correctly translated
        - missing_terms: List of terms not found in translation
        - adherence_rate: Percentage of terms correctly handled
    
    Thesis Contribution #3: This metric is used for evaluation and ablations.
    """
    source_matches = glossary.find_matches(source_text, case_sensitive=case_sensitive)
    
    # Deduplicate by source term
    unique_terms = {}
    for match in source_matches:
        key = match.entry.source.lower() if not case_sensitive else match.entry.source
        if key not in unique_terms:
            unique_terms[key] = match.entry
    
    found = []
    missing = []
    
    for entry in unique_terms.values():
        target_term = entry.target
        if case_sensitive:
            present = target_term in translated_text
        else:
            present = target_term.lower() in translated_text.lower()
        
        if present:
            found.append(entry.source)
        else:
            missing.append(entry.source)
    
    total = len(unique_terms)
    return {
        "total_terms": total,
        "found_terms": len(found),
        "found_list": found,
        "missing_terms": missing,
        "adherence_rate": len(found) / total if total > 0 else 1.0,
    }


# ============================================================================
# Built-in Glossary
# ============================================================================

def get_default_glossary() -> Glossary:
    """Return a built-in EN→FR glossary for scientific/technical terms.
    
    This provides a starting point for users without custom glossaries.
    """
    entries = [
        # General scientific terms
        GlossaryEntry("abstract", "résumé", "general"),
        GlossaryEntry("acknowledgments", "remerciements", "general"),
        GlossaryEntry("algorithm", "algorithme", "cs"),
        GlossaryEntry("analysis", "analyse", "general"),
        GlossaryEntry("approach", "approche", "general"),
        GlossaryEntry("assumption", "hypothèse", "general"),
        GlossaryEntry("background", "contexte", "general"),
        GlossaryEntry("benchmark", "référence", "cs"),
        GlossaryEntry("conclusion", "conclusion", "general"),
        GlossaryEntry("contribution", "contribution", "general"),
        GlossaryEntry("dataset", "jeu de données", "cs"),
        GlossaryEntry("discussion", "discussion", "general"),
        GlossaryEntry("equation", "équation", "math"),
        GlossaryEntry("experiment", "expérience", "general"),
        GlossaryEntry("figure", "figure", "general"),
        GlossaryEntry("framework", "cadre", "general"),
        GlossaryEntry("hypothesis", "hypothèse", "general"),
        GlossaryEntry("implementation", "implémentation", "cs"),
        GlossaryEntry("introduction", "introduction", "general"),
        GlossaryEntry("limitation", "limitation", "general"),
        GlossaryEntry("literature", "littérature", "general"),
        GlossaryEntry("method", "méthode", "general"),
        GlossaryEntry("methodology", "méthodologie", "general"),
        GlossaryEntry("model", "modèle", "general"),
        GlossaryEntry("neural network", "réseau de neurones", "ml"),
        GlossaryEntry("objective", "objectif", "general"),
        GlossaryEntry("observation", "observation", "general"),
        GlossaryEntry("optimization", "optimisation", "math"),
        GlossaryEntry("parameter", "paramètre", "general"),
        GlossaryEntry("performance", "performance", "general"),
        GlossaryEntry("pipeline", "pipeline", "cs"),
        GlossaryEntry("prediction", "prédiction", "ml"),
        GlossaryEntry("preprocessing", "prétraitement", "cs"),
        GlossaryEntry("probability", "probabilité", "math"),
        GlossaryEntry("proof", "preuve", "math"),
        GlossaryEntry("proposition", "proposition", "math"),
        GlossaryEntry("related work", "travaux connexes", "general"),
        GlossaryEntry("result", "résultat", "general"),
        GlossaryEntry("section", "section", "general"),
        GlossaryEntry("simulation", "simulation", "general"),
        GlossaryEntry("state of the art", "état de l'art", "general"),
        GlossaryEntry("table", "tableau", "general"),
        GlossaryEntry("theorem", "théorème", "math"),
        GlossaryEntry("training", "entraînement", "ml"),
        GlossaryEntry("validation", "validation", "general"),
        
        # ML/NLP specific
        GlossaryEntry("attention mechanism", "mécanisme d'attention", "ml"),
        GlossaryEntry("batch size", "taille de lot", "ml"),
        GlossaryEntry("deep learning", "apprentissage profond", "ml"),
        GlossaryEntry("embedding", "plongement", "ml"),
        GlossaryEntry("fine-tuning", "ajustement fin", "ml"),
        GlossaryEntry("gradient descent", "descente de gradient", "ml"),
        GlossaryEntry("hyperparameter", "hyperparamètre", "ml"),
        GlossaryEntry("inference", "inférence", "ml"),
        GlossaryEntry("language model", "modèle de langue", "nlp"),
        GlossaryEntry("learning rate", "taux d'apprentissage", "ml"),
        GlossaryEntry("loss function", "fonction de perte", "ml"),
        GlossaryEntry("machine learning", "apprentissage automatique", "ml"),
        GlossaryEntry("machine translation", "traduction automatique", "nlp"),
        GlossaryEntry("natural language processing", "traitement automatique du langage naturel", "nlp"),
        GlossaryEntry("overfitting", "surapprentissage", "ml"),
        GlossaryEntry("pretrained", "pré-entraîné", "ml"),
        GlossaryEntry("prompt", "invite", "nlp"),
        GlossaryEntry("tokenization", "tokenisation", "nlp"),
        GlossaryEntry("transformer", "transformeur", "ml"),
    ]
    
    return Glossary(
        entries=entries,
        name="default_en_fr",
        source_lang="en",
        target_lang="fr",
    )

