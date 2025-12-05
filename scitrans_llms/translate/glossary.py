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
    
    Expanded to 500+ terms covering:
    - General scientific vocabulary
    - Machine Learning & AI
    - Mathematics & Statistics
    - Physics & Chemistry
    - Computer Science
    - Natural Language Processing
    - Document structure terminology
    
    Thesis Contribution #1: Terminology-constrained translation.
    """
    entries = [
        # ===== General Scientific Terms =====
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
        GlossaryEntry("objective", "objectif", "general"),
        GlossaryEntry("observation", "observation", "general"),
        GlossaryEntry("parameter", "paramètre", "general"),
        GlossaryEntry("performance", "performance", "general"),
        GlossaryEntry("related work", "travaux connexes", "general"),
        GlossaryEntry("result", "résultat", "general"),
        GlossaryEntry("section", "section", "general"),
        GlossaryEntry("simulation", "simulation", "general"),
        GlossaryEntry("state of the art", "état de l'art", "general"),
        GlossaryEntry("table", "tableau", "general"),
        GlossaryEntry("validation", "validation", "general"),
        GlossaryEntry("evaluation", "évaluation", "general"),
        GlossaryEntry("metric", "métrique", "general"),
        GlossaryEntry("measure", "mesure", "general"),
        GlossaryEntry("score", "score", "general"),
        GlossaryEntry("baseline", "ligne de base", "ml"),
        GlossaryEntry("comparison", "comparaison", "general"),
        GlossaryEntry("improvement", "amélioration", "general"),
        GlossaryEntry("advancement", "avancement", "general"),
        GlossaryEntry("progress", "progrès", "general"),
        GlossaryEntry("development", "développement", "general"),
        GlossaryEntry("research", "recherche", "general"),
        GlossaryEntry("study", "étude", "general"),
        GlossaryEntry("investigation", "investigation", "general"),
        GlossaryEntry("review", "revue", "general"),
        GlossaryEntry("survey", "enquête", "general"),
        GlossaryEntry("overview", "aperçu", "general"),
        GlossaryEntry("summary", "résumé", "general"),
        GlossaryEntry("description", "description", "general"),
        GlossaryEntry("explanation", "explication", "general"),
        GlossaryEntry("interpretation", "interprétation", "general"),
        GlossaryEntry("understanding", "compréhension", "general"),
        GlossaryEntry("knowledge", "connaissance", "general"),
        GlossaryEntry("information", "information", "general"),
        GlossaryEntry("data", "données", "general"),
        GlossaryEntry("evidence", "preuve", "general"),
        GlossaryEntry("demonstration", "démonstration", "general"),
        GlossaryEntry("example", "exemple", "general"),
        GlossaryEntry("instance", "instance", "general"),
        GlossaryEntry("case", "cas", "general"),
        GlossaryEntry("scenario", "scénario", "general"),
        GlossaryEntry("situation", "situation", "general"),
        GlossaryEntry("condition", "condition", "general"),
        GlossaryEntry("context", "contexte", "general"),
        GlossaryEntry("environment", "environnement", "general"),
        GlossaryEntry("configuration", "configuration", "general"),
        GlossaryEntry("setup", "configuration", "general"),
        GlossaryEntry("organization", "organisation", "general"),
        GlossaryEntry("structure", "structure", "general"),
        GlossaryEntry("architecture", "architecture", "general"),
        GlossaryEntry("design", "conception", "general"),
        GlossaryEntry("strategy", "stratégie", "general"),
        GlossaryEntry("technique", "technique", "general"),
        GlossaryEntry("procedure", "procédure", "general"),
        GlossaryEntry("protocol", "protocole", "general"),
        GlossaryEntry("workflow", "flux de travail", "general"),
        GlossaryEntry("process", "processus", "general"),
        GlossaryEntry("step", "étape", "general"),
        GlossaryEntry("phase", "phase", "general"),
        GlossaryEntry("stage", "stade", "general"),
        GlossaryEntry("level", "niveau", "general"),
        GlossaryEntry("layer", "couche", "general"),
        GlossaryEntry("component", "composant", "general"),
        GlossaryEntry("element", "élément", "general"),
        GlossaryEntry("factor", "facteur", "general"),
        GlossaryEntry("variable", "variable", "general"),
        GlossaryEntry("constant", "constante", "general"),
        GlossaryEntry("coefficient", "coefficient", "general"),
        GlossaryEntry("ratio", "ratio", "general"),
        GlossaryEntry("rate", "taux", "general"),
        GlossaryEntry("proportion", "proportion", "general"),
        GlossaryEntry("percentage", "pourcentage", "general"),
        GlossaryEntry("amount", "quantité", "general"),
        GlossaryEntry("quantity", "quantité", "general"),
        GlossaryEntry("value", "valeur", "general"),
        GlossaryEntry("magnitude", "amplitude", "general"),
        GlossaryEntry("range", "plage", "general"),
        GlossaryEntry("interval", "intervalle", "general"),
        GlossaryEntry("boundary", "limite", "general"),
        GlossaryEntry("threshold", "seuil", "general"),
        GlossaryEntry("limit", "limite", "general"),
        GlossaryEntry("maximum", "maximum", "general"),
        GlossaryEntry("minimum", "minimum", "general"),
        GlossaryEntry("average", "moyenne", "general"),
        GlossaryEntry("mean", "moyenne", "stats"),
        GlossaryEntry("median", "médiane", "stats"),
        GlossaryEntry("mode", "mode", "stats"),
        
        # ===== Machine Learning & AI =====
        GlossaryEntry("neural network", "réseau de neurones", "ml"),
        GlossaryEntry("deep learning", "apprentissage profond", "ml"),
        GlossaryEntry("machine learning", "apprentissage automatique", "ml"),
        GlossaryEntry("artificial intelligence", "intelligence artificielle", "ml"),
        GlossaryEntry("supervised learning", "apprentissage supervisé", "ml"),
        GlossaryEntry("unsupervised learning", "apprentissage non supervisé", "ml"),
        GlossaryEntry("reinforcement learning", "apprentissage par renforcement", "ml"),
        GlossaryEntry("semi-supervised learning", "apprentissage semi-supervisé", "ml"),
        GlossaryEntry("self-supervised learning", "apprentissage auto-supervisé", "ml"),
        GlossaryEntry("transfer learning", "apprentissage par transfert", "ml"),
        GlossaryEntry("federated learning", "apprentissage fédéré", "ml"),
        GlossaryEntry("continual learning", "apprentissage continu", "ml"),
        GlossaryEntry("meta-learning", "méta-apprentissage", "ml"),
        GlossaryEntry("few-shot learning", "apprentissage à partir de peu d'exemples", "ml"),
        GlossaryEntry("zero-shot learning", "apprentissage sans exemple", "ml"),
        GlossaryEntry("attention mechanism", "mécanisme d'attention", "ml"),
        GlossaryEntry("self-attention", "auto-attention", "ml"),
        GlossaryEntry("multi-head attention", "attention multi-tête", "ml"),
        GlossaryEntry("cross-attention", "attention croisée", "ml"),
        GlossaryEntry("batch size", "taille de lot", "ml"),
        GlossaryEntry("mini-batch", "mini-lot", "ml"),
        GlossaryEntry("epoch", "époque", "ml"),
        GlossaryEntry("iteration", "itération", "ml"),
        GlossaryEntry("embedding", "plongement", "ml"),
        GlossaryEntry("word embedding", "plongement de mots", "ml"),
        GlossaryEntry("positional embedding", "plongement positionnel", "ml"),
        GlossaryEntry("fine-tuning", "ajustement fin", "ml"),
        GlossaryEntry("pre-training", "pré-entraînement", "ml"),
        GlossaryEntry("gradient descent", "descente de gradient", "ml"),
        GlossaryEntry("stochastic gradient descent", "descente de gradient stochastique", "ml"),
        GlossaryEntry("backpropagation", "rétropropagation", "ml"),
        GlossaryEntry("forward pass", "passe avant", "ml"),
        GlossaryEntry("backward pass", "passe arrière", "ml"),
        GlossaryEntry("hyperparameter", "hyperparamètre", "ml"),
        GlossaryEntry("inference", "inférence", "ml"),
        GlossaryEntry("prediction", "prédiction", "ml"),
        GlossaryEntry("classification", "classification", "ml"),
        GlossaryEntry("regression", "régression", "ml"),
        GlossaryEntry("clustering", "regroupement", "ml"),
        GlossaryEntry("segmentation", "segmentation", "ml"),
        GlossaryEntry("detection", "détection", "ml"),
        GlossaryEntry("recognition", "reconnaissance", "ml"),
        GlossaryEntry("generation", "génération", "ml"),
        GlossaryEntry("training", "entraînement", "ml"),
        GlossaryEntry("testing", "test", "ml"),
        GlossaryEntry("learning rate", "taux d'apprentissage", "ml"),
        GlossaryEntry("loss function", "fonction de perte", "ml"),
        GlossaryEntry("cost function", "fonction de coût", "ml"),
        GlossaryEntry("objective function", "fonction objectif", "ml"),
        GlossaryEntry("activation function", "fonction d'activation", "ml"),
        GlossaryEntry("sigmoid", "sigmoïde", "ml"),
        GlossaryEntry("softmax", "softmax", "ml"),
        GlossaryEntry("relu", "relu", "ml"),
        GlossaryEntry("tanh", "tanh", "ml"),
        GlossaryEntry("dropout", "abandon", "ml"),
        GlossaryEntry("regularization", "régularisation", "ml"),
        GlossaryEntry("normalization", "normalisation", "ml"),
        GlossaryEntry("batch normalization", "normalisation par lot", "ml"),
        GlossaryEntry("layer normalization", "normalisation de couche", "ml"),
        GlossaryEntry("overfitting", "surapprentissage", "ml"),
        GlossaryEntry("underfitting", "sous-apprentissage", "ml"),
        GlossaryEntry("generalization", "généralisation", "ml"),
        GlossaryEntry("bias", "biais", "ml"),
        GlossaryEntry("variance", "variance", "ml"),
        GlossaryEntry("pretrained", "pré-entraîné", "ml"),
        GlossaryEntry("pretrained model", "modèle pré-entraîné", "ml"),
        GlossaryEntry("foundation model", "modèle de fondation", "ml"),
        GlossaryEntry("large language model", "grand modèle de langue", "ml"),
        GlossaryEntry("transformer", "transformeur", "ml"),
        GlossaryEntry("encoder", "encodeur", "ml"),
        GlossaryEntry("decoder", "décodeur", "ml"),
        GlossaryEntry("encoder-decoder", "encodeur-décodeur", "ml"),
        GlossaryEntry("autoencoder", "auto-encodeur", "ml"),
        GlossaryEntry("variational autoencoder", "auto-encodeur variationnel", "ml"),
        GlossaryEntry("generative adversarial network", "réseau antagoniste génératif", "ml"),
        GlossaryEntry("convolutional neural network", "réseau de neurones convolutif", "ml"),
        GlossaryEntry("recurrent neural network", "réseau de neurones récurrent", "ml"),
        GlossaryEntry("long short-term memory", "mémoire à court terme longue", "ml"),
        GlossaryEntry("gated recurrent unit", "unité récurrente à portes", "ml"),
        GlossaryEntry("residual connection", "connexion résiduelle", "ml"),
        GlossaryEntry("skip connection", "connexion de saut", "ml"),
        GlossaryEntry("pooling", "regroupement", "ml"),
        GlossaryEntry("max pooling", "regroupement maximum", "ml"),
        GlossaryEntry("average pooling", "regroupement moyen", "ml"),
        GlossaryEntry("convolution", "convolution", "ml"),
        GlossaryEntry("kernel", "noyau", "ml"),
        GlossaryEntry("filter", "filtre", "ml"),
        GlossaryEntry("stride", "pas", "ml"),
        GlossaryEntry("padding", "rembourrage", "ml"),
        GlossaryEntry("feature map", "carte de caractéristiques", "ml"),
        GlossaryEntry("feature extraction", "extraction de caractéristiques", "ml"),
        GlossaryEntry("feature engineering", "ingénierie des caractéristiques", "ml"),
        GlossaryEntry("dimensionality reduction", "réduction de dimensionnalité", "ml"),
        GlossaryEntry("principal component analysis", "analyse en composantes principales", "ml"),
        
        # ===== Metrics & Evaluation =====
        GlossaryEntry("accuracy", "exactitude", "ml"),
        GlossaryEntry("precision", "précision", "ml"),
        GlossaryEntry("recall", "rappel", "ml"),
        GlossaryEntry("f1 score", "score F1", "ml"),
        GlossaryEntry("f-measure", "mesure F", "ml"),
        GlossaryEntry("area under curve", "aire sous la courbe", "ml"),
        GlossaryEntry("roc curve", "courbe ROC", "ml"),
        GlossaryEntry("confusion matrix", "matrice de confusion", "ml"),
        GlossaryEntry("true positive", "vrai positif", "ml"),
        GlossaryEntry("false positive", "faux positif", "ml"),
        GlossaryEntry("true negative", "vrai négatif", "ml"),
        GlossaryEntry("false negative", "faux négatif", "ml"),
        GlossaryEntry("cross-validation", "validation croisée", "ml"),
        GlossaryEntry("k-fold cross-validation", "validation croisée à k plis", "ml"),
        GlossaryEntry("train-test split", "division entraînement-test", "ml"),
        GlossaryEntry("validation set", "ensemble de validation", "ml"),
        GlossaryEntry("test set", "ensemble de test", "ml"),
        GlossaryEntry("training set", "ensemble d'entraînement", "ml"),
        GlossaryEntry("hold-out set", "ensemble de réserve", "ml"),
        GlossaryEntry("perplexity", "perplexité", "nlp"),
        GlossaryEntry("bleu score", "score BLEU", "nlp"),
        GlossaryEntry("rouge score", "score ROUGE", "nlp"),
        GlossaryEntry("meteor score", "score METEOR", "nlp"),
        GlossaryEntry("word error rate", "taux d'erreur de mots", "nlp"),
        GlossaryEntry("character error rate", "taux d'erreur de caractères", "nlp"),
        
        # ===== Natural Language Processing =====
        GlossaryEntry("natural language processing", "traitement automatique du langage naturel", "nlp"),
        GlossaryEntry("natural language understanding", "compréhension du langage naturel", "nlp"),
        GlossaryEntry("natural language generation", "génération de langage naturel", "nlp"),
        GlossaryEntry("language model", "modèle de langue", "nlp"),
        GlossaryEntry("machine translation", "traduction automatique", "nlp"),
        GlossaryEntry("neural machine translation", "traduction automatique neuronale", "nlp"),
        GlossaryEntry("statistical machine translation", "traduction automatique statistique", "nlp"),
        GlossaryEntry("tokenization", "tokenisation", "nlp"),
        GlossaryEntry("tokenizer", "tokeniseur", "nlp"),
        GlossaryEntry("subword", "sous-mot", "nlp"),
        GlossaryEntry("byte pair encoding", "encodage par paires d'octets", "nlp"),
        GlossaryEntry("wordpiece", "morceau de mot", "nlp"),
        GlossaryEntry("sentencepiece", "morceau de phrase", "nlp"),
        GlossaryEntry("vocabulary", "vocabulaire", "nlp"),
        GlossaryEntry("out-of-vocabulary", "hors vocabulaire", "nlp"),
        GlossaryEntry("named entity recognition", "reconnaissance d'entités nommées", "nlp"),
        GlossaryEntry("part-of-speech tagging", "étiquetage morphosyntaxique", "nlp"),
        GlossaryEntry("dependency parsing", "analyse en dépendances", "nlp"),
        GlossaryEntry("constituency parsing", "analyse en constituants", "nlp"),
        GlossaryEntry("semantic role labeling", "étiquetage de rôles sémantiques", "nlp"),
        GlossaryEntry("coreference resolution", "résolution de coréférence", "nlp"),
        GlossaryEntry("sentiment analysis", "analyse de sentiment", "nlp"),
        GlossaryEntry("text classification", "classification de texte", "nlp"),
        GlossaryEntry("question answering", "réponse aux questions", "nlp"),
        GlossaryEntry("reading comprehension", "compréhension de lecture", "nlp"),
        GlossaryEntry("summarization", "résumé automatique", "nlp"),
        GlossaryEntry("text generation", "génération de texte", "nlp"),
        GlossaryEntry("dialogue system", "système de dialogue", "nlp"),
        GlossaryEntry("chatbot", "agent conversationnel", "nlp"),
        GlossaryEntry("prompt", "invite", "nlp"),
        GlossaryEntry("prompt engineering", "ingénierie d'invites", "nlp"),
        GlossaryEntry("in-context learning", "apprentissage en contexte", "nlp"),
        GlossaryEntry("instruction tuning", "ajustement par instructions", "nlp"),
        GlossaryEntry("sequence-to-sequence", "séquence à séquence", "nlp"),
        GlossaryEntry("beam search", "recherche en faisceau", "nlp"),
        GlossaryEntry("greedy decoding", "décodage glouton", "nlp"),
        GlossaryEntry("sampling", "échantillonnage", "nlp"),
        GlossaryEntry("temperature", "température", "nlp"),
        GlossaryEntry("top-k sampling", "échantillonnage top-k", "nlp"),
        GlossaryEntry("nucleus sampling", "échantillonnage par noyau", "nlp"),
        GlossaryEntry("repetition penalty", "pénalité de répétition", "nlp"),
        GlossaryEntry("length penalty", "pénalité de longueur", "nlp"),
        
        # ===== Mathematics =====
        GlossaryEntry("probability", "probabilité", "math"),
        GlossaryEntry("distribution", "distribution", "math"),
        GlossaryEntry("normal distribution", "distribution normale", "math"),
        GlossaryEntry("gaussian distribution", "distribution gaussienne", "math"),
        GlossaryEntry("uniform distribution", "distribution uniforme", "math"),
        GlossaryEntry("exponential distribution", "distribution exponentielle", "math"),
        GlossaryEntry("conditional probability", "probabilité conditionnelle", "math"),
        GlossaryEntry("joint probability", "probabilité jointe", "math"),
        GlossaryEntry("marginal probability", "probabilité marginale", "math"),
        GlossaryEntry("bayesian", "bayésien", "math"),
        GlossaryEntry("likelihood", "vraisemblance", "math"),
        GlossaryEntry("maximum likelihood", "maximum de vraisemblance", "math"),
        GlossaryEntry("posterior", "postérieur", "math"),
        GlossaryEntry("prior", "a priori", "math"),
        GlossaryEntry("expectation", "espérance", "math"),
        GlossaryEntry("expected value", "valeur espérée", "math"),
        GlossaryEntry("standard deviation", "écart-type", "math"),
        GlossaryEntry("covariance", "covariance", "math"),
        GlossaryEntry("correlation", "corrélation", "math"),
        GlossaryEntry("entropy", "entropie", "math"),
        GlossaryEntry("cross-entropy", "entropie croisée", "math"),
        GlossaryEntry("mutual information", "information mutuelle", "math"),
        GlossaryEntry("divergence", "divergence", "math"),
        GlossaryEntry("kl divergence", "divergence KL", "math"),
        GlossaryEntry("derivative", "dérivée", "math"),
        GlossaryEntry("partial derivative", "dérivée partielle", "math"),
        GlossaryEntry("gradient", "gradient", "math"),
        GlossaryEntry("hessian", "hessien", "math"),
        GlossaryEntry("jacobian", "jacobien", "math"),
        GlossaryEntry("integral", "intégrale", "math"),
        GlossaryEntry("integration", "intégration", "math"),
        GlossaryEntry("differentiation", "différentiation", "math"),
        GlossaryEntry("optimization", "optimisation", "math"),
        GlossaryEntry("convex optimization", "optimisation convexe", "math"),
        GlossaryEntry("constraint", "contrainte", "math"),
        GlossaryEntry("lagrangian", "lagrangien", "math"),
        GlossaryEntry("linear algebra", "algèbre linéaire", "math"),
        GlossaryEntry("matrix", "matrice", "math"),
        GlossaryEntry("vector", "vecteur", "math"),
        GlossaryEntry("tensor", "tenseur", "math"),
        GlossaryEntry("scalar", "scalaire", "math"),
        GlossaryEntry("dot product", "produit scalaire", "math"),
        GlossaryEntry("cross product", "produit vectoriel", "math"),
        GlossaryEntry("matrix multiplication", "multiplication matricielle", "math"),
        GlossaryEntry("eigenvalue", "valeur propre", "math"),
        GlossaryEntry("eigenvector", "vecteur propre", "math"),
        GlossaryEntry("singular value decomposition", "décomposition en valeurs singulières", "math"),
        GlossaryEntry("norm", "norme", "math"),
        GlossaryEntry("euclidean distance", "distance euclidienne", "math"),
        GlossaryEntry("cosine similarity", "similarité cosinus", "math"),
        GlossaryEntry("theorem", "théorème", "math"),
        GlossaryEntry("lemma", "lemme", "math"),
        GlossaryEntry("corollary", "corollaire", "math"),
        GlossaryEntry("proof", "preuve", "math"),
        GlossaryEntry("proposition", "proposition", "math"),
        GlossaryEntry("conjecture", "conjecture", "math"),
        GlossaryEntry("axiom", "axiome", "math"),
        GlossaryEntry("definition", "définition", "math"),
        
        # ===== Statistics =====
        GlossaryEntry("statistical significance", "significativité statistique", "stats"),
        GlossaryEntry("p-value", "valeur p", "stats"),
        GlossaryEntry("confidence interval", "intervalle de confiance", "stats"),
        GlossaryEntry("hypothesis testing", "test d'hypothèse", "stats"),
        GlossaryEntry("null hypothesis", "hypothèse nulle", "stats"),
        GlossaryEntry("alternative hypothesis", "hypothèse alternative", "stats"),
        GlossaryEntry("type I error", "erreur de type I", "stats"),
        GlossaryEntry("type II error", "erreur de type II", "stats"),
        GlossaryEntry("sample size", "taille d'échantillon", "stats"),
        GlossaryEntry("population", "population", "stats"),
        GlossaryEntry("random sampling", "échantillonnage aléatoire", "stats"),
        GlossaryEntry("stratified sampling", "échantillonnage stratifié", "stats"),
        GlossaryEntry("bootstrap", "bootstrap", "stats"),
        GlossaryEntry("t-test", "test t", "stats"),
        GlossaryEntry("chi-square test", "test du chi-carré", "stats"),
        GlossaryEntry("anova", "ANOVA", "stats"),
        GlossaryEntry("linear regression", "régression linéaire", "stats"),
        GlossaryEntry("logistic regression", "régression logistique", "stats"),
        GlossaryEntry("multivariate analysis", "analyse multivariée", "stats"),
        GlossaryEntry("factor analysis", "analyse factorielle", "stats"),
        
        # ===== Computer Science =====
        GlossaryEntry("pipeline", "pipeline", "cs"),
        GlossaryEntry("preprocessing", "prétraitement", "cs"),
        GlossaryEntry("postprocessing", "post-traitement", "cs"),
        GlossaryEntry("runtime", "temps d'exécution", "cs"),
        GlossaryEntry("memory", "mémoire", "cs"),
        GlossaryEntry("storage", "stockage", "cs"),
        GlossaryEntry("computation", "calcul", "cs"),
        GlossaryEntry("parallel processing", "traitement parallèle", "cs"),
        GlossaryEntry("distributed computing", "calcul distribué", "cs"),
        GlossaryEntry("cloud computing", "infonuagique", "cs"),
        GlossaryEntry("gpu", "GPU", "cs"),
        GlossaryEntry("cpu", "CPU", "cs"),
        GlossaryEntry("tpu", "TPU", "cs"),
        GlossaryEntry("hardware", "matériel", "cs"),
        GlossaryEntry("software", "logiciel", "cs"),
        GlossaryEntry("library", "bibliothèque", "cs"),
        GlossaryEntry("module", "module", "cs"),
        GlossaryEntry("function", "fonction", "cs"),
        GlossaryEntry("class", "classe", "cs"),
        GlossaryEntry("object", "objet", "cs"),
        GlossaryEntry("interface", "interface", "cs"),
        GlossaryEntry("api", "API", "cs"),
        GlossaryEntry("database", "base de données", "cs"),
        GlossaryEntry("query", "requête", "cs"),
        GlossaryEntry("index", "index", "cs"),
        GlossaryEntry("cache", "cache", "cs"),
        GlossaryEntry("buffer", "tampon", "cs"),
        GlossaryEntry("queue", "file d'attente", "cs"),
        GlossaryEntry("stack", "pile", "cs"),
        GlossaryEntry("tree", "arbre", "cs"),
        GlossaryEntry("graph", "graphe", "cs"),
        GlossaryEntry("node", "nœud", "cs"),
        GlossaryEntry("edge", "arête", "cs"),
        GlossaryEntry("path", "chemin", "cs"),
        GlossaryEntry("search", "recherche", "cs"),
        GlossaryEntry("sorting", "tri", "cs"),
        GlossaryEntry("hashing", "hachage", "cs"),
        GlossaryEntry("encryption", "chiffrement", "cs"),
        GlossaryEntry("compression", "compression", "cs"),
        GlossaryEntry("serialization", "sérialisation", "cs"),
        GlossaryEntry("parsing", "analyse syntaxique", "cs"),
        GlossaryEntry("rendering", "rendu", "cs"),
        GlossaryEntry("debugging", "débogage", "cs"),
        GlossaryEntry("testing", "test", "cs"),
        GlossaryEntry("deployment", "déploiement", "cs"),
        GlossaryEntry("scalability", "scalabilité", "cs"),
        GlossaryEntry("latency", "latence", "cs"),
        GlossaryEntry("throughput", "débit", "cs"),
        GlossaryEntry("bandwidth", "bande passante", "cs"),
        GlossaryEntry("concurrency", "concurrence", "cs"),
        GlossaryEntry("synchronization", "synchronisation", "cs"),
        GlossaryEntry("asynchronous", "asynchrone", "cs"),
        GlossaryEntry("callback", "rappel", "cs"),
        GlossaryEntry("thread", "fil d'exécution", "cs"),
        GlossaryEntry("process", "processus", "cs"),
        
        # ===== Physics =====
        GlossaryEntry("energy", "énergie", "physics"),
        GlossaryEntry("force", "force", "physics"),
        GlossaryEntry("mass", "masse", "physics"),
        GlossaryEntry("velocity", "vitesse", "physics"),
        GlossaryEntry("acceleration", "accélération", "physics"),
        GlossaryEntry("momentum", "quantité de mouvement", "physics"),
        GlossaryEntry("frequency", "fréquence", "physics"),
        GlossaryEntry("wavelength", "longueur d'onde", "physics"),
        GlossaryEntry("amplitude", "amplitude", "physics"),
        GlossaryEntry("phase", "phase", "physics"),
        GlossaryEntry("spectrum", "spectre", "physics"),
        GlossaryEntry("quantum", "quantique", "physics"),
        GlossaryEntry("particle", "particule", "physics"),
        GlossaryEntry("wave", "onde", "physics"),
        GlossaryEntry("field", "champ", "physics"),
        GlossaryEntry("potential", "potentiel", "physics"),
        GlossaryEntry("charge", "charge", "physics"),
        GlossaryEntry("current", "courant", "physics"),
        GlossaryEntry("voltage", "tension", "physics"),
        GlossaryEntry("resistance", "résistance", "physics"),
        GlossaryEntry("capacitance", "capacité", "physics"),
        GlossaryEntry("inductance", "inductance", "physics"),
        GlossaryEntry("magnetic", "magnétique", "physics"),
        GlossaryEntry("electric", "électrique", "physics"),
        GlossaryEntry("electromagnetic", "électromagnétique", "physics"),
        GlossaryEntry("thermal", "thermique", "physics"),
        GlossaryEntry("temperature", "température", "physics"),
        GlossaryEntry("heat", "chaleur", "physics"),
        GlossaryEntry("pressure", "pression", "physics"),
        GlossaryEntry("density", "densité", "physics"),
        GlossaryEntry("viscosity", "viscosité", "physics"),
        
        # ===== Chemistry =====
        GlossaryEntry("molecule", "molécule", "chemistry"),
        GlossaryEntry("atom", "atome", "chemistry"),
        GlossaryEntry("electron", "électron", "chemistry"),
        GlossaryEntry("proton", "proton", "chemistry"),
        GlossaryEntry("neutron", "neutron", "chemistry"),
        GlossaryEntry("ion", "ion", "chemistry"),
        GlossaryEntry("bond", "liaison", "chemistry"),
        GlossaryEntry("compound", "composé", "chemistry"),
        GlossaryEntry("element", "élément", "chemistry"),
        GlossaryEntry("reaction", "réaction", "chemistry"),
        GlossaryEntry("catalyst", "catalyseur", "chemistry"),
        GlossaryEntry("enzyme", "enzyme", "chemistry"),
        GlossaryEntry("protein", "protéine", "chemistry"),
        GlossaryEntry("acid", "acide", "chemistry"),
        GlossaryEntry("base", "base", "chemistry"),
        GlossaryEntry("ph", "pH", "chemistry"),
        GlossaryEntry("solution", "solution", "chemistry"),
        GlossaryEntry("concentration", "concentration", "chemistry"),
        GlossaryEntry("solvent", "solvant", "chemistry"),
        GlossaryEntry("solute", "soluté", "chemistry"),
        
        # ===== Document Structure =====
        GlossaryEntry("chapter", "chapitre", "general"),
        GlossaryEntry("paragraph", "paragraphe", "general"),
        GlossaryEntry("sentence", "phrase", "general"),
        GlossaryEntry("word", "mot", "general"),
        GlossaryEntry("character", "caractère", "general"),
        GlossaryEntry("heading", "titre", "general"),
        GlossaryEntry("subheading", "sous-titre", "general"),
        GlossaryEntry("caption", "légende", "general"),
        GlossaryEntry("footnote", "note de bas de page", "general"),
        GlossaryEntry("reference", "référence", "general"),
        GlossaryEntry("citation", "citation", "general"),
        GlossaryEntry("bibliography", "bibliographie", "general"),
        GlossaryEntry("appendix", "annexe", "general"),
        GlossaryEntry("index", "index", "general"),
        GlossaryEntry("glossary", "glossaire", "general"),
        GlossaryEntry("preface", "préface", "general"),
        GlossaryEntry("foreword", "avant-propos", "general"),
        GlossaryEntry("abbreviation", "abréviation", "general"),
        GlossaryEntry("acronym", "acronyme", "general"),
        GlossaryEntry("symbol", "symbole", "general"),
        GlossaryEntry("notation", "notation", "general"),
        GlossaryEntry("formula", "formule", "general"),
        GlossaryEntry("diagram", "diagramme", "general"),
        GlossaryEntry("chart", "graphique", "general"),
        GlossaryEntry("illustration", "illustration", "general"),
        GlossaryEntry("photograph", "photographie", "general"),
        GlossaryEntry("image", "image", "general"),
        GlossaryEntry("layout", "mise en page", "general"),
        GlossaryEntry("format", "format", "general"),
        GlossaryEntry("margin", "marge", "general"),
        GlossaryEntry("spacing", "espacement", "general"),
        GlossaryEntry("alignment", "alignement", "general"),
        GlossaryEntry("indentation", "indentation", "general"),
        GlossaryEntry("font", "police", "general"),
        GlossaryEntry("typeface", "police de caractères", "general"),
        GlossaryEntry("bold", "gras", "general"),
        GlossaryEntry("italic", "italique", "general"),
        GlossaryEntry("underline", "souligné", "general"),
        
        # ===== Additional Academic Terms =====
        GlossaryEntry("peer review", "examen par les pairs", "general"),
        GlossaryEntry("double-blind", "double aveugle", "general"),
        GlossaryEntry("reproducibility", "reproductibilité", "general"),
        GlossaryEntry("replication", "réplication", "general"),
        GlossaryEntry("ethics", "éthique", "general"),
        GlossaryEntry("consent", "consentement", "general"),
        GlossaryEntry("privacy", "confidentialité", "general"),
        GlossaryEntry("anonymization", "anonymisation", "general"),
        GlossaryEntry("bias", "biais", "general"),
        GlossaryEntry("fairness", "équité", "general"),
        GlossaryEntry("transparency", "transparence", "general"),
        GlossaryEntry("accountability", "responsabilité", "general"),
        GlossaryEntry("open source", "source ouverte", "cs"),
        GlossaryEntry("open access", "accès libre", "general"),
        GlossaryEntry("preprint", "prépublication", "general"),
        GlossaryEntry("postprint", "post-publication", "general"),
        GlossaryEntry("manuscript", "manuscrit", "general"),
        GlossaryEntry("draft", "brouillon", "general"),
        GlossaryEntry("revision", "révision", "general"),
        GlossaryEntry("submission", "soumission", "general"),
        GlossaryEntry("acceptance", "acceptation", "general"),
        GlossaryEntry("rejection", "rejet", "general"),
        GlossaryEntry("workshop", "atelier", "general"),
        GlossaryEntry("conference", "conférence", "general"),
        GlossaryEntry("symposium", "symposium", "general"),
        GlossaryEntry("journal", "revue", "general"),
        GlossaryEntry("proceedings", "actes", "general"),
        GlossaryEntry("volume", "volume", "general"),
        GlossaryEntry("issue", "numéro", "general"),
        GlossaryEntry("impact factor", "facteur d'impact", "general"),
        GlossaryEntry("citation count", "nombre de citations", "general"),
        GlossaryEntry("h-index", "indice h", "general"),
    ]
    
    # Deduplicate entries by source term
    seen = set()
    unique_entries = []
    for entry in entries:
        key = entry.source.lower()
        if key not in seen:
            seen.add(key)
            unique_entries.append(entry)
    
    return Glossary(
        entries=unique_entries,
        name="default_en_fr",
        source_lang="en",
        target_lang="fr",
    )


def merge_glossaries(
    *glossaries: Glossary,
    include_default: bool = True,
) -> Glossary:
    """Merge multiple glossaries into one.
    
    This is useful for combining user-provided glossaries with
    the default built-in glossary.
    
    Args:
        *glossaries: Glossaries to merge
        include_default: Whether to include the default glossary
        
    Returns:
        Merged Glossary with all entries
    """
    from pathlib import Path
    
    all_entries = []
    seen_sources = set()
    
    # Start with provided glossaries
    for gloss in glossaries:
        if gloss:
            for entry in gloss.entries:
                if entry.source.lower() not in seen_sources:
                    all_entries.append(entry)
                    seen_sources.add(entry.source.lower())
    
    # Try to load user glossaries from config directory
    try:
        from scitrans_llms.config import GLOSSARY_DIR
        if GLOSSARY_DIR.exists():
            for csv_file in GLOSSARY_DIR.glob("*.csv"):
                try:
                    user_gloss = load_glossary_csv(csv_file)
                    for entry in user_gloss.entries:
                        if entry.source.lower() not in seen_sources:
                            all_entries.append(entry)
                            seen_sources.add(entry.source.lower())
                except Exception:
                    pass  # Skip invalid files
    except ImportError:
        pass  # config module not available
    
    # Add default glossary last (user entries take precedence)
    if include_default:
        default = get_default_glossary()
        for entry in default.entries:
            if entry.source.lower() not in seen_sources:
                all_entries.append(entry)
                seen_sources.add(entry.source.lower())
    
    return Glossary(
        entries=all_entries,
        name="merged",
        source_lang="en",
        target_lang="fr",
    )

