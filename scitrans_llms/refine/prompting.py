from __future__ import annotations

"""Prompt helpers and lightweight self-evaluation for translations."""

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..translate.memory import TranslationMemory

FRENCH_SIGNAL_WORDS = {
    "le",
    "la",
    "les",
    "des",
    "de",
    "et",
    "pour",
    "avec",
    "sur",
    "dans",
    "par",
    "selon",
    "ainsi",
}

# Language-specific guidelines for more accurate translations
LANGUAGE_GUIDELINES = {
    "fr": {
        "name": "French",
        "articles": "Use appropriate articles (le, la, les, un, une, des)",
        "formal": "Use formal/academic register (nous/on instead of je)",
        "numbers": "Use French number formatting (1,5 instead of 1.5 for decimals, spaces for thousands: 1 000)",
        "punctuation": "Use French punctuation rules (space before : ; ? !)",
        "terms": {
            "machine learning": "apprentissage automatique",
            "deep learning": "apprentissage profond",
            "state of the art": "état de l'art",
            "neural network": "réseau de neurones",
            "dataset": "jeu de données",
            "benchmark": "banc d'essai / référence",
        }
    },
    "en": {
        "name": "English",
        "articles": "Use appropriate articles (the, a, an)",
        "formal": "Use academic register with passive voice where appropriate",
        "numbers": "Use English number formatting (1.5 for decimals, commas for thousands: 1,000)",
        "punctuation": "Standard English punctuation (no space before : ; ? !)",
        "terms": {}
    },
    "de": {
        "name": "German",
        "articles": "Use appropriate articles with correct gender (der, die, das)",
        "formal": "Use formal academic register",
        "numbers": "Use German number formatting (1,5 for decimals, points for thousands: 1.000)",
        "punctuation": "Standard German punctuation",
        "terms": {}
    },
}

# Precise domain-specific instructions
DOMAIN_PROMPTS = {
    "scientific": """
CRITICAL RULES for scientific translation:
1. PRESERVE all technical terms exactly as they appear in the glossary
2. NEVER translate: variable names, function names, code, URLs, DOIs, mathematical notation
3. PRESERVE all placeholders like <<MATH_001>>, <<CODE_002>>, <<SECNUM_003>> EXACTLY
4. MAINTAIN the original paragraph structure and sentence boundaries
5. USE consistent terminology throughout - if you translate a term one way, always translate it the same way
6. KEEP numerical values and units unchanged (e.g., "5.3 MHz" stays as "5.3 MHz")
7. TRANSLATE figure/table captions but preserve the numbering (e.g., "Figure 1" → "Figure 1" in French)
""",
    "legal": """
CRITICAL RULES for legal translation:
1. PRESERVE exact legal terminology from the glossary
2. MAINTAIN formal register throughout
3. DO NOT paraphrase legal definitions - translate literally when precision is required
4. KEEP article and section numbers unchanged
""",
    "medical": """
CRITICAL RULES for medical translation:
1. PRESERVE all medical/pharmaceutical terms from the glossary
2. KEEP drug names, dosages, and measurements unchanged
3. MAINTAIN clinical precision - no approximations
4. USE standard medical terminology for the target language
""",
}


def build_prompt(
    src_lang: str,
    tgt_lang: str,
    glossary: Dict[str, str],
    memory: Optional[TranslationMemory] = None,
    domain: str = "scientific",
    include_examples: bool = True,
) -> str:
    """Build a precise translation prompt with domain-specific instructions.
    
    Args:
        src_lang: Source language code (e.g., 'en')
        tgt_lang: Target language code (e.g., 'fr')
        glossary: Dictionary of source -> target term mappings
        memory: Optional translation memory for context
        domain: Translation domain ('scientific', 'legal', 'medical')
        include_examples: Whether to include few-shot examples
        
    Returns:
        Complete prompt string for the LLM
    """
    src_info = LANGUAGE_GUIDELINES.get(src_lang, {"name": src_lang})
    tgt_info = LANGUAGE_GUIDELINES.get(tgt_lang, {"name": tgt_lang})
    
    base = [
        f"You are an expert {domain} translator specializing in {src_info.get('name', src_lang)} to {tgt_info.get('name', tgt_lang)} translation.",
        "",
        "YOUR TASK: Translate the following text accurately while preserving technical precision.",
        "",
    ]
    
    # Add domain-specific rules
    domain_rules = DOMAIN_PROMPTS.get(domain, DOMAIN_PROMPTS["scientific"])
    base.append(domain_rules)
    
    # Add target language guidelines
    if tgt_lang in LANGUAGE_GUIDELINES:
        tgt_guidelines = LANGUAGE_GUIDELINES[tgt_lang]
        base.extend([
            "",
            f"TARGET LANGUAGE RULES ({tgt_guidelines.get('name', tgt_lang)}):",
            f"- {tgt_guidelines.get('articles', 'Use appropriate articles')}",
            f"- {tgt_guidelines.get('formal', 'Use formal register')}",
            f"- {tgt_guidelines.get('numbers', 'Use standard number formatting')}",
            f"- {tgt_guidelines.get('punctuation', 'Use standard punctuation')}",
        ])
    
    # Add memory context if available
    if memory:
        context = memory.contextual_prompt()
        if context:
            base.extend(["", "PREVIOUS TRANSLATIONS (maintain consistency):", context])
    
    # Add glossary with emphasis
    if glossary:
        entries = list(glossary.items())[:100]  # Increased limit
        formatted = "\n".join(f"  • {k} → {v}" for k, v in entries)
        base.extend([
            "",
            "MANDATORY GLOSSARY (use these translations EXACTLY):",
            formatted,
        ])
    
    # Add few-shot examples for better quality
    if include_examples and src_lang == "en" and tgt_lang == "fr":
        base.extend([
            "",
            "EXAMPLES of correct translations:",
            "  IN:  'The model achieves state-of-the-art performance on the benchmark.'",
            "  OUT: 'Le modèle atteint des performances à l'état de l'art sur le banc d'essai.'",
            "",
            "  IN:  'We use a learning rate of $\\alpha = 0.001$ with batch size 32.'",
            "  OUT: 'Nous utilisons un taux d'apprentissage de $\\alpha = 0.001$ avec une taille de lot de 32.'",
            "",
            "  IN:  '1.1 Background and related work'",
            "  OUT: '1.1 Contexte et travaux connexes'",
        ])
    
    base.extend([
        "",
        "Now translate the following text:",
        "",
    ])
    
    return "\n".join(base)


def build_precise_prompt(
    text: str,
    src_lang: str,
    tgt_lang: str,
    glossary: Dict[str, str],
    context_before: List[str] = None,
    context_after: List[str] = None,
) -> str:
    """Build a highly precise prompt for a single text segment.
    
    This is used for critical translations where maximum accuracy is needed.
    """
    parts = [
        f"Translate this {src_lang.upper()} text to {tgt_lang.upper()}.",
        "",
    ]
    
    # Add surrounding context
    if context_before:
        parts.append("CONTEXT BEFORE (already translated):")
        for ctx in context_before[-3:]:  # Last 3 sentences
            parts.append(f"  {ctx}")
        parts.append("")
    
    # Add glossary requirements
    if glossary:
        # Find relevant terms in the text
        relevant = [(k, v) for k, v in glossary.items() if k.lower() in text.lower()]
        if relevant:
            parts.append("REQUIRED TERMINOLOGY (must use these exact translations):")
            for src, tgt in relevant:
                parts.append(f"  '{src}' → '{tgt}'")
            parts.append("")
    
    parts.extend([
        "TEXT TO TRANSLATE:",
        f">>> {text}",
        "",
        "TRANSLATION (preserve all placeholders like <<MATH_001>>, maintain structure):",
    ])
    
    return "\n".join(parts)


@dataclass
class TranslationEvaluation:
    changed_ratio: float
    french_signals: int
    empty: bool

    @property
    def acceptable(self) -> bool:
        return not self.empty and (self.french_signals >= 2 or self.changed_ratio >= 0.25)


def evaluate_translation(source: str, translated: str) -> TranslationEvaluation:
    if translated is None:
        return TranslationEvaluation(changed_ratio=0.0, french_signals=0, empty=True)

    src_tokens = [t for t in source.split() if t.strip()]
    tgt_tokens = [t for t in translated.split() if t.strip()]
    empty = len(tgt_tokens) == 0
    overlap = 0
    for s, t in zip(src_tokens, tgt_tokens):
        if s.lower() == t.lower():
            overlap += 1
    changed_ratio = 1.0 - (overlap / max(len(src_tokens), 1))
    french_signals = sum(1 for t in tgt_tokens if t.lower().strip(",.;:()[]") in FRENCH_SIGNAL_WORDS)
    return TranslationEvaluation(changed_ratio=changed_ratio, french_signals=french_signals, empty=empty)


def refine_prompt(base_prompt: str, evaluation: TranslationEvaluation, iteration: int) -> str:
    reinforcements: List[str] = []
    if evaluation.empty:
        reinforcements.append("Your previous output was empty. Provide a complete translation for every sentence.")
    if evaluation.french_signals < 2:
        reinforcements.append("Use fluent French phrasing; avoid copying English words directly.")
    if evaluation.changed_ratio < 0.25:
        reinforcements.append("Rephrase the text fully in the target language, preserving meaning and placeholders.")
    reinforcements.append(f"This is refinement attempt {iteration + 1} of 4.")
    return base_prompt + "\n" + "\n".join(reinforcements) + "\n"
