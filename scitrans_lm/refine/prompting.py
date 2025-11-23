from __future__ import annotations

"""Prompt helpers and lightweight self-evaluation for translations.

The goal of this module is to centralize prompt construction and simple
quality heuristics so all backends (LLM or dictionary) can benefit from
the same guidance. The heuristics are intentionally cheap: they avoid
network calls and focus on surface-level indicators to decide whether a
translation attempt should be retried.
"""

from dataclasses import dataclass
from typing import Dict, List


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


def build_prompt(src_lang: str, tgt_lang: str, glossary: Dict[str, str]) -> str:
    base = [
        f"You are a meticulous scientific translator. Translate from {src_lang} to {tgt_lang}.",
        "Respect placeholders such as [[FORMULA_0001]] and never alter them.",
        "Preserve sentence boundaries and keep the scientific tone concise.",
    ]
    if glossary:
        entries = list(glossary.items())[:80]
        formatted = "\n".join(f"- '{k}' -> '{v}'" for k, v in entries)
        base.append("Enforce the following glossary exactly:")
        base.append(formatted)
    return "\n".join(base) + "\n"


@dataclass
class TranslationEvaluation:
    changed_ratio: float
    french_signals: int
    empty: bool

    @property
    def acceptable(self) -> bool:
        # Accept if meaningful French hints or at least a quarter of tokens changed
        return not self.empty and (self.french_signals >= 2 or self.changed_ratio >= 0.25)


def evaluate_translation(source: str, translated: str) -> TranslationEvaluation:
    if translated is None:
        return TranslationEvaluation(changed_ratio=0.0, french_signals=0, empty=True)

    src_tokens = [t for t in source.split() if t.strip()]
    tgt_tokens = [t for t in translated.split() if t.strip()]
    empty = len(tgt_tokens) == 0
    # Changed ratio: proportion of tokens that differ ignoring case
    overlap = 0
    for s, t in zip(src_tokens, tgt_tokens):
        if s.lower() == t.lower():
            overlap += 1
    changed_ratio = 1.0 - (overlap / max(len(src_tokens), 1))
    french_signals = sum(1 for t in tgt_tokens if t.lower().strip(",.;:()[]") in FRENCH_SIGNAL_WORDS)
    return TranslationEvaluation(changed_ratio=changed_ratio, french_signals=french_signals, empty=empty)


def refine_prompt(base_prompt: str, evaluation: TranslationEvaluation, iteration: int) -> str:
    """Tighten the prompt based on observed issues.

    Each iteration makes the instructions more explicit. After the fourth
    attempt the caller should fall back to the last produced text.
    """

    reinforcements: List[str] = []
    if evaluation.empty:
        reinforcements.append("Your previous output was empty. Provide a complete translation for every sentence.")
    if evaluation.french_signals < 2:
        reinforcements.append("Use fluent French phrasing; avoid copying English words directly.")
    if evaluation.changed_ratio < 0.25:
        reinforcements.append("Rephrase the text fully in the target language, preserving meaning and placeholders.")
    reinforcements.append(f"This is refinement attempt {iteration + 1} of 4.")
    return base_prompt + "\n" + "\n".join(reinforcements) + "\n"

