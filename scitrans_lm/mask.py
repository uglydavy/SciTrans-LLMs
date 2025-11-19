"""Utilities for masking and preserving non-translatable content."""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

PLACEHOLDER_TEMPLATE = "[[FORMULA_{:04d}]]"

MATH_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(r"\$\$(.+?)\$\$", re.DOTALL),
    re.compile(r"\$(.+?)\$", re.DOTALL),
    re.compile(r"\\\[(.+?)\\\]", re.DOTALL),
    re.compile(r"\\\((.+?)\\\)", re.DOTALL),
    re.compile(r"`([^`]+)`", re.DOTALL),
)


def mask_protected_segments(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Replace inline formulas/code with placeholders."""

    placeholders: List[Tuple[str, str]] = []

    def _repl(match: re.Match[str]) -> str:
        token = PLACEHOLDER_TEMPLATE.format(len(placeholders) + 1)
        placeholders.append((token, match.group(0)))
        return token

    masked = text
    for pattern in MATH_PATTERNS:
        masked = pattern.sub(_repl, masked)
    return masked, placeholders


def unmask(text: str, placeholders: Sequence[Tuple[str, str]]) -> str:
    """Restore placeholder tokens with their original content."""

    restored = text
    for token, original in placeholders:
        restored = restored.replace(token, original)
    return restored


def looks_like_formula_block(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 3:
        return False
    math_keywords = ("\\frac", "\\sum", "\\int", "\\begin{equation}", "\\alpha", "\\beta")
    if any(kw in stripped for kw in math_keywords):
        return True
    math_chars = set("=<>±×÷∑∏√^_{}[]")
    score = sum(1 for ch in stripped if ch in math_chars)
    return score / max(1, len(stripped)) > 0.08


def looks_like_numeric_table(text: str) -> bool:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    numeric_lines = 0
    for line in lines:
        digits = sum(ch.isdigit() for ch in line)
        if digits / max(1, len(line)) > 0.45:
            numeric_lines += 1
    return numeric_lines / len(lines) > 0.6

