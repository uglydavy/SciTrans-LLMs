from __future__ import annotations
import re

def fix_spacing(text: str) -> str:
    # Single space after sentence-ending punctuation
    text = re.sub(r"([.!?])\s{2,}", r"\1 ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s{3,}", "  ", text)
    return text

def normalize(text: str) -> str:
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = fix_spacing(t)
    return t.strip()