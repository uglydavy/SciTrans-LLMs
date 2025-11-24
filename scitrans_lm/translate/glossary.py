from __future__ import annotations
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

"""Glossary loading and enforcement helpers."""

import csv
import re
from pathlib import Path
from typing import Dict

from ..config import DEFAULT_GLOSSARY, GLOSSARY_DIR


def _load_docx_terms(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    try:
        import docx
    except Exception:
        return mapping
    try:
        doc = docx.Document(path)
    except Exception:
        return mapping
    for para in doc.paragraphs:
        if ":" in para.text:
            src, tgt = para.text.split(":", 1)
            src = src.strip()
            tgt = tgt.strip()
            if src and tgt:
                mapping[src.lower()] = tgt
    return mapping


def _load_txt_terms(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if "," in line:
            src, tgt = line.split(",", 1)
        elif ":" in line:
            src, tgt = line.split(":", 1)
        else:
            continue
        src = src.strip()
        tgt = tgt.strip()
        if src and tgt:
            mapping[src.lower()] = tgt
    return mapping


def load_default_glossary() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if DEFAULT_GLOSSARY.exists():
        with DEFAULT_GLOSSARY.open("r", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                s = (row.get("source") or "").strip()
                t = (row.get("target") or "").strip()
                if s and t:
                    mapping[s.lower()] = t
    return mapping


def load_user_glossaries() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for p in GLOSSARY_DIR.glob("*"):
        if p.name == DEFAULT_GLOSSARY.name:
            continue
        if p.suffix.lower() == ".csv":
            with p.open("r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    s = (row.get("source") or "").strip()
                    t = (row.get("target") or "").strip()
                    if s and t:
                        mapping[s.lower()] = t
        elif p.suffix.lower() in {".txt"}:
            mapping.update(_load_txt_terms(p))
        elif p.suffix.lower() in {".docx"}:
            mapping.update(_load_docx_terms(p))
    return mapping


def merge_glossaries() -> Dict[str, str]:
    d = load_default_glossary()
    u = load_user_glossaries()
    d.update(u)
    return d


def inject_prompt_instructions(mapping: Dict[str, str], src: str, tgt: str) -> str:
    prefix = f"Please translate from {src} to {tgt}. Preserve placeholder tokens like [[FORMULA_0001]]."
    if not mapping:
        return prefix + "\n"
    pairs = "\n".join([f"- '{k}' -> '{v}'" for k, v in list(mapping.items())[:100]])
    return f"{prefix}\nEnsure these terms are enforced as-is:\n{pairs}\n"


def enforce_post(text: str, mapping: Dict[str, str]) -> str:
    """Simple post-processing: replace exact term matches case-insensitively."""
    if not mapping:
        return text

    def repl(match):
        src = match.group(0)
        return mapping.get(src.lower(), src)

    terms = sorted(mapping.keys(), key=len, reverse=True)
    for term in terms:
        pattern = re.compile(re.escape(term), flags=re.IGNORECASE)
        text = pattern.sub(mapping[term], text)
    return text
