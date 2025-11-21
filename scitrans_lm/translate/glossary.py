
from __future__ import annotations
import csv
import re
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import requests

from ..config import DEFAULT_GLOSSARY, GLOSSARY_DIR

REMOTE_GLOSSARY_URL = "https://ftp.freedict.org/pub/FreeDict/wordlists/eng-fra.txt"

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

def _parse_rows_from_csv(path: Path) -> Iterable[Tuple[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            s = (row.get("source") or "").strip()
            t = (row.get("target") or "").strip()
            if s and t:
                yield s, t


def _parse_rows_from_txt(path: Path) -> Iterable[Tuple[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            if "," in line:
                parts = line.split(",", 1)
            elif "\t" in line:
                parts = line.split("\t", 1)
            else:
                continue
            s, t = (parts[0] or "").strip(), (parts[1] or "").strip()
            if s and t:
                yield s, t


def load_user_glossaries() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for p in GLOSSARY_DIR.glob("*"):
        if p.name == DEFAULT_GLOSSARY.name:
            continue
        rows: Iterable[Tuple[str, str]] = []
        if p.suffix.lower() == ".csv":
            rows = _parse_rows_from_csv(p)
        elif p.suffix.lower() in {".txt", ".tsv"}:
            rows = _parse_rows_from_txt(p)
        for s, t in rows:
            mapping[s.lower()] = t
    return mapping

def merge_glossaries() -> Dict[str, str]:
    d = load_default_glossary()
    u = load_user_glossaries()
    d.update(u)
    return d


def download_remote_glossary(url: str | None = None, dest: Path | None = None, limit: int = 5000) -> Path | None:
    """Download a bilingual wordlist and convert to CSV for offline use.

    The default source is the FreeDict English-French wordlist which ships as
    simple tab-separated lines: ``english<TAB>french``. Any endpoint returning
    ``source,target`` rows will also work. The glossary is written into the
    configured glossary directory and picked up automatically by
    :func:`merge_glossaries`.
    """

    dest = dest or (GLOSSARY_DIR / "remote_en_fr.csv")
    url = url or REMOTE_GLOSSARY_URL
    if dest.exists():
        # Reuse files updated within the last 7 days to avoid hammering servers
        if time.time() - dest.stat().st_mtime < 7 * 24 * 3600:
            return dest
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except Exception as exc:  # pragma: no cover - network specific
        warnings.warn(
            f"Unable to download remote glossary from {url}: {exc}", RuntimeWarning
        )
        return None

    lines = resp.text.splitlines()
    rows: List[Tuple[str, str]] = []
    for line in lines:
        if len(rows) >= limit:
            break
        if not line.strip() or line.startswith("#"):
            continue
        if "," in line:
            parts = line.split(",", 1)
        elif "\t" in line:
            parts = line.split("\t", 1)
        else:
            continue
        s, t = (parts[0] or "").strip(), (parts[1] or "").strip()
        if s and t:
            rows.append((s, t))

    if not rows:
        warnings.warn(
            f"Remote glossary at {url} was empty or unreadable.", RuntimeWarning
        )
        return None

    with dest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target"])
        for s, t in rows:
            writer.writerow([s, t])
    return dest

def inject_prompt_instructions(mapping: Dict[str,str], src: str, tgt: str) -> str:
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
    # Replace longer terms first to avoid partial overlaps
    terms = sorted(mapping.keys(), key=len, reverse=True)
    for term in terms:
        pattern = re.compile(re.escape(term), flags=re.IGNORECASE)
        text = pattern.sub(mapping[term], text)
    return text
