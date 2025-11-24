from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .prompting import evaluate_translation


def _glossary_hits(text: str, glossary: Dict[str, str]) -> int:
    if not glossary:
        return 0
    hits = 0
    lowered = text.lower()
    for src in glossary:
        if src in lowered:
            hits += 1
    return hits


@dataclass
class Candidate:
    text: str
    score: float
    detail: Dict[str, float]


def rerank_candidates(source: str, candidates: Iterable[str], glossary: Dict[str, str]) -> Candidate:
    best = Candidate(text=source, score=-1.0, detail={})
    for cand in candidates:
        eval_res = evaluate_translation(source, cand)
        glossary_hit = _glossary_hits(cand or "", glossary)
        score = (eval_res.changed_ratio * 0.5) + (eval_res.french_signals * 0.3) + (glossary_hit * 0.2)
        if eval_res.empty:
            score -= 1.0
        if score > best.score:
            best = Candidate(
                text=cand or source,
                score=score,
                detail={
                    "changed_ratio": eval_res.changed_ratio,
                    "french_signals": eval_res.french_signals,
                    "glossary_hit": glossary_hit,
                },
            )
    return best
