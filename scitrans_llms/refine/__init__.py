"""
Refinement module for post-translation improvement.

This module provides:
- Document-level refinement pass
- Coherence checking and improvement
- Glossary enforcement post-processing
- Candidate reranking
- LLM-based refinement

Thesis Contribution #2: Document-level LLM refinement
to improve coherence, pronouns, and terminology consistency.
"""

from scitrans_llms.refine.base import (
    Refiner,
    RefinementResult,
    NoOpRefiner,
    GlossaryRefiner,
    PlaceholderValidator,
    CompositeRefiner,
    create_refiner,
)

__all__ = [
    "Refiner",
    "RefinementResult",
    "NoOpRefiner",
    "GlossaryRefiner",
    "PlaceholderValidator",
    "CompositeRefiner",
    "create_refiner",
]

# Lazy imports for LLM refiners
def __getattr__(name):
    if name in ("LLMRefiner", "CoherenceRefiner", "StyleRefiner", "RefinementConfig"):
        from scitrans_llms.refine import llm
        return getattr(llm, name)
    if name in ("CandidateReranker", "RerankedTranslator", "CandidateScore", "RerankingResult"):
        from scitrans_llms.refine import rerank
        return getattr(rerank, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
