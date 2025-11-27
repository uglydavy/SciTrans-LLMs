"""
Evaluation module for measuring translation quality.

This module provides:
- BLEU, chrF++ metrics via SacreBLEU
- COMET neural evaluation (when available)
- Glossary adherence metrics
- Numeric consistency checking
- Layout fidelity measures
- Ablation study framework
- Baseline comparison tools

Thesis Contribution #3: Research-grade evaluation for ablations.
"""

from scitrans_llms.eval.metrics import (
    compute_bleu,
    compute_chrf,
    compute_glossary_adherence,
    compute_numeric_consistency,
    compute_placeholder_preservation,
    evaluate_translation,
    EvaluationResult,
)

__all__ = [
    # Metrics
    "compute_bleu",
    "compute_chrf",
    "compute_glossary_adherence",
    "compute_numeric_consistency",
    "compute_placeholder_preservation",
    "evaluate_translation",
    "EvaluationResult",
]

# Lazy imports for runner and ablation
def __getattr__(name):
    if name in ("EvaluationRunner", "run_evaluation", "EvaluationConfig"):
        from scitrans_llms.eval import runner
        return getattr(runner, name)
    if name in ("AblationStudy", "AblationConfig", "run_ablation"):
        from scitrans_llms.eval import ablation
        return getattr(ablation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
