"""
Evaluation metrics for translation quality assessment.

This module implements:
- Standard MT metrics (BLEU, chrF++)
- Domain-specific metrics (glossary adherence, numeric preservation)
- Layout fidelity metrics (for PDF evaluation)

Thesis Contribution #3: These metrics enable systematic evaluation
and ablation studies.

Design Philosophy:
- Each metric returns structured results
- Metrics can be computed at document or segment level
- Easy integration with evaluation scripts
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from scitrans_next.translate.glossary import Glossary, check_glossary_adherence


@dataclass
class EvaluationResult:
    """Complete evaluation results for a translation.
    
    Attributes:
        bleu: BLEU score (0-100)
        chrf: chrF++ score (0-100)
        glossary_adherence: Percentage of glossary terms correctly translated
        numeric_consistency: Percentage of numbers preserved
        details: Additional metric-specific details
    """
    bleu: Optional[float] = None
    chrf: Optional[float] = None
    glossary_adherence: Optional[float] = None
    numeric_consistency: Optional[float] = None
    details: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "bleu": self.bleu,
            "chrf": self.chrf,
            "glossary_adherence": self.glossary_adherence,
            "numeric_consistency": self.numeric_consistency,
            "details": self.details,
        }
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = ["Evaluation Results:"]
        if self.bleu is not None:
            lines.append(f"  BLEU: {self.bleu:.2f}")
        if self.chrf is not None:
            lines.append(f"  chrF++: {self.chrf:.2f}")
        if self.glossary_adherence is not None:
            lines.append(f"  Glossary Adherence: {self.glossary_adherence:.1%}")
        if self.numeric_consistency is not None:
            lines.append(f"  Numeric Consistency: {self.numeric_consistency:.1%}")
        return "\n".join(lines)


def compute_bleu(
    hypothesis: str | list[str],
    reference: str | list[str],
) -> float:
    """Compute BLEU score using SacreBLEU.
    
    Args:
        hypothesis: System output (single string or list of segments)
        reference: Reference translation (single string or list)
        
    Returns:
        BLEU score (0-100 scale)
    """
    try:
        import sacrebleu
    except ImportError:
        raise ImportError("sacrebleu is required for BLEU computation. Install with: pip install sacrebleu")
    
    # Normalize to lists
    if isinstance(hypothesis, str):
        hypothesis = [hypothesis]
    if isinstance(reference, str):
        reference = [[reference]]
    else:
        reference = [[r] for r in reference]
    
    # Flatten references for sacrebleu format
    refs = [[r[0] for r in reference]]
    
    bleu = sacrebleu.corpus_bleu(hypothesis, refs)
    return bleu.score


def compute_chrf(
    hypothesis: str | list[str],
    reference: str | list[str],
) -> float:
    """Compute chrF++ score using SacreBLEU.
    
    chrF++ is often more robust than BLEU for morphologically
    rich languages like French.
    
    Args:
        hypothesis: System output
        reference: Reference translation
        
    Returns:
        chrF++ score (0-100 scale)
    """
    try:
        import sacrebleu
    except ImportError:
        raise ImportError("sacrebleu is required for chrF++ computation. Install with: pip install sacrebleu")
    
    # Normalize to lists
    if isinstance(hypothesis, str):
        hypothesis = [hypothesis]
    if isinstance(reference, str):
        reference = [[reference]]
    else:
        reference = [[r] for r in reference]
    
    refs = [[r[0] for r in reference]]
    
    chrf = sacrebleu.corpus_chrf(hypothesis, refs)
    return chrf.score


def compute_glossary_adherence(
    hypothesis: str,
    source: str,
    glossary: Glossary,
) -> dict:
    """Measure how well a translation adheres to glossary terms.
    
    Args:
        hypothesis: Translated text
        source: Original source text
        glossary: Glossary to check against
        
    Returns:
        Dictionary with adherence rate and details
    """
    return check_glossary_adherence(
        translated_text=hypothesis,
        source_text=source,
        glossary=glossary,
    )


def compute_numeric_consistency(
    hypothesis: str,
    source: str,
) -> dict:
    """Check that numbers are preserved between source and translation.
    
    Numbers (integers, decimals, percentages) should typically
    appear unchanged in translations.
    
    Args:
        hypothesis: Translated text
        source: Original source text
        
    Returns:
        Dictionary with consistency rate and details
    """
    # Extract numbers from source
    number_pattern = re.compile(r'\b\d+(?:[.,]\d+)?%?\b')
    
    source_numbers = set(number_pattern.findall(source))
    hyp_numbers = set(number_pattern.findall(hypothesis))
    
    # Numbers that should appear in translation
    expected = source_numbers
    found = source_numbers & hyp_numbers
    missing = source_numbers - hyp_numbers
    
    total = len(expected)
    consistency = len(found) / total if total > 0 else 1.0
    
    return {
        "consistency_rate": consistency,
        "total_numbers": total,
        "found": list(found),
        "missing": list(missing),
    }


def compute_placeholder_preservation(
    hypothesis: str,
    source_masked: str,
) -> dict:
    """Check that all placeholders from masked source appear in translation.
    
    This validates that protected content (formulas, code, etc.)
    was correctly preserved through translation.
    
    Args:
        hypothesis: Translated text
        source_masked: Source text with placeholders
        
    Returns:
        Dictionary with preservation rate and details
    """
    placeholder_pattern = re.compile(r'<<[A-Z]+_\d{3}>>')
    
    source_placeholders = set(placeholder_pattern.findall(source_masked))
    hyp_placeholders = set(placeholder_pattern.findall(hypothesis))
    
    expected = source_placeholders
    found = source_placeholders & hyp_placeholders
    missing = source_placeholders - hyp_placeholders
    extra = hyp_placeholders - source_placeholders
    
    total = len(expected)
    preservation = len(found) / total if total > 0 else 1.0
    
    return {
        "preservation_rate": preservation,
        "total_placeholders": total,
        "found": list(found),
        "missing": list(missing),
        "extra": list(extra),
    }


def evaluate_translation(
    hypothesis: str,
    reference: str,
    source: str,
    glossary: Optional[Glossary] = None,
    source_masked: Optional[str] = None,
) -> EvaluationResult:
    """Compute all evaluation metrics for a translation.
    
    Args:
        hypothesis: System output
        reference: Reference translation
        source: Original source text
        glossary: Optional glossary for adherence checking
        source_masked: Optional masked source for placeholder checking
        
    Returns:
        EvaluationResult with all metrics
    """
    result = EvaluationResult()
    details = {}
    
    # Standard MT metrics
    try:
        result.bleu = compute_bleu(hypothesis, reference)
        result.chrf = compute_chrf(hypothesis, reference)
    except ImportError:
        details["mt_metrics_error"] = "sacrebleu not installed"
    
    # Glossary adherence
    if glossary:
        gloss_result = compute_glossary_adherence(hypothesis, source, glossary)
        result.glossary_adherence = gloss_result["adherence_rate"]
        details["glossary"] = gloss_result
    
    # Numeric consistency
    num_result = compute_numeric_consistency(hypothesis, source)
    result.numeric_consistency = num_result["consistency_rate"]
    details["numeric"] = num_result
    
    # Placeholder preservation
    if source_masked:
        placeholder_result = compute_placeholder_preservation(hypothesis, source_masked)
        details["placeholders"] = placeholder_result
    
    result.details = details
    return result

