"""
Evaluation runner for systematic translation assessment.

This module provides:
- Batch evaluation of translation outputs
- Multi-metric scoring (BLEU, chrF++, COMET, custom)
- Result aggregation and reporting
- Export to various formats (JSON, CSV, LaTeX)

Thesis Contribution #3: Research-grade evaluation infrastructure.
"""

from __future__ import annotations

import json
import csv
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Iterator
from datetime import datetime

from scitrans_llms.eval.metrics import (
    compute_bleu,
    compute_chrf,
    compute_glossary_adherence,
    compute_numeric_consistency,
    compute_placeholder_preservation,
    EvaluationResult,
)
from scitrans_llms.translate.glossary import Glossary


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""
    compute_bleu: bool = True
    compute_chrf: bool = True
    compute_comet: bool = False  # Requires unbabel-comet
    compute_glossary: bool = True
    compute_numeric: bool = True
    compute_placeholder: bool = True
    
    # Reference settings
    reference_path: Optional[Path] = None
    source_path: Optional[Path] = None
    
    # Output settings
    output_dir: Optional[Path] = None
    output_format: str = "json"  # json, csv, latex


@dataclass
class SegmentResult:
    """Evaluation result for a single segment."""
    segment_id: str
    source: str
    hypothesis: str
    reference: str
    bleu: Optional[float] = None
    chrf: Optional[float] = None
    comet: Optional[float] = None
    glossary_adherence: Optional[float] = None
    numeric_consistency: Optional[float] = None
    placeholder_preservation: Optional[float] = None


@dataclass
class EvaluationReport:
    """Complete evaluation report for a translation run."""
    run_id: str
    timestamp: str
    config: dict
    
    # Aggregate scores
    bleu: Optional[float] = None
    chrf: Optional[float] = None
    comet: Optional[float] = None
    glossary_adherence: Optional[float] = None
    numeric_consistency: Optional[float] = None
    placeholder_preservation: Optional[float] = None
    
    # Per-segment results
    segment_results: list[SegmentResult] = field(default_factory=list)
    
    # Metadata
    num_segments: int = 0
    source_file: Optional[str] = None
    hypothesis_file: Optional[str] = None
    reference_file: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "config": self.config,
            "scores": {
                "bleu": self.bleu,
                "chrf": self.chrf,
                "comet": self.comet,
                "glossary_adherence": self.glossary_adherence,
                "numeric_consistency": self.numeric_consistency,
                "placeholder_preservation": self.placeholder_preservation,
            },
            "num_segments": self.num_segments,
            "source_file": self.source_file,
            "hypothesis_file": self.hypothesis_file,
            "reference_file": self.reference_file,
            "segment_results": [asdict(s) for s in self.segment_results],
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def to_latex_table(self) -> str:
        """Generate LaTeX table of results."""
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\begin{tabular}{lc}",
            r"\toprule",
            r"Metric & Score \\",
            r"\midrule",
        ]
        
        if self.bleu is not None:
            lines.append(f"BLEU & {self.bleu:.2f} \\\\")
        if self.chrf is not None:
            lines.append(f"chrF++ & {self.chrf:.2f} \\\\")
        if self.comet is not None:
            lines.append(f"COMET & {self.comet:.4f} \\\\")
        if self.glossary_adherence is not None:
            lines.append(f"Glossary Adherence & {self.glossary_adherence:.1%} \\\\")
        if self.numeric_consistency is not None:
            lines.append(f"Numeric Consistency & {self.numeric_consistency:.1%} \\\\")
        
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            f"\\caption{{Translation Evaluation Results ({self.run_id})}}",
            r"\end{table}",
        ])
        
        return "\n".join(lines)
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Evaluation Report: {self.run_id}",
            f"Timestamp: {self.timestamp}",
            f"Segments: {self.num_segments}",
            "",
            "Scores:",
        ]
        
        if self.bleu is not None:
            lines.append(f"  BLEU:       {self.bleu:.2f}")
        if self.chrf is not None:
            lines.append(f"  chrF++:     {self.chrf:.2f}")
        if self.comet is not None:
            lines.append(f"  COMET:      {self.comet:.4f}")
        if self.glossary_adherence is not None:
            lines.append(f"  Glossary:   {self.glossary_adherence:.1%}")
        if self.numeric_consistency is not None:
            lines.append(f"  Numeric:    {self.numeric_consistency:.1%}")
        if self.placeholder_preservation is not None:
            lines.append(f"  Placeholder: {self.placeholder_preservation:.1%}")
        
        return "\n".join(lines)


class EvaluationRunner:
    """Run evaluations on translation outputs.
    
    Usage:
        runner = EvaluationRunner(config)
        report = runner.evaluate(
            hypotheses=["translation1", "translation2"],
            references=["reference1", "reference2"],
            sources=["source1", "source2"],
        )
        print(report.summary())
    """
    
    def __init__(
        self,
        config: Optional[EvaluationConfig] = None,
        glossary: Optional[Glossary] = None,
    ):
        self.config = config or EvaluationConfig()
        self.glossary = glossary
        self._comet_model = None
    
    def evaluate(
        self,
        hypotheses: list[str],
        references: list[str],
        sources: Optional[list[str]] = None,
        sources_masked: Optional[list[str]] = None,
        run_id: Optional[str] = None,
    ) -> EvaluationReport:
        """Run full evaluation on translation outputs.
        
        Args:
            hypotheses: System translations
            references: Reference translations
            sources: Original source texts (for glossary/numeric checks)
            sources_masked: Masked source texts (for placeholder checks)
            run_id: Optional identifier for this run
            
        Returns:
            EvaluationReport with all scores
        """
        if len(hypotheses) != len(references):
            raise ValueError("hypotheses and references must have same length")
        
        if sources and len(sources) != len(hypotheses):
            raise ValueError("sources must have same length as hypotheses")
        
        run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report = EvaluationReport(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            config=asdict(self.config) if hasattr(self.config, '__dataclass_fields__') else {},
            num_segments=len(hypotheses),
        )
        
        # Corpus-level metrics
        if self.config.compute_bleu:
            try:
                report.bleu = compute_bleu(hypotheses, references)
            except Exception:
                pass
        
        if self.config.compute_chrf:
            try:
                report.chrf = compute_chrf(hypotheses, references)
            except Exception:
                pass
        
        if self.config.compute_comet:
            try:
                report.comet = self._compute_comet(hypotheses, references, sources)
            except Exception:
                pass
        
        # Segment-level and aggregate custom metrics
        glossary_scores = []
        numeric_scores = []
        placeholder_scores = []
        
        for i, (hyp, ref) in enumerate(zip(hypotheses, references)):
            seg_result = SegmentResult(
                segment_id=str(i),
                source=sources[i] if sources else "",
                hypothesis=hyp,
                reference=ref,
            )
            
            if sources and self.config.compute_glossary and self.glossary:
                result = compute_glossary_adherence(hyp, sources[i], self.glossary)
                seg_result.glossary_adherence = result["adherence_rate"]
                glossary_scores.append(result["adherence_rate"])
            
            if sources and self.config.compute_numeric:
                result = compute_numeric_consistency(hyp, sources[i])
                seg_result.numeric_consistency = result["consistency_rate"]
                numeric_scores.append(result["consistency_rate"])
            
            if sources_masked and self.config.compute_placeholder:
                result = compute_placeholder_preservation(hyp, sources_masked[i])
                seg_result.placeholder_preservation = result["preservation_rate"]
                placeholder_scores.append(result["preservation_rate"])
            
            report.segment_results.append(seg_result)
        
        # Aggregate custom metrics
        if glossary_scores:
            report.glossary_adherence = sum(glossary_scores) / len(glossary_scores)
        if numeric_scores:
            report.numeric_consistency = sum(numeric_scores) / len(numeric_scores)
        if placeholder_scores:
            report.placeholder_preservation = sum(placeholder_scores) / len(placeholder_scores)
        
        return report
    
    def _compute_comet(
        self,
        hypotheses: list[str],
        references: list[str],
        sources: Optional[list[str]],
    ) -> float:
        """Compute COMET score if available."""
        if not sources:
            return None
        
        try:
            from comet import download_model, load_from_checkpoint
        except ImportError:
            return None
        
        if self._comet_model is None:
            model_path = download_model("Unbabel/wmt22-comet-da")
            self._comet_model = load_from_checkpoint(model_path)
        
        data = [
            {"src": src, "mt": hyp, "ref": ref}
            for src, hyp, ref in zip(sources, hypotheses, references)
        ]
        
        output = self._comet_model.predict(data, batch_size=8, gpus=0)
        return output.system_score
    
    def evaluate_files(
        self,
        hypothesis_file: Path,
        reference_file: Path,
        source_file: Optional[Path] = None,
        run_id: Optional[str] = None,
    ) -> EvaluationReport:
        """Evaluate from files (one segment per line)."""
        hypotheses = Path(hypothesis_file).read_text().strip().split("\n")
        references = Path(reference_file).read_text().strip().split("\n")
        
        sources = None
        if source_file:
            sources = Path(source_file).read_text().strip().split("\n")
        
        report = self.evaluate(hypotheses, references, sources, run_id=run_id)
        report.hypothesis_file = str(hypothesis_file)
        report.reference_file = str(reference_file)
        report.source_file = str(source_file) if source_file else None
        
        return report
    
    def save_report(self, report: EvaluationReport, output_path: Path):
        """Save evaluation report to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == ".json":
            output_path.write_text(report.to_json())
        elif output_path.suffix == ".tex":
            output_path.write_text(report.to_latex_table())
        elif output_path.suffix == ".csv":
            self._save_csv(report, output_path)
        else:
            output_path.write_text(report.summary())
    
    def _save_csv(self, report: EvaluationReport, path: Path):
        """Save segment results to CSV."""
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "segment_id", "bleu", "chrf", "glossary", "numeric", "placeholder"
            ])
            for seg in report.segment_results:
                writer.writerow([
                    seg.segment_id,
                    seg.bleu or "",
                    seg.chrf or "",
                    seg.glossary_adherence or "",
                    seg.numeric_consistency or "",
                    seg.placeholder_preservation or "",
                ])


def run_evaluation(
    hypotheses: list[str],
    references: list[str],
    sources: Optional[list[str]] = None,
    glossary: Optional[Glossary] = None,
) -> EvaluationReport:
    """Convenience function for quick evaluation."""
    runner = EvaluationRunner(glossary=glossary)
    return runner.evaluate(hypotheses, references, sources)

