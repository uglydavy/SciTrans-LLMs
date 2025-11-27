"""
Experiment runner for systematic translation evaluation.

Provides:
- Unified interface for running experiments
- Multiple configuration support
- Automatic result collection
- Progress tracking and logging
- Export to multiple formats

Usage:
    runner = ExperimentRunner(corpus, output_dir="results/")
    results = runner.run_all_experiments()
    runner.export_results()
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

from scitrans_next.models import Document
from scitrans_next.pipeline import TranslationPipeline, PipelineConfig
from scitrans_next.translate.glossary import Glossary, get_default_glossary
from scitrans_next.eval.runner import EvaluationRunner, EvaluationReport
from scitrans_next.eval.ablation import AblationStudy, AblationConfig
from scitrans_next.experiments.corpus import Corpus, CorpusDocument


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    name: str
    description: str = ""
    
    # Translation settings
    backend: str = "dummy"
    model: Optional[str] = None
    
    # Feature toggles
    enable_glossary: bool = True
    enable_context: bool = True
    enable_refinement: bool = True
    enable_masking: bool = True
    
    # Other settings
    refiner_mode: str = "default"
    num_candidates: int = 1
    
    def to_pipeline_config(self, glossary: Optional[Glossary] = None) -> PipelineConfig:
        """Convert to PipelineConfig."""
        kwargs = {}
        if self.model:
            kwargs["model"] = self.model
        
        return PipelineConfig(
            translator_backend=self.backend,
            translator_kwargs=kwargs,
            enable_glossary=self.enable_glossary,
            enable_context=self.enable_context,
            enable_refinement=self.enable_refinement,
            enable_masking=self.enable_masking,
            glossary=glossary if self.enable_glossary else None,
            refiner_mode=self.refiner_mode,
            num_candidates=self.num_candidates,
        )
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExperimentResult:
    """Result of a single experiment."""
    config: ExperimentConfig
    evaluation: EvaluationReport
    hypotheses: list[str]
    duration_seconds: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    errors: list[str] = field(default_factory=list)
    
    @property
    def success(self) -> bool:
        return len(self.errors) == 0
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Experiment: {self.config.name}",
            f"  Backend: {self.config.backend}",
            f"  Duration: {self.duration_seconds:.1f}s",
        ]
        if self.evaluation.bleu:
            lines.append(f"  BLEU: {self.evaluation.bleu:.2f}")
        if self.evaluation.chrf:
            lines.append(f"  chrF++: {self.evaluation.chrf:.2f}")
        if self.evaluation.glossary_adherence:
            lines.append(f"  Glossary: {self.evaluation.glossary_adherence:.1%}")
        return "\n".join(lines)


class ExperimentRunner:
    """Run systematic experiments on a corpus.
    
    Supports:
    - Single configuration experiments
    - Ablation studies
    - Baseline comparisons
    - Full experiment suites
    """
    
    def __init__(
        self,
        corpus: Corpus,
        output_dir: str | Path = "results",
        glossary: Optional[Glossary] = None,
    ):
        self.corpus = corpus
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.glossary = glossary or corpus.glossary or get_default_glossary()
        self.results: list[ExperimentResult] = []
    
    def run_experiment(
        self,
        config: ExperimentConfig,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> ExperimentResult:
        """Run a single experiment with the given configuration."""
        progress_callback = progress_callback or (lambda m, p: None)
        
        pipeline_config = config.to_pipeline_config(self.glossary)
        pipeline = TranslationPipeline(pipeline_config)
        
        all_hypotheses = []
        all_sources = []
        errors = []
        
        start_time = time.time()
        
        for i, doc in enumerate(self.corpus.documents):
            progress_callback(
                f"Translating {doc.doc_id}...",
                (i + 1) / len(self.corpus.documents)
            )
            
            try:
                # Create Document and translate
                trans_doc = doc.to_document()
                result = pipeline.translate(trans_doc)
                
                # Collect translations
                for block in trans_doc.all_blocks:
                    if block.is_translatable:
                        all_hypotheses.append(block.translated_text or "")
                        all_sources.append(block.source_text)
                
            except Exception as e:
                errors.append(f"{doc.doc_id}: {str(e)}")
        
        duration = time.time() - start_time
        
        # Evaluate
        progress_callback("Evaluating...", 1.0)
        
        eval_runner = EvaluationRunner(glossary=self.glossary)
        evaluation = eval_runner.evaluate(
            hypotheses=all_hypotheses,
            references=self.corpus.get_all_references()[:len(all_hypotheses)],
            sources=all_sources,
            run_id=config.name,
        )
        
        result = ExperimentResult(
            config=config,
            evaluation=evaluation,
            hypotheses=all_hypotheses,
            duration_seconds=duration,
            errors=errors,
        )
        
        self.results.append(result)
        return result
    
    def run_ablation(
        self,
        base_backend: str = "dummy",
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> list[ExperimentResult]:
        """Run ablation study testing each component."""
        configs = [
            ExperimentConfig(
                name="full",
                description="Full system with all features",
                backend=base_backend,
                enable_glossary=True,
                enable_context=True,
                enable_refinement=True,
                enable_masking=True,
            ),
            ExperimentConfig(
                name="no_glossary",
                description="Without glossary enforcement",
                backend=base_backend,
                enable_glossary=False,
                enable_context=True,
                enable_refinement=True,
                enable_masking=True,
            ),
            ExperimentConfig(
                name="no_context",
                description="Without document context",
                backend=base_backend,
                enable_glossary=True,
                enable_context=False,
                enable_refinement=True,
                enable_masking=True,
            ),
            ExperimentConfig(
                name="no_refinement",
                description="Without refinement pass",
                backend=base_backend,
                enable_glossary=True,
                enable_context=True,
                enable_refinement=False,
                enable_masking=True,
            ),
            ExperimentConfig(
                name="no_masking",
                description="Without placeholder masking",
                backend=base_backend,
                enable_glossary=True,
                enable_context=True,
                enable_refinement=True,
                enable_masking=False,
            ),
            ExperimentConfig(
                name="minimal",
                description="Minimal system (no features)",
                backend=base_backend,
                enable_glossary=False,
                enable_context=False,
                enable_refinement=False,
                enable_masking=False,
            ),
        ]
        
        results = []
        for i, config in enumerate(configs):
            if progress_callback:
                progress_callback(f"Running {config.name}...", i / len(configs))
            result = self.run_experiment(config)
            results.append(result)
        
        return results
    
    def run_backend_comparison(
        self,
        backends: list[str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> list[ExperimentResult]:
        """Compare different translation backends."""
        backends = backends or ["dummy", "dictionary"]
        
        results = []
        for i, backend in enumerate(backends):
            if progress_callback:
                progress_callback(f"Testing {backend}...", i / len(backends))
            
            config = ExperimentConfig(
                name=f"backend_{backend}",
                description=f"Using {backend} backend",
                backend=backend,
            )
            
            try:
                result = self.run_experiment(config)
                results.append(result)
            except Exception as e:
                print(f"Backend {backend} failed: {e}")
        
        return results
    
    def export_results(
        self,
        prefix: str = "experiment",
        formats: list[str] = None,
    ) -> dict[str, Path]:
        """Export results to various formats.
        
        Args:
            prefix: Filename prefix
            formats: List of formats ('json', 'csv', 'latex', 'summary')
            
        Returns:
            Dict mapping format to output file path
        """
        formats = formats or ["json", "latex", "summary"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outputs = {}
        
        # JSON export
        if "json" in formats:
            json_path = self.output_dir / f"{prefix}_{timestamp}.json"
            data = {
                "corpus": self.corpus.name,
                "timestamp": timestamp,
                "results": [
                    {
                        "config": r.config.to_dict(),
                        "evaluation": r.evaluation.to_dict(),
                        "duration_seconds": r.duration_seconds,
                        "errors": r.errors,
                    }
                    for r in self.results
                ],
            }
            json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            outputs["json"] = json_path
        
        # LaTeX table
        if "latex" in formats:
            latex_path = self.output_dir / f"{prefix}_{timestamp}.tex"
            latex_content = self._generate_latex_table()
            latex_path.write_text(latex_content)
            outputs["latex"] = latex_path
        
        # Summary text
        if "summary" in formats:
            summary_path = self.output_dir / f"{prefix}_{timestamp}.txt"
            summary = self._generate_summary()
            summary_path.write_text(summary)
            outputs["summary"] = summary_path
        
        return outputs
    
    def _generate_latex_table(self) -> str:
        """Generate LaTeX results table."""
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Translation Experiment Results}",
            r"\label{tab:experiments}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Configuration & BLEU & chrF++ & Gloss. Adh. & Time (s) \\",
            r"\midrule",
        ]
        
        for r in self.results:
            name = r.config.name.replace("_", r"\_")
            bleu = f"{r.evaluation.bleu:.2f}" if r.evaluation.bleu else "---"
            chrf = f"{r.evaluation.chrf:.2f}" if r.evaluation.chrf else "---"
            gloss = f"{r.evaluation.glossary_adherence:.1%}" if r.evaluation.glossary_adherence else "---"
            time_s = f"{r.duration_seconds:.1f}"
            lines.append(f"{name} & {bleu} & {chrf} & {gloss} & {time_s} \\\\")
        
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])
        
        return "\n".join(lines)
    
    def _generate_summary(self) -> str:
        """Generate text summary of results."""
        lines = [
            "=" * 60,
            "EXPERIMENT RESULTS SUMMARY",
            "=" * 60,
            f"Corpus: {self.corpus.name}",
            f"Documents: {len(self.corpus.documents)}",
            f"Segments: {self.corpus.total_segments}",
            "",
            "-" * 60,
            f"{'Configuration':<20} {'BLEU':>8} {'chrF++':>8} {'Gloss%':>8} {'Time':>8}",
            "-" * 60,
        ]
        
        for r in self.results:
            name = r.config.name[:20]
            bleu = f"{r.evaluation.bleu:.2f}" if r.evaluation.bleu else "N/A"
            chrf = f"{r.evaluation.chrf:.2f}" if r.evaluation.chrf else "N/A"
            gloss = f"{r.evaluation.glossary_adherence:.1%}" if r.evaluation.glossary_adherence else "N/A"
            time_s = f"{r.duration_seconds:.1f}s"
            lines.append(f"{name:<20} {bleu:>8} {chrf:>8} {gloss:>8} {time_s:>8}")
        
        lines.append("-" * 60)
        
        # Best results
        if self.results:
            best_bleu = max((r for r in self.results if r.evaluation.bleu), 
                           key=lambda r: r.evaluation.bleu, default=None)
            if best_bleu:
                lines.append(f"\nBest BLEU: {best_bleu.config.name} ({best_bleu.evaluation.bleu:.2f})")
        
        return "\n".join(lines)


def run_experiment(
    corpus: Corpus,
    config: ExperimentConfig,
    output_dir: str = "results",
) -> ExperimentResult:
    """Convenience function to run a single experiment."""
    runner = ExperimentRunner(corpus, output_dir)
    return runner.run_experiment(config)

