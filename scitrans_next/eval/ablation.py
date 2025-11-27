"""
Ablation study framework for systematic experiments.

This module provides:
- Configuration of ablation variables
- Systematic execution of translation runs
- Result collection and comparison
- Statistical analysis of differences

Thesis Contribution #3: Ablation studies to validate
each component's contribution.

Ablation variables:
- With/without glossary enforcement
- With/without document-level context
- With/without refinement pass
- With/without masking
- Different translator backends
"""

from __future__ import annotations

import json
import itertools
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Callable
from datetime import datetime

from scitrans_next.models import Document
from scitrans_next.pipeline import TranslationPipeline, PipelineConfig, PipelineResult
from scitrans_next.eval.runner import EvaluationRunner, EvaluationReport
from scitrans_next.translate.glossary import Glossary


@dataclass
class AblationVariable:
    """A single variable in an ablation study."""
    name: str
    values: list
    description: str = ""
    
    def __iter__(self):
        return iter(self.values)


@dataclass
class AblationConfig:
    """Configuration for an ablation study.
    
    Defines which variables to test and their values.
    """
    name: str
    description: str = ""
    
    # Core ablation toggles
    test_glossary: bool = True  # With vs without glossary
    test_context: bool = True   # With vs without document context
    test_refinement: bool = True  # With vs without refinement
    test_masking: bool = True   # With vs without masking
    
    # Additional variables
    backends: list[str] = field(default_factory=lambda: ["dummy"])
    
    # Evaluation settings
    num_runs: int = 1  # Repeat each config for variance
    
    def get_variables(self) -> list[AblationVariable]:
        """Get list of ablation variables to test."""
        variables = []
        
        if self.test_glossary:
            variables.append(AblationVariable(
                name="enable_glossary",
                values=[True, False],
                description="Glossary enforcement",
            ))
        
        if self.test_context:
            variables.append(AblationVariable(
                name="enable_context",
                values=[True, False],
                description="Document-level context",
            ))
        
        if self.test_refinement:
            variables.append(AblationVariable(
                name="enable_refinement",
                values=[True, False],
                description="Refinement pass",
            ))
        
        if self.test_masking:
            variables.append(AblationVariable(
                name="enable_masking",
                values=[True, False],
                description="Placeholder masking",
            ))
        
        if len(self.backends) > 1:
            variables.append(AblationVariable(
                name="translator_backend",
                values=self.backends,
                description="Translation backend",
            ))
        
        return variables
    
    def get_configurations(self) -> list[dict]:
        """Generate all configuration combinations."""
        variables = self.get_variables()
        
        if not variables:
            return [{}]
        
        # Get all combinations
        names = [v.name for v in variables]
        value_lists = [list(v.values) for v in variables]
        
        configs = []
        for values in itertools.product(*value_lists):
            config = dict(zip(names, values))
            configs.append(config)
        
        return configs


@dataclass
class AblationRun:
    """Result of a single ablation run."""
    config: dict
    config_name: str
    evaluation: EvaluationReport
    translation_stats: dict
    duration_seconds: float


@dataclass
class AblationStudy:
    """Complete ablation study with results.
    
    Usage:
        study = AblationStudy(config=AblationConfig(name="main"))
        study.run(documents, references, sources)
        print(study.summary())
        study.save("results/ablation.json")
    """
    config: AblationConfig
    runs: list[AblationRun] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def run(
        self,
        documents: list[Document],
        references: list[list[str]],
        sources: Optional[list[list[str]]] = None,
        glossary: Optional[Glossary] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ):
        """Execute the ablation study.
        
        Args:
            documents: List of documents to translate
            references: List of reference translations (per document, per block)
            sources: List of source texts (per document, per block)
            glossary: Glossary to use (when enabled)
            progress_callback: Optional progress updates
        """
        import time
        
        configurations = self.config.get_configurations()
        total_runs = len(configurations) * self.config.num_runs * len(documents)
        current_run = 0
        
        eval_runner = EvaluationRunner(glossary=glossary)
        
        for config_dict in configurations:
            config_name = self._config_to_name(config_dict)
            
            for run_num in range(self.config.num_runs):
                # Aggregate results across documents
                all_hypotheses = []
                all_references = []
                all_sources = []
                total_stats = {}
                
                start_time = time.time()
                
                for doc_idx, doc in enumerate(documents):
                    current_run += 1
                    if progress_callback:
                        progress_callback(
                            f"Running {config_name} (doc {doc_idx+1}/{len(documents)})",
                            current_run / total_runs,
                        )
                    
                    # Create pipeline config
                    pipeline_config = PipelineConfig(
                        translator_backend=config_dict.get("translator_backend", "dummy"),
                        enable_glossary=config_dict.get("enable_glossary", True),
                        enable_context=config_dict.get("enable_context", True),
                        enable_refinement=config_dict.get("enable_refinement", True),
                        enable_masking=config_dict.get("enable_masking", True),
                        glossary=glossary if config_dict.get("enable_glossary", True) else None,
                    )
                    
                    # Run translation
                    pipeline = TranslationPipeline(pipeline_config)
                    result = pipeline.translate(doc)
                    
                    # Collect outputs
                    hyps = [b.translated_text or "" for b in doc.all_blocks if b.is_translatable]
                    all_hypotheses.extend(hyps)
                    
                    if doc_idx < len(references):
                        all_references.extend(references[doc_idx])
                    
                    if sources and doc_idx < len(sources):
                        all_sources.extend(sources[doc_idx])
                    
                    # Aggregate stats
                    for key, value in result.stats.items():
                        total_stats[key] = total_stats.get(key, 0) + value
                
                duration = time.time() - start_time
                
                # Evaluate
                eval_report = eval_runner.evaluate(
                    hypotheses=all_hypotheses,
                    references=all_references[:len(all_hypotheses)],
                    sources=all_sources[:len(all_hypotheses)] if all_sources else None,
                    run_id=f"{config_name}_run{run_num}",
                )
                
                self.runs.append(AblationRun(
                    config=config_dict,
                    config_name=config_name,
                    evaluation=eval_report,
                    translation_stats=total_stats,
                    duration_seconds=duration,
                ))
    
    def _config_to_name(self, config: dict) -> str:
        """Create a readable name for a configuration."""
        parts = []
        for key, value in sorted(config.items()):
            if isinstance(value, bool):
                parts.append(f"{key.replace('enable_', '')}={'Y' if value else 'N'}")
            else:
                parts.append(f"{key}={value}")
        return "_".join(parts) if parts else "default"
    
    def get_results_table(self) -> list[dict]:
        """Get results as a table (list of dicts)."""
        rows = []
        for run in self.runs:
            row = {
                "config": run.config_name,
                **run.config,
                "bleu": run.evaluation.bleu,
                "chrf": run.evaluation.chrf,
                "glossary_adherence": run.evaluation.glossary_adherence,
                "numeric_consistency": run.evaluation.numeric_consistency,
                "duration_seconds": run.duration_seconds,
            }
            rows.append(row)
        return rows
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Ablation Study: {self.config.name}",
            f"Timestamp: {self.timestamp}",
            f"Total runs: {len(self.runs)}",
            "",
            "Results:",
            "-" * 60,
        ]
        
        # Header
        header = f"{'Configuration':<30} {'BLEU':>8} {'chrF++':>8} {'Gloss%':>8}"
        lines.append(header)
        lines.append("-" * 60)
        
        # Rows
        for run in self.runs:
            name = run.config_name[:30]
            bleu = f"{run.evaluation.bleu:.2f}" if run.evaluation.bleu else "N/A"
            chrf = f"{run.evaluation.chrf:.2f}" if run.evaluation.chrf else "N/A"
            gloss = f"{run.evaluation.glossary_adherence:.1%}" if run.evaluation.glossary_adherence else "N/A"
            lines.append(f"{name:<30} {bleu:>8} {chrf:>8} {gloss:>8}")
        
        return "\n".join(lines)
    
    def to_latex_table(self) -> str:
        """Generate LaTeX results table."""
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\begin{tabular}{l" + "c" * 4 + "}",
            r"\toprule",
            r"Configuration & BLEU & chrF++ & Gloss. & Time (s) \\",
            r"\midrule",
        ]
        
        for run in self.runs:
            name = run.config_name.replace("_", r"\_")
            bleu = f"{run.evaluation.bleu:.2f}" if run.evaluation.bleu else "---"
            chrf = f"{run.evaluation.chrf:.2f}" if run.evaluation.chrf else "---"
            gloss = f"{run.evaluation.glossary_adherence:.1%}" if run.evaluation.glossary_adherence else "---"
            time_s = f"{run.duration_seconds:.1f}"
            lines.append(f"{name} & {bleu} & {chrf} & {gloss} & {time_s} \\\\")
        
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            f"\\caption{{Ablation Study: {self.config.name}}}",
            r"\end{table}",
        ])
        
        return "\n".join(lines)
    
    def save(self, path: Path):
        """Save study results to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "config": asdict(self.config),
            "timestamp": self.timestamp,
            "runs": [
                {
                    "config": run.config,
                    "config_name": run.config_name,
                    "evaluation": run.evaluation.to_dict(),
                    "translation_stats": run.translation_stats,
                    "duration_seconds": run.duration_seconds,
                }
                for run in self.runs
            ],
        }
        
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    
    def analyze_component_impact(self) -> dict[str, dict]:
        """Analyze the impact of each component.
        
        Compares runs with/without each ablation variable
        to quantify each component's contribution.
        """
        impacts = {}
        
        for variable in self.config.get_variables():
            var_name = variable.name
            
            # Find runs with and without this feature
            with_feature = [r for r in self.runs if r.config.get(var_name, True)]
            without_feature = [r for r in self.runs if not r.config.get(var_name, True)]
            
            if with_feature and without_feature:
                # Average BLEU difference
                with_bleu = [r.evaluation.bleu for r in with_feature if r.evaluation.bleu]
                without_bleu = [r.evaluation.bleu for r in without_feature if r.evaluation.bleu]
                
                if with_bleu and without_bleu:
                    bleu_diff = sum(with_bleu)/len(with_bleu) - sum(without_bleu)/len(without_bleu)
                else:
                    bleu_diff = None
                
                # Average glossary difference
                with_gloss = [r.evaluation.glossary_adherence for r in with_feature 
                             if r.evaluation.glossary_adherence]
                without_gloss = [r.evaluation.glossary_adherence for r in without_feature 
                                if r.evaluation.glossary_adherence]
                
                if with_gloss and without_gloss:
                    gloss_diff = sum(with_gloss)/len(with_gloss) - sum(without_gloss)/len(without_gloss)
                else:
                    gloss_diff = None
                
                impacts[var_name] = {
                    "bleu_impact": bleu_diff,
                    "glossary_impact": gloss_diff,
                    "description": variable.description,
                }
        
        return impacts


def run_ablation(
    documents: list[Document],
    references: list[list[str]],
    sources: Optional[list[list[str]]] = None,
    glossary: Optional[Glossary] = None,
    config: Optional[AblationConfig] = None,
) -> AblationStudy:
    """Convenience function to run an ablation study."""
    config = config or AblationConfig(name="ablation")
    study = AblationStudy(config=config)
    study.run(documents, references, sources, glossary)
    return study

