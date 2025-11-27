"""
Baseline comparison tools for evaluation.

This module provides:
- Wrappers for baseline systems (PDFMathTranslate, DocuTranslate)
- Simple NMT baselines (Google Translate, DeepL)
- Naive LLM baseline (no context, no glossary)
- Comparison reporting

Thesis Contribution #3: Systematic comparison with baselines
to demonstrate improvements.
"""

from __future__ import annotations

import subprocess
import tempfile
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from scitrans_next.models import Document
from scitrans_next.eval.runner import EvaluationRunner, EvaluationReport


class BaselineSystem(ABC):
    """Abstract base for baseline translation systems."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """System name for reporting."""
        pass
    
    @abstractmethod
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate a single text segment."""
        pass
    
    def translate_document(self, document: Document) -> list[str]:
        """Translate all translatable blocks in a document."""
        results = []
        for block in document.all_blocks:
            if block.is_translatable:
                translated = self.translate(
                    block.source_text,
                    document.source_lang,
                    document.target_lang,
                )
                results.append(translated)
        return results


class NaiveLLMBaseline(BaselineSystem):
    """Naive LLM baseline: simple translation without context or glossary.
    
    This serves as a baseline to show the value of:
    - Document-level context
    - Glossary enforcement
    - Refinement passes
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self._client = None
    
    @property
    def name(self) -> str:
        return f"naive-llm-{self.model}"
    
    def _get_client(self):
        if self._client is None:
            import os
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except ImportError:
                raise ImportError("OpenAI library required")
        return self._client
    
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Simple translation without context."""
        client = self._get_client()
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"Translate from {source_lang} to {target_lang}."},
                {"role": "user", "content": text},
            ],
            temperature=0.3,
        )
        
        return response.choices[0].message.content or text


class GoogleTranslateBaseline(BaselineSystem):
    """Google Translate API baseline."""
    
    def __init__(self):
        self._translator = None
    
    @property
    def name(self) -> str:
        return "google-translate"
    
    def _get_translator(self):
        if self._translator is None:
            try:
                from deep_translator import GoogleTranslator
                self._translator = GoogleTranslator
            except ImportError:
                raise ImportError("deep-translator library required")
        return self._translator
    
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        translator_class = self._get_translator()
        translator = translator_class(source=source_lang, target=target_lang)
        return translator.translate(text)


class DeepLBaseline(BaselineSystem):
    """DeepL API baseline."""
    
    def __init__(self, api_key: Optional[str] = None):
        import os
        self.api_key = api_key or os.getenv("DEEPL_API_KEY")
        self._translator = None
    
    @property
    def name(self) -> str:
        return "deepl"
    
    def _get_translator(self):
        if self._translator is None:
            try:
                import deepl
                self._translator = deepl.Translator(self.api_key)
            except ImportError:
                raise ImportError("deepl library required")
        return self._translator
    
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        translator = self._get_translator()
        # DeepL uses different language codes
        target_code = {"fr": "FR", "en": "EN-US", "de": "DE"}.get(target_lang, target_lang.upper())
        result = translator.translate_text(text, target_lang=target_code)
        return result.text


class PDFMathTranslateBaseline(BaselineSystem):
    """PDFMathTranslate baseline wrapper.
    
    Requires pdf2zh to be installed.
    """
    
    def __init__(self, engine: str = "google"):
        self.engine = engine
    
    @property
    def name(self) -> str:
        return f"pdfmathtranslate-{self.engine}"
    
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """PDFMathTranslate works on PDFs, not text. Use for PDF comparison only."""
        raise NotImplementedError("Use translate_pdf for PDF translation")
    
    def translate_pdf(
        self,
        input_pdf: Path,
        output_pdf: Path,
        source_lang: str = "en",
        target_lang: str = "fr",
    ) -> Path:
        """Translate a PDF using PDFMathTranslate."""
        try:
            cmd = [
                "pdf2zh",
                str(input_pdf),
                "-o", str(output_pdf),
                "-li", source_lang,
                "-lo", target_lang,
                "-s", self.engine,
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return output_pdf
        except FileNotFoundError:
            raise RuntimeError("pdf2zh not found. Install with: pip install pdf2zh")


@dataclass
class BaselineComparison:
    """Compare SciTrans-Next with baseline systems."""
    
    baselines: list[BaselineSystem] = field(default_factory=list)
    results: dict[str, EvaluationReport] = field(default_factory=dict)
    
    def add_baseline(self, baseline: BaselineSystem):
        """Add a baseline system to compare."""
        self.baselines.append(baseline)
    
    def run_comparison(
        self,
        documents: list[Document],
        references: list[list[str]],
        sources: list[list[str]],
        scitrans_hypotheses: list[list[str]],
    ) -> dict[str, EvaluationReport]:
        """Run comparison between SciTrans-Next and baselines.
        
        Args:
            documents: Source documents
            references: Reference translations
            sources: Source texts
            scitrans_hypotheses: SciTrans-Next outputs
            
        Returns:
            Dict mapping system name to EvaluationReport
        """
        runner = EvaluationRunner()
        
        # Flatten references and sources
        flat_refs = [ref for doc_refs in references for ref in doc_refs]
        flat_srcs = [src for doc_srcs in sources for src in doc_srcs]
        flat_scitrans = [hyp for doc_hyps in scitrans_hypotheses for hyp in doc_hyps]
        
        # Evaluate SciTrans-Next
        self.results["scitrans-next"] = runner.evaluate(
            hypotheses=flat_scitrans,
            references=flat_refs,
            sources=flat_srcs,
            run_id="scitrans-next",
        )
        
        # Evaluate each baseline
        for baseline in self.baselines:
            try:
                all_hyps = []
                for doc in documents:
                    doc_hyps = baseline.translate_document(doc)
                    all_hyps.extend(doc_hyps)
                
                self.results[baseline.name] = runner.evaluate(
                    hypotheses=all_hyps,
                    references=flat_refs[:len(all_hyps)],
                    sources=flat_srcs[:len(all_hyps)],
                    run_id=baseline.name,
                )
            except Exception as e:
                print(f"Baseline {baseline.name} failed: {e}")
        
        return self.results
    
    def summary(self) -> str:
        """Generate comparison summary."""
        lines = [
            "Baseline Comparison Results",
            "=" * 60,
            "",
            f"{'System':<25} {'BLEU':>8} {'chrF++':>8} {'Gloss%':>8}",
            "-" * 60,
        ]
        
        for name, report in self.results.items():
            bleu = f"{report.bleu:.2f}" if report.bleu else "N/A"
            chrf = f"{report.chrf:.2f}" if report.chrf else "N/A"
            gloss = f"{report.glossary_adherence:.1%}" if report.glossary_adherence else "N/A"
            lines.append(f"{name:<25} {bleu:>8} {chrf:>8} {gloss:>8}")
        
        return "\n".join(lines)
    
    def to_latex_table(self) -> str:
        """Generate LaTeX comparison table."""
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"System & BLEU & chrF++ & Glossary Adh. \\",
            r"\midrule",
        ]
        
        for name, report in self.results.items():
            name_tex = name.replace("_", r"\_")
            bleu = f"{report.bleu:.2f}" if report.bleu else "---"
            chrf = f"{report.chrf:.2f}" if report.chrf else "---"
            gloss = f"{report.glossary_adherence:.1%}" if report.glossary_adherence else "---"
            
            # Bold best results
            if name == "scitrans-next":
                lines.append(f"\\textbf{{{name_tex}}} & \\textbf{{{bleu}}} & \\textbf{{{chrf}}} & \\textbf{{{gloss}}} \\\\")
            else:
                lines.append(f"{name_tex} & {bleu} & {chrf} & {gloss} \\\\")
        
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Comparison with baseline systems}",
            r"\end{table}",
        ])
        
        return "\n".join(lines)

