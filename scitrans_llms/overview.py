"""Lightweight component map for SciTrans-LM.

CLI and GUI users can surface this summary to quickly locate modules without
hunting through folders.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List


@dataclass
class Component:
    """Represents a functional area of the project."""

    name: str
    responsibility: str
    key_files: List[str]
    notes: str = ""

    def to_dict(self) -> Dict[str, str]:
        data = asdict(self)
        data["key_files"] = ", ".join(self.key_files)
        return data


def get_component_map() -> List[Component]:
    """Return a curated list of major components and their locations."""

    return [
        Component(
            name="CLI",
            responsibility="Entry points for translation, inspection, evaluation, and diagnostics.",
            key_files=["scitrans_llms/cli.py"],
            notes="Typer commands wrap the same pipeline used by the GUI.",
        ),
        Component(
            name="GUI",
            responsibility="Gradio interface with Translate, Debug/QA, Pipeline Lab, and System Check tabs.",
            key_files=["scitrans_llms/gui.py"],
            notes="Shares translation pipeline and diagnostics with the CLI.",
        ),
        Component(
            name="Pipeline",
            responsibility="Layout-aware extraction, masking, translation, reranking, and rendering.",
            key_files=["scitrans_llms/pipeline.py", "scitrans_llms/render/pdf.py"],
            notes="Backed by YOLO layout detection when available; uses glossary enforcement and translation memory.",
        ),
        Component(
            name="Ingestion",
            responsibility="Document parsing, layout analysis, and classification of text blocks.",
            key_files=["scitrans_llms/ingest/pdf.py", "scitrans_llms/ingest/analyzer.py"],
            notes="Provides block kinds used to skip figures/formulas and normalise tables.",
        ),
        Component(
            name="Masking",
            responsibility="Protect formulas, numbers, and inline code before translation then restore them.",
            key_files=["scitrans_llms/masking.py"],
            notes="Used by both CLI and GUI to avoid corrupting structured content.",
        ),
        Component(
            name="Translation backends",
            responsibility="Pluggable translators (OpenAI, DeepL, Google, DeepSeek, Perplexity, offline dictionary).",
            key_files=["scitrans_llms/translate/backends.py", "scitrans_llms/translate/glossary.py"],
            notes="Glossary enforcement and dictionary fallback keep terminology consistent.",
        ),
        Component(
            name="Refinement",
            responsibility="Prompt construction, iterative refinement, reranking, and BLEU scoring.",
            key_files=["scitrans_llms/refine/prompting.py", "scitrans_llms/refine/rerank.py", "scitrans_llms/refine/scoring.py"],
            notes="Quality loops rerank multiple attempts; BLEU scoring aids quick evaluation.",
        ),
        Component(
            name="Bootstrap & assets",
            responsibility="Ensures YOLO layout weights and default glossaries exist before runs.",
            key_files=["scitrans_llms/bootstrap.py", "scitrans_llms/config.py"],
            notes="Creates placeholders and keeps data/glossary populated.",
        ),
        Component(
            name="Diagnostics",
            responsibility="Environment health checks for dependencies, assets, and API keys.",
            key_files=["scitrans_llms/diagnostics.py"],
            notes="Surfaced via CLI doctor and GUI System Check tab.",
        ),
    ]


__all__ = ["Component", "get_component_map"]
