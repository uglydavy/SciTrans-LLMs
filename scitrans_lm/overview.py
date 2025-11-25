"""Lightweight component map for SciTrans-LM.

This module centralises a human-readable overview of the codebase so both
CLI and GUI users can discover where to extend or debug specific stages of
the pipeline without hunting through folders.
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
            key_files=["scitrans_lm/cli.py"],
            notes="Typer commands wrap the same pipeline used by the GUI.",
        ),
        Component(
            name="GUI",
            responsibility="Gradio interface with Translate, Debug/QA, Pipeline Lab, and System Check tabs.",
            key_files=["scitrans_lm/gui.py"],
            notes="Shares translation pipeline and diagnostics with the CLI.",
        ),
        Component(
            name="Pipeline",
            responsibility="Layout-aware extraction, masking, translation, reranking, and rendering.",
            key_files=["scitrans_lm/pipeline.py", "scitrans_lm/render/pdf.py"],
            notes="Backed by YOLO layout detection when available; uses glossary enforcement and translation memory.",
        ),
        Component(
            name="Ingestion",
            responsibility="Document parsing, layout analysis, and classification of text blocks.",
            key_files=["scitrans_lm/ingest/pdf.py", "scitrans_lm/ingest/analyzer.py"],
            notes="Provides block kinds used to skip figures/formulas and normalise tables.",
        ),
        Component(
            name="Masking",
            responsibility="Protect formulas, numbers, and inline code before translation then restore them.",
            key_files=["scitrans_lm/mask.py"],
            notes="Used by both CLI and GUI to avoid corrupting structured content.",
        ),
        Component(
            name="Translation backends",
            responsibility="Pluggable translators (OpenAI, DeepL, Google, DeepSeek, Perplexity, offline dictionary).",
            key_files=["scitrans_lm/translate/backends.py", "scitrans_lm/translate/glossary.py"],
            notes="Glossary enforcement and dictionary fallback keep terminology consistent.",
        ),
        Component(
            name="Refinement",
            responsibility="Prompt construction, iterative refinement, reranking, and BLEU scoring.",
            key_files=["scitrans_lm/refine/prompting.py", "scitrans_lm/refine/rerank.py", "scitrans_lm/refine/scoring.py"],
            notes="Quality loops rerank multiple attempts; BLEU scoring aids quick evaluation.",
        ),
        Component(
            name="Bootstrap & assets",
            responsibility="Ensures YOLO layout weights and default glossaries exist before runs.",
            key_files=["scitrans_lm/bootstrap.py", "scitrans_lm/config.py"],
            notes="Creates placeholders and keeps data/glossary populated.",
        ),
        Component(
            name="Diagnostics",
            responsibility="Environment health checks for dependencies, assets, and API keys.",
            key_files=["scitrans_lm/diagnostics.py"],
            notes="Surfaced via CLI doctor and GUI System Check tab.",
        ),
    ]


__all__ = ["Component", "get_component_map"]
