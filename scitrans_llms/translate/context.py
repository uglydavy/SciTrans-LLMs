"""
Document-level context management for coherent translation.

This module provides tools for:
- Maintaining translation context across segments
- Building context windows for LLM prompts
- Extracting document summaries and entity lists

Thesis Contribution #2: Document-level LLM context and refinement.
This is what distinguishes our approach from segment-level translation.

Design Philosophy:
- Context is built incrementally as translation proceeds
- Multiple context strategies are supported (sliding window, summary, etc.)
- Context can be serialized for debugging/analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from scitrans_llms.models import Document, Segment, Block
from scitrans_llms.translate.glossary import Glossary


@dataclass
class ContextWindow:
    """A window of recent translations for context.
    
    This implements a sliding window over translated segments,
    used to provide context to the translator.
    
    Attributes:
        max_segments: Maximum number of segments to keep
        max_tokens: Approximate token limit (rough estimate)
        source_segments: Source text of recent segments
        target_segments: Translated text of recent segments
    """
    max_segments: int = 5
    max_tokens: int = 2000  # Rough estimate, not exact tokenization
    source_segments: list[str] = field(default_factory=list)
    target_segments: list[str] = field(default_factory=list)
    
    def add(self, source: str, target: str) -> None:
        """Add a translated segment to the context window."""
        self.source_segments.append(source)
        self.target_segments.append(target)
        
        # Trim to max_segments
        while len(self.source_segments) > self.max_segments:
            self.source_segments.pop(0)
            self.target_segments.pop(0)
        
        # Trim to approximate token limit
        self._trim_to_token_limit()
    
    def _trim_to_token_limit(self) -> None:
        """Remove oldest segments if we exceed token limit."""
        # Rough estimate: 1 token ≈ 4 characters
        while self._estimate_tokens() > self.max_tokens and len(self.source_segments) > 1:
            self.source_segments.pop(0)
            self.target_segments.pop(0)
    
    def _estimate_tokens(self) -> int:
        """Rough token count estimate."""
        total_chars = sum(len(s) for s in self.source_segments)
        total_chars += sum(len(s) for s in self.target_segments)
        return total_chars // 4
    
    def format_for_prompt(self, source_lang: str = "EN", target_lang: str = "FR") -> str:
        """Format the context window for LLM prompt injection."""
        if not self.source_segments:
            return ""
        
        lines = ["### Previous translations for context:"]
        for src, tgt in zip(self.source_segments, self.target_segments):
            # Truncate long segments
            src_trunc = src[:300] + "..." if len(src) > 300 else src
            tgt_trunc = tgt[:300] + "..." if len(tgt) > 300 else tgt
            lines.append(f"[{source_lang}] {src_trunc}")
            lines.append(f"[{target_lang}] {tgt_trunc}")
            lines.append("")
        
        return "\n".join(lines)
    
    def clear(self) -> None:
        """Clear all context."""
        self.source_segments.clear()
        self.target_segments.clear()


@dataclass
class DocumentContext:
    """Full context for document-level translation.
    
    This aggregates all context needed for coherent translation:
    - Sliding window of recent translations
    - Document summary (if extracted)
    - Named entities and key terms
    - Glossary
    - Document structure information
    
    Thesis Contribution #2: This is the core abstraction enabling
    document-level LLM translation.
    """
    document: Document
    glossary: Optional[Glossary] = None
    context_window: ContextWindow = field(default_factory=ContextWindow)
    summary: Optional[str] = None
    entities: list[str] = field(default_factory=list)
    current_segment_index: int = 0
    current_block_index: int = 0
    
    @classmethod
    def from_document(
        cls,
        doc: Document,
        glossary: Optional[Glossary] = None,
        window_size: int = 5,
    ) -> DocumentContext:
        """Create a context from a document."""
        return cls(
            document=doc,
            glossary=glossary or (Glossary(entries=list(
                Glossary(entries=[]).entries
            )) if not doc.glossary_terms else Glossary(
                entries=[],  # Could populate from doc.glossary_terms
            )),
            context_window=ContextWindow(max_segments=window_size),
        )
    
    def add_translation(self, source: str, target: str) -> None:
        """Record a completed translation for context."""
        self.context_window.add(source, target)
    
    def get_current_segment(self) -> Optional[Segment]:
        """Get the segment currently being translated."""
        if self.current_segment_index < len(self.document.segments):
            return self.document.segments[self.current_segment_index]
        return None
    
    def advance_segment(self) -> None:
        """Move to the next segment."""
        self.current_segment_index += 1
        self.current_block_index = 0
    
    def build_system_prompt(
        self,
        include_glossary: bool = True,
        include_context: bool = True,
        include_summary: bool = True,
    ) -> str:
        """Build a complete system prompt for LLM translation.
        
        This assembles all context components into a single prompt.
        """
        source_lang = self.document.source_lang.upper()
        target_lang = self.document.target_lang.upper()
        
        parts = [
            f"You are an expert translator specializing in scientific and technical documents.",
            f"Translate the following text from {source_lang} to {target_lang}.",
            "",
            "Guidelines:",
            "- Preserve all placeholders exactly as they appear (e.g., <<MATH_001>>)",
            "- Maintain the technical accuracy and terminology of the source",
            "- Keep the same paragraph structure",
            "- Preserve all formatting markers",
        ]
        
        if include_summary and self.summary:
            parts.extend([
                "",
                "### Document summary:",
                self.summary,
            ])
        
        if include_glossary and self.glossary and len(self.glossary) > 0:
            parts.extend([
                "",
                self.glossary.to_prompt_string(max_entries=30),
            ])
        
        if include_context:
            context_str = self.context_window.format_for_prompt(source_lang, target_lang)
            if context_str:
                parts.extend([
                    "",
                    context_str,
                ])
        
        return "\n".join(parts)
    
    def get_translation_stats(self) -> dict:
        """Return statistics about translation progress."""
        total_segments = len(self.document.segments)
        total_blocks = len(self.document.all_blocks)
        translated_blocks = sum(
            1 for b in self.document.all_blocks if b.translated_text is not None
        )
        
        return {
            "total_segments": total_segments,
            "current_segment": self.current_segment_index,
            "total_blocks": total_blocks,
            "translated_blocks": translated_blocks,
            "progress_percent": (translated_blocks / total_blocks * 100) if total_blocks > 0 else 0,
            "context_window_size": len(self.context_window.source_segments),
        }


def extract_document_summary(doc: Document, max_chars: int = 500) -> str:
    """Extract a brief summary from document structure.
    
    This creates a simple summary from:
    - Document title
    - Section headings
    - First paragraph
    
    A more sophisticated version could use an LLM to generate
    a proper summary.
    """
    parts = []
    
    if doc.title:
        parts.append(f"Title: {doc.title}")
    
    # Collect section headings
    headings = []
    for segment in doc.segments:
        if segment.title:
            headings.append(segment.title)
        for block in segment.blocks:
            if block.block_type.name == "HEADING":
                headings.append(block.source_text)
    
    if headings:
        parts.append("Sections: " + ", ".join(headings[:5]))
    
    # First paragraph as abstract proxy
    for block in doc.all_blocks:
        if block.block_type.name == "PARAGRAPH" and len(block.source_text) > 50:
            first_para = block.source_text[:300]
            if len(block.source_text) > 300:
                first_para += "..."
            parts.append(f"Opening: {first_para}")
            break
    
    summary = "\n".join(parts)
    return summary[:max_chars]


def extract_entities(doc: Document) -> list[str]:
    """Extract named entities and key terms from document.
    
    This is a simple keyword extraction. Could be extended with:
    - NER (spaCy, transformers)
    - TF-IDF key term extraction
    - Citation extraction
    """
    import re
    
    # Simple heuristic: find capitalized multi-word phrases
    entities = set()
    
    for block in doc.all_blocks:
        if not block.source_text:
            continue
        
        # Find capitalized phrases (2+ words)
        pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
        matches = re.findall(pattern, block.source_text)
        entities.update(matches)
    
    # Return sorted by frequency (approximate)
    return sorted(entities)[:20]

