"""
Core data models for SciTrans-Next.

These models provide a clean abstraction over documents that works for both
plain text (for testing/development) and PDFs (for production).

Design Philosophy:
- Immutable-ish: models use frozen dataclasses where possible
- Serializable: all models can be converted to/from JSON for debugging
- Layout-aware: coordinates are optional but fully supported for PDF reconstruction
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import json
import uuid


class BlockType(Enum):
    """Types of content blocks in a document.
    
    These correspond to layout detection categories and determine
    how each block should be processed during translation.
    """
    PARAGRAPH = auto()      # Regular text - translate normally
    HEADING = auto()        # Section headings - translate, preserve structure
    LIST_ITEM = auto()      # Bullet/numbered items - translate
    TABLE = auto()          # Tables - protect or translate cells
    FIGURE = auto()         # Figures/images - protect entirely  
    EQUATION = auto()       # Math formulas - protect (placeholder)
    CODE = auto()           # Code blocks - protect
    CAPTION = auto()        # Figure/table captions - translate
    FOOTNOTE = auto()       # Footnotes - translate
    REFERENCE = auto()      # Bibliography entries - partial protection
    HEADER = auto()         # Page headers - often skip
    FOOTER = auto()         # Page footers - often skip
    UNKNOWN = auto()        # Fallback - treat as paragraph


@dataclass
class BoundingBox:
    """Bounding box for layout-aware rendering.
    
    Coordinates are in PDF points (1/72 inch) from top-left origin.
    """
    x0: float
    y0: float
    x1: float
    y1: float
    page: int = 0
    
    @property
    def width(self) -> float:
        return self.x1 - self.x0
    
    @property
    def height(self) -> float:
        return self.y1 - self.y0
    
    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)
    
    def to_dict(self) -> dict:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1, "page": self.page}
    
    @classmethod
    def from_dict(cls, d: dict) -> BoundingBox:
        return cls(x0=d["x0"], y0=d["y0"], x1=d["x1"], y1=d["y1"], page=d.get("page", 0))


@dataclass
class Block:
    """A single content block within a document.
    
    This is the atomic unit of translation. Each block has:
    - source_text: original text content
    - block_type: semantic type (paragraph, heading, equation, etc.)
    - bbox: optional layout coordinates for PDF reconstruction
    - translated_text: filled in after translation
    - metadata: flexible dict for additional info (font, style, etc.)
    
    Thesis Contribution #1: Blocks can be marked as "protected" based on type,
    and the masking system will insert placeholders before translation.
    """
    source_text: str
    block_type: BlockType = BlockType.PARAGRAPH
    block_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    bbox: Optional[BoundingBox] = None
    translated_text: Optional[str] = None
    masked_text: Optional[str] = None  # Text with placeholders inserted
    metadata: dict = field(default_factory=dict)
    
    @property
    def is_protected(self) -> bool:
        """Whether this block should be protected (not translated)."""
        return self.block_type in {
            BlockType.EQUATION, 
            BlockType.CODE, 
            BlockType.FIGURE,
        }
    
    @property
    def is_translatable(self) -> bool:
        """Whether this block contains text that should be translated."""
        return self.block_type in {
            BlockType.PARAGRAPH,
            BlockType.HEADING,
            BlockType.LIST_ITEM,
            BlockType.CAPTION,
            BlockType.FOOTNOTE,
        }
    
    def to_dict(self) -> dict:
        return {
            "block_id": self.block_id,
            "source_text": self.source_text,
            "block_type": self.block_type.name,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "translated_text": self.translated_text,
            "masked_text": self.masked_text,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> Block:
        return cls(
            block_id=d.get("block_id", str(uuid.uuid4())[:8]),
            source_text=d["source_text"],
            block_type=BlockType[d.get("block_type", "PARAGRAPH")],
            bbox=BoundingBox.from_dict(d["bbox"]) if d.get("bbox") else None,
            translated_text=d.get("translated_text"),
            masked_text=d.get("masked_text"),
            metadata=d.get("metadata", {}),
        )


@dataclass
class Segment:
    """A logical segment of a document (e.g., a section or subsection).
    
    Segments group related blocks and are used for:
    - Document-level context: translation can use previous segments
    - Coherence: pronouns and terminology should be consistent within segments
    - Evaluation: metrics can be computed per-segment
    
    Thesis Contribution #2: Segments provide the context window for
    document-level LLM translation and refinement.
    """
    blocks: list[Block]
    segment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: Optional[str] = None  # Section heading if available
    metadata: dict = field(default_factory=dict)
    
    @property
    def source_text(self) -> str:
        """Concatenated source text of all blocks."""
        return "\n\n".join(b.source_text for b in self.blocks if b.source_text)
    
    @property
    def translated_text(self) -> str:
        """Concatenated translated text of all blocks."""
        return "\n\n".join(
            b.translated_text or "" for b in self.blocks if b.is_translatable
        )
    
    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "title": self.title,
            "blocks": [b.to_dict() for b in self.blocks],
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> Segment:
        return cls(
            segment_id=d.get("segment_id", str(uuid.uuid4())[:8]),
            title=d.get("title"),
            blocks=[Block.from_dict(b) for b in d.get("blocks", [])],
            metadata=d.get("metadata", {}),
        )


@dataclass  
class Document:
    """A complete document ready for translation.
    
    The Document is the top-level container that holds:
    - segments: logical divisions of the document
    - metadata: title, authors, source language, etc.
    - glossary_terms: extracted or user-provided terminology
    
    Documents can be created from:
    - Plain text (for testing): Document.from_text()
    - PDF files (for production): via ingest module
    - JSON (for debugging): Document.from_dict()
    """
    segments: list[Segment]
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    source_lang: str = "en"
    target_lang: str = "fr"
    title: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    glossary_terms: dict[str, str] = field(default_factory=dict)  # source -> target
    
    @property
    def all_blocks(self) -> list[Block]:
        """Flat list of all blocks across all segments."""
        return [block for segment in self.segments for block in segment.blocks]
    
    @property
    def source_text(self) -> str:
        """Full source text of the document."""
        return "\n\n---\n\n".join(s.source_text for s in self.segments)
    
    @property
    def translated_text(self) -> str:
        """Full translated text of the document."""
        return "\n\n---\n\n".join(s.translated_text for s in self.segments)
    
    @classmethod
    def from_text(
        cls,
        text: str,
        source_lang: str = "en",
        target_lang: str = "fr",
        paragraph_separator: str = "\n\n",
    ) -> Document:
        """Create a Document from plain text.
        
        This is the simplest way to create a document for testing.
        Each double-newline-separated paragraph becomes a block.
        All blocks are grouped into a single segment.
        
        Args:
            text: Plain text content
            source_lang: Source language code (default: "en")
            target_lang: Target language code (default: "fr")
            paragraph_separator: String that separates paragraphs
            
        Returns:
            Document instance ready for translation
        """
        paragraphs = [p.strip() for p in text.split(paragraph_separator) if p.strip()]
        blocks = [
            Block(source_text=p, block_type=BlockType.PARAGRAPH)
            for p in paragraphs
        ]
        segment = Segment(blocks=blocks, title="Main Content")
        return cls(
            segments=[segment],
            source_lang=source_lang,
            target_lang=target_lang,
        )
    
    @classmethod
    def from_paragraphs(
        cls,
        paragraphs: list[str],
        source_lang: str = "en",
        target_lang: str = "fr",
    ) -> Document:
        """Create a Document from a list of paragraphs."""
        blocks = [
            Block(source_text=p, block_type=BlockType.PARAGRAPH)
            for p in paragraphs if p.strip()
        ]
        segment = Segment(blocks=blocks, title="Main Content")
        return cls(
            segments=[segment],
            source_lang=source_lang,
            target_lang=target_lang,
        )
    
    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "title": self.title,
            "segments": [s.to_dict() for s in self.segments],
            "metadata": self.metadata,
            "glossary_terms": self.glossary_terms,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> Document:
        return cls(
            doc_id=d.get("doc_id", str(uuid.uuid4())[:8]),
            source_lang=d.get("source_lang", "en"),
            target_lang=d.get("target_lang", "fr"),
            title=d.get("title"),
            segments=[Segment.from_dict(s) for s in d.get("segments", [])],
            metadata=d.get("metadata", {}),
            glossary_terms=d.get("glossary_terms", {}),
        )
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize document to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> Document:
        """Deserialize document from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def summary(self) -> str:
        """Return a human-readable summary of the document."""
        total_blocks = len(self.all_blocks)
        translatable = sum(1 for b in self.all_blocks if b.is_translatable)
        protected = sum(1 for b in self.all_blocks if b.is_protected)
        return (
            f"Document '{self.title or self.doc_id}'\n"
            f"  Direction: {self.source_lang} → {self.target_lang}\n"
            f"  Segments: {len(self.segments)}\n"
            f"  Blocks: {total_blocks} total, {translatable} translatable, {protected} protected\n"
            f"  Glossary terms: {len(self.glossary_terms)}"
        )

