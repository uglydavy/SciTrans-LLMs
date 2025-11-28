from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from scitrans_llms.models import Block, BlockType
from .pdf import parse_pdf


@dataclass
class BlockSummary:
    page_index: int
    kind: str
    bbox: Tuple[float, float, float, float]
    text_preview: str


def classify_block(block: Block) -> str:
    """Classify a block based on its content and type."""
    txt = block.source_text.strip()
    lower = txt.lower()
    
    # Use block type first if available
    if block.block_type == BlockType.FIGURE:
        return "figure"
    if block.block_type == BlockType.CAPTION:
        return "caption"
    if block.block_type == BlockType.HEADING:
        return "heading"
    if block.block_type == BlockType.TABLE:
        return "table"
    
    # Heuristic classification based on content
    if lower.startswith("figure") or lower.startswith("fig."):
        return "figure_caption"
    if lower.startswith("table"):
        return "table_caption"
    if len(txt.split()) <= 6 and txt.isupper():
        return "heading"
    if any(k in lower for k in ("abstract", "résumé")) and len(txt) < 240:
        return "abstract"
    if txt.endswith(":"):
        return "label"
    
    return "paragraph"


def analyze_document(pdf_path: str, pages: Optional[List[int]] = None, notes=None) -> List[BlockSummary]:
    """Analyze a PDF document and return block summaries.
    
    Args:
        pdf_path: Path to PDF file
        pages: Optional list of page numbers to analyze (0-indexed)
        notes: Optional notes (unused, for compatibility)
        
    Returns:
        List of BlockSummary objects describing each block
    """
    # Parse the PDF document
    doc = parse_pdf(pdf_path, pages=pages)
    
    summaries: List[BlockSummary] = []
    
    # Iterate through all segments and blocks
    for segment in doc.segments:
        for block in segment.blocks:
            kind = classify_block(block)
            text = block.source_text
            preview = text[:140].replace("\n", " ") + ("..." if len(text) > 140 else "")
            
            # Extract bbox as tuple
            if block.bbox:
                bbox_tuple = (block.bbox.x0, block.bbox.y0, block.bbox.x1, block.bbox.y1)
                page_index = block.bbox.page
            else:
                bbox_tuple = (0.0, 0.0, 0.0, 0.0)
                page_index = 0
            
            summaries.append(
                BlockSummary(
                    page_index=page_index,
                    kind=kind,
                    bbox=bbox_tuple,
                    text_preview=preview
                )
            )
    
    return summaries
