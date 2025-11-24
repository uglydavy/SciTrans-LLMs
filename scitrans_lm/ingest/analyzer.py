from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .pdf import Block, extract_blocks


@dataclass
class BlockSummary:
    page_index: int
    kind: str
    bbox: Tuple[float, float, float, float]
    text_preview: str


def classify_block(block: Block) -> str:
    txt = block.text.strip()
    lower = txt.lower()
    if lower.startswith("figure") or lower.startswith("fig."):
        return "figure_caption"
    if lower.startswith("table"):
        return "table_caption"
    if len(txt.split()) <= 6 and txt.isupper():
        return "heading"
    if any(k in lower for k in ("abstract", "résumé")) and len(txt) < 240:
        return "abstract"
    if txt.endswith(":" ):
        return "label"
    return block.kind or "paragraph"


def analyze_document(pdf_path: str, pages: List[int]) -> List[BlockSummary]:
    blocks, _ = extract_blocks(pdf_path, pages)
    summaries: List[BlockSummary] = []
    for blk in blocks:
        kind = classify_block(blk)
        preview = blk.text[:140].replace("\n", " ") + ("..." if len(blk.text) > 140 else "")
        summaries.append(BlockSummary(page_index=blk.page_index, kind=kind, bbox=blk.bbox, text_preview=preview))
    return summaries
