from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import fitz  # PyMuPDF


@dataclass
class Block:
    page_index: int
    bbox: Tuple[float, float, float, float]
    text: str
    kind: str = "paragraph"  # simple default


def extract_blocks(pdf_path: str, page_indices: List[int]) -> Tuple[List[Block], Tuple[int, int]]:
    """Extract basic text blocks (bbox + text) for specified pages using PyMuPDF.

    Returns (blocks, size) where size=(width,height) of last page processed (for reference).
    """
    blocks: List[Block] = []
    doc = fitz.open(pdf_path)
    size = (0, 0)
    for pi in page_indices:
        page = doc.load_page(pi)
        size = (page.rect.width, page.rect.height)
        for x0, y0, x1, y1, txt, _, _ in page.get_text("blocks"):  # type: ignore
            if not txt or not txt.strip():
                continue
            blocks.append(Block(page_index=pi, bbox=(x0, y0, x1, y1), text=txt.strip()))
    doc.close()
    return blocks, size