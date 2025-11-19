
from __future__ import annotations
from typing import List, Dict, Tuple
from dataclasses import dataclass
import io
import fitz  # PyMuPDF

@dataclass
class BlockOut:
    page_index: int
    bbox: Tuple[float, float, float, float]
    text: str

def render_overlay(input_pdf: str, output_pdf: str, blocks_out: List[BlockOut], pages: List[int]) -> None:
    src = fitz.open(input_pdf)
    out = fitz.open()
    page_map = {pi: [] for pi in pages}
    for b in blocks_out:
        page_map.setdefault(b.page_index, []).append(b)

    for i, pi in enumerate(pages):
        sp = src.load_page(pi)
        np = out.new_page(width=sp.rect.width, height=sp.rect.height)
        # draw original page as background
        pix = sp.get_pixmap(alpha=False)
        img_bytes = pix.tobytes("png")
        np.insert_image(sp.rect, stream=img_bytes)
        # overlay translated text on original blocks
        for b in page_map.get(pi, []):
            if not b.text.strip():
                continue
            rect = fitz.Rect(*b.bbox)
            np.draw_rect(rect, color=(1,1,1), fill=(1,1,1), width=0)
            np.insert_textbox(rect, b.text, fontsize=10, fontname="helv", align=0)
        # ensure contents are wrapped for proper rendering on some versions
        try:
            np.wrap_contents()  # public API
        except Exception:
            pass
    out.save(output_pdf)
    out.close()
    src.close()
