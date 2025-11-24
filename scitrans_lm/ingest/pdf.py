from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import fitz  # PyMuPDF


@dataclass
class Block:
    page_index: int
    bbox: Tuple[float, float, float, float]
    text: str
    kind: str = "paragraph"  # simple default


def extract_blocks(
    pdf_path: str,
    page_indices: List[int],
    progress: Optional[Callable[[str], None]] = None,
    notes: Optional[List[str]] = None,
) -> Tuple[List[Block], Tuple[int, int]]:
    """Extract basic text blocks (bbox + text) for specified pages using PyMuPDF.

    Falls back to per-page OCR (pytesseract) when no selectable text is found so
    image-only PDFs still produce content instead of blank translations. Notes
    and progress callbacks receive human-readable messages about fallbacks.

    Returns (blocks, size) where size=(width,height) of last page processed (for reference).
    """

    def _log(msg: str) -> None:
        if progress:
            progress(msg)
        if notes is not None:
            notes.append(msg)

    blocks: List[Block] = []
    doc = fitz.open(pdf_path)
    size = (0, 0)
    for pi in page_indices:
        page = doc.load_page(pi)
        size = (page.rect.width, page.rect.height)
        page_blocks = 0
        for x0, y0, x1, y1, txt, _, _ in page.get_text("blocks"):  # type: ignore
            if not txt or not txt.strip():
                continue
            blocks.append(Block(page_index=pi, bbox=(x0, y0, x1, y1), text=txt.strip()))
            page_blocks += 1

        if page_blocks == 0:
            try:
                import pytesseract
                from PIL import Image

                pix = page.get_pixmap(alpha=False, matrix=fitz.Matrix(2, 2))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = pytesseract.image_to_string(img)
                if text and text.strip():
                    _log(f"OCR fallback used on page {pi + 1} (no selectable text detected).")
                    blocks.append(
                        Block(
                            page_index=pi,
                            bbox=(0.0, 0.0, page.rect.width, page.rect.height),
                            text=text.strip(),
                        )
                    )
                else:
                    _log(f"Page {pi + 1} contained no extractable text even after OCR.")
            except Exception as exc:  # noqa: BLE001
                _log(
                    "OCR fallback unavailable; install pytesseract and pillow for image-only PDFs. "
                    f"(page {pi + 1}, reason: {exc})"
                )
    doc.close()
    return blocks, size
