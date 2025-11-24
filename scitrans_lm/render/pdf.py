from __future__ import annotations
from typing import List, Tuple
from dataclasses import dataclass
import io

import fitz  # PyMuPDF
from PIL import Image


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
        # draw original page as background, compressing to keep downloads lean
        pix = sp.get_pixmap(alpha=False, matrix=fitz.Matrix(1, 1))
        img_bytes = _pixmap_to_jpeg(pix)
        np.insert_image(sp.rect, stream=img_bytes)
        # overlay translated text on original blocks
        for b in page_map.get(pi, []):
            if not b.text.strip():
                continue
            rect = fitz.Rect(*b.bbox)
            np.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1), width=0)
            np.insert_textbox(rect, b.text, fontsize=10, fontname="helv", align=0)
        # ensure contents are wrapped for proper rendering on some versions
        try:
            np.wrap_contents()
        except Exception:
            pass
    out.save(output_pdf, deflate=True, garbage=4)
    out.close()
    src.close()


def _pixmap_to_jpeg(pix: fitz.Pixmap, quality: int = 80) -> bytes:
    """Return JPEG bytes, handling PyMuPDF API variants defensively.

    The `quality` kwarg disappeared in some builds, while others only accept a
    positional format string. We try a small cascade of safe calls before
    falling back to Pillow or uncompressed bytes so that overlay rendering
    never crashes with ``TypeError: tobytes() got an unexpected keyword
    argument 'quality'``.
    """

    attempts = [
        lambda: pix.tobytes("jpeg", quality=quality),
        lambda: pix.tobytes("jpeg"),
        lambda: pix.tobytes(),
    ]
    for fn in attempts:
        try:
            return fn()
        except TypeError:
            # Older PyMuPDF builds raise TypeError for unexpected kwargs
            continue
        except Exception:
            continue

    try:
        mode = "RGB" if pix.colorspace and getattr(pix.colorspace, "n", 3) == 3 else "CMYK"
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()
    except Exception:
        # Final fallback: raw bytes (may be large but keeps pipeline alive)
        try:
            return pix.tobytes()
        except Exception:
            return b""
