"""
Rendering module for producing translated PDF output.

This module handles:
- PDF rendering with layout preservation
- Font matching and fallback
- Text positioning and sizing
- Figure and table preservation

Thesis Contribution #1: Layout-preserving translation
requires faithful visual reconstruction.
"""

from scitrans_next.render.pdf import (
    render_pdf,
    PDFRenderer,
    FontMapper,
)

__all__ = [
    "render_pdf",
    "PDFRenderer",
    "FontMapper",
]
