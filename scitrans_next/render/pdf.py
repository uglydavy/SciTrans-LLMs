"""
PDF rendering with layout preservation.

This module reconstructs translated PDFs by:
1. Starting from the original PDF (preserving images, layout)
2. Overlaying translated text at original positions
3. Matching fonts and sizes as closely as possible

Approaches:
- Overlay mode: Add translation as text layer over original
- Replace mode: White out original text, insert translation
- Hybrid: Combine both for best results
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from scitrans_next.models import Document, Block, BoundingBox


@dataclass
class FontSpec:
    """Font specification for rendering."""
    name: str
    size: float
    is_bold: bool = False
    is_italic: bool = False
    color: tuple = (0, 0, 0)


class FontMapper:
    """Maps source fonts to available fonts.
    
    Handles:
    - Font substitution when source font unavailable
    - Size adjustment for different scripts
    - Fallback to default fonts
    """
    
    # Common font substitutions
    SUBSTITUTIONS = {
        "Times": "Times-Roman",
        "TimesNewRoman": "Times-Roman",
        "Times New Roman": "Times-Roman",
        "Arial": "Helvetica",
        "Helvetica": "Helvetica",
        "Courier": "Courier",
        "CourierNew": "Courier",
        "Courier New": "Courier",
    }
    
    # Default fonts
    DEFAULT_SERIF = "Times-Roman"
    DEFAULT_SANS = "Helvetica"
    DEFAULT_MONO = "Courier"
    
    def __init__(self):
        self._available_fonts: set[str] = set()
        self._cache: dict[str, str] = {}
    
    def map_font(self, source_font: str) -> str:
        """Map a source font name to an available font."""
        if source_font in self._cache:
            return self._cache[source_font]
        
        # Try direct substitution
        for pattern, replacement in self.SUBSTITUTIONS.items():
            if pattern.lower() in source_font.lower():
                self._cache[source_font] = replacement
                return replacement
        
        # Classify and use default
        lower = source_font.lower()
        if any(mono in lower for mono in ["mono", "courier", "consolas", "code"]):
            result = self.DEFAULT_MONO
        elif any(sans in lower for sans in ["arial", "helvetica", "gothic", "sans"]):
            result = self.DEFAULT_SANS
        else:
            result = self.DEFAULT_SERIF
        
        self._cache[source_font] = result
        return result
    
    def adjust_size_for_language(
        self,
        size: float,
        source_lang: str,
        target_lang: str,
    ) -> float:
        """Adjust font size for target language.
        
        Some languages (e.g., German, French) tend to be longer,
        so we may need to slightly reduce font size.
        """
        # Language expansion factors (approximate)
        expansion = {
            "fr": 1.15,  # French is ~15% longer than English
            "de": 1.20,  # German is ~20% longer
            "es": 1.10,  # Spanish is ~10% longer
            "it": 1.10,  # Italian is ~10% longer
            "pt": 1.10,  # Portuguese is ~10% longer
        }
        
        if source_lang == "en" and target_lang in expansion:
            factor = expansion[target_lang]
            # Reduce size proportionally (but not too much)
            return max(size * 0.9, size / (factor ** 0.3))
        
        return size


@dataclass
class RenderConfig:
    """Configuration for PDF rendering."""
    mode: str = "overlay"  # "overlay", "replace", or "hybrid"
    preserve_images: bool = True
    preserve_formatting: bool = True
    adjust_font_size: bool = True
    fallback_font: str = "Helvetica"
    text_color: tuple = (0, 0, 0)
    # For replace mode
    background_color: tuple = (255, 255, 255)


class PDFRenderer:
    """Render translated Document back to PDF.
    
    Usage:
        renderer = PDFRenderer()
        renderer.render(
            document=translated_doc,
            source_pdf="original.pdf",
            output_path="translated.pdf"
        )
    """
    
    def __init__(self, config: Optional[RenderConfig] = None):
        self.config = config or RenderConfig()
        self.font_mapper = FontMapper()
    
    def render(
        self,
        document: Document,
        source_pdf: str | Path,
        output_path: str | Path,
    ) -> Path:
        """Render a translated document to PDF.
        
        Args:
            document: Translated Document with blocks
            source_pdf: Path to original PDF
            output_path: Path for output PDF
            
        Returns:
            Path to rendered PDF
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF is required. Install with: pip install PyMuPDF")
        
        source_pdf = Path(source_pdf)
        output_path = Path(output_path)
        
        if not source_pdf.exists():
            raise FileNotFoundError(f"Source PDF not found: {source_pdf}")
        
        # Open source PDF
        doc = fitz.open(str(source_pdf))
        
        # Group blocks by page
        blocks_by_page = self._group_blocks_by_page(document)
        
        # Process each page
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = blocks_by_page.get(page_num, [])
            
            if self.config.mode == "replace":
                self._render_replace_mode(page, blocks)
            else:  # overlay or hybrid
                self._render_overlay_mode(page, blocks)
        
        # Save output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(output_path))
        doc.close()
        
        return output_path
    
    def _group_blocks_by_page(self, document: Document) -> dict[int, list[Block]]:
        """Group blocks by their page number."""
        blocks_by_page: dict[int, list[Block]] = {}
        
        for block in document.all_blocks:
            if block.bbox:
                page = block.bbox.page
                if page not in blocks_by_page:
                    blocks_by_page[page] = []
                blocks_by_page[page].append(block)
        
        return blocks_by_page
    
    def _render_overlay_mode(self, page, blocks: list[Block]):
        """Render translations as overlay on original text."""
        import fitz
        
        for block in blocks:
            if not block.translated_text or not block.bbox:
                continue
            
            if not block.is_translatable:
                continue
            
            bbox = block.bbox
            rect = fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
            
            # Get font info
            font_name = block.metadata.get("font_name", "Helvetica")
            font_size = block.metadata.get("font_size", 11)
            
            # Map font
            mapped_font = self.font_mapper.map_font(font_name)
            
            # Adjust size if needed
            if self.config.adjust_font_size:
                font_size = self.font_mapper.adjust_size_for_language(
                    font_size, "en", "fr"
                )
            
            # First, white out the original text area
            page.draw_rect(rect, color=None, fill=(1, 1, 1))
            
            # Insert translated text
            text = block.translated_text
            
            # Use text writer for better control
            try:
                # Fit text to rectangle
                rc = page.insert_textbox(
                    rect,
                    text,
                    fontsize=font_size,
                    fontname=mapped_font,
                    color=self.config.text_color,
                    align=fitz.TEXT_ALIGN_LEFT,
                )
                
                # If text didn't fit, try smaller font
                if rc < 0:
                    smaller_size = font_size * 0.85
                    page.draw_rect(rect, color=None, fill=(1, 1, 1))
                    page.insert_textbox(
                        rect,
                        text,
                        fontsize=smaller_size,
                        fontname=mapped_font,
                        color=self.config.text_color,
                        align=fitz.TEXT_ALIGN_LEFT,
                    )
            except Exception:
                # Fallback: simple text insert
                page.insert_text(
                    (bbox.x0, bbox.y0 + font_size),
                    text,
                    fontsize=font_size,
                    fontname="helv",
                )
    
    def _render_replace_mode(self, page, blocks: list[Block]):
        """Replace original text with translations."""
        import fitz
        
        # First pass: white out all text areas
        for block in blocks:
            if not block.bbox or not block.is_translatable:
                continue
            
            bbox = block.bbox
            rect = fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
            
            # Expand rect slightly to cover fully
            rect = rect + fitz.Rect(-2, -2, 2, 2)
            
            # Draw white rectangle
            page.draw_rect(rect, color=None, fill=(1, 1, 1))
        
        # Second pass: insert translations
        self._render_overlay_mode(page, blocks)


def render_pdf(
    document: Document,
    source_pdf: str | Path,
    output_path: str | Path,
    mode: str = "overlay",
) -> Path:
    """Convenience function to render a translated document.
    
    Args:
        document: Translated Document
        source_pdf: Path to original PDF
        output_path: Output path for translated PDF
        mode: Rendering mode ("overlay" or "replace")
        
    Returns:
        Path to output PDF
    """
    config = RenderConfig(mode=mode)
    renderer = PDFRenderer(config)
    return renderer.render(document, source_pdf, output_path)

