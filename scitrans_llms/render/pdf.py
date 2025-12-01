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

Fixes:
- Better handling of blocks without bbox
- Improved text replacement using redaction
- Fallback to full page replacement when bbox unavailable
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict

from scitrans_llms.models import Document, Block, BoundingBox


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
        
        # Check if we have proper bbox info
        has_bbox = any(block.bbox for block in document.all_blocks)
        
        # Always use text search-based replacement for reliability
        # This finds source text in PDF and replaces with translated text
        # More reliable than bbox-based replacement which can miss text
        self._render_by_text_search_all(doc, document)
        
        # Save output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(output_path))
        doc.close()
        
        return output_path
    
    def _group_blocks_by_page(self, document: Document) -> Dict[int, List[Block]]:
        """Group blocks by their page number."""
        blocks_by_page: Dict[int, List[Block]] = {}
        
        for block in document.all_blocks:
            if block.bbox:
                page = block.bbox.page
                if page not in blocks_by_page:
                    blocks_by_page[page] = []
                blocks_by_page[page].append(block)
        
        return blocks_by_page
    
    def _render_by_text_search_all(self, doc, document: Document):
        """Replace ALL source text with translations using text search.
        
        This method searches for source text in the PDF and replaces it.
        It handles blocks that may contain merged text by searching for
        sentence-level chunks.
        """
        import fitz
        import re
        
        # Build a mapping of source -> translated for all blocks
        translations = {}
        for block in document.all_blocks:
            if not block.is_translatable or not block.translated_text:
                continue
            if block.translated_text.strip() == block.source_text.strip():
                continue
            
            source = block.source_text.strip()
            if len(source) >= 3:
                translations[source] = block.translated_text
        
        # Process each page
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # For each translation pair
            for source, translated in translations.items():
                # Try different search strategies
                search_keys = []
                
                # 1. Try first 60 chars (for titles/headings)
                if len(source) <= 60:
                    search_keys.append(source)
                else:
                    search_keys.append(source[:60])
                
                # 2. Try first sentence or phrase
                sentences = re.split(r'[.!?]\s+', source)
                if sentences and len(sentences[0]) >= 10:
                    search_keys.append(sentences[0][:50])
                
                # 3. For text starting with common patterns
                words = source.split()[:8]
                if len(words) >= 3:
                    search_keys.append(' '.join(words))
                
                # Try each search key
                for search_key in search_keys:
                    if len(search_key) < 5:
                        continue
                    
                    instances = page.search_for(search_key)
                    
                    for rect in instances:
                        try:
                            font_size = max(6, min(10, rect.height * 0.65))
                            
                            # Use proportional translated text
                            trans_text = translated
                            if len(translated) > 150:
                                # Truncate long translations proportionally
                                ratio = len(search_key) / len(source)
                                trans_len = int(len(translated) * ratio * 1.2)
                                trans_text = translated[:trans_len]
                            
                            page.add_redact_annot(
                                rect,
                                text=trans_text[:200],
                                fontname="helv",
                                fontsize=font_size,
                                fill=(1, 1, 1),
                                text_color=(0, 0, 0),
                            )
                        except Exception:
                            continue
            
            # Apply all redactions for this page
            try:
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
            except Exception:
                pass
    
    def _render_by_text_search(self, doc, document: Document):
        """Render by finding source text in PDF and replacing it.
        
        This is the fallback when blocks don't have bbox info.
        Uses PyMuPDF redaction for cleaner text replacement.
        """
        import fitz
        
        replaced_count = 0
        for block in document.all_blocks:
            if not block.is_translatable:
                continue
            if not block.translated_text:
                continue
            if block.translated_text.strip() == block.source_text.strip():
                continue
            
            # Search for source text in all pages
            source_text = block.source_text.strip()
            if len(source_text) < 3:
                continue
            
            # Use first 80 chars for search (avoid matching issues with long text)
            search_text = source_text[:80] if len(source_text) > 80 else source_text
            
            # Try to find and replace on each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Search for text instances
                text_instances = page.search_for(search_text, quads=True)
                
                if not text_instances:
                    continue
                
                for quad in text_instances:
                    try:
                        # Convert quad to rect
                        rect = quad.rect if hasattr(quad, 'rect') else fitz.Rect(quad)
                        
                        # Expand rect slightly for better coverage
                        rect = rect + fitz.Rect(-2, -2, 2, 2)
                        
                        # Use redaction annotation for cleaner removal
                        page.add_redact_annot(rect, fill=(1, 1, 1))
                        page.apply_redactions()
                        
                        # Calculate font size from rect height
                        font_size = max(6, min(12, rect.height * 0.75))
                        
                        # Insert translated text
                        try:
                            page.insert_textbox(
                                rect,
                                block.translated_text,
                                fontsize=font_size,
                                fontname="helv",
                                color=(0, 0, 0),
                                align=fitz.TEXT_ALIGN_LEFT,
                            )
                        except:
                            # Fallback: insert as text at position
                            page.insert_text(
                                fitz.Point(rect.x0, rect.y0 + font_size),
                                block.translated_text[:200],  # Limit length
                                fontsize=font_size,
                                fontname="helv",
                            )
                        
                        replaced_count += 1
                        break  # Only replace first instance of this block
                    except Exception as e:
                        # Continue to next instance
                        continue
                
                if replaced_count > 0:
                    break  # Found on this page, don't search other pages
        
        return replaced_count
    
    def _render_overlay_mode(self, page, blocks: list[Block]):
        """Render translations as overlay using redaction for proper text replacement."""
        import fitz
        
        for block in blocks:
            if not block.translated_text or not block.bbox:
                continue
            
            if not block.is_translatable:
                continue
            
            # Skip if translation is same as source
            if block.translated_text.strip() == block.source_text.strip():
                continue
            
            bbox = block.bbox
            rect = fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
            
            # Get font info
            font_size = block.metadata.get("font_size", min(11, rect.height * 0.7))
            
            text = block.translated_text
            
            try:
                # Use redaction for clean text replacement
                page.add_redact_annot(
                    rect,
                    text=text,
                    fontname="helv",
                    fontsize=font_size,
                    fill=(1, 1, 1),
                    text_color=(0, 0, 0),
                )
            except Exception:
                # Fallback: draw white rect and insert text
                page.draw_rect(rect, color=None, fill=(1, 1, 1))
                try:
                    page.insert_textbox(
                        rect,
                        text,
                        fontsize=font_size,
                        fontname="helv",
                        color=(0, 0, 0),
                    )
                except:
                    page.insert_text(
                        (bbox.x0, bbox.y0 + font_size),
                        text[:200],
                        fontsize=font_size,
                        fontname="helv",
                    )
        
        # Apply all redactions
        try:
            page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
        except Exception:
            pass
    
    def _render_replace_mode(self, page, blocks: list[Block]):
        """Replace original text with translations using redaction.
        
        This properly removes original text from the PDF text layer
        and inserts translated text in its place.
        """
        import fitz
        
        # Collect all areas to redact with their translations
        for block in blocks:
            if not block.bbox or not block.is_translatable:
                continue
            if not block.translated_text:
                continue
            # Skip if translation is same as source
            if block.translated_text.strip() == block.source_text.strip():
                continue
            
            bbox = block.bbox
            rect = fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
            
            # Expand rect slightly
            rect = rect + fitz.Rect(-1, -1, 1, 1)
            
            # Get font size from metadata or estimate from rect height
            font_size = block.metadata.get("font_size", min(11, rect.height * 0.7))
            
            try:
                # Add redaction annotation - this marks text for removal
                page.add_redact_annot(
                    rect,
                    text=block.translated_text,  # Replacement text
                    fontname="helv",
                    fontsize=font_size,
                    fill=(1, 1, 1),  # White background
                    text_color=(0, 0, 0),  # Black text
                )
            except Exception:
                # If redaction fails, use overlay method
                page.draw_rect(rect, color=None, fill=(1, 1, 1))
                try:
                    page.insert_textbox(
                        rect,
                        block.translated_text,
                        fontsize=font_size,
                        fontname="helv",
                        color=(0, 0, 0),
                    )
                except:
                    pass
        
        # Apply all redactions at once - this removes original text
        try:
            page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
        except Exception:
            pass  # Some PDFs may not support redaction


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

