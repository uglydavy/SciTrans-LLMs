"""
PDF rendering with layout preservation.

This module reconstructs translated PDFs by:
1. Starting from the original PDF (preserving images, layout)
2. Overlaying translated text at original positions
3. Matching fonts and sizes as closely as possible

Approaches:
- BBox mode: Use extracted bounding boxes for precise placement
- Search mode: Find source text and replace (fallback)
- Hybrid: Combine both for best results

Key improvements:
- Better text fitting with automatic size adjustment
- Support for multi-column layouts
- Graceful fallback when exact positioning fails
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Tuple

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
    """Maps source fonts to available fonts."""
    
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
    
    DEFAULT_SERIF = "Times-Roman"
    DEFAULT_SANS = "Helvetica"
    DEFAULT_MONO = "Courier"
    
    def __init__(self):
        self._cache: dict[str, str] = {}
    
    def map_font(self, source_font: str) -> str:
        """Map a source font name to an available font."""
        if source_font in self._cache:
            return self._cache[source_font]
        
        for pattern, replacement in self.SUBSTITUTIONS.items():
            if pattern.lower() in source_font.lower():
                self._cache[source_font] = replacement
                return replacement
        
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
        """Adjust font size for target language expansion."""
        expansion = {
            "fr": 1.15,
            "de": 1.20,
            "es": 1.10,
            "it": 1.10,
            "pt": 1.10,
        }
        
        if source_lang == "en" and target_lang in expansion:
            factor = expansion[target_lang]
            return max(size * 0.85, size / (factor ** 0.4))
        
        return size


@dataclass
class RenderConfig:
    """Configuration for PDF rendering."""
    mode: str = "hybrid"  # "bbox", "search", or "hybrid"
    preserve_images: bool = True
    adjust_font_size: bool = True
    min_font_size: float = 6.0
    max_font_size: float = 14.0
    fallback_font: str = "helv"
    text_color: tuple = (0, 0, 0)
    background_color: tuple = (1, 1, 1)


class PDFRenderer:
    """Render translated Document back to PDF with layout preservation.
    
    This is a core thesis contribution: maintaining document layout
    while replacing text content.
    """
    
    def __init__(self, config: Optional[RenderConfig] = None):
        self.config = config or RenderConfig()
        self.font_mapper = FontMapper()
        self._stats = {"replaced": 0, "skipped": 0, "errors": 0}
    
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
            import fitz
        except ImportError:
            raise ImportError("PyMuPDF is required. Install with: pip install PyMuPDF")
        
        source_pdf = Path(source_pdf)
        output_path = Path(output_path)
        
        if not source_pdf.exists():
            raise FileNotFoundError(f"Source PDF not found: {source_pdf}")
        
        # Reset stats
        self._stats = {"replaced": 0, "skipped": 0, "errors": 0}
        
        # Open source PDF
        doc = fitz.open(str(source_pdf))
        
        # Group blocks by page
        blocks_by_page = self._group_blocks_by_page(document)
        
        # Determine best rendering approach
        has_bbox = any(block.bbox for block in document.all_blocks if block.is_translatable)
        
        if self.config.mode == "bbox" and has_bbox:
            self._render_bbox_mode(doc, blocks_by_page)
        elif self.config.mode == "search":
            self._render_search_mode(doc, document)
        else:  # hybrid
            # Try bbox first, then search for remaining
            if has_bbox:
                self._render_bbox_mode(doc, blocks_by_page)
            self._render_search_mode(doc, document, only_unhandled=True)
        
        # Save output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(output_path), garbage=4, deflate=True)
        doc.close()
        
        return output_path
    
    @property
    def stats(self) -> dict:
        """Get rendering statistics."""
        return self._stats.copy()
    
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
    
    def _render_bbox_mode(self, doc, blocks_by_page: Dict[int, List[Block]]):
        """Render using bounding box positions."""
        import fitz
        
        for page_num, blocks in blocks_by_page.items():
            if page_num >= len(doc):
                continue
            
            page = doc[page_num]
            
            for block in blocks:
                if not block.is_translatable or not block.translated_text:
                    self._stats["skipped"] += 1
                    continue
                
                if block.translated_text.strip() == block.source_text.strip():
                    self._stats["skipped"] += 1
                    continue
                
                try:
                    self._replace_block_bbox(page, block)
                    self._stats["replaced"] += 1
                except Exception as e:
                    self._stats["errors"] += 1
            
            # Apply all redactions for this page
            try:
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
            except Exception:
                pass
    
    def _replace_block_bbox(self, page, block: Block):
        """Replace a single block using its bounding box."""
        import fitz
        
        bbox = block.bbox
        rect = fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
        
        # Expand rect slightly for better coverage
        rect = rect + fitz.Rect(-1, -1, 1, 1)
        
        # Calculate optimal font size
        font_size = self._calculate_font_size(block, rect)
        
        text = block.translated_text.strip()
        
        # Add redaction annotation with replacement text
        page.add_redact_annot(
            rect,
            text=text,
            fontname=self.config.fallback_font,
            fontsize=font_size,
            fill=self.config.background_color,
            text_color=self.config.text_color,
        )
    
    def _calculate_font_size(self, block: Block, rect) -> float:
        """Calculate optimal font size for text in rect."""
        # Get metadata font size if available
        meta_size = block.metadata.get("font_size", 0)
        
        # Estimate from rect height
        height_size = rect.height * 0.65
        
        # Use metadata if reasonable, otherwise estimate
        if 6 <= meta_size <= 16:
            size = meta_size
        else:
            size = height_size
        
        # Adjust for language expansion
        if self.config.adjust_font_size:
            size = self.font_mapper.adjust_size_for_language(size, "en", "fr")
        
        # Clamp to reasonable range
        return max(self.config.min_font_size, min(self.config.max_font_size, size))
    
    def _render_search_mode(self, doc, document: Document, only_unhandled: bool = False):
        """Render by searching for source text and replacing."""
        import fitz
        
        # Build translation mapping
        translations = {}
        for block in document.all_blocks:
            if not block.is_translatable or not block.translated_text:
                continue
            if block.translated_text.strip() == block.source_text.strip():
                continue
            
            # Skip if already handled via bbox
            if only_unhandled and block.bbox:
                continue
            
            source = block.source_text.strip()
            if len(source) >= 5:
                translations[source] = block.translated_text
        
        if not translations:
            return
        
        # Process each page
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            for source, translated in translations.items():
                self._search_and_replace(page, source, translated)
            
            # Apply redactions
            try:
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
            except Exception:
                pass
    
    def _search_and_replace(self, page, source: str, translated: str):
        """Search for source text and replace with translation."""
        import fitz
        
        # Generate search keys from source
        search_keys = self._generate_search_keys(source)
        
        for search_key in search_keys:
            if len(search_key) < 5:
                continue
            
            instances = page.search_for(search_key)
            
            for rect in instances:
                try:
                    font_size = max(6, min(11, rect.height * 0.7))
                    
                    # Get proportional translation
                    trans_text = self._get_proportional_text(
                        translated, source, search_key
                    )
                    
                    page.add_redact_annot(
                        rect,
                        text=trans_text,
                        fontname=self.config.fallback_font,
                        fontsize=font_size,
                        fill=self.config.background_color,
                        text_color=self.config.text_color,
                    )
                    self._stats["replaced"] += 1
                    return  # Only replace first match
                except Exception:
                    self._stats["errors"] += 1
    
    def _generate_search_keys(self, source: str) -> List[str]:
        """Generate search keys for finding source text."""
        keys = []
        
        # Full text for short sources
        if len(source) <= 80:
            keys.append(source)
        else:
            keys.append(source[:80])
        
        # First sentence
        sentences = re.split(r'[.!?]\s+', source)
        if sentences and len(sentences[0]) >= 15:
            keys.append(sentences[0][:60])
        
        # First few words
        words = source.split()[:10]
        if len(words) >= 4:
            keys.append(' '.join(words))
        
        return keys
    
    def _get_proportional_text(
        self,
        translated: str,
        source: str,
        search_key: str
    ) -> str:
        """Get proportional chunk of translated text."""
        if len(translated) <= 200:
            return translated
        
        # Calculate proportion
        ratio = len(search_key) / max(len(source), 1)
        target_len = int(len(translated) * ratio * 1.3)  # +30% for safety
        target_len = max(50, min(target_len, 300))
        
        return translated[:target_len]


def render_pdf(
    document: Document,
    source_pdf: str | Path,
    output_path: str | Path,
    mode: str = "hybrid",
) -> Path:
    """Convenience function to render a translated document.
    
    Args:
        document: Translated Document with blocks
        source_pdf: Path to original PDF
        output_path: Output path for translated PDF
        mode: Rendering mode ("bbox", "search", or "hybrid")
        
    Returns:
        Path to output PDF
    """
    config = RenderConfig(mode=mode)
    renderer = PDFRenderer(config)
    return renderer.render(document, source_pdf, output_path)
