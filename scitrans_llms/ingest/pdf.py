"""
PDF parsing with layout detection.

This module extracts structured content from PDFs while preserving
layout information for faithful reconstruction.

Approach:
1. PyMuPDF text extraction with coordinate tracking
2. DocLayout-YOLO for visual layout detection (required)
3. MinerU (magic-pdf) as high-quality fallback if YOLO/text extraction fails

The parser produces a Document with:
- Blocks for each content region (paragraph, heading, figure, etc.)
- Bounding boxes for layout preservation
- Font and style metadata for rendering
"""

from __future__ import annotations

import io
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Iterator

from scitrans_llms.models import (
    Document, Segment, Block, BlockType, BoundingBox
)

try:
    import magic_pdf
    MINERU_AVAILABLE = True
except ImportError:
    MINERU_AVAILABLE = False


@dataclass
class TextSpan:
    """A span of text with position and style information."""
    text: str
    bbox: BoundingBox
    font_name: str = ""
    font_size: float = 12.0
    is_bold: bool = False
    is_italic: bool = False
    color: tuple = (0, 0, 0)
    
    @property
    def is_likely_heading(self) -> bool:
        """Heuristic: larger or bold text is likely a heading."""
        return self.font_size > 14 or self.is_bold


@dataclass
class PageContent:
    """Extracted content from a single PDF page."""
    page_num: int
    width: float
    height: float
    spans: list[TextSpan] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)
    image_bytes: Optional[bytes] = None


class LayoutDetector(ABC):
    """Abstract base for layout detection strategies."""
    
    @abstractmethod
    def detect(self, page: PageContent) -> list[tuple[BoundingBox, BlockType]]:
        """Detect layout regions on a page.
        
        Returns list of (bounding_box, block_type) pairs.
        """
        pass


class HeuristicLayoutDetector(LayoutDetector):
    """Rule-based layout detection without ML models.
    
    Uses heuristics based on:
    - Font size (headings are larger)
    - Position (headers/footers at page edges)
    - Text patterns (equations contain math symbols)
    - Spacing (paragraphs are separated by whitespace)
    """
    
    def __init__(
        self,
        heading_size_threshold: float = 14.0,
        header_margin: float = 50.0,
        footer_margin: float = 50.0,
    ):
        self.heading_size_threshold = heading_size_threshold
        self.header_margin = header_margin
        self.footer_margin = footer_margin
    
    def detect(self, page: PageContent) -> list[tuple[BoundingBox, BlockType]]:
        """Classify spans into block types based on heuristics."""
        results = []
        
        for span in page.spans:
            block_type = self._classify_span(span, page)
            results.append((span.bbox, block_type))
        
        return results
    
    def _classify_span(self, span: TextSpan, page: PageContent) -> BlockType:
        """Classify a single text span."""
        text = span.text.strip()
        bbox = span.bbox
        
        # Check for page header/footer
        if bbox.y0 < self.header_margin:
            return BlockType.HEADER
        if bbox.y0 > page.height - self.footer_margin:
            return BlockType.FOOTER
        
        # Check for equations (LaTeX patterns)
        if self._looks_like_equation(text):
            return BlockType.EQUATION
        
        # Check for code (monospace, special patterns)
        if self._looks_like_code(text, span.font_name):
            return BlockType.CODE
        
        # Check for section headings with numbering (higher priority)
        if self._looks_like_numbered_section(text):
            return BlockType.HEADING
        
        # Check for headings (font size, bold, short text)
        if span.is_likely_heading and len(text) < 200:
            return BlockType.HEADING
        
        # Check for list items
        if self._looks_like_list_item(text):
            return BlockType.LIST_ITEM
        
        # Check for captions (starts with Figure/Table)
        if self._looks_like_caption(text):
            return BlockType.CAPTION
        
        # Check for references
        if self._looks_like_reference(text):
            return BlockType.REFERENCE
        
        # Default: paragraph
        return BlockType.PARAGRAPH
    
    def _looks_like_numbered_section(self, text: str) -> bool:
        """Check if text looks like a numbered section heading.
        
        Detects patterns like:
        - I. Introduction
        - 1.1 Background
        - 2. Methodology
        - 3.5 Results
        - Chapter 1: Title
        """
        patterns = [
            r'^[IVXLCDM]+\.\s+[A-Z]',  # Roman numerals: I. Introduction
            r'^\d+\.\d+\s+[A-Z]',       # Hierarchical: 1.1 Title
            r'^\d+\.\s+[A-Z]',          # Simple: 1. Title
            r'^Chapter\s+\d+',          # Chapter 1
            r'^Section\s+\d+',          # Section 1
            r'^\d+\s+[A-Z][A-Z\s]{2,}', # Number then TITLE
        ]
        # Must be relatively short to be a heading
        if len(text) > 150:
            return False
        return any(re.match(p, text) for p in patterns)
    
    def _looks_like_equation(self, text: str) -> bool:
        """Check if text looks like a mathematical equation."""
        # LaTeX delimiters
        if re.search(r'\$.*\$|\\\[|\\\]|\\begin\{equation', text):
            return True
        # High density of math symbols
        math_chars = set('∑∫∂∇αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ±×÷≤≥≠≈∞∈∉⊂⊃∪∩')
        if sum(1 for c in text if c in math_chars) > len(text) * 0.1:
            return True
        return False
    
    def _looks_like_code(self, text: str, font_name: str) -> bool:
        """Check if text looks like code."""
        # Monospace fonts
        if any(mono in font_name.lower() for mono in ['mono', 'courier', 'consolas']):
            return True
        # Code patterns
        if re.search(r'^\s*(def |class |import |from |if |for |while |return )', text):
            return True
        if re.search(r'[{}\[\]();]', text) and '=' in text:
            return True
        return False
    
    def _looks_like_list_item(self, text: str) -> bool:
        """Check if text is a list item or section heading.
        
        Enhanced to handle various numbering formats:
        - I. Introduction (Roman numerals)
        - 1.1 Something (hierarchical)
        - 2. Something (simple numbered)
        - 3.5 Something (decimal sections)
        - a) Something (lettered)
        """
        patterns = [
            r'^[\s]*[-•●○◦▪▸►]\s',      # Bullet points
            r'^[\s]*\d+[.)]\s',          # Numbered lists: 1. 2. 1) 2)
            r'^[\s]*\d+\.\d+[.)]*\s',    # Hierarchical: 1.1 1.2 2.1 3.5
            r'^[\s]*\d+\.\d+\.\d+[.)]*\s', # Deep hierarchical: 1.1.1
            r'^[\s]*[a-z][.)]\s',        # Lettered lists: a. b. a) b)
            r'^[\s]*[ivxlcdm]+[.)]\s',   # Roman numerals (lowercase): i. ii. iii.
            r'^[\s]*[IVXLCDM]+[.)]\s',   # Roman numerals (uppercase): I. II. III.
            r'^[\s]*\([a-z0-9]+\)\s',    # Parenthesized: (a) (1)
            r'^[\s]*\[[a-z0-9]+\]\s',    # Bracketed: [1] [a]
        ]
        return any(re.match(p, text, re.IGNORECASE) for p in patterns)
    
    def _looks_like_caption(self, text: str) -> bool:
        """Check if text is a figure/table caption."""
        return bool(re.match(
            r'^(Figure|Fig\.|Table|Tab\.|Listing|Algorithm)\s*\d',
            text,
            re.IGNORECASE
        ))
    
    def _looks_like_reference(self, text: str) -> bool:
        """Check if text is a bibliography reference."""
        # Starts with [number] or number.
        if re.match(r'^\[\d+\]|\d+\.\s+[A-Z]', text):
            return True
        # Contains DOI or arXiv
        if 'doi:' in text.lower() or 'arxiv:' in text.lower():
            return True
        return False


class YOLOLayoutDetector(LayoutDetector):
    """Layout detection using DocLayout-YOLO.
    
    This provides more accurate detection of:
    - Figures and images
    - Tables
    - Multi-column layouts
    - Complex document structures
    
    Requires: ultralytics, model weights
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model if available."""
        try:
            from ultralytics import YOLO
            if self.model_path and Path(self.model_path).exists():
                self.model = YOLO(self.model_path)
            else:
                # Try default location
                default_path = Path(__file__).parent.parent / "data" / "layout" / "layout_model.pt"
                if default_path.exists():
                    self.model = YOLO(str(default_path))
        except ImportError:
            self.model = None  # ultralytics not installed
    
    @property
    def is_available(self) -> bool:
        return self.model is not None
    
    def detect(self, page: PageContent) -> list[tuple[BoundingBox, BlockType]]:
        """Run YOLO detection on page image."""
        if not self.model:
            raise RuntimeError(
                "DocLayout-YOLO model not loaded. Install `ultralytics` and ensure "
                "data/layout/layout_model.pt is present."
            )
        if not page.image_bytes:
            raise RuntimeError("Page image bytes are required for YOLO detection.")
        
        from PIL import Image
        
        image = Image.open(io.BytesIO(page.image_bytes))
        detections = []
        
        results = self.model.predict(image, verbose=False)
        for result in results:
            if not hasattr(result, "boxes") or result.boxes is None:
                continue
            xyxy = result.boxes.xyxy.tolist()
            classes = result.boxes.cls.tolist()
            names = result.names or {}
            
            for coords, cls_idx in zip(xyxy, classes):
                label = names.get(int(cls_idx))
                if not label:
                    continue
                block_type = self.LABEL_MAP.get(label.lower())
                if not block_type:
                    continue
                x0, y0, x1, y1 = coords
                detections.append((
                    BoundingBox(
                        x0=float(x0),
                        y0=float(y0),
                        x1=float(x1),
                        y1=float(y1),
                        page=page.page_num,
                    ),
                    block_type,
                ))
        
        return detections
    
    # Class mapping from YOLO labels to BlockType
    LABEL_MAP = {
        "text": BlockType.PARAGRAPH,
        "title": BlockType.HEADING,
        "figure": BlockType.FIGURE,
        "table": BlockType.TABLE,
        "formula": BlockType.EQUATION,
        "list": BlockType.LIST_ITEM,
        "caption": BlockType.CAPTION,
        "footnote": BlockType.FOOTNOTE,
        "header": BlockType.HEADER,
        "footer": BlockType.FOOTER,
    }


class PDFParser:
    """Main PDF parser combining text extraction and layout detection.
    
    Usage:
        parser = PDFParser()
        doc = parser.parse("paper.pdf")
        
        for block in doc.all_blocks:
            print(f"{block.block_type.name}: {block.source_text[:50]}...")
    """
    
    def __init__(
        self,
        layout_detector: Optional[LayoutDetector] = None,
        extract_images: bool = True,
        merge_spans: bool = True,
    ):
        self.layout_detector = layout_detector or YOLOLayoutDetector()
        self.extract_images = extract_images
        self.merge_spans = merge_spans
    
    def parse(
        self,
        pdf_path: str | Path,
        pages: Optional[list[int]] = None,
        source_lang: str = "en",
        target_lang: str = "fr",
    ) -> Document:
        """Parse a PDF file into a Document.
        
        Args:
            pdf_path: Path to PDF file
            pages: Optional list of page numbers (0-indexed)
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Document with extracted blocks and layout info
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF is required. Install with: pip install PyMuPDF")
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        doc = fitz.open(str(pdf_path))
        
        # Determine pages to process
        if pages is None:
            page_nums = list(range(len(doc)))
        else:
            page_nums = [p for p in pages if 0 <= p < len(doc)]
        
        segments = []
        
        for page_num in page_nums:
            page = doc[page_num]
            page_content = self._extract_page(page, page_num)
            segment = self._page_to_segment(page_content, page_num)
            if segment.blocks:  # Only add non-empty segments
                segments.append(segment)
        
        doc.close()
        
        return Document(
            segments=segments,
            source_lang=source_lang,
            target_lang=target_lang,
            title=pdf_path.stem,
            metadata={
                "source_file": str(pdf_path),
                "total_pages": len(page_nums),
            }
        )
    
    def _extract_page(self, page, page_num: int) -> PageContent:
        """Extract text spans and images from a page."""
        import fitz
        
        content = PageContent(
            page_num=page_num,
            width=page.rect.width,
            height=page.rect.height,
        )
        
        # Render page once for YOLO detection
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(1.2, 1.2))
            content.image_bytes = pix.tobytes("png")
        except Exception:
            content.image_bytes = None
        
        # Extract text with detailed info
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        
        for block in blocks:
            if block["type"] == 0:  # Text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                        
                        bbox_coords = span.get("bbox", [0, 0, 0, 0])
                        content.spans.append(TextSpan(
                            text=text,
                            bbox=BoundingBox(
                                x0=bbox_coords[0],
                                y0=bbox_coords[1],
                                x1=bbox_coords[2],
                                y1=bbox_coords[3],
                                page=page_num,
                            ),
                            font_name=span.get("font", ""),
                            font_size=span.get("size", 12.0),
                            is_bold="bold" in span.get("font", "").lower(),
                            is_italic="italic" in span.get("font", "").lower(),
                            color=span.get("color", 0),
                        ))
            
            elif block["type"] == 1 and self.extract_images:  # Image block
                bbox = block.get("bbox", [0, 0, 0, 0])
                content.images.append({
                    "bbox": BoundingBox(
                        x0=bbox[0], y0=bbox[1],
                        x1=bbox[2], y1=bbox[3],
                        page=page_num,
                    ),
                    "width": block.get("width", 0),
                    "height": block.get("height", 0),
                })
        
        return content
    
    def _page_to_segment(self, page: PageContent, page_num: int) -> Segment:
        """Convert extracted page content to a Segment."""
        blocks = []
        
        # Group spans into logical blocks
        if self.merge_spans:
            grouped_spans = self._group_spans(page.spans)
        else:
            grouped_spans = [[s] for s in page.spans]
        
        # Detect layout and classify blocks
        layout_results = self.layout_detector.detect(page)
        
        for span_group in grouped_spans:
            if not span_group:
                continue
            
            # Merge text from spans
            text = " ".join(s.text for s in span_group)
            
            # Use first span's bbox (could compute union)
            bbox = span_group[0].bbox
            
            # Find matching layout classification
            block_type = self._find_block_type(bbox, layout_results, span_group[0])
            
            # Extract font metadata and structural markers
            metadata = {
                "font_name": span_group[0].font_name,
                "font_size": span_group[0].font_size,
                "is_bold": span_group[0].is_bold,
                "is_italic": span_group[0].is_italic,
            }
            
            # Extract and preserve structural markers (numbering, bullets, etc.)
            structural_info = self._extract_structural_markers(text)
            if structural_info:
                metadata.update(structural_info)
            
            blocks.append(Block(
                source_text=text,
                block_type=block_type,
                bbox=bbox,
                metadata=metadata,
            ))
        
        # Add image blocks
        for img in page.images:
            blocks.append(Block(
                source_text="[IMAGE]",
                block_type=BlockType.FIGURE,
                bbox=img["bbox"],
                metadata={"width": img["width"], "height": img["height"]},
            ))
        
        # Sort blocks by reading order (top-to-bottom, left-to-right)
        blocks.sort(key=lambda b: (b.bbox.y0 if b.bbox else 0, b.bbox.x0 if b.bbox else 0))
        
        return Segment(
            blocks=blocks,
            title=f"Page {page_num + 1}",
            metadata={"page_num": page_num},
        )
    
    def _group_spans(self, spans: list[TextSpan]) -> list[list[TextSpan]]:
        """Group spans that belong to the same logical block.
        
        Uses proximity and formatting to determine grouping.
        """
        if not spans:
            return []
        
        # Sort by position
        sorted_spans = sorted(spans, key=lambda s: (s.bbox.y0, s.bbox.x0))
        
        groups = []
        current_group = [sorted_spans[0]]
        
        for span in sorted_spans[1:]:
            prev = current_group[-1]
            
            # Check if this span continues the current group
            vertical_gap = span.bbox.y0 - prev.bbox.y1
            same_size = abs(span.font_size - prev.font_size) < 1
            
            # Group if close vertically and same formatting
            if vertical_gap < span.font_size * 1.5 and same_size:
                current_group.append(span)
            else:
                groups.append(current_group)
                current_group = [span]
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _find_block_type(
        self,
        bbox: BoundingBox,
        layout_results: list[tuple[BoundingBox, BlockType]],
        span: TextSpan,
    ) -> BlockType:
        """Find the block type for a bbox using layout results."""
        # Check layout detection results
        for layout_bbox, block_type in layout_results:
            if self._bboxes_overlap(bbox, layout_bbox):
                return block_type
        
        # Fallback: use span properties
        if span.is_likely_heading:
            return BlockType.HEADING
        
        return BlockType.PARAGRAPH
    
    def _bboxes_overlap(self, a: BoundingBox, b: BoundingBox) -> bool:
        """Check if two bounding boxes overlap significantly."""
        # Compute intersection
        x0 = max(a.x0, b.x0)
        y0 = max(a.y0, b.y0)
        x1 = min(a.x1, b.x1)
        y1 = min(a.y1, b.y1)
        
        if x1 <= x0 or y1 <= y0:
            return False
        
        intersection = (x1 - x0) * (y1 - y0)
        area_a = a.width * a.height
        
        # Overlap if intersection > 50% of smaller area
        return intersection > area_a * 0.5
    
    def _extract_structural_markers(self, text: str) -> dict:
        """Extract structural markers for alignment preservation.
        
        This identifies:
        - Section numbers (1.1, I., etc.)
        - Bullet types (•, -, *, etc.)
        - List numbers
        - Indentation level (from bbox)
        
        Returns:
            Dictionary with structural metadata
        """
        result = {}
        
        # Check for section numbering
        section_patterns = [
            (r'^([IVXLCDM]+)\.\s+', 'roman_upper'),
            (r'^([ivxlcdm]+)\.\s+', 'roman_lower'),
            (r'^(\d+(?:\.\d+)+)\s+', 'numeric_hierarchical'),  # 1.1, 3.5, 1.2.3
            (r'^(\d+)[.)]\s+', 'numeric'),  # 1. or 1) or 2.
            (r'^([a-z])[.)]\s+', 'letter_lower'),
            (r'^([A-Z])[.)]\s+', 'letter_upper'),
        ]
        
        for pattern, marker_type in section_patterns:
            match = re.match(pattern, text)
            if match:
                result['section_number'] = match.group(1)
                result['numbering_style'] = marker_type
                result['has_numbering'] = True
                break
        
        # Check for bullet points
        bullet_pattern = r'^([-•●○◦▪▸►*+])\s+'
        bullet_match = re.match(bullet_pattern, text)
        if bullet_match:
            result['bullet_char'] = bullet_match.group(1)
            result['has_bullet'] = True
        
        # Calculate indentation level from leading spaces
        leading_spaces = len(text) - len(text.lstrip())
        if leading_spaces > 0:
            result['indent_level'] = leading_spaces // 2  # Assume 2 spaces per level
            result['indent_chars'] = leading_spaces
        
        return result


class MinerUPDFParser:
    """PDF parser using minerU (magic-pdf) for robust extraction.
    
    MinerU provides better handling of:
    - Complex layouts (multi-column, tables)
    - Mathematical formulas (LaTeX extraction)
    - Reading order detection
    - Structure preservation
    
    Falls back to PyMuPDF if minerU is not available.
    """
    
    def __init__(self):
        self.available = MINERU_AVAILABLE
    
    def parse(
        self,
        pdf_path: str | Path,
        pages: Optional[list[int]] = None,
        source_lang: str = "en",
        target_lang: str = "fr",
    ) -> Document:
        """Parse PDF using minerU.
        
        Args:
            pdf_path: Path to PDF file
            pages: Optional list of page numbers (0-indexed)
            source_lang: Source language
            target_lang: Target language
            
        Returns:
            Parsed Document
        """
        if not self.available:
            # Fall back to PyMuPDF parser
            return PDFParser().parse(pdf_path, pages, source_lang, target_lang)
        
        try:
            from magic_pdf.pipe.UNIPipe import UNIPipe
            from magic_pdf.pipe.OCRPipe import OCRPipe
            import json
            
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF not found: {pdf_path}")
            
            # Read PDF bytes
            pdf_bytes = pdf_path.read_bytes()
            
            # Try UNI pipe first (faster, no OCR)
            try:
                pipe = UNIPipe(pdf_bytes, {})
                pipe.pipe_classify()
                pipe.pipe_parse()
                result = pipe.pipe_mk_markdown()
            except Exception:
                # Fall back to OCR pipe if needed
                pipe = OCRPipe(pdf_bytes, {})
                pipe.pipe_classify()
                pipe.pipe_parse()
                result = pipe.pipe_mk_markdown()
            
            # Convert minerU output to our Document format
            return self._convert_mineru_result(result, pdf_path, source_lang, target_lang, pages)
            
        except Exception as e:
            # If minerU fails, fall back to PyMuPDF
            print(f"minerU extraction failed: {e}, falling back to PyMuPDF")
            return PDFParser().parse(pdf_path, pages, source_lang, target_lang)
    
    def _convert_mineru_result(
        self,
        result: dict,
        pdf_path: Path,
        source_lang: str,
        target_lang: str,
        pages: Optional[list[int]] = None,
    ) -> Document:
        """Convert minerU output to Document format."""
        segments = []
        
        # minerU returns structured markdown
        # Parse it into our Block/Segment structure
        content = result.get('content', '')
        
        # Split by pages or sections
        sections = content.split('\n\n')
        
        current_blocks = []
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # Classify the block type based on content
            block_type = self._classify_mineru_section(section)
            
            block = Block(
                source_text=section,
                block_type=block_type,
                metadata={"source": "mineru"}
            )
            current_blocks.append(block)
        
        # Create a single segment with all blocks
        if current_blocks:
            segments.append(Segment(
                blocks=current_blocks,
                title=pdf_path.stem,
            ))
        
        return Document(
            segments=segments,
            source_lang=source_lang,
            target_lang=target_lang,
            title=pdf_path.stem,
            metadata={
                "source_file": str(pdf_path),
                "extractor": "mineru",
            }
        )
    
    def _classify_mineru_section(self, text: str) -> BlockType:
        """Classify a section from minerU output."""
        text = text.strip()
        
        # Check for markdown headings
        if text.startswith('#'):
            return BlockType.HEADING
        
        # Check for equations ($$...$$)
        if text.startswith('$$') and text.endswith('$$'):
            return BlockType.EQUATION
        
        # Check for code blocks (```...```)
        if text.startswith('```') and text.endswith('```'):
            return BlockType.CODE
        
        # Check for captions
        if re.match(r'^(Figure|Fig\.|Table|Tab\.)', text, re.IGNORECASE):
            return BlockType.CAPTION
        
        # Check for lists
        if re.match(r'^[\s]*[-•*]\s', text) or re.match(r'^[\s]*\d+[.)]\s', text):
            return BlockType.LIST_ITEM
        
        # Default: paragraph
        return BlockType.PARAGRAPH


class PDFMinerParser:
    """Enhanced PDF parser using pdfminer.six for better layout extraction.
    
    PDFMiner provides more accurate text extraction with:
    - Better text block segmentation
    - Accurate bounding boxes
    - Reading order detection
    - Font information
    """
    
    def parse(
        self,
        pdf_path: str | Path,
        pages: Optional[list[int]] = None,
        source_lang: str = "en",
        target_lang: str = "fr",
    ) -> Document:
        """Parse PDF using pdfminer.six with enhanced layout analysis."""
        try:
            from pdfminer.high_level import extract_pages
            from pdfminer.layout import (
                LAParams, LTTextBox, LTTextLine, LTChar, 
                LTFigure, LTTextBoxHorizontal
            )
        except ImportError as e:
            import warnings
            warnings.warn(f"pdfminer not available ({e}), falling back to PyMuPDF")
            return PDFParser().parse(pdf_path, pages, source_lang, target_lang)
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Configure pdfminer for better layout
        laparams = LAParams(
            line_margin=0.5,
            word_margin=0.1,
            char_margin=2.0,
            boxes_flow=0.5,
            detect_vertical=False,
        )
        
        all_blocks = []
        
        # Extract pages
        for page_num, page_layout in enumerate(extract_pages(str(pdf_path), laparams=laparams)):
            # Filter pages if specified
            if pages is not None and page_num not in pages:
                continue
            
            page_width = page_layout.width
            page_height = page_layout.height
            
            # Process each element on the page
            for element in page_layout:
                if isinstance(element, LTTextBox):
                    # Extract text and bounding box
                    text = element.get_text().strip()
                    if not text:
                        continue
                    
                    # PDFMiner uses bottom-left origin, convert to top-left
                    x0, y0, x1, y1 = element.bbox
                    bbox = BoundingBox(
                        x0=x0,
                        y0=page_height - y1,  # Convert to top-left origin
                        x1=x1,
                        y1=page_height - y0,
                        page=page_num,
                    )
                    
                    # Detect block type and font info
                    block_type, font_info = self._analyze_text_box(element)
                    
                    block = Block(
                        source_text=text,
                        block_type=block_type,
                        bbox=bbox,
                        metadata={
                            "font_size": font_info.get("size", 11),
                            "font_name": font_info.get("name", ""),
                            "is_bold": font_info.get("bold", False),
                            "source": "pdfminer",
                        }
                    )
                    all_blocks.append(block)
        
        # Create segments (group by page)
        segments = []
        blocks_by_page = {}
        for block in all_blocks:
            page = block.bbox.page if block.bbox else 0
            if page not in blocks_by_page:
                blocks_by_page[page] = []
            blocks_by_page[page].append(block)
        
        for page_num in sorted(blocks_by_page.keys()):
            segments.append(Segment(
                blocks=blocks_by_page[page_num],
                title=f"Page {page_num + 1}",
            ))
        
        return Document(
            segments=segments,
            source_lang=source_lang,
            target_lang=target_lang,
            title=pdf_path.stem,
            metadata={
                "source_file": str(pdf_path),
                "extractor": "pdfminer",
                "total_blocks": len(all_blocks),
            }
        )
    
    def _analyze_text_box(self, text_box) -> tuple:
        """Analyze a text box to determine type and font info."""
        from pdfminer.layout import LTChar, LTAnno
        
        text = text_box.get_text().strip()
        font_sizes = []
        font_names = []
        
        # Collect font info from characters
        for line in text_box:
            for char in line:
                if isinstance(char, LTChar):
                    font_sizes.append(char.size)
                    font_names.append(char.fontname)
        
        # Get dominant font info
        avg_size = sum(font_sizes) / len(font_sizes) if font_sizes else 11
        font_name = font_names[0] if font_names else ""
        is_bold = "Bold" in font_name or "bold" in font_name
        
        font_info = {
            "size": avg_size,
            "name": font_name,
            "bold": is_bold,
        }
        
        # Classify block type
        block_type = BlockType.PARAGRAPH
        
        # Heading detection
        if avg_size > 14 or is_bold:
            if len(text) < 100:
                block_type = BlockType.HEADING
        
        # Section number detection
        if re.match(r'^[\d.]+\s+[A-Z]', text) or re.match(r'^[IVX]+\.\s', text):
            block_type = BlockType.HEADING
        
        # Abstract/Keywords detection
        if text.upper().startswith(('ABSTRACT', 'RÉSUMÉ', 'KEYWORDS')):
            block_type = BlockType.HEADING
        
        # Equation detection
        if re.search(r'\$\$.*\$\$|\\\[.*\\\]', text):
            block_type = BlockType.EQUATION
        
        # Caption detection
        if re.match(r'^(Figure|Fig\.|Table|Tab\.)\s*\d', text, re.IGNORECASE):
            block_type = BlockType.CAPTION
        
        # Reference detection
        if re.match(r'^\[\d+\]', text):
            block_type = BlockType.REFERENCE
        
        return block_type, font_info


def parse_pdf(
    pdf_path: str | Path,
    pages: Optional[list[int]] = None,
    source_lang: str = "en",
    target_lang: str = "fr",
) -> Document:
    """Convenience function to parse a PDF.
    
    Args:
        pdf_path: Path to PDF file
        pages: Optional list of page numbers (0-indexed)
        source_lang: Source language
        target_lang: Target language
        
    Returns:
        Parsed Document
    """
    last_error: Exception | None = None
    
    # Preferred: PyMuPDF + pdfminer text with DocLayout-YOLO classification
    try:
        parser = PDFParser(layout_detector=YOLOLayoutDetector())
        return parser.parse(pdf_path, pages, source_lang, target_lang)
    except Exception as e:
        last_error = e
    
    # High-quality fallback: minerU (requires magic-pdf)
    if MINERU_AVAILABLE:
        try:
            return MinerUPDFParser().parse(pdf_path, pages, source_lang, target_lang)
        except Exception as e:
            last_error = e
    
    raise RuntimeError(
        "PDF extraction failed. Ensure DocLayout-YOLO weights are present "
        "and magic-pdf (minerU) is installed."
    ) from last_error


# Compatibility layer for old API
class _LegacyBlock:
    """Legacy block format for compatibility with old analyzer code."""
    def __init__(self, block: Block):
        self.text = block.source_text
        self.page_index = block.bbox.page if block.bbox else 0
        self.kind = block.block_type.name.lower() if block.block_type else "paragraph"
        if block.bbox:
            self.bbox = (block.bbox.x0, block.bbox.y0, block.bbox.x1, block.bbox.y1)
        else:
            self.bbox = (0.0, 0.0, 0.0, 0.0)


def extract_blocks(
    pdf_path: str | Path,
    pages: Optional[list[int]] = None,
    notes: Optional[dict] = None,
) -> tuple[list[_LegacyBlock], dict]:
    """Extract blocks from PDF (legacy compatibility function).
    
    This function provides compatibility with old code that expects
    blocks with .text, .page_index, .kind, and .bbox attributes.
    
    Args:
        pdf_path: Path to PDF file
        pages: Optional list of page numbers (0-indexed)
        notes: Optional notes dict (ignored, kept for compatibility)
        
    Returns:
        Tuple of (list of legacy blocks, metadata dict)
    """
    doc = parse_pdf(pdf_path, pages=pages)
    legacy_blocks = [_LegacyBlock(block) for block in doc.all_blocks]
    metadata = {
        "total_pages": len(doc.segments),
        "total_blocks": len(legacy_blocks),
    }
    return legacy_blocks, metadata

