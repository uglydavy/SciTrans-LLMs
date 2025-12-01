"""
Enhanced YOLO-based layout detection with page rendering.

This module provides a complete implementation of YOLO layout detection
that renders PDF pages to images and runs detection.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from scitrans_llms.models import BoundingBox, BlockType


def detect_layout_yolo(
    pdf_path: str | Path,
    page_num: int,
    model_path: Optional[str] = None,
    conf_threshold: float = 0.25,
) -> list[tuple[BoundingBox, BlockType]]:
    """Run YOLO layout detection on a PDF page.
    
    Args:
        pdf_path: Path to PDF file
        page_num: Page number (0-indexed)
        model_path: Optional path to YOLO model weights
        conf_threshold: Confidence threshold for detections
        
    Returns:
        List of (BoundingBox, BlockType) tuples
    """
    try:
        import fitz  # PyMuPDF
        from PIL import Image
        from scitrans_llms.yolo.predictor import LayoutPredictor
    except ImportError as e:
        print(f"Required libraries not available: {e}")
        return []
    
    try:
        # Open PDF and get page
        doc = fitz.open(str(pdf_path))
        if page_num >= len(doc):
            doc.close()
            return []
        
        page = doc[page_num]
        
        # Render page to image at high resolution
        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better detection
        pix = page.get_pixmap(matrix=mat)
        
        # Save to temporary image file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            pix.save(tmp_path)
        
        doc.close()
        
        # Run YOLO detection
        predictor = LayoutPredictor(model_path=Path(model_path) if model_path else None)
        detections = predictor.detect(tmp_path, conf=conf_threshold)
        
        # Convert detections to our format
        results = []
        for det in detections:
            # Scale bbox back from 2x resolution
            bbox = BoundingBox(
                x0=det.bbox[0] / 2.0,
                y0=det.bbox[1] / 2.0,
                x1=det.bbox[2] / 2.0,
                y1=det.bbox[3] / 2.0,
                page=page_num,
            )
            
            # Map YOLO label to BlockType
            block_type = _map_yolo_label(det.label)
            results.append((bbox, block_type))
        
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)
        
        return results
        
    except Exception as e:
        print(f"YOLO detection failed: {e}")
        return []


def _map_yolo_label(label: str) -> BlockType:
    """Map YOLO detection label to BlockType."""
    label_map = {
        "text": BlockType.PARAGRAPH,
        "title": BlockType.HEADING,
        "figure": BlockType.FIGURE,
        "figure_caption": BlockType.CAPTION,
        "table": BlockType.TABLE,
        "table_caption": BlockType.CAPTION,
        "formula": BlockType.EQUATION,
        "equation": BlockType.EQUATION,
        "list": BlockType.LIST_ITEM,
        "caption": BlockType.CAPTION,
        "footnote": BlockType.FOOTNOTE,
        "header": BlockType.HEADER,
        "footer": BlockType.FOOTER,
    }
    
    return label_map.get(label.lower(), BlockType.PARAGRAPH)


class YOLOLayoutParser:
    """PDF parser that uses YOLO for layout detection.
    
    This is an enhanced parser that:
    1. Renders PDF pages to images
    2. Runs YOLO detection to identify regions
    3. Extracts text from detected regions
    4. Preserves layout and structure
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
    
    def parse_page(
        self,
        pdf_path: str | Path,
        page_num: int,
    ) -> list[tuple[str, BlockType, BoundingBox]]:
        """Parse a single page using YOLO detection.
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)
            
        Returns:
            List of (text, block_type, bbox) tuples in reading order
        """
        try:
            import fitz
        except ImportError:
            return []
        
        # Get YOLO detections
        detections = detect_layout_yolo(pdf_path, page_num, self.model_path)
        
        if not detections:
            return []
        
        # Open PDF to extract text from detected regions
        doc = fitz.open(str(pdf_path))
        page = doc[page_num]
        
        results = []
        for bbox, block_type in detections:
            # Extract text from this region
            rect = fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
            text = page.get_text("text", clip=rect).strip()
            
            if text:
                results.append((text, block_type, bbox))
        
        doc.close()
        
        # Sort by reading order (top-to-bottom, left-to-right)
        results.sort(key=lambda x: (x[2].y0, x[2].x0))
        
        return results


__all__ = ['detect_layout_yolo', 'YOLOLayoutParser']

