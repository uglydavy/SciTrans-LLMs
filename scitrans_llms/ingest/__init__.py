"""
Ingestion module for parsing PDF documents.

This module provides:
- PDF parsing with PyMuPDF
- Layout detection (DocLayout-YOLO or heuristic fallback)
- Block segmentation and classification
- Structure extraction (headings, paragraphs, tables, figures)

Thesis Contribution #1: Layout-preserving translation requires
accurate extraction of document structure.
"""

from scitrans_llms.ingest.pdf import (
    parse_pdf,
    PDFParser,
    LayoutDetector,
    HeuristicLayoutDetector,
    extract_blocks,
    SimpleBlock,
)

__all__ = [
    "parse_pdf",
    "PDFParser",
    "LayoutDetector",
    "HeuristicLayoutDetector",
    "extract_blocks",
    "SimpleBlock",
]
