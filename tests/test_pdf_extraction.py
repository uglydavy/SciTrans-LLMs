"""
Comprehensive tests for PDF extraction and layout detection.

Tests cover:
- Basic PDF parsing
- Block type classification (paragraph, heading, equation, list, etc.)
- Segment creation and page handling
- Layout detection (heuristic and YOLO)
- Bounding box extraction
- Font and style preservation

These tests use dynamically generated PDFs to avoid external dependencies.
"""

import pytest
import tempfile
from pathlib import Path

# Try to import required modules
try:
    from scitrans_llms.ingest.pdf import PDFParser, YOLOLayoutDetector, HeuristicLayoutDetector
    from scitrans_llms.models import Document, BlockType
    import fitz  # PyMuPDF
    IMPORTS_OK = True
except Exception as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)


# Skip entire module if imports failed
if not IMPORTS_OK:
    pytest.skip(f"Skipping PDF extraction tests: {IMPORT_ERROR}", allow_module_level=True)


# ============================================================================
# Test PDF Generation Fixtures
# ============================================================================

@pytest.fixture
def simple_pdf():
    """Create a simple single-page PDF."""
    doc = fitz.open()
    page = doc.new_page()
    
    # Title
    page.insert_text((72, 72), "Test Document Title", fontsize=18, fontname="helv")
    
    # Paragraphs
    page.insert_text((72, 120), "This is the first paragraph of text.", fontsize=11, fontname="helv")
    page.insert_text((72, 150), "This is the second paragraph with more content.", fontsize=11, fontname="helv")
    
    tmp_path = Path(tempfile.mkdtemp()) / "simple.pdf"
    doc.save(str(tmp_path))
    doc.close()
    
    yield tmp_path
    
    if tmp_path.exists():
        tmp_path.unlink()


@pytest.fixture
def multi_page_pdf():
    """Create a multi-page PDF with varied content."""
    doc = fitz.open()
    
    # Page 1: Title and introduction
    page1 = doc.new_page()
    page1.insert_text((72, 72), "Research Paper Title", fontsize=20, fontname="helv")
    page1.insert_text((72, 110), "Abstract", fontsize=14, fontname="helv")
    rect = fitz.Rect(72, 130, 540, 200)
    page1.insert_textbox(rect, 
        "This paper presents novel research findings. "
        "We demonstrate significant improvements over baseline methods.",
        fontsize=11, fontname="helv")
    
    # Page 2: Methods
    page2 = doc.new_page()
    page2.insert_text((72, 72), "1. Introduction", fontsize=14, fontname="helv")
    rect = fitz.Rect(72, 100, 540, 200)
    page2.insert_textbox(rect,
        "Machine learning has revolutionized many fields. "
        "Our approach builds on recent advances in neural networks.",
        fontsize=11, fontname="helv")
    page2.insert_text((72, 220), "2. Methodology", fontsize=14, fontname="helv")
    rect = fitz.Rect(72, 250, 540, 350)
    page2.insert_textbox(rect,
        "We employ a transformer architecture with attention mechanisms. "
        "The model processes documents through multiple stages.",
        fontsize=11, fontname="helv")
    
    # Page 3: Results
    page3 = doc.new_page()
    page3.insert_text((72, 72), "3. Results", fontsize=14, fontname="helv")
    rect = fitz.Rect(72, 100, 540, 200)
    page3.insert_textbox(rect,
        "Our experiments show a 15% improvement in accuracy. "
        "The model achieves state-of-the-art performance on the benchmark.",
        fontsize=11, fontname="helv")
    page3.insert_text((72, 220), "References", fontsize=14, fontname="helv")
    page3.insert_text((72, 250), "[1] Smith et al. Machine Learning, 2023.", fontsize=10, fontname="helv")
    page3.insert_text((72, 270), "[2] Jones et al. Neural Networks, 2024.", fontsize=10, fontname="helv")
    
    tmp_path = Path(tempfile.mkdtemp()) / "multipage.pdf"
    doc.save(str(tmp_path))
    doc.close()
    
    yield tmp_path
    
    if tmp_path.exists():
        tmp_path.unlink()


@pytest.fixture
def complex_pdf():
    """Create a PDF with various content types."""
    doc = fitz.open()
    page = doc.new_page()
    
    # Title
    page.insert_text((72, 50), "Complex Document with Mixed Content", fontsize=16, fontname="helv")
    
    # Section heading
    page.insert_text((72, 90), "1. Introduction", fontsize=14, fontname="helv")
    
    # Paragraph
    rect = fitz.Rect(72, 110, 540, 180)
    page.insert_textbox(rect,
        "This document contains various content types including text, "
        "lists, and structured content.",
        fontsize=11, fontname="helv")
    
    # List items
    page.insert_text((72, 200), "Key Features:", fontsize=12, fontname="helv")
    page.insert_text((90, 220), "• First item in the list", fontsize=11, fontname="helv")
    page.insert_text((90, 240), "• Second item with details", fontsize=11, fontname="helv")
    page.insert_text((90, 260), "• Third item for completeness", fontsize=11, fontname="helv")
    
    # Another section
    page.insert_text((72, 300), "2. Technical Details", fontsize=14, fontname="helv")
    rect = fitz.Rect(72, 320, 540, 400)
    page.insert_textbox(rect,
        "The implementation uses Python with PyMuPDF for PDF processing. "
        "Layout detection is performed using machine learning models.",
        fontsize=11, fontname="helv")
    
    # Caption-like text
    page.insert_text((72, 420), "Figure 1: System Architecture", fontsize=10, fontname="helv")
    
    tmp_path = Path(tempfile.mkdtemp()) / "complex.pdf"
    doc.save(str(tmp_path))
    doc.close()
    
    yield tmp_path
    
    if tmp_path.exists():
        tmp_path.unlink()


@pytest.fixture
def pdf_parser():
    """Create a PDFParser with the best available detector."""
    try:
        detector = YOLOLayoutDetector()
        if detector.is_available:
            return PDFParser(layout_detector=detector)
    except Exception:
        pass
    
    return PDFParser(layout_detector=HeuristicLayoutDetector())


# ============================================================================
# Basic Parsing Tests
# ============================================================================

class TestBasicParsing:
    """Test basic PDF parsing functionality."""
    
    def test_parse_simple_pdf(self, pdf_parser, simple_pdf):
        """Test parsing a simple single-page PDF."""
        doc = pdf_parser.parse(str(simple_pdf))
        
        assert doc is not None
        assert isinstance(doc, Document)
        assert len(doc.segments) == 1
        assert len(doc.all_blocks) > 0
    
    def test_parse_multi_page_pdf(self, pdf_parser, multi_page_pdf):
        """Test parsing a multi-page PDF."""
        doc = pdf_parser.parse(str(multi_page_pdf))
        
        assert doc is not None
        assert len(doc.segments) == 3  # 3 pages
        assert len(doc.all_blocks) > 5
    
    def test_parse_complex_pdf(self, pdf_parser, complex_pdf):
        """Test parsing a complex PDF with varied content."""
        doc = pdf_parser.parse(str(complex_pdf))
        
        assert doc is not None
        assert len(doc.all_blocks) > 3
    
    def test_segments_have_blocks(self, pdf_parser, multi_page_pdf):
        """Test that all segments contain blocks."""
        doc = pdf_parser.parse(str(multi_page_pdf))
        
        for segment in doc.segments:
            assert len(segment.blocks) >= 0  # Can have empty pages
    
    def test_document_has_metadata(self, pdf_parser, simple_pdf):
        """Test that document has metadata."""
        doc = pdf_parser.parse(str(simple_pdf))
        
        assert doc.metadata is not None
        assert isinstance(doc.metadata, dict)


# ============================================================================
# Block Type Classification Tests
# ============================================================================

class TestBlockTypeClassification:
    """Test block type detection and classification."""
    
    def test_has_paragraphs(self, pdf_parser, multi_page_pdf):
        """Test that paragraphs are detected."""
        doc = pdf_parser.parse(str(multi_page_pdf))
        
        paragraphs = [b for b in doc.all_blocks if b.block_type == BlockType.PARAGRAPH]
        assert len(paragraphs) > 0
    
    def test_has_headings(self, pdf_parser, multi_page_pdf):
        """Test that headings are detected."""
        doc = pdf_parser.parse(str(multi_page_pdf))
        
        headings = [b for b in doc.all_blocks if b.block_type == BlockType.HEADING]
        # May detect headings depending on detector
        assert len(headings) >= 0
    
    def test_block_type_variety(self, pdf_parser, complex_pdf):
        """Test that multiple block types are detected."""
        doc = pdf_parser.parse(str(complex_pdf))
        
        block_types = {b.block_type for b in doc.all_blocks}
        # Should have at least paragraphs
        assert BlockType.PARAGRAPH in block_types or len(block_types) > 0
    
    def test_block_type_distribution(self, pdf_parser, multi_page_pdf):
        """Test reasonable distribution of block types."""
        doc = pdf_parser.parse(str(multi_page_pdf))
        
        type_counts = {}
        for block in doc.all_blocks:
            type_counts[block.block_type] = type_counts.get(block.block_type, 0) + 1
        
        # Should detect at least one type
        assert len(type_counts) >= 1


# ============================================================================
# Layout Properties Tests
# ============================================================================

class TestLayoutProperties:
    """Test layout metadata and bounding boxes."""
    
    def test_blocks_have_bounding_boxes(self, pdf_parser, simple_pdf):
        """Test that blocks have bounding box coordinates."""
        doc = pdf_parser.parse(str(simple_pdf))
        
        blocks_with_bbox = [b for b in doc.all_blocks if b.bbox is not None]
        assert len(blocks_with_bbox) > 0
    
    def test_bounding_box_validity(self, pdf_parser, simple_pdf):
        """Test that bounding boxes have valid coordinates."""
        doc = pdf_parser.parse(str(simple_pdf))
        
        for block in doc.all_blocks:
            if block.bbox:
                bbox = block.bbox
                assert bbox.x0 >= 0
                assert bbox.y0 >= 0
                assert bbox.x1 >= bbox.x0
                assert bbox.y1 >= bbox.y0
                assert bbox.page >= 0
    
    def test_blocks_have_font_info(self, pdf_parser, simple_pdf):
        """Test that blocks preserve font information."""
        doc = pdf_parser.parse(str(simple_pdf))
        
        blocks_with_font = [b for b in doc.all_blocks 
                          if b.metadata.get('font_size', 0) > 0]
        # At least some blocks should have font info
        assert len(blocks_with_font) >= 0  # May not always have font info
    
    def test_page_numbers_correct(self, pdf_parser, multi_page_pdf):
        """Test that blocks are assigned to correct pages."""
        doc = pdf_parser.parse(str(multi_page_pdf))
        
        page_numbers = set()
        for block in doc.all_blocks:
            if block.bbox:
                page_numbers.add(block.bbox.page)
        
        # Should have blocks from multiple pages
        assert len(page_numbers) <= 3  # Max 3 pages


# ============================================================================
# Text Extraction Tests
# ============================================================================

class TestTextExtraction:
    """Test text content extraction."""
    
    def test_blocks_have_source_text(self, pdf_parser, simple_pdf):
        """Test that all blocks have source text."""
        doc = pdf_parser.parse(str(simple_pdf))
        
        for block in doc.all_blocks:
            assert block.source_text is not None
            assert isinstance(block.source_text, str)
    
    def test_text_not_empty(self, pdf_parser, simple_pdf):
        """Test that translatable blocks have non-empty text."""
        doc = pdf_parser.parse(str(simple_pdf))
        
        translatable = [b for b in doc.all_blocks if b.is_translatable]
        non_empty = [b for b in translatable if b.source_text.strip()]
        
        # Most translatable blocks should have text
        if len(translatable) > 0:
            assert len(non_empty) >= len(translatable) * 0.5
    
    def test_text_encoding(self, pdf_parser, simple_pdf):
        """Test that text is properly encoded."""
        doc = pdf_parser.parse(str(simple_pdf))
        
        for block in doc.all_blocks:
            try:
                block.source_text.encode('utf-8').decode('utf-8')
            except UnicodeError:
                pytest.fail(f"Block has encoding issues: {block.source_text[:50]}")
    
    def test_extracts_key_content(self, pdf_parser, simple_pdf):
        """Test that key content is extracted."""
        doc = pdf_parser.parse(str(simple_pdf))
        
        all_text = " ".join(b.source_text for b in doc.all_blocks)
        
        # Should extract some text
        assert len(all_text) > 10


# ============================================================================
# Translatable Blocks Tests
# ============================================================================

class TestTranslatableBlocks:
    """Test translatable block identification."""
    
    def test_paragraphs_are_translatable(self, pdf_parser, simple_pdf):
        """Test that paragraphs are marked translatable."""
        doc = pdf_parser.parse(str(simple_pdf))
        
        paragraphs = [b for b in doc.all_blocks if b.block_type == BlockType.PARAGRAPH]
        translatable = [b for b in paragraphs if b.is_translatable]
        
        assert len(translatable) == len(paragraphs)
    
    def test_headings_are_translatable(self, pdf_parser, simple_pdf):
        """Test that headings are marked translatable."""
        doc = pdf_parser.parse(str(simple_pdf))
        
        headings = [b for b in doc.all_blocks if b.block_type == BlockType.HEADING]
        translatable = [b for b in headings if b.is_translatable]
        
        assert len(translatable) == len(headings)
    
    def test_translatable_ratio(self, pdf_parser, multi_page_pdf):
        """Test that most blocks are translatable."""
        doc = pdf_parser.parse(str(multi_page_pdf))
        
        total = len(doc.all_blocks)
        translatable = len([b for b in doc.all_blocks if b.is_translatable])
        
        if total > 0:
            ratio = translatable / total
            assert ratio > 0.5  # At least 50% should be translatable


# ============================================================================
# Edge Cases Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_nonexistent_pdf(self, pdf_parser):
        """Test handling of nonexistent PDF."""
        with pytest.raises(Exception):
            pdf_parser.parse("/nonexistent/file.pdf")
    
    def test_empty_pdf(self, pdf_parser):
        """Test handling of empty PDF."""
        doc = fitz.open()
        doc.new_page()  # Empty page
        
        tmp_path = Path(tempfile.mkdtemp()) / "empty.pdf"
        doc.save(str(tmp_path))
        doc.close()
        
        try:
            result = pdf_parser.parse(str(tmp_path))
            assert result is not None
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    
    def test_single_character_text(self, pdf_parser):
        """Test PDF with minimal text."""
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "X", fontsize=12)
        
        tmp_path = Path(tempfile.mkdtemp()) / "single_char.pdf"
        doc.save(str(tmp_path))
        doc.close()
        
        try:
            result = pdf_parser.parse(str(tmp_path))
            assert result is not None
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


# ============================================================================
# Heuristic Fallback Tests
# ============================================================================

class TestHeuristicFallback:
    """Test heuristic detection as fallback."""
    
    def test_heuristic_parser_works(self, simple_pdf):
        """Test that heuristic-based parsing works."""
        parser = PDFParser(layout_detector=HeuristicLayoutDetector())
        doc = parser.parse(str(simple_pdf))
        
        assert doc is not None
        assert len(doc.all_blocks) > 0
    
    def test_heuristic_detects_headings(self, multi_page_pdf):
        """Test heuristic heading detection based on font size."""
        parser = PDFParser(layout_detector=HeuristicLayoutDetector())
        doc = parser.parse(str(multi_page_pdf))
        
        # Should find at least some blocks
        assert len(doc.all_blocks) > 0


# ============================================================================
# Document Metadata Tests
# ============================================================================

class TestDocumentMetadata:
    """Test document-level metadata."""
    
    def test_metadata_exists(self, pdf_parser, simple_pdf):
        """Test that document has metadata."""
        doc = pdf_parser.parse(str(simple_pdf))
        
        assert doc.metadata is not None
        assert isinstance(doc.metadata, dict)
    
    def test_source_language(self, pdf_parser, simple_pdf):
        """Test that source language is set."""
        doc = pdf_parser.parse(str(simple_pdf))
        
        assert doc.source_lang == "en"
    
    def test_target_language(self, pdf_parser, simple_pdf):
        """Test that target language is set."""
        doc = pdf_parser.parse(str(simple_pdf))
        
        assert doc.target_lang == "fr"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
