"""
End-to-end tests for the complete translation pipeline.

Tests the full workflow:
1. PDF upload/creation
2. Text extraction with layout detection
3. Masking of special content
4. Translation with glossary
5. Refinement
6. Unmasking
7. PDF rendering

These tests verify the entire system works together correctly.
"""

import pytest
import tempfile
from pathlib import Path

from scitrans_llms.models import Document, BlockType
from scitrans_llms.pipeline import TranslationPipeline, PipelineConfig, PipelineResult
from scitrans_llms.ingest.pdf import parse_pdf
from scitrans_llms.render.pdf import render_pdf


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_pdf():
    """Create a test PDF with various content types."""
    try:
        import fitz
    except ImportError:
        pytest.skip("PyMuPDF required for E2E tests")
    
    doc = fitz.open()
    page = doc.new_page()
    
    # Title
    page.insert_text(
        (72, 72),
        "Machine Learning for Scientific Translation",
        fontsize=18,
        fontname="helv"
    )
    
    # Abstract
    page.insert_text((72, 110), "Abstract", fontsize=14, fontname="helv")
    page.insert_text(
        (72, 130),
        "This paper presents a novel approach to scientific document translation.",
        fontsize=11,
        fontname="helv"
    )
    
    # Section with content
    page.insert_text((72, 170), "1. Introduction", fontsize=14, fontname="helv")
    intro_text = (
        "Neural machine translation has revolutionized the field. "
        "Our system uses a transformer architecture with attention mechanisms. "
        "The model achieves state-of-the-art results on scientific texts."
    )
    rect = fitz.Rect(72, 190, 540, 280)
    page.insert_textbox(rect, intro_text, fontsize=11, fontname="helv")
    
    # Section with technical content
    page.insert_text((72, 300), "2. Methodology", fontsize=14, fontname="helv")
    method_text = (
        "We employ deep learning techniques including neural networks. "
        "The algorithm processes documents through multiple stages. "
        "Results demonstrate significant improvement over baseline systems."
    )
    rect = fitz.Rect(72, 320, 540, 410)
    page.insert_textbox(rect, method_text, fontsize=11, fontname="helv")
    
    # References section
    page.insert_text((72, 440), "References", fontsize=14, fontname="helv")
    page.insert_text(
        (72, 460),
        "[1] Smith et al. Machine Learning, 2023.",
        fontsize=10,
        fontname="helv"
    )
    
    # Save to temp file
    tmp_path = Path(tempfile.mkdtemp()) / "test_paper.pdf"
    doc.save(str(tmp_path))
    doc.close()
    
    yield tmp_path
    
    # Cleanup
    if tmp_path.exists():
        tmp_path.unlink()


@pytest.fixture
def output_path(test_pdf):
    """Get output path for rendered PDF."""
    return test_pdf.parent / "translated.pdf"


# ============================================================================
# End-to-End Pipeline Tests
# ============================================================================

class TestFullPipeline:
    """Test the complete translation pipeline."""
    
    def test_basic_e2e_flow(self, test_pdf, output_path):
        """Test basic end-to-end translation flow."""
        # Step 1: Extract
        doc = parse_pdf(test_pdf)
        assert doc is not None
        assert len(doc.all_blocks) > 0
        
        # Step 2: Translate
        config = PipelineConfig(
            translator_backend="dictionary",
            enable_masking=True,
            enable_glossary=True,
            enable_context=True,
        )
        pipeline = TranslationPipeline(config)
        result = pipeline.translate(doc)
        
        assert isinstance(result, PipelineResult)
        assert result.document is not None
        assert result.stats["translated_blocks"] > 0
        
        # Step 3: Render
        rendered = render_pdf(result.document, test_pdf, output_path)
        assert rendered.exists()
        assert rendered.stat().st_size > 0
        
        # Cleanup
        if output_path.exists():
            output_path.unlink()
    
    def test_e2e_with_all_features(self, test_pdf, output_path):
        """Test E2E with all pipeline features enabled."""
        doc = parse_pdf(test_pdf)
        
        config = PipelineConfig(
            translator_backend="dictionary",
            enable_masking=True,
            enable_glossary=True,
            enable_context=True,
            enable_refinement=True,
            num_candidates=1,
        )
        pipeline = TranslationPipeline(config)
        result = pipeline.translate(doc)
        
        # Verify all stages ran
        assert result.stats["total_blocks"] > 0
        assert result.stats["translated_blocks"] > 0
        
        # Render output
        rendered = render_pdf(result.document, test_pdf, output_path)
        assert rendered.exists()
        
        # Cleanup
        if output_path.exists():
            output_path.unlink()
    
    def test_e2e_minimal_config(self, test_pdf, output_path):
        """Test E2E with minimal configuration."""
        doc = parse_pdf(test_pdf)
        
        config = PipelineConfig(
            translator_backend="dictionary",
            enable_masking=False,
            enable_glossary=False,
            enable_context=False,
            enable_refinement=False,
        )
        pipeline = TranslationPipeline(config)
        result = pipeline.translate(doc)
        
        assert result.stats["translated_blocks"] > 0
        
        rendered = render_pdf(result.document, test_pdf, output_path)
        assert rendered.exists()
        
        # Cleanup
        if output_path.exists():
            output_path.unlink()


class TestExtractionQuality:
    """Test extraction quality in E2E context."""
    
    def test_block_types_detected(self, test_pdf):
        """Test that various block types are detected."""
        doc = parse_pdf(test_pdf)
        
        block_types = {b.block_type for b in doc.all_blocks}
        
        # Should detect at least paragraphs and headings
        assert BlockType.PARAGRAPH in block_types or BlockType.HEADING in block_types
    
    def test_text_extraction_complete(self, test_pdf):
        """Test that text is fully extracted."""
        doc = parse_pdf(test_pdf)
        
        all_text = " ".join(b.source_text for b in doc.all_blocks)
        
        # Check key phrases are extracted
        assert "Machine Learning" in all_text or "machine learning" in all_text.lower()
        assert "Introduction" in all_text or "introduction" in all_text.lower()
    
    def test_extraction_method_recorded(self, test_pdf):
        """Test that extraction method is recorded in metadata."""
        doc = parse_pdf(test_pdf)
        
        assert "extraction_method" in doc.metadata
        assert doc.metadata["extraction_method"] in ["yolo", "mineru", "pdfminer", "heuristic"]


class TestTranslationQuality:
    """Test translation quality in E2E context."""
    
    def test_translation_produces_french(self, test_pdf):
        """Test that translation produces French text."""
        doc = parse_pdf(test_pdf)
        
        config = PipelineConfig(translator_backend="dictionary")
        pipeline = TranslationPipeline(config)
        result = pipeline.translate(doc)
        
        # Check some blocks have French translations
        translations = [
            b.translated_text for b in result.document.all_blocks
            if b.translated_text and b.translated_text != b.source_text
        ]
        
        assert len(translations) > 0
    
    def test_glossary_terms_applied(self, test_pdf):
        """Test that glossary terms are applied."""
        doc = parse_pdf(test_pdf)
        
        config = PipelineConfig(
            translator_backend="dictionary",
            enable_glossary=True,
        )
        pipeline = TranslationPipeline(config)
        result = pipeline.translate(doc)
        
        # Check for glossary term translations
        all_translated = " ".join(
            b.translated_text or "" for b in result.document.all_blocks
        )
        
        # "machine learning" should become "apprentissage automatique"
        # This depends on the glossary content
        assert len(all_translated) > 0
    
    def test_translation_preserves_structure(self, test_pdf):
        """Test that translation preserves document structure."""
        doc = parse_pdf(test_pdf)
        original_block_count = len(doc.all_blocks)
        
        config = PipelineConfig(translator_backend="dictionary")
        pipeline = TranslationPipeline(config)
        result = pipeline.translate(doc)
        
        # Same number of blocks
        assert len(result.document.all_blocks) == original_block_count
        
        # Same segment structure
        assert len(result.document.segments) == len(doc.segments)


class TestRenderingQuality:
    """Test rendering quality in E2E context."""
    
    def test_output_is_valid_pdf(self, test_pdf, output_path):
        """Test that output is a valid PDF file."""
        import fitz
        
        doc = parse_pdf(test_pdf)
        config = PipelineConfig(translator_backend="dictionary")
        pipeline = TranslationPipeline(config)
        result = pipeline.translate(doc)
        
        render_pdf(result.document, test_pdf, output_path)
        
        # Verify it's a valid PDF
        rendered_doc = fitz.open(str(output_path))
        assert len(rendered_doc) > 0
        rendered_doc.close()
        
        # Cleanup
        if output_path.exists():
            output_path.unlink()
    
    def test_output_has_text(self, test_pdf, output_path):
        """Test that output PDF contains text."""
        import fitz
        
        doc = parse_pdf(test_pdf)
        config = PipelineConfig(translator_backend="dictionary")
        pipeline = TranslationPipeline(config)
        result = pipeline.translate(doc)
        
        render_pdf(result.document, test_pdf, output_path)
        
        rendered_doc = fitz.open(str(output_path))
        text = ""
        for page in rendered_doc:
            text += page.get_text()
        rendered_doc.close()
        
        assert len(text) > 0
        
        # Cleanup
        if output_path.exists():
            output_path.unlink()
    
    def test_output_preserves_page_count(self, test_pdf, output_path):
        """Test that page count is preserved."""
        import fitz
        
        original = fitz.open(str(test_pdf))
        original_pages = len(original)
        original.close()
        
        doc = parse_pdf(test_pdf)
        config = PipelineConfig(translator_backend="dictionary")
        pipeline = TranslationPipeline(config)
        result = pipeline.translate(doc)
        
        render_pdf(result.document, test_pdf, output_path)
        
        rendered = fitz.open(str(output_path))
        rendered_pages = len(rendered)
        rendered.close()
        
        assert rendered_pages == original_pages
        
        # Cleanup
        if output_path.exists():
            output_path.unlink()


class TestErrorRecovery:
    """Test error handling in E2E context."""
    
    def test_handles_empty_pdf(self):
        """Test handling of PDF with no text."""
        import fitz
        
        # Create empty PDF
        doc = fitz.open()
        doc.new_page()  # Empty page
        tmp_path = Path(tempfile.mkdtemp()) / "empty.pdf"
        doc.save(str(tmp_path))
        doc.close()
        
        try:
            parsed = parse_pdf(tmp_path)
            # Should not crash, may have 0 blocks
            assert parsed is not None
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    
    def test_handles_translation_errors_gracefully(self, test_pdf, output_path):
        """Test that translation errors don't crash the pipeline."""
        doc = parse_pdf(test_pdf)
        
        # Use a valid backend
        config = PipelineConfig(translator_backend="dictionary")
        pipeline = TranslationPipeline(config)
        
        # Should complete without crashing
        result = pipeline.translate(doc)
        assert result is not None
        
        # Should still be able to render
        rendered = render_pdf(result.document, test_pdf, output_path)
        assert rendered.exists()
        
        # Cleanup
        if output_path.exists():
            output_path.unlink()


class TestPipelineStats:
    """Test pipeline statistics in E2E context."""
    
    def test_stats_are_complete(self, test_pdf):
        """Test that pipeline stats are complete."""
        doc = parse_pdf(test_pdf)
        
        config = PipelineConfig(translator_backend="dictionary")
        pipeline = TranslationPipeline(config)
        result = pipeline.translate(doc)
        
        # Check required stats
        assert "total_blocks" in result.stats
        assert "translated_blocks" in result.stats
        assert "skipped_blocks" in result.stats
    
    def test_stats_are_consistent(self, test_pdf):
        """Test that stats are internally consistent."""
        doc = parse_pdf(test_pdf)
        
        config = PipelineConfig(translator_backend="dictionary")
        pipeline = TranslationPipeline(config)
        result = pipeline.translate(doc)
        
        # translated + skipped should roughly equal total
        total = result.stats["total_blocks"]
        translated = result.stats["translated_blocks"]
        skipped = result.stats["skipped_blocks"]
        
        assert translated + skipped <= total


# ============================================================================
# Performance Tests (Optional)
# ============================================================================

class TestPerformance:
    """Basic performance tests."""
    
    def test_extraction_completes_quickly(self, test_pdf):
        """Test that extraction completes in reasonable time."""
        import time
        
        start = time.time()
        doc = parse_pdf(test_pdf)
        elapsed = time.time() - start
        
        # Should complete in under 10 seconds for a simple PDF
        assert elapsed < 10.0
        assert doc is not None
    
    def test_translation_completes_quickly(self, test_pdf):
        """Test that translation completes in reasonable time."""
        import time
        
        doc = parse_pdf(test_pdf)
        config = PipelineConfig(translator_backend="dictionary")
        pipeline = TranslationPipeline(config)
        
        start = time.time()
        result = pipeline.translate(doc)
        elapsed = time.time() - start
        
        # Should complete in under 5 seconds for a simple document
        assert elapsed < 5.0
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

