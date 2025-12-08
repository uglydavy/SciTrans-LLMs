"""
Tests for PDF rendering and output generation.

Tests cover:
- PDF creation from translated documents
- Layout preservation during rendering
- Font mapping and size adjustment
- Rendering modes (bbox, search, hybrid)
- Error handling and fallbacks
"""

import pytest
import tempfile
from pathlib import Path

from scitrans_llms.models import Document, Segment, Block, BlockType, BoundingBox
from scitrans_llms.render.pdf import PDFRenderer, RenderConfig, render_pdf, FontMapper


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_document():
    """Create a sample translated document for rendering tests."""
    blocks = [
        Block(
            source_text="Introduction",
            translated_text="Introduction",
            block_type=BlockType.HEADING,
            bbox=BoundingBox(x0=72, y0=72, x1=300, y1=90, page=0),
            metadata={"font_size": 16}
        ),
        Block(
            source_text="This is a test paragraph with some content.",
            translated_text="Ceci est un paragraphe de test avec du contenu.",
            block_type=BlockType.PARAGRAPH,
            bbox=BoundingBox(x0=72, y0=100, x1=540, y1=150, page=0),
            metadata={"font_size": 11}
        ),
        Block(
            source_text="Machine learning is important.",
            translated_text="L'apprentissage automatique est important.",
            block_type=BlockType.PARAGRAPH,
            bbox=BoundingBox(x0=72, y0=160, x1=540, y1=200, page=0),
            metadata={"font_size": 11}
        ),
    ]
    
    segment = Segment(blocks=blocks, title="Page 1")
    
    return Document(
        segments=[segment],
        source_lang="en",
        target_lang="fr",
        title="Test Document",
    )


@pytest.fixture
def source_pdf():
    """Create a temporary source PDF for rendering tests."""
    try:
        import fitz
    except ImportError:
        pytest.skip("PyMuPDF required for rendering tests")
    
    doc = fitz.open()
    page = doc.new_page()
    
    # Add content matching the sample document
    page.insert_text((72, 85), "Introduction", fontsize=16, fontname="helv")
    page.insert_text((72, 120), "This is a test paragraph with some content.", fontsize=11, fontname="helv")
    page.insert_text((72, 180), "Machine learning is important.", fontsize=11, fontname="helv")
    
    # Save to temp file
    tmp_path = Path(tempfile.mkdtemp()) / "source.pdf"
    doc.save(str(tmp_path))
    doc.close()
    
    yield tmp_path
    
    # Cleanup
    if tmp_path.exists():
        tmp_path.unlink()


# ============================================================================
# FontMapper Tests
# ============================================================================

class TestFontMapper:
    """Test font mapping functionality."""
    
    def test_map_times_new_roman(self):
        mapper = FontMapper()
        assert mapper.map_font("Times New Roman") == "Times-Roman"
        assert mapper.map_font("TimesNewRoman") == "Times-Roman"
    
    def test_map_arial(self):
        mapper = FontMapper()
        assert mapper.map_font("Arial") == "Helvetica"
    
    def test_map_courier(self):
        mapper = FontMapper()
        assert mapper.map_font("Courier New") == "Courier"
        assert mapper.map_font("CourierNew") == "Courier"
    
    def test_fallback_to_serif(self):
        mapper = FontMapper()
        # Unknown font should fall back to serif
        result = mapper.map_font("UnknownSerifFont")
        assert result == "Times-Roman"
    
    def test_fallback_to_sans(self):
        mapper = FontMapper()
        # Sans-serif patterns
        assert mapper.map_font("SomeGothicFont") == "Helvetica"
        assert mapper.map_font("MySansFont") == "Helvetica"
    
    def test_fallback_to_mono(self):
        mapper = FontMapper()
        # Monospace patterns - must contain "mono", "courier", "consolas", or "code"
        assert mapper.map_font("MyMonoFont") == "Courier"
        assert mapper.map_font("ConsolasFont") == "Courier"
        assert mapper.map_font("CodeFont") == "Courier"
    
    def test_caching(self):
        mapper = FontMapper()
        # First call
        result1 = mapper.map_font("TestFont")
        # Second call should use cache
        result2 = mapper.map_font("TestFont")
        assert result1 == result2
    
    def test_adjust_size_for_french(self):
        mapper = FontMapper()
        # French text is ~15% longer, so size should be reduced
        adjusted = mapper.adjust_size_for_language(12.0, "en", "fr")
        assert adjusted < 12.0
        assert adjusted > 10.0  # Not too small
    
    def test_adjust_size_for_german(self):
        mapper = FontMapper()
        # German is ~20% longer
        adjusted = mapper.adjust_size_for_language(12.0, "en", "de")
        assert adjusted < 12.0
    
    def test_no_adjustment_for_same_language(self):
        mapper = FontMapper()
        adjusted = mapper.adjust_size_for_language(12.0, "en", "en")
        assert adjusted == 12.0


# ============================================================================
# RenderConfig Tests
# ============================================================================

class TestRenderConfig:
    """Test render configuration."""
    
    def test_default_config(self):
        config = RenderConfig()
        assert config.mode == "hybrid"
        assert config.preserve_images is True
        assert config.min_font_size == 6.0
        assert config.max_font_size == 14.0
    
    def test_custom_config(self):
        config = RenderConfig(
            mode="bbox",
            min_font_size=8.0,
            max_font_size=12.0,
        )
        assert config.mode == "bbox"
        assert config.min_font_size == 8.0
        assert config.max_font_size == 12.0


# ============================================================================
# PDFRenderer Tests
# ============================================================================

class TestPDFRenderer:
    """Test PDF rendering functionality."""
    
    def test_renderer_initialization(self):
        renderer = PDFRenderer()
        assert renderer.config is not None
        assert renderer.font_mapper is not None
    
    def test_renderer_with_custom_config(self):
        config = RenderConfig(mode="search")
        renderer = PDFRenderer(config)
        assert renderer.config.mode == "search"
    
    def test_render_creates_output_file(self, sample_document, source_pdf):
        renderer = PDFRenderer()
        output_path = source_pdf.parent / "output.pdf"
        
        result = renderer.render(sample_document, source_pdf, output_path)
        
        assert result.exists()
        assert result.stat().st_size > 0
        
        # Cleanup
        if output_path.exists():
            output_path.unlink()
    
    def test_render_preserves_page_count(self, sample_document, source_pdf):
        import fitz
        
        renderer = PDFRenderer()
        output_path = source_pdf.parent / "output.pdf"
        
        renderer.render(sample_document, source_pdf, output_path)
        
        # Check page count
        original = fitz.open(str(source_pdf))
        rendered = fitz.open(str(output_path))
        
        assert len(rendered) == len(original)
        
        original.close()
        rendered.close()
        
        # Cleanup
        if output_path.exists():
            output_path.unlink()
    
    def test_render_bbox_mode(self, sample_document, source_pdf):
        config = RenderConfig(mode="bbox")
        renderer = PDFRenderer(config)
        output_path = source_pdf.parent / "output_bbox.pdf"
        
        result = renderer.render(sample_document, source_pdf, output_path)
        assert result.exists()
        
        # Cleanup
        if output_path.exists():
            output_path.unlink()
    
    def test_render_search_mode(self, sample_document, source_pdf):
        config = RenderConfig(mode="search")
        renderer = PDFRenderer(config)
        output_path = source_pdf.parent / "output_search.pdf"
        
        result = renderer.render(sample_document, source_pdf, output_path)
        assert result.exists()
        
        # Cleanup
        if output_path.exists():
            output_path.unlink()
    
    def test_render_hybrid_mode(self, sample_document, source_pdf):
        config = RenderConfig(mode="hybrid")
        renderer = PDFRenderer(config)
        output_path = source_pdf.parent / "output_hybrid.pdf"
        
        result = renderer.render(sample_document, source_pdf, output_path)
        assert result.exists()
        
        # Cleanup
        if output_path.exists():
            output_path.unlink()
    
    def test_render_nonexistent_source(self, sample_document):
        renderer = PDFRenderer()
        
        with pytest.raises(FileNotFoundError):
            renderer.render(
                sample_document,
                "/nonexistent/source.pdf",
                "/tmp/output.pdf"
            )
    
    def test_render_creates_parent_dirs(self, sample_document, source_pdf):
        renderer = PDFRenderer()
        output_path = source_pdf.parent / "subdir" / "nested" / "output.pdf"
        
        result = renderer.render(sample_document, source_pdf, output_path)
        assert result.exists()
        
        # Cleanup
        if output_path.exists():
            output_path.unlink()
        output_path.parent.rmdir()
        output_path.parent.parent.rmdir()
    
    def test_stats_tracking(self, sample_document, source_pdf):
        renderer = PDFRenderer()
        output_path = source_pdf.parent / "output.pdf"
        
        renderer.render(sample_document, source_pdf, output_path)
        
        stats = renderer.stats
        assert "replaced" in stats
        assert "skipped" in stats
        assert "errors" in stats
        
        # Cleanup
        if output_path.exists():
            output_path.unlink()


# ============================================================================
# render_pdf Convenience Function Tests
# ============================================================================

class TestRenderPdfFunction:
    """Test the render_pdf convenience function."""
    
    def test_render_pdf_default_mode(self, sample_document, source_pdf):
        output_path = source_pdf.parent / "output.pdf"
        
        result = render_pdf(sample_document, source_pdf, output_path)
        
        assert result.exists()
        
        # Cleanup
        if output_path.exists():
            output_path.unlink()
    
    def test_render_pdf_custom_mode(self, sample_document, source_pdf):
        output_path = source_pdf.parent / "output.pdf"
        
        result = render_pdf(sample_document, source_pdf, output_path, mode="bbox")
        
        assert result.exists()
        
        # Cleanup
        if output_path.exists():
            output_path.unlink()


# ============================================================================
# Edge Cases
# ============================================================================

class TestRenderingEdgeCases:
    """Test edge cases in rendering."""
    
    def test_empty_document(self, source_pdf):
        """Test rendering a document with no blocks."""
        empty_doc = Document(
            segments=[Segment(blocks=[], title="Empty")],
            source_lang="en",
            target_lang="fr",
        )
        
        renderer = PDFRenderer()
        output_path = source_pdf.parent / "output_empty.pdf"
        
        result = renderer.render(empty_doc, source_pdf, output_path)
        assert result.exists()
        
        # Cleanup
        if output_path.exists():
            output_path.unlink()
    
    def test_blocks_without_bbox(self, source_pdf):
        """Test rendering blocks without bounding boxes."""
        doc = Document(
            segments=[Segment(
                blocks=[
                    Block(
                        source_text="Test text",
                        translated_text="Texte de test",
                        block_type=BlockType.PARAGRAPH,
                        bbox=None,  # No bbox
                    )
                ],
                title="Page 1"
            )],
            source_lang="en",
            target_lang="fr",
        )
        
        config = RenderConfig(mode="search")
        renderer = PDFRenderer(config)
        output_path = source_pdf.parent / "output_nobbox.pdf"
        
        result = renderer.render(doc, source_pdf, output_path)
        assert result.exists()
        
        # Cleanup
        if output_path.exists():
            output_path.unlink()
    
    def test_untranslated_blocks(self, source_pdf):
        """Test that untranslated blocks are skipped."""
        doc = Document(
            segments=[Segment(
                blocks=[
                    Block(
                        source_text="Test text",
                        translated_text=None,  # Not translated
                        block_type=BlockType.PARAGRAPH,
                        bbox=BoundingBox(x0=72, y0=100, x1=500, y1=120, page=0),
                    )
                ],
                title="Page 1"
            )],
            source_lang="en",
            target_lang="fr",
        )
        
        renderer = PDFRenderer()
        output_path = source_pdf.parent / "output_untrans.pdf"
        
        result = renderer.render(doc, source_pdf, output_path)
        assert result.exists()
        
        # Cleanup
        if output_path.exists():
            output_path.unlink()
    
    def test_same_source_and_translation(self, source_pdf):
        """Test that identical translations are skipped."""
        doc = Document(
            segments=[Segment(
                blocks=[
                    Block(
                        source_text="Test",
                        translated_text="Test",  # Same as source
                        block_type=BlockType.PARAGRAPH,
                        bbox=BoundingBox(x0=72, y0=100, x1=500, y1=120, page=0),
                    )
                ],
                title="Page 1"
            )],
            source_lang="en",
            target_lang="fr",
        )
        
        renderer = PDFRenderer()
        output_path = source_pdf.parent / "output_same.pdf"
        
        result = renderer.render(doc, source_pdf, output_path)
        
        # Should be skipped (same text), but file still created
        assert result.exists()
        assert renderer.stats["skipped"] >= 1
        
        # Cleanup
        if output_path.exists():
            output_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

