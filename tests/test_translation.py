"""
Comprehensive tests for end-to-end translation pipeline.

Tests cover:
- Basic text translation
- Document translation with masking
- Glossary application
- Context handling
- Dictionary backend functionality
- Pipeline configuration
- Error handling
"""

import pytest
from pathlib import Path
from scitrans_llms.pipeline import (
    TranslationPipeline,
    PipelineConfig,
    translate_text,
)
from scitrans_llms.models import Document, BlockType
from scitrans_llms.masking import MaskConfig
from scitrans_llms.translate.glossary import Glossary, GlossaryEntry


class TestBasicTextTranslation:
    """Test basic text translation functionality."""
    
    def test_simple_text_translation(self):
        """Test translating simple text."""
        result = translate_text(
            "Hello world",
            source_lang="en",
            target_lang="fr",
            backend="free"
        )
        
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
        
    def test_scientific_text_translation(self):
        """Test translating scientific text."""
        text = "Machine learning is a subset of artificial intelligence."
        result = translate_text(
            text,
            source_lang="en",
            target_lang="fr",
            backend="free"
        )
        
        assert result is not None
        assert len(result) > 0
        
    def test_multiple_sentences(self):
        """Test translating multiple sentences."""
        text = "This is sentence one. This is sentence two."
        result = translate_text(
            text,
            source_lang="en",
            target_lang="fr",
            backend="free"
        )
        
        assert result is not None
        assert len(result) > 0


class TestDocumentTranslation:
    """Test document-level translation."""
    
    def test_simple_document(self):
        """Test translating simple document."""
        doc = Document.from_text(
            "This is a test document.",
            source_language="en"
        )
        
        config = PipelineConfig(
            source_language="en",
            target_language="fr",
            translator_backend="free"
        )
        
        pipeline = TranslationPipeline(config)
        result = pipeline.translate(doc)
        
        assert result.success
        assert result.translated_doc is not None
        assert len(result.translated_doc.all_blocks) > 0
        
    def test_multi_paragraph_document(self):
        """Test translating multi-paragraph document."""
        paragraphs = [
            "First paragraph.",
            "Second paragraph.",
            "Third paragraph."
        ]
        doc = Document.from_paragraphs(paragraphs, source_language="en")
        
        config = PipelineConfig(
            source_language="en",
            target_language="fr",
            translator_backend="free"
        )
        
        pipeline = TranslationPipeline(config)
        result = pipeline.translate(doc)
        
        assert result.success
        translated_blocks = [b for b in result.translated_doc.all_blocks 
                           if b.is_translatable and b.translated_text]
        assert len(translated_blocks) == 3
        
    def test_translation_preserves_structure(self):
        """Test that translation preserves document structure."""
        paragraphs = ["Para 1", "Para 2", "Para 3"]
        doc = Document.from_paragraphs(paragraphs, source_language="en")
        
        config = PipelineConfig(
            source_language="en",
            target_language="fr",
            translator_backend="free"
        )
        
        pipeline = TranslationPipeline(config)
        result = pipeline.translate(doc)
        
        # Structure should be preserved
        assert len(result.translated_doc.segments) == len(doc.segments)
        assert len(result.translated_doc.all_blocks) == len(doc.all_blocks)


class TestMaskingIntegration:
    """Test masking integration in translation pipeline."""
    
    def test_translate_with_math(self):
        """Test translating text with math equations."""
        text = "The formula $E=mc^2$ is Einstein's equation."
        doc = Document.from_text(text, source_language="en")
        
        config = PipelineConfig(
            source_language="en",
            target_language="fr",
            translator_backend="free",
            mask_config=MaskConfig(enabled=True)
        )
        
        pipeline = TranslationPipeline(config)
        result = pipeline.translate(doc)
        
        assert result.success
        # Math should be preserved in translated text
        translated_block = result.translated_doc.all_blocks[0]
        assert "$E=mc^2$" in translated_block.translated_text
        
    def test_translate_with_url(self):
        """Test translating text with URLs."""
        text = "Visit https://example.com for more information."
        doc = Document.from_text(text, source_language="en")
        
        config = PipelineConfig(
            source_language="en",
            target_language="fr",
            translator_backend="free",
            mask_config=MaskConfig(enabled=True)
        )
        
        pipeline = TranslationPipeline(config)
        result = pipeline.translate(doc)
        
        assert result.success
        # URL should be preserved
        translated_block = result.translated_doc.all_blocks[0]
        assert "https://example.com" in translated_block.translated_text
        
    def test_translate_with_code(self):
        """Test translating text with code snippets."""
        text = "Use the `print()` function to output text."
        doc = Document.from_text(text, source_language="en")
        
        config = PipelineConfig(
            source_language="en",
            target_language="fr",
            translator_backend="free",
            mask_config=MaskConfig(enabled=True)
        )
        
        pipeline = TranslationPipeline(config)
        result = pipeline.translate(doc)
        
        assert result.success
        # Code should be preserved
        translated_block = result.translated_doc.all_blocks[0]
        assert "`print()`" in translated_block.translated_text
        
    def test_masking_stats(self):
        """Test that masking stats are tracked."""
        text = "Formula $x=1$ and URL https://example.com"
        doc = Document.from_text(text, source_language="en")
        
        config = PipelineConfig(
            source_language="en",
            target_language="fr",
            translator_backend="free",
            mask_config=MaskConfig(enabled=True)
        )
        
        pipeline = TranslationPipeline(config)
        result = pipeline.translate(doc)
        
        assert result.success
        assert result.stats["masks_applied"] >= 2


class TestGlossaryIntegration:
    """Test glossary integration in translation."""
    
    def test_translate_with_glossary(self):
        """Test translating with custom glossary."""
        glossary = Glossary()
        glossary.add_entry(GlossaryEntry(
            source_term="machine learning",
            target_term="apprentissage automatique",
            source_language="en",
            target_language="fr"
        ))
        
        text = "Machine learning is important."
        doc = Document.from_text(text, source_language="en")
        
        config = PipelineConfig(
            source_language="en",
            target_language="fr",
            translator_backend="free",
            glossary=glossary
        )
        
        pipeline = TranslationPipeline(config)
        result = pipeline.translate(doc)
        
        assert result.success
        # Glossary term should be used
        translated = result.translated_doc.all_blocks[0].translated_text
        assert "apprentissage automatique" in translated.lower()
        
    def test_glossary_case_insensitive(self):
        """Test that glossary matching is case-insensitive."""
        glossary = Glossary()
        glossary.add_entry(GlossaryEntry(
            source_term="neural network",
            target_term="réseau neuronal",
            source_language="en",
            target_language="fr"
        ))
        
        text = "Neural Networks are powerful."  # Capitalized
        doc = Document.from_text(text, source_language="en")
        
        config = PipelineConfig(
            source_language="en",
            target_language="fr",
            translator_backend="free",
            glossary=glossary
        )
        
        pipeline = TranslationPipeline(config)
        result = pipeline.translate(doc)
        
        assert result.success
        # Should still match case-insensitively
        translated = result.translated_doc.all_blocks[0].translated_text
        assert "réseau neuronal" in translated.lower()


class TestDictionaryBackend:
    """Test dictionary-based translation backend."""
    
    def test_dictionary_translation(self):
        """Test basic dictionary translation."""
        text = "Hello world"
        result = translate_text(
            text,
            source_lang="en",
            target_lang="fr",
            backend="dictionary"
        )
        
        assert result is not None
        assert len(result) > 0
        
    def test_dictionary_with_unknown_words(self):
        """Test dictionary handling of unknown words."""
        # Use text likely to have some untranslated words
        text = "supercalifragilisticexpialidocious"
        result = translate_text(
            text,
            source_lang="en",
            target_lang="fr",
            backend="dictionary"
        )
        
        # Should still return something (might be unchanged)
        assert result is not None
        
    def test_dictionary_scientific_terms(self):
        """Test dictionary with scientific terms."""
        text = "neural network algorithm"
        result = translate_text(
            text,
            source_lang="en",
            target_lang="fr",
            backend="dictionary"
        )
        
        assert result is not None
        assert len(result) > 0


class TestPipelineConfiguration:
    """Test pipeline configuration options."""
    
    def test_default_config(self):
        """Test default pipeline configuration."""
        config = PipelineConfig(
            source_language="en",
            target_language="fr",
            translator_backend="free"
        )
        
        assert config.source_language == "en"
        assert config.target_language == "fr"
        assert config.translator_backend == "free"
        
    def test_custom_mask_config(self):
        """Test custom mask configuration."""
        mask_config = MaskConfig(enabled=False)
        config = PipelineConfig(
            source_language="en",
            target_language="fr",
            translator_backend="free",
            mask_config=mask_config
        )
        
        assert config.mask_config.enabled is False
        
    def test_context_window_config(self):
        """Test context window configuration."""
        config = PipelineConfig(
            source_language="en",
            target_language="fr",
            translator_backend="free",
            context_window=5
        )
        
        assert config.context_window == 5


class TestTranslationStats:
    """Test translation statistics tracking."""
    
    def test_stats_tracking(self):
        """Test that statistics are tracked."""
        doc = Document.from_text("Test document.", source_language="en")
        
        config = PipelineConfig(
            source_language="en",
            target_language="fr",
            translator_backend="free"
        )
        
        pipeline = TranslationPipeline(config)
        result = pipeline.translate(doc)
        
        assert "total_blocks" in result.stats
        assert "translated_blocks" in result.stats
        assert result.stats["total_blocks"] > 0
        
    def test_masking_stats_when_enabled(self):
        """Test masking stats when masking is enabled."""
        text = "Formula $x=1$ here."
        doc = Document.from_text(text, source_language="en")
        
        config = PipelineConfig(
            source_language="en",
            target_language="fr",
            translator_backend="free",
            mask_config=MaskConfig(enabled=True)
        )
        
        pipeline = TranslationPipeline(config)
        result = pipeline.translate(doc)
        
        assert "masks_applied" in result.stats
        assert result.stats["masks_applied"] > 0
        
    def test_masking_stats_when_disabled(self):
        """Test masking stats when masking is disabled."""
        text = "Formula $x=1$ here."
        doc = Document.from_text(text, source_language="en")
        
        config = PipelineConfig(
            source_language="en",
            target_language="fr",
            translator_backend="free",
            mask_config=MaskConfig(enabled=False)
        )
        
        pipeline = TranslationPipeline(config)
        result = pipeline.translate(doc)
        
        # No masking should occur
        assert result.stats.get("masks_applied", 0) == 0


class TestErrorHandling:
    """Test error handling in translation pipeline."""
    
    def test_empty_document(self):
        """Test translating empty document."""
        doc = Document.from_text("", source_language="en")
        
        config = PipelineConfig(
            source_language="en",
            target_language="fr",
            translator_backend="free"
        )
        
        pipeline = TranslationPipeline(config)
        result = pipeline.translate(doc)
        
        # Should handle gracefully
        assert result.translated_doc is not None
        
    def test_invalid_backend(self):
        """Test with invalid backend."""
        with pytest.raises(Exception):
            config = PipelineConfig(
                source_language="en",
                target_language="fr",
                translator_backend="nonexistent_backend"
            )
            pipeline = TranslationPipeline(config)
            
    def test_unsupported_language(self):
        """Test with unsupported language pair."""
        # Most backends support en-fr, but may not support obscure pairs
        # This test verifies graceful handling
        doc = Document.from_text("Test", source_language="en")
        
        config = PipelineConfig(
            source_language="en",
            target_language="zz",  # Invalid language code
            translator_backend="free"
        )
        
        pipeline = TranslationPipeline(config)
        # Should either succeed with fallback or fail gracefully
        try:
            result = pipeline.translate(doc)
            assert result is not None
        except Exception as e:
            # If it fails, it should be a handled exception
            assert str(e)  # Has error message


class TestContextHandling:
    """Test document context handling."""
    
    def test_context_with_multiple_segments(self):
        """Test context handling across segments."""
        paragraphs = [
            "Neural networks are computational models.",
            "They consist of interconnected nodes.",
            "These nodes process information."
        ]
        doc = Document.from_paragraphs(paragraphs, source_language="en")
        
        config = PipelineConfig(
            source_language="en",
            target_language="fr",
            translator_backend="free",
            context_window=2
        )
        
        pipeline = TranslationPipeline(config)
        result = pipeline.translate(doc)
        
        assert result.success
        # All blocks should be translated
        translated_blocks = [b for b in result.translated_doc.all_blocks 
                           if b.translated_text]
        assert len(translated_blocks) == 3


class TestTranslatableBlocks:
    """Test handling of translatable vs non-translatable blocks."""
    
    def test_skip_non_translatable(self):
        """Test that non-translatable blocks are skipped."""
        # Create document with mixed block types
        doc = Document.from_text("Test", source_language="en")
        
        # Manually mark one block as non-translatable
        if len(doc.all_blocks) > 0:
            doc.all_blocks[0]._block_type = BlockType.EQUATION
        
        config = PipelineConfig(
            source_language="en",
            target_language="fr",
            translator_backend="free"
        )
        
        pipeline = TranslationPipeline(config)
        result = pipeline.translate(doc)
        
        # Non-translatable blocks should not be translated
        equation_blocks = [b for b in result.translated_doc.all_blocks 
                          if b.block_type == BlockType.EQUATION]
        for block in equation_blocks:
            assert not block.translated_text or block.translated_text == block.source_text


class TestBackendAvailability:
    """Test different backend availability."""
    
    def test_free_backend_available(self):
        """Test that free backend is always available."""
        result = translate_text(
            "Test",
            source_lang="en",
            target_lang="fr",
            backend="free"
        )
        
        assert result is not None
        
    def test_dictionary_backend_available(self):
        """Test that dictionary backend is available."""
        result = translate_text(
            "Test",
            source_lang="en",
            target_lang="fr",
            backend="dictionary"
        )
        
        assert result is not None


class TestComplexDocuments:
    """Test translation of complex documents."""
    
    def test_document_with_mixed_content(self):
        """Test document with math, URLs, and regular text."""
        text = """
        Machine learning models use equations like $y = wx + b$.
        See more at https://example.com.
        The code `model.fit(X, y)` trains the model.
        """
        doc = Document.from_text(text, source_language="en")
        
        config = PipelineConfig(
            source_language="en",
            target_language="fr",
            translator_backend="free",
            mask_config=MaskConfig(enabled=True)
        )
        
        pipeline = TranslationPipeline(config)
        result = pipeline.translate(doc)
        
        assert result.success
        translated = result.translated_doc.all_blocks[0].translated_text
        
        # All protected content should be preserved
        assert "$y = wx + b$" in translated
        assert "https://example.com" in translated
        assert "`model.fit(X, y)`" in translated


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
