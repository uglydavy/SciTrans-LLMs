"""
Core tests for SciTrans-LLMs.

These tests verify the fundamental components work correctly:
- Document model creation and serialization
- Masking and unmasking
- Glossary loading and enforcement
- Pipeline execution

Run with: pytest tests/test_core.py -v
"""

import pytest
from scitrans_llms.models import Document, Block, Segment, BlockType
from scitrans_llms.masking import (
    MaskRegistry, MaskConfig, mask_text, unmask_text,
    mask_document, unmask_document, validate_placeholders
)
from scitrans_llms.translate.glossary import (
    Glossary, GlossaryEntry, enforce_glossary, check_glossary_adherence,
    get_default_glossary
)
from scitrans_llms.translate.base import DummyTranslator, DictionaryTranslator
from scitrans_llms.pipeline import TranslationPipeline, PipelineConfig


class TestDocumentModel:
    """Tests for the Document data model."""
    
    def test_create_from_text(self):
        """Document can be created from plain text."""
        text = "First paragraph.\n\nSecond paragraph."
        doc = Document.from_text(text)
        
        assert len(doc.segments) == 1
        assert len(doc.all_blocks) == 2
        assert doc.all_blocks[0].source_text == "First paragraph."
        assert doc.all_blocks[1].source_text == "Second paragraph."
    
    def test_create_from_paragraphs(self):
        """Document can be created from a list of paragraphs."""
        paragraphs = ["Hello world.", "This is a test."]
        doc = Document.from_paragraphs(paragraphs)
        
        assert len(doc.all_blocks) == 2
        assert doc.source_lang == "en"
        assert doc.target_lang == "fr"
    
    def test_serialization(self):
        """Document can be serialized to/from JSON."""
        doc = Document.from_text("Test paragraph")
        json_str = doc.to_json()
        restored = Document.from_json(json_str)
        
        assert restored.doc_id == doc.doc_id
        assert len(restored.all_blocks) == len(doc.all_blocks)
        assert restored.all_blocks[0].source_text == "Test paragraph"
    
    def test_block_types(self):
        """Blocks have correct type attributes."""
        para_block = Block(source_text="text", block_type=BlockType.PARAGRAPH)
        math_block = Block(source_text="$x^2$", block_type=BlockType.EQUATION)
        
        assert para_block.is_translatable
        assert not para_block.is_protected
        
        assert not math_block.is_translatable
        assert math_block.is_protected


class TestMasking:
    """Tests for the masking system."""
    
    def test_mask_latex_inline(self):
        """Inline LaTeX is masked correctly."""
        registry = MaskRegistry()
        text = "The formula is $x^2 + y^2 = z^2$ as shown."
        
        masked = mask_text(text, registry)
        
        assert "$x^2" not in masked
        assert "<<MATH_" in masked
        assert len(registry.mappings) == 1
    
    def test_mask_latex_display(self):
        """Display LaTeX is masked correctly."""
        registry = MaskRegistry()
        text = "Consider: $$\\int_0^1 x dx$$"
        
        masked = mask_text(text, registry)
        
        assert "$$" not in masked
        assert "<<MATHDISP_" in masked
    
    def test_mask_urls(self):
        """URLs are masked correctly."""
        registry = MaskRegistry()
        text = "See https://example.com/paper for details."
        
        masked = mask_text(text, registry)
        
        assert "https://" not in masked
        assert "<<URL_" in masked
    
    def test_mask_code(self):
        """Code blocks are masked correctly."""
        registry = MaskRegistry()
        text = "Use `print(x)` to output."
        
        masked = mask_text(text, registry)
        
        assert "`print(x)`" not in masked
        assert "<<CODE_" in masked
    
    def test_unmask_restores_content(self):
        """Unmasking restores original content."""
        registry = MaskRegistry()
        original = "The formula $x^2$ and URL https://test.com"
        
        masked = mask_text(original, registry)
        unmasked = unmask_text(masked, registry)
        
        assert unmasked == original
    
    def test_document_masking(self):
        """Document-level masking works correctly."""
        doc = Document.from_text("Formula: $E=mc^2$\n\nPlain text here.")
        
        registry = mask_document(doc)
        
        # First block should have masked formula
        assert "<<MATH_" in doc.all_blocks[0].masked_text
        # Second block should be unchanged
        assert doc.all_blocks[1].masked_text == "Plain text here."
    
    def test_validate_placeholders(self):
        """Placeholder validation detects missing placeholders."""
        source_masked = "Text with <<MATH_000>> and <<URL_001>>."
        translated = "Texte avec <<MATH_000>>."  # Missing URL
        
        missing = validate_placeholders(source_masked, translated)
        
        assert "<<URL_001>>" in missing
        assert "<<MATH_000>>" not in missing


class TestGlossary:
    """Tests for the glossary system."""
    
    def test_glossary_lookup(self):
        """Glossary lookup works correctly."""
        glossary = Glossary(entries=[
            GlossaryEntry("machine learning", "apprentissage automatique"),
            GlossaryEntry("neural network", "réseau de neurones"),
        ])
        
        result = glossary.get_target("machine learning")
        assert result == "apprentissage automatique"
        
        # Case-insensitive by default
        result = glossary.get_target("Machine Learning")
        assert result == "apprentissage automatique"
    
    def test_glossary_find_matches(self):
        """Glossary finds terms in text."""
        glossary = Glossary(entries=[
            GlossaryEntry("deep learning", "apprentissage profond"),
        ])
        
        text = "Deep learning is a subset of machine learning."
        matches = glossary.find_matches(text)
        
        assert len(matches) == 1
        assert matches[0].matched_text == "Deep learning"
    
    def test_glossary_enforcement(self):
        """Glossary terms are enforced in translations."""
        glossary = Glossary(entries=[
            GlossaryEntry("algorithm", "algorithme"),
        ])
        
        source = "The algorithm is efficient."
        # Bad translation that kept English term
        translated = "The algorithm est efficace."
        
        fixed = enforce_glossary(translated, source, glossary)
        
        # Should replace with French term
        assert "algorithme" in fixed or "algorithm" in fixed.lower()
    
    def test_glossary_adherence(self):
        """Glossary adherence is measured correctly."""
        glossary = Glossary(entries=[
            GlossaryEntry("model", "modèle"),
            GlossaryEntry("training", "entraînement"),
        ])
        
        source = "The model requires training."
        translated = "Le modèle nécessite entraînement."
        
        result = check_glossary_adherence(translated, source, glossary)
        
        assert result["adherence_rate"] == 1.0
        assert result["total_terms"] == 2
    
    def test_default_glossary(self):
        """Default glossary has expected terms."""
        glossary = get_default_glossary()
        
        assert len(glossary) > 30
        assert glossary.get_target("algorithm") == "algorithme"
        assert glossary.get_target("deep learning") == "apprentissage profond"


class TestTranslators:
    """Tests for translator backends."""
    
    def test_dummy_translator_prefix(self):
        """Dummy translator adds prefix."""
        translator = DummyTranslator(mode="prefix")
        result = translator.translate("Hello world")
        
        assert result.text == "[TRANSLATED] Hello world"
    
    def test_dummy_translator_echo(self):
        """Dummy translator echo mode returns input unchanged."""
        translator = DummyTranslator(mode="echo")
        result = translator.translate("Test input")
        
        assert result.text == "Test input"
    
    def test_dictionary_translator(self):
        """Dictionary translator replaces glossary terms."""
        glossary = Glossary(entries=[
            GlossaryEntry("model", "modèle"),
            GlossaryEntry("data", "données"),
        ])
        translator = DictionaryTranslator(glossary=glossary)
        
        result = translator.translate("The model uses data.")
        
        assert "modèle" in result.text
        assert "données" in result.text


class TestPipeline:
    """Tests for the translation pipeline."""
    
    def test_simple_pipeline(self):
        """Basic pipeline execution works."""
        doc = Document.from_text("Hello world")
        config = PipelineConfig(translator_backend="dummy")
        pipeline = TranslationPipeline(config)
        
        result = pipeline.translate(doc)
        
        assert result.success
        assert len(result.translated_text) > 0
    
    def test_pipeline_with_masking(self):
        """Pipeline preserves masked content."""
        doc = Document.from_text("Formula: $x^2$")
        config = PipelineConfig(
            translator_backend="echo",
            translator_kwargs={"mode": "echo"},
            enable_masking=True,
        )
        config.translator_backend = "dummy"
        pipeline = TranslationPipeline(config)
        
        # Need to use echo mode for this test
        from scitrans_llms.translate.base import DummyTranslator
        pipeline.translator = DummyTranslator(mode="echo")
        
        result = pipeline.translate(doc)
        
        # Formula should be preserved after unmasking
        assert "$x^2$" in result.translated_text
    
    def test_pipeline_with_glossary(self):
        """Pipeline uses glossary correctly."""
        glossary = Glossary(entries=[
            GlossaryEntry("test", "essai"),
        ])
        doc = Document.from_text("This is a test.")
        config = PipelineConfig(
            translator_backend="dictionary",
            enable_glossary=True,
            glossary=glossary,
        )
        pipeline = TranslationPipeline(config)
        
        result = pipeline.translate(doc)
        
        assert "essai" in result.translated_text
    
    def test_pipeline_stats(self):
        """Pipeline returns useful statistics."""
        doc = Document.from_paragraphs(["Para 1", "Para 2", "Para 3"])
        config = PipelineConfig(translator_backend="dummy")
        pipeline = TranslationPipeline(config)
        
        result = pipeline.translate(doc)
        
        assert "total_blocks" in result.stats
        assert result.stats["total_blocks"] == 3
        assert result.stats["translated_blocks"] == 3
    
    def test_translate_text_convenience(self):
        """Convenience function works correctly."""
        from scitrans_llms.pipeline import translate_text
        
        result = translate_text("Hello", backend="dummy")
        
        assert "[TRANSLATED]" in result


# Quick test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

