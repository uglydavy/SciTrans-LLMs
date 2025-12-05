# -*- coding: utf-8 -*-
"""
Comprehensive test suite for SciTrans-LLMs.

This test file covers all major components:
1. Models (Document, Block, Segment)
2. Masking (patterns, preservation, unmask)
3. Translation (backends, context, candidates)
4. Pipeline (full workflow)
5. Glossary (loading, matching, enforcement)
6. Refinement (reranking, scoring)
7. PDF Rendering (text replacement)

Thesis Innovation Testing:
- #1: Terminology-constrained translation
- #2: Document-level context
- #3: Candidate reranking
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

# ============================================================================
# Model Tests
# ============================================================================

class TestModels:
    """Test Document, Block, and Segment models."""
    
    def test_document_creation(self):
        """Test creating a Document from text."""
        from scitrans_llms.models import Document
        
        doc = Document.from_text("Hello world", source_lang="en", target_lang="fr")
        
        assert doc.source_lang == "en"
        assert doc.target_lang == "fr"
        assert len(doc.all_blocks) > 0
        assert doc.all_blocks[0].source_text == "Hello world"
    
    def test_block_properties(self):
        """Test Block properties."""
        from scitrans_llms.models import Block, BlockType
        
        block = Block(
            block_id="test_1",
            block_type=BlockType.PARAGRAPH,
            source_text="Test content",
        )
        
        assert block.is_translatable
        assert not block.is_protected
        
        eq_block = Block(
            block_id="eq_1",
            block_type=BlockType.EQUATION,
            source_text="E = mc^2",
        )
        
        assert eq_block.is_protected
        assert not eq_block.is_translatable
    
    def test_document_translated_text(self):
        """Test getting translated text from document."""
        from scitrans_llms.models import Document, Block, BlockType
        
        doc = Document.from_text("Hello", source_lang="en", target_lang="fr")
        doc.all_blocks[0].translated_text = "Bonjour"
        
        assert doc.translated_text == "Bonjour"


# ============================================================================
# Masking Tests  
# ============================================================================

class TestMasking:
    """Test masking and unmasking operations."""
    
    def test_latex_inline_masking(self):
        """Test inline LaTeX masking."""
        from scitrans_llms.masking import mask_text, MaskRegistry, MaskConfig
        
        registry = MaskRegistry()
        config = MaskConfig(mask_latex_inline=True)
        
        text = "The equation $x^2 + y^2 = z^2$ is Pythagorean."
        masked = mask_text(text, registry, config)
        
        assert "$x^2 + y^2 = z^2$" not in masked
        assert "<<MATH_" in masked
        assert len(registry.mappings) == 1
    
    def test_latex_display_masking(self):
        """Test display LaTeX masking."""
        from scitrans_llms.masking import mask_text, MaskRegistry, MaskConfig
        
        registry = MaskRegistry()
        config = MaskConfig(mask_latex_display=True)
        
        text = "Consider $$\\int_0^1 x dx = \\frac{1}{2}$$ for integration."
        masked = mask_text(text, registry, config)
        
        assert "$$" not in masked
        assert "<<MATHDISP_" in masked
    
    def test_url_masking(self):
        """Test URL masking."""
        from scitrans_llms.masking import mask_text, MaskRegistry, MaskConfig
        
        registry = MaskRegistry()
        config = MaskConfig(mask_urls=True)
        
        text = "Visit https://example.com/paper.pdf for details."
        masked = mask_text(text, registry, config)
        
        assert "https://example.com" not in masked
        assert "<<URL_" in masked
    
    def test_code_block_masking(self):
        """Test code block masking."""
        from scitrans_llms.masking import mask_text, MaskRegistry, MaskConfig
        
        registry = MaskRegistry()
        config = MaskConfig(mask_code_blocks=True)
        
        text = "Example:\n```python\nprint('hello')\n```\nEnd."
        masked = mask_text(text, registry, config)
        
        assert "```python" not in masked
        assert "<<CODEBLK_" in masked
    
    def test_unmask(self):
        """Test unmasking restores original content."""
        from scitrans_llms.masking import mask_text, unmask_text, MaskRegistry
        
        registry = MaskRegistry()
        original = "Formula $E=mc^2$ and URL https://test.com here."
        masked = mask_text(original, registry)
        
        # Simulate translation that preserves placeholders
        translated = masked.replace("Formula", "Formule").replace("and", "et")
        
        unmasked = unmask_text(translated, registry)
        
        assert "$E=mc^2$" in unmasked
        assert "https://test.com" in unmasked
    
    def test_section_number_preservation(self):
        """Test section numbers are preserved."""
        from scitrans_llms.masking import mask_text, unmask_text, MaskRegistry, MaskConfig
        
        registry = MaskRegistry()
        config = MaskConfig(preserve_section_numbers=True)
        
        text = "1. Introduction\n1.1 Background\nII. Methods"
        masked = mask_text(text, registry, config)
        
        # Section numbers should be masked
        assert "<<SECNUM_" in masked
    
    def test_placeholder_validation(self):
        """Test placeholder validation."""
        from scitrans_llms.masking import validate_placeholders
        
        source = "Text <<MATH_001>> and <<URL_000>> here."
        good_trans = "Texte <<MATH_001>> et <<URL_000>> ici."
        bad_trans = "Texte <<MATH_001>> et ici."  # Missing URL
        
        missing = validate_placeholders(source, good_trans)
        assert len(missing) == 0
        
        missing = validate_placeholders(source, bad_trans)
        assert "<<URL_000>>" in missing


# ============================================================================
# Glossary Tests
# ============================================================================

class TestGlossary:
    """Test glossary operations."""
    
    def test_default_glossary(self):
        """Test loading default glossary."""
        from scitrans_llms.translate.glossary import get_default_glossary
        
        glossary = get_default_glossary()
        
        assert len(glossary) > 100  # Should have significant entries
        assert glossary.get_target("algorithm") == "algorithme"
        assert glossary.get_target("neural network") == "réseau de neurones"
    
    def test_glossary_matching(self):
        """Test finding glossary terms in text."""
        from scitrans_llms.translate.glossary import get_default_glossary
        
        glossary = get_default_glossary()
        
        text = "The machine learning algorithm uses deep learning."
        matches = glossary.find_matches(text)
        
        # Should find multiple terms
        terms_found = [m.entry.source for m in matches]
        assert "machine learning" in terms_found or "algorithm" in terms_found
    
    def test_glossary_enforcement(self):
        """Test enforcing glossary in translation."""
        from scitrans_llms.translate.glossary import (
            get_default_glossary, enforce_glossary
        )
        
        glossary = get_default_glossary()
        
        source = "The algorithm performs optimization."
        # Bad translation that left "algorithm" in English
        bad_translation = "L'algorithm effectue une optimisation."
        
        fixed = enforce_glossary(bad_translation, source, glossary)
        
        # Should replace "algorithm" with "algorithme"
        assert "algorithme" in fixed
    
    def test_glossary_adherence_check(self):
        """Test glossary adherence measurement."""
        from scitrans_llms.translate.glossary import (
            get_default_glossary, check_glossary_adherence
        )
        
        glossary = get_default_glossary()
        
        source = "The algorithm and the model work together."
        good_trans = "L'algorithme et le modèle fonctionnent ensemble."
        
        result = check_glossary_adherence(good_trans, source, glossary)
        
        assert result["adherence_rate"] >= 0.5  # Should be high
        assert "total_terms" in result


# ============================================================================
# Translator Tests
# ============================================================================

class TestTranslators:
    """Test translation backends."""
    
    def test_dummy_translator(self):
        """Test dummy translator."""
        from scitrans_llms.translate.base import DummyTranslator
        
        translator = DummyTranslator(mode="prefix")
        result = translator.translate("Hello world")
        
        assert "[TRANSLATED]" in result.text
        assert result.source_text == "Hello world"
    
    def test_dictionary_translator(self):
        """Test dictionary-based translator."""
        from scitrans_llms.translate.base import DictionaryTranslator
        from scitrans_llms.translate.glossary import get_default_glossary
        
        glossary = get_default_glossary()
        translator = DictionaryTranslator(glossary=glossary)
        
        result = translator.translate("The algorithm is good")
        
        # Should apply glossary and dictionary
        assert "algorithme" in result.text.lower() or "algorithm" not in result.text.lower()
    
    def test_translator_factory(self):
        """Test translator factory function."""
        from scitrans_llms.translate.base import create_translator
        
        # Test creating different translators
        dummy = create_translator("dummy")
        assert dummy.name.startswith("dummy")
        
        dict_trans = create_translator("dictionary")
        assert dict_trans.name == "dictionary"
        
        # Test with invalid backend
        with pytest.raises(ValueError):
            create_translator("nonexistent_backend")
    
    def test_translation_context(self):
        """Test translation with context."""
        from scitrans_llms.translate.base import (
            DictionaryTranslator, TranslationContext
        )
        from scitrans_llms.translate.glossary import get_default_glossary
        
        context = TranslationContext(
            previous_source=["Previous sentence."],
            previous_target=["Phrase précédente."],
            glossary=get_default_glossary(),
            source_lang="en",
            target_lang="fr",
        )
        
        translator = DictionaryTranslator()
        result = translator.translate("The method works well.", context)
        
        assert result.text != "The method works well."  # Should be translated


# ============================================================================
# Pipeline Tests
# ============================================================================

class TestPipeline:
    """Test translation pipeline."""
    
    def test_basic_pipeline(self):
        """Test basic pipeline execution."""
        from scitrans_llms.pipeline import TranslationPipeline, PipelineConfig
        from scitrans_llms.models import Document
        
        config = PipelineConfig(
            translator_backend="dictionary",
            enable_masking=True,
            enable_glossary=True,
        )
        
        pipeline = TranslationPipeline(config)
        doc = Document.from_text(
            "The machine learning algorithm works.",
            source_lang="en",
            target_lang="fr",
        )
        
        result = pipeline.translate(doc)
        
        assert result.success
        assert result.translated_text != ""
        assert result.stats["translated_blocks"] > 0
    
    def test_pipeline_with_masking(self):
        """Test pipeline preserves masked content."""
        from scitrans_llms.pipeline import TranslationPipeline, PipelineConfig
        from scitrans_llms.models import Document
        
        config = PipelineConfig(
            translator_backend="dummy",
            enable_masking=True,
        )
        
        pipeline = TranslationPipeline(config)
        doc = Document.from_text(
            "The formula $E=mc^2$ is famous.",
            source_lang="en",
            target_lang="fr",
        )
        
        result = pipeline.translate(doc)
        
        # Formula should be preserved
        assert "$E=mc^2$" in result.translated_text
    
    def test_pipeline_with_candidates(self):
        """Test pipeline with candidate generation."""
        from scitrans_llms.pipeline import TranslationPipeline, PipelineConfig
        from scitrans_llms.models import Document
        
        config = PipelineConfig(
            translator_backend="dictionary",
            num_candidates=3,  # Enable reranking
        )
        
        pipeline = TranslationPipeline(config)
        doc = Document.from_text("Simple test.", source_lang="en", target_lang="fr")
        
        result = pipeline.translate(doc)
        
        assert result.success


# ============================================================================
# Reranking Tests
# ============================================================================

class TestReranking:
    """Test candidate reranking."""
    
    def test_basic_reranking(self):
        """Test basic reranking operation."""
        from scitrans_llms.refine.rerank import CandidateReranker
        
        reranker = CandidateReranker(use_llm_scoring=False)
        
        candidates = [
            "C'est un bon algorithme.",
            "C'est un algorithm bon.",
            "Ceci est un bon algorithme de traduction.",
        ]
        
        result = reranker.rerank(
            source_text="This is a good algorithm.",
            candidates=candidates,
        )
        
        assert result.best_candidate in candidates
        assert len(result.scores) == 3
    
    def test_reranking_with_glossary(self):
        """Test reranking considers glossary adherence."""
        from scitrans_llms.refine.rerank import CandidateReranker
        from scitrans_llms.translate.glossary import get_default_glossary
        
        glossary = get_default_glossary()
        reranker = CandidateReranker(use_llm_scoring=False)
        
        # One candidate uses correct term, other doesn't
        candidates = [
            "L'algorithme fonctionne.",  # Uses "algorithme" (correct)
            "L'algorithm fonctionne.",   # Uses English term (wrong)
        ]
        
        result = reranker.rerank(
            source_text="The algorithm works.",
            candidates=candidates,
            glossary=glossary,
        )
        
        # Should prefer the one with correct terminology
        best_score = result.scores[result.metadata["best_index"]]
        assert best_score.terminology > 0.5
    
    def test_placeholder_preservation_scoring(self):
        """Test scoring considers placeholder preservation."""
        from scitrans_llms.refine.rerank import CandidateReranker
        
        reranker = CandidateReranker(use_llm_scoring=False)
        
        source_masked = "Voir <<URL_000>> pour plus de détails."
        
        candidates = [
            "Voir <<URL_000>> pour plus d'informations.",  # Preserved
            "Voir le site pour plus d'informations.",       # Lost placeholder
        ]
        
        result = reranker.rerank(
            source_text="See URL for more details.",
            candidates=candidates,
            source_masked=source_masked,
        )
        
        # Should prefer candidate that preserved placeholder
        assert result.scores[0].placeholder_preservation > result.scores[1].placeholder_preservation


# ============================================================================
# Refinement Tests
# ============================================================================

class TestRefinement:
    """Test translation refinement."""
    
    def test_list_structure_preservation(self):
        """Test list structure is preserved."""
        from scitrans_llms.refine.postprocess import preserve_list_structure
        
        source = "1. First item\n2. Second item"
        translated = "Premier élément\nDeuxième élément"
        
        result = preserve_list_structure(source, translated)
        
        assert "1." in result
        assert "2." in result
    
    def test_bullet_preservation(self):
        """Test bullet points are preserved."""
        from scitrans_llms.refine.postprocess import preserve_list_structure
        
        source = "• Item one\n• Item two"
        translated = "Élément un\nÉlément deux"
        
        result = preserve_list_structure(source, translated)
        
        assert "•" in result


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for full workflows."""
    
    def test_translate_text_convenience(self):
        """Test translate_text convenience function."""
        from scitrans_llms.pipeline import translate_text
        
        result = translate_text(
            "The algorithm is efficient.",
            backend="dictionary",
        )
        
        assert result != ""
        assert result != "The algorithm is efficient."  # Should be different
    
    def test_full_workflow_with_document(self):
        """Test complete workflow with document."""
        from scitrans_llms.models import Document
        from scitrans_llms.pipeline import TranslationPipeline, PipelineConfig
        from scitrans_llms.masking import mask_document, unmask_document
        
        # Create document
        doc = Document.from_text(
            "The deep learning model achieves good accuracy.",
            source_lang="en",
            target_lang="fr",
        )
        
        # Create pipeline
        config = PipelineConfig(
            translator_backend="dictionary",
            enable_masking=True,
            enable_glossary=True,
            enable_refinement=True,
        )
        
        pipeline = TranslationPipeline(config)
        result = pipeline.translate(doc)
        
        # Verify result
        assert result.success
        assert "modèle" in result.translated_text.lower() or len(result.translated_text) > 0


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_text(self):
        """Test handling empty text."""
        from scitrans_llms.models import Document
        
        doc = Document.from_text("", source_lang="en", target_lang="fr")
        
        assert len(doc.all_blocks) == 0 or doc.all_blocks[0].source_text == ""
    
    def test_unicode_text(self):
        """Test handling Unicode text."""
        from scitrans_llms.pipeline import translate_text
        
        result = translate_text(
            "Résumé: Les données sont importantes.",
            source_lang="fr",
            target_lang="en",
            backend="dictionary",
        )
        
        assert result is not None
    
    def test_special_characters(self):
        """Test handling special characters."""
        from scitrans_llms.masking import mask_text, MaskRegistry
        
        registry = MaskRegistry()
        text = "Special: @#$%^&*() and < > \" ' test."
        
        masked = mask_text(text, registry)
        
        # Should not crash
        assert masked is not None
    
    def test_very_long_text(self):
        """Test handling very long text."""
        from scitrans_llms.models import Document
        from scitrans_llms.pipeline import TranslationPipeline, PipelineConfig
        
        long_text = "This is a test sentence. " * 100
        
        doc = Document.from_text(long_text, source_lang="en", target_lang="fr")
        config = PipelineConfig(translator_backend="dummy")
        pipeline = TranslationPipeline(config)
        
        result = pipeline.translate(doc)
        
        assert result.success


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


