"""
Comprehensive tests for masking functionality.

Tests cover:
- LaTeX equation masking (inline and display)
- URL masking
- Code block masking
- Placeholder preservation during translation
- Unmasking and restoration
- Mask registry tracking
- Document-level masking
"""

import pytest
from scitrans_llms.masking import (
    mask_text,
    unmask_text,
    mask_document,
    unmask_document,
    validate_placeholders,
    MaskConfig,
    MaskRegistry,
)
from scitrans_llms.models import Document, BlockType


class TestInlineMathMasking:
    """Test inline LaTeX math equation masking."""
    
    def test_single_inline_equation(self):
        """Test masking of single inline equation."""
        text = "The formula $E = mc^2$ is famous."
        registry = MaskRegistry()
        masked = mask_text(text, registry)
        
        assert "$E = mc^2$" not in masked
        assert "MATH_" in masked
        assert len(registry.mappings) == 1
        
    def test_multiple_inline_equations(self):
        """Test masking of multiple inline equations."""
        text = "We have $x = 5$ and $y = 10$."
        registry = MaskRegistry()
        masked = mask_text(text, registry)
        
        assert "$x = 5$" not in masked
        assert "$y = 10$" not in masked
        assert len(registry.mappings) == 2
        
    def test_complex_inline_equation(self):
        """Test masking of complex inline equation."""
        text = "The integral $\\int_0^\\infty e^{-x^2} dx$ converges."
        registry = MaskRegistry()
        masked = mask_text(text, registry)
        
        assert "\\int" not in masked
        assert "MATH_" in masked
        
    def test_inline_unmask(self):
        """Test unmasking restores inline equations."""
        text = "The formula $E = mc^2$ is famous."
        registry = MaskRegistry()
        masked = mask_text(text, registry)
        unmasked = unmask_text(masked, registry)
        
        assert unmasked == text


class TestDisplayMathMasking:
    """Test display LaTeX math equation masking."""
    
    def test_double_dollar_equation(self):
        """Test masking of $$ display equation."""
        text = "Consider:\n$$E = mc^2$$\nThis is Einstein's equation."
        registry = MaskRegistry()
        masked = mask_text(text, registry)
        
        assert "$$E = mc^2$$" not in masked
        assert "MATH_" in masked
        
    def test_bracket_equation(self):
        """Test masking of \\[ \\] display equation."""
        text = "The equation:\n\\[x^2 + y^2 = r^2\\]\nis a circle."
        registry = MaskRegistry()
        masked = mask_text(text, registry)
        
        assert "\\[x^2" not in masked
        assert "MATH_" in masked
        
    def test_equation_environment(self):
        """Test masking of equation environment."""
        text = """
        The formula is:
        \\begin{equation}
        f(x) = \\sum_{i=0}^n a_i x^i
        \\end{equation}
        """
        registry = MaskRegistry()
        masked = mask_text(text, registry)
        
        assert "\\begin{equation}" not in masked
        assert "MATH_" in masked
        
    def test_align_environment(self):
        """Test masking of align environment."""
        text = """
        \\begin{align}
        x &= a + b \\\\
        y &= c + d
        \\end{align}
        """
        registry = MaskRegistry()
        masked = mask_text(text, registry)
        
        assert "\\begin{align}" not in masked
        assert "MATH_" in masked


class TestURLMasking:
    """Test URL masking."""
    
    def test_http_url(self):
        """Test masking of http URL."""
        text = "Visit http://example.com for more info."
        registry = MaskRegistry()
        masked = mask_text(text, registry)
        
        assert "http://example.com" not in masked
        assert "URL_" in masked
        
    def test_https_url(self):
        """Test masking of https URL."""
        text = "Check https://secure.example.com/path"
        registry = MaskRegistry()
        masked = mask_text(text, registry)
        
        assert "https://secure.example.com" not in masked
        assert "URL_" in masked
        
    def test_multiple_urls(self):
        """Test masking of multiple URLs."""
        text = "Visit http://example.com and https://other.com"
        registry = MaskRegistry()
        masked = mask_text(text, registry)
        
        assert len(registry.mappings) == 2
        assert "URL_" in masked
        
    def test_url_with_query_params(self):
        """Test masking of URL with query parameters."""
        text = "Search: https://example.com/search?q=test&page=1"
        registry = MaskRegistry()
        masked = mask_text(text, registry)
        
        assert "?q=test" not in masked
        assert "URL_" in masked
        
    def test_url_unmask(self):
        """Test unmasking restores URLs."""
        text = "Visit https://example.com"
        registry = MaskRegistry()
        masked = mask_text(text, registry)
        unmasked = unmask_text(masked, registry)
        
        assert unmasked == text


class TestCodeMasking:
    """Test code block masking."""
    
    def test_inline_code(self):
        """Test masking of inline code."""
        text = "Use the `print()` function."
        registry = MaskRegistry()
        masked = mask_text(text, registry)
        
        assert "`print()`" not in masked
        assert "CODE_" in masked
        
    def test_code_block(self):
        """Test masking of code block."""
        text = """
        Example:
        ```python
        def hello():
            print("Hello")
        ```
        """
        registry = MaskRegistry()
        masked = mask_text(text, registry)
        
        assert "```python" not in masked
        assert "CODEBLK_" in masked
        
    def test_multiple_inline_code(self):
        """Test masking of multiple inline code snippets."""
        text = "Use `var x = 1` and `var y = 2`."
        registry = MaskRegistry()
        masked = mask_text(text, registry)
        
        assert len(registry.mappings) == 2
        
    def test_code_unmask(self):
        """Test unmasking restores code."""
        text = "Use `print()` function."
        registry = MaskRegistry()
        masked = mask_text(text, registry)
        unmasked = unmask_text(masked, registry)
        
        assert unmasked == text


class TestMixedMasking:
    """Test masking of mixed content types."""
    
    def test_math_and_url(self):
        """Test masking both math and URL."""
        text = "See $E=mc^2$ at https://example.com"
        registry = MaskRegistry()
        masked = mask_text(text, registry)
        
        assert "$E=mc^2$" not in masked
        assert "https://example.com" not in masked
        assert len(registry.mappings) == 2
        
    def test_math_and_code(self):
        """Test masking both math and code."""
        text = "Compute $\\sum x_i$ using `sum(x)`"
        registry = MaskRegistry()
        masked = mask_text(text, registry)
        
        assert "$\\sum x_i$" not in masked
        assert "`sum(x)`" not in masked
        assert len(registry.mappings) == 2
        
    def test_all_types_together(self):
        """Test masking math, URL, and code together."""
        text = "The formula $f(x)$ is at https://example.com, use `compute()`"
        registry = MaskRegistry()
        masked = mask_text(text, registry)
        
        assert len(registry.mappings) == 3
        
    def test_mixed_unmask(self):
        """Test unmasking restores all types correctly."""
        text = "Formula $E=mc^2$ at https://example.com with `code()`"
        registry = MaskRegistry()
        masked = mask_text(text, registry)
        unmasked = unmask_text(masked, registry)
        
        assert unmasked == text


class TestPlaceholderPreservation:
    """Test placeholder preservation during translation."""
    
    def test_validate_all_placeholders_present(self):
        """Test validation passes when all placeholders present."""
        original = "Text with MATH_0 and URL_1 placeholders"
        translated = "Texte avec MATH_0 et URL_1 espaces réservés"
        registry = MaskRegistry()
        registry.mappings = {"MATH_0": "$x$", "URL_1": "http://example.com"}
        
        # Should not raise
        validate_placeholders(original, translated, registry)
        
    def test_validate_missing_placeholder(self):
        """Test validation fails when placeholder missing."""
        original = "Text with MATH_0 placeholder"
        translated = "Texte sans espace réservé"  # Missing MATH_0
        registry = MaskRegistry()
        registry.mappings = {"MATH_0": "$x$"}
        
        with pytest.raises(ValueError):
            validate_placeholders(original, translated, registry)
            
    def test_validate_extra_placeholder(self):
        """Test validation when extra placeholder added."""
        original = "Text with MATH_0"
        translated = "Texte avec MATH_0 et MATH_1"  # Extra MATH_1
        registry = MaskRegistry()
        registry.mappings = {"MATH_0": "$x$"}
        
        # Should not raise - extra placeholders are okay
        validate_placeholders(original, translated, registry)
        
    def test_placeholder_order_independent(self):
        """Test validation is order-independent."""
        original = "Text with MATH_0 and URL_1"
        translated = "Texte avec URL_1 et MATH_0"  # Different order
        registry = MaskRegistry()
        registry.mappings = {"MATH_0": "$x$", "URL_1": "http://example.com"}
        
        # Should not raise
        validate_placeholders(original, translated, registry)


class TestDocumentMasking:
    """Test document-level masking."""
    
    def test_mask_simple_document(self):
        """Test masking a simple document."""
        doc = Document.from_text(
            "The equation $E=mc^2$ is fundamental.",
            source_language="en"
        )
        config = MaskConfig(enabled=True)
        
        masked_doc, registry = mask_document(doc, config)
        
        assert len(registry.mappings) > 0
        # Check that source text is masked
        for block in masked_doc.all_blocks:
            if block.is_translatable and "$" in block.source_text:
                # Original has equation, masked should not
                assert "$E=mc^2$" not in block.masked_text
                
    def test_unmask_simple_document(self):
        """Test unmasking a simple document."""
        doc = Document.from_text(
            "The equation $E=mc^2$ is fundamental.",
            source_language="en"
        )
        config = MaskConfig(enabled=True)
        
        masked_doc, registry = mask_document(doc, config)
        
        # Simulate translation
        for block in masked_doc.all_blocks:
            if block.is_translatable:
                block.translated_text = block.masked_text  # Keep placeholders
        
        unmasked_doc = unmask_document(masked_doc, registry)
        
        # Should restore equations in translated text
        for block in unmasked_doc.all_blocks:
            if block.is_translatable and block.translated_text:
                assert "MATH_" not in block.translated_text
                
    def test_document_with_multiple_blocks(self):
        """Test masking document with multiple blocks."""
        paragraphs = [
            "First paragraph with $x=1$.",
            "Second paragraph with $y=2$.",
            "Third at https://example.com"
        ]
        doc = Document.from_paragraphs(paragraphs, source_language="en")
        config = MaskConfig(enabled=True)
        
        masked_doc, registry = mask_document(doc, config)
        
        assert len(registry.mappings) >= 3  # At least 2 math + 1 URL
        
    def test_masking_disabled(self):
        """Test that masking can be disabled."""
        doc = Document.from_text(
            "The equation $E=mc^2$ is fundamental.",
            source_language="en"
        )
        config = MaskConfig(enabled=False)
        
        masked_doc, registry = mask_document(doc, config)
        
        # No masks should be created
        assert len(registry.mappings) == 0
        # Text should be unchanged
        for orig_block, masked_block in zip(doc.all_blocks, masked_doc.all_blocks):
            assert orig_block.source_text == masked_block.masked_text


class TestMaskRegistry:
    """Test MaskRegistry functionality."""
    
    def test_registry_stores_masks(self):
        """Test that registry stores masks correctly."""
        registry = MaskRegistry()
        text = "Formula $E=mc^2$ and URL https://example.com"
        masked = mask_text(text, registry)
        
        assert len(registry.mappings) == 2
        assert any("$E=mc^2$" in v for v in registry.mappings.values())
        assert any("https://example.com" in v for v in registry.mappings.values())
        
    def test_registry_unique_keys(self):
        """Test that registry generates unique keys."""
        registry = MaskRegistry()
        text = "Multiple $a$ and $b$ and $c$"
        masked = mask_text(text, registry)
        
        keys = list(registry.mappings.keys())
        assert len(keys) == len(set(keys))  # All unique
        
    def test_registry_retrieval(self):
        """Test retrieving values from registry."""
        registry = MaskRegistry()
        text = "Formula $E=mc^2$"
        masked = mask_text(text, registry)
        
        # Get the mask key from masked text
        import re
        match = re.search(r'MATH_\d+', masked)
        assert match
        key = match.group(0)
        
        assert key in registry.mappings
        assert "$E=mc^2$" in registry.mappings[key]


class TestMaskConfig:
    """Test MaskConfig configuration."""
    
    def test_default_config(self):
        """Test default MaskConfig settings."""
        config = MaskConfig()
        assert config.enabled is True
        
    def test_custom_config(self):
        """Test custom MaskConfig settings."""
        config = MaskConfig(
            enabled=True,
            preserve_structure=True
        )
        assert config.enabled is True
        assert config.preserve_structure is True


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_text(self):
        """Test masking empty text."""
        text = ""
        registry = MaskRegistry()
        masked = mask_text(text, registry)
        
        assert masked == ""
        assert len(registry.mappings) == 0
        
    def test_text_without_maskable_content(self):
        """Test masking text with nothing to mask."""
        text = "Just plain text."
        registry = MaskRegistry()
        masked = mask_text(text, registry)
        
        assert masked == text
        assert len(registry.mappings) == 0
        
    def test_nested_delimiters(self):
        """Test handling of nested delimiters."""
        text = "Complex: $a + (b * $c$)$ equation"
        registry = MaskRegistry()
        masked = mask_text(text, registry)
        
        # Should handle gracefully (exact behavior may vary)
        assert "MATH_" in masked or text == masked
        
    def test_malformed_equation(self):
        """Test handling of malformed equation."""
        text = "Unclosed equation $x + y"
        registry = MaskRegistry()
        masked = mask_text(text, registry)
        
        # Should handle gracefully without crashing
        assert masked is not None
        
    def test_very_long_equation(self):
        """Test masking very long equation."""
        equation = "$" + "x^2 + " * 100 + "1$"
        text = f"Long equation: {equation}"
        registry = MaskRegistry()
        masked = mask_text(text, registry)
        
        assert "MATH_" in masked
        assert len(registry.mappings) == 1


class TestRoundTripConsistency:
    """Test that mask/unmask round trips are consistent."""
    
    def test_roundtrip_simple(self):
        """Test simple round trip."""
        text = "Formula $E=mc^2$ is famous."
        registry = MaskRegistry()
        masked = mask_text(text, registry)
        unmasked = unmask_text(masked, registry)
        assert unmasked == text
        
    def test_roundtrip_complex(self):
        """Test complex round trip with multiple types."""
        text = """
        The formula $E=mc^2$ is at https://example.com.
        Use `code()` to compute $\\int_0^1 x dx$.
        """
        registry = MaskRegistry()
        masked = mask_text(text, registry)
        unmasked = unmask_text(masked, registry)
        assert unmasked == text
        
    def test_roundtrip_document(self):
        """Test document round trip."""
        paragraphs = [
            "Formula $x=1$ here.",
            "URL https://example.com there.",
            "Code `print()` everywhere."
        ]
        doc = Document.from_paragraphs(paragraphs, source_language="en")
        config = MaskConfig(enabled=True)
        
        masked_doc, registry = mask_document(doc, config)
        
        # Simulate translation that preserves placeholders
        for block in masked_doc.all_blocks:
            if block.is_translatable:
                block.translated_text = block.masked_text
        
        unmasked_doc = unmask_document(masked_doc, registry)
        
        # Check that original content is restored
        for orig_block, unmasked_block in zip(doc.all_blocks, unmasked_doc.all_blocks):
            if orig_block.is_translatable:
                # Translated text should have masks replaced
                assert "MATH_" not in unmasked_block.translated_text
                assert "URL_" not in unmasked_block.translated_text
                assert "CODE_" not in unmasked_block.translated_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
