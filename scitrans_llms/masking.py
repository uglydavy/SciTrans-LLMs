"""
Masking module for protecting non-translatable content.

This module handles the insertion and restoration of placeholders for:
- Mathematical formulas (LaTeX, MathML)
- Code blocks
- URLs and email addresses
- Special formatting tokens

Thesis Contribution #1: Terminology-constrained, layout-preserving translation
requires robust masking to prevent corruption of formulas and code.

Design:
- Each mask type has a unique prefix (e.g., <<MATH_001>>)
- Masks are reversible: original content is stored and can be restored
- Masks work at the Block level for fine-grained control
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from scitrans_llms.models import Block, Document


@dataclass
class MaskRegistry:
    """Stores mappings between placeholders and original content.
    
    This registry tracks all masked content for a document, allowing
    reliable restoration after translation.
    """
    mappings: dict[str, str] = field(default_factory=dict)  # placeholder -> original
    counters: dict[str, int] = field(default_factory=dict)  # prefix -> count
    
    def register(self, prefix: str, original: str) -> str:
        """Register content and return a placeholder."""
        count = self.counters.get(prefix, 0)
        self.counters[prefix] = count + 1
        placeholder = f"<<{prefix}_{count:03d}>>"
        self.mappings[placeholder] = original
        return placeholder
    
    def restore(self, text: str) -> str:
        """Restore all placeholders in text with original content."""
        result = text
        for placeholder, original in self.mappings.items():
            result = result.replace(placeholder, original)
        return result
    
    def clear(self) -> None:
        """Clear all mappings."""
        self.mappings.clear()
        self.counters.clear()


# ============================================================================
# Pattern Definitions
# ============================================================================

# LaTeX inline math: $...$
LATEX_INLINE_PATTERN = re.compile(r'\$(?!\$)(.+?)(?<!\$)\$', re.DOTALL)

# LaTeX display math: $$...$$ or \[...\]
LATEX_DISPLAY_PATTERN = re.compile(
    r'(\$\$.+?\$\$|\\\[.+?\\\])', 
    re.DOTALL
)

# LaTeX environments: \begin{equation}...\end{equation}, etc.
LATEX_ENV_PATTERN = re.compile(
    r'\\begin\{(equation|align|gather|multline|eqnarray)\*?\}.*?\\end\{\1\*?\}',
    re.DOTALL
)

# Inline LaTeX commands that shouldn't be translated
LATEX_COMMAND_PATTERN = re.compile(
    r'\\(?:frac|sqrt|sum|int|prod|lim|log|ln|sin|cos|tan|exp|alpha|beta|gamma|'
    r'delta|epsilon|theta|lambda|mu|sigma|omega|pi|infty|partial|nabla|cdot|times|'
    r'leq|geq|neq|approx|equiv|subset|supset|cap|cup|forall|exists|rightarrow|'
    r'leftarrow|Rightarrow|Leftarrow|text|mathrm|mathbf|mathit|mathcal)\b'
)

# Code blocks (fenced)
CODE_BLOCK_PATTERN = re.compile(r'```[\s\S]*?```', re.MULTILINE)

# Inline code
INLINE_CODE_PATTERN = re.compile(r'`[^`\n]+`')

# URLs
URL_PATTERN = re.compile(
    r'https?://[^\s<>\[\]()"\']+'
    r'|www\.[^\s<>\[\]()"\']+'
)

# Email addresses
EMAIL_PATTERN = re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+')

# DOI patterns
DOI_PATTERN = re.compile(r'10\.\d{4,}/[^\s]+')

# Numbers with units (to preserve formatting)
NUMBER_UNIT_PATTERN = re.compile(
    r'\b\d+(?:\.\d+)?(?:\s*(?:Hz|kHz|MHz|GHz|nm|μm|mm|cm|m|km|'
    r'mg|g|kg|ms|s|min|h|°C|°F|K|V|mV|A|mA|W|kW|MW|J|kJ|MJ|Pa|kPa|MPa|'
    r'mol|mmol|μmol|L|mL|μL|%|ppm|ppb))\b',
    re.IGNORECASE
)


# ============================================================================
# Masking Functions
# ============================================================================

@dataclass
class MaskConfig:
    """Configuration for what content types to mask."""
    mask_latex_inline: bool = True
    mask_latex_display: bool = True
    mask_latex_env: bool = True
    mask_code_blocks: bool = True
    mask_inline_code: bool = True
    mask_urls: bool = True
    mask_emails: bool = True
    mask_dois: bool = True
    mask_numbers_units: bool = False  # Often want these translated contextually


def mask_text(
    text: str,
    registry: MaskRegistry,
    config: MaskConfig | None = None,
) -> str:
    """Apply all configured masks to text.
    
    Args:
        text: Input text to mask
        registry: MaskRegistry to store mappings
        config: Optional MaskConfig (uses defaults if None)
        
    Returns:
        Text with placeholders inserted
    """
    if config is None:
        config = MaskConfig()
    
    result = text
    
    # Order matters: mask larger patterns first to avoid nested issues
    
    # LaTeX environments (largest structures)
    if config.mask_latex_env:
        result = _apply_pattern(result, LATEX_ENV_PATTERN, registry, "MATHENV")
    
    # LaTeX display math
    if config.mask_latex_display:
        result = _apply_pattern(result, LATEX_DISPLAY_PATTERN, registry, "MATHDISP")
    
    # Code blocks (before inline to avoid conflicts)
    if config.mask_code_blocks:
        result = _apply_pattern(result, CODE_BLOCK_PATTERN, registry, "CODEBLK")
    
    # LaTeX inline math
    if config.mask_latex_inline:
        result = _apply_pattern(result, LATEX_INLINE_PATTERN, registry, "MATH")
    
    # Inline code
    if config.mask_inline_code:
        result = _apply_pattern(result, INLINE_CODE_PATTERN, registry, "CODE")
    
    # DOIs (before URLs to catch them specifically)
    if config.mask_dois:
        result = _apply_pattern(result, DOI_PATTERN, registry, "DOI")
    
    # URLs
    if config.mask_urls:
        result = _apply_pattern(result, URL_PATTERN, registry, "URL")
    
    # Emails
    if config.mask_emails:
        result = _apply_pattern(result, EMAIL_PATTERN, registry, "EMAIL")
    
    # Numbers with units
    if config.mask_numbers_units:
        result = _apply_pattern(result, NUMBER_UNIT_PATTERN, registry, "NUM")
    
    return result


def _apply_pattern(
    text: str,
    pattern: re.Pattern,
    registry: MaskRegistry,
    prefix: str,
) -> str:
    """Apply a regex pattern and replace matches with placeholders."""
    def replacer(match: re.Match) -> str:
        return registry.register(prefix, match.group(0))
    return pattern.sub(replacer, text)


def unmask_text(text: str, registry: MaskRegistry) -> str:
    """Restore all placeholders in text.
    
    Args:
        text: Text with placeholders
        registry: MaskRegistry containing mappings
        
    Returns:
        Text with original content restored
    """
    return registry.restore(text)


# ============================================================================
# Block-Level Operations
# ============================================================================

def mask_block(
    block: Block,
    registry: MaskRegistry,
    config: MaskConfig | None = None,
) -> None:
    """Mask content in a single block (mutates block.masked_text).
    
    For protected blocks (equations, code), the entire content is masked.
    For translatable blocks, specific patterns within the text are masked.
    """
    if block.is_protected:
        # Entire block is protected - mask it completely
        prefix = {
            "EQUATION": "MATHBLK",
            "CODE": "CODEBLK", 
            "FIGURE": "FIG",
        }.get(block.block_type.name, "PROT")
        placeholder = registry.register(prefix, block.source_text)
        block.masked_text = placeholder
    else:
        # Apply pattern-based masking within the text
        block.masked_text = mask_text(block.source_text, registry, config)


def unmask_block(block: Block, registry: MaskRegistry) -> None:
    """Restore placeholders in a block's translated text (mutates block)."""
    if block.translated_text:
        block.translated_text = unmask_text(block.translated_text, registry)


def mask_document(
    doc: Document,
    config: MaskConfig | None = None,
) -> MaskRegistry:
    """Apply masking to all blocks in a document.
    
    Args:
        doc: Document to mask
        config: Optional MaskConfig
        
    Returns:
        MaskRegistry containing all mappings (needed for unmasking)
    """
    registry = MaskRegistry()
    for block in doc.all_blocks:
        mask_block(block, registry, config)
    return registry


def unmask_document(doc: Document, registry: MaskRegistry) -> None:
    """Restore all placeholders in a document's translated blocks."""
    for block in doc.all_blocks:
        unmask_block(block, registry)


# ============================================================================
# Utility Functions
# ============================================================================

def count_placeholders(text: str) -> dict[str, int]:
    """Count placeholders by type in text.
    
    Useful for validation: the same placeholders should appear
    in both masked source and translated text.
    """
    pattern = re.compile(r'<<([A-Z]+)_\d{3}>>')
    counts: dict[str, int] = {}
    for match in pattern.finditer(text):
        prefix = match.group(1)
        counts[prefix] = counts.get(prefix, 0) + 1
    return counts


def validate_placeholders(source_masked: str, translated: str) -> list[str]:
    """Check that all placeholders from source appear in translation.
    
    Returns:
        List of missing placeholders (empty if all present)
    """
    source_placeholders = set(re.findall(r'<<[A-Z]+_\d{3}>>', source_masked))
    translated_placeholders = set(re.findall(r'<<[A-Z]+_\d{3}>>', translated))
    
    missing = source_placeholders - translated_placeholders
    return list(missing)


def extract_placeholders(text: str) -> list[str]:
    """Extract all placeholders from text."""
    return re.findall(r'<<[A-Z]+_\d{3}>>', text)

