from __future__ import annotations
import re
from typing import Tuple, Optional

# Patterns for list markers that should be preserved
LIST_MARKER_PATTERNS = [
    r'^(\s*)([-•●○◦▪▸►])\s+',           # Bullet points
    r'^(\s*)(\d+\.)\s+',                  # Numbered: 1. 2. 3.
    r'^(\s*)(\d+\))\s+',                  # Numbered: 1) 2) 3)
    r'^(\s*)(\d+\.\d+\.?)\s+',            # Hierarchical: 1.1 1.2 2.1
    r'^(\s*)(\d+\.\d+\.\d+\.?)\s+',       # Deep hierarchical: 1.1.1
    r'^(\s*)([a-z]\.)\s+',                # Lettered: a. b. c.
    r'^(\s*)([a-z]\))\s+',                # Lettered: a) b) c)
    r'^(\s*)([ivxlcdm]+\.)\s+',           # Roman: i. ii. iii.
    r'^(\s*)([ivxlcdm]+\))\s+',           # Roman: i) ii) iii)
    r'^(\s*)(\([a-z0-9]+\))\s+',          # Parenthesized: (a) (1)
    r'^(\s*)(\[[a-z0-9]+\])\s+',          # Bracketed: [1] [a]
]


def extract_list_marker(text: str) -> Tuple[str, str, str]:
    """Extract list marker from text if present.
    
    Returns:
        Tuple of (indent, marker, content) or ('', '', text) if no marker
    """
    for pattern in LIST_MARKER_PATTERNS:
        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            indent = match.group(1)
            marker = match.group(2)
            content = text[match.end():]
            return (indent, marker, content)
    return ('', '', text)


def restore_list_marker(indent: str, marker: str, translated_content: str) -> str:
    """Restore list marker to translated content.
    
    Args:
        indent: Original indentation
        marker: Original list marker (e.g., "1.", "•", "a)")
        translated_content: Translated text content
        
    Returns:
        Text with list marker restored
    """
    if marker:
        return f"{indent}{marker} {translated_content.strip()}"
    return translated_content


def preserve_list_structure(source: str, translated: str) -> str:
    """Preserve list structure from source in translated text.
    
    If source has a list marker, ensure the translated text keeps it.
    """
    indent, marker, _ = extract_list_marker(source)
    if marker:
        # Check if translated already has a marker
        _, trans_marker, trans_content = extract_list_marker(translated)
        if trans_marker:
            # Replace with original marker
            return restore_list_marker(indent, marker, trans_content)
        else:
            # Add original marker
            return restore_list_marker(indent, marker, translated)
    return translated


def fix_spacing(text: str) -> str:
    # Single space after sentence-ending punctuation
    text = re.sub(r"([.!?])\s{2,}", r"\1 ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s{3,}", "  ", text)
    return text


def normalize(text: str) -> str:
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = fix_spacing(t)
    return t.strip()