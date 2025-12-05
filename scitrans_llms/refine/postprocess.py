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
    
    If source has list markers, ensure the translated text keeps them.
    Handles both single-line and multi-line text.
    """
    # Handle multi-line text: process each line separately
    source_lines = source.split('\n')
    translated_lines = translated.split('\n')
    
    # If source has multiple lines with markers, try to restore them
    if len(source_lines) > 1:
        # Extract markers from source lines
        source_markers = []
        for line in source_lines:
            indent, marker, _ = extract_list_marker(line)
            source_markers.append((indent, marker))
        
        # Count how many source lines have markers
        markers_count = sum(1 for _, m in source_markers if m)
        
        if markers_count > 0:
            # Build result with markers restored
            result_lines = []
            marker_idx = 0
            
            for i, trans_line in enumerate(translated_lines):
                trans_line = trans_line.strip()
                if not trans_line:
                    result_lines.append('')
                    continue
                
                # Find next marker to apply
                while marker_idx < len(source_markers):
                    indent, marker = source_markers[marker_idx]
                    marker_idx += 1
                    if marker:
                        # Check if translated line already has a marker
                        _, existing_marker, content = extract_list_marker(trans_line)
                        if existing_marker:
                            result_lines.append(restore_list_marker(indent, marker, content))
                        else:
                            result_lines.append(restore_list_marker(indent, marker, trans_line))
                        break
                else:
                    # No more markers, keep line as-is
                    result_lines.append(trans_line)
            
            return '\n'.join(result_lines)
    
    # Single line handling
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