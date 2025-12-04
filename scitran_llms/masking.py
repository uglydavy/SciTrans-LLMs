"""Masking module to protect non-translatable content."""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass
import hashlib


@dataclass
class MaskToken:
    """A masked token."""
    token: str
    content: str
    pattern: str


class Masker:
    """Masks and unmasks protected content."""
    
    def __init__(self):
        """Initialize masker."""
        self.masks: Dict[str, str] = {}
        self.patterns = {
            "latex_display": (r"\$\$[^$]+\$\$", "LATEX"),
            "latex_inline": (r"\$[^$]+\$", "LATEX"),
            "code_block": (r"```[^`]*```", "CODE"),
            "code_inline": (r"`[^`]+`", "CODE"),
            "url": (r"https?://[^\s]+", "URL"),
            "cite": (r"\\cite\{[^}]+\}", "CITE"),
            "ref": (r"\\ref\{[^}]+\}", "REF"),
            "label": (r"\\label\{[^}]+\}", "LABEL"),
        }
    
    def mask(self, text: str) -> str:
        """Mask protected content in text."""
        masked = text
        
        for name, (pattern, prefix) in self.patterns.items():
            matches = re.finditer(pattern, masked)
            for match in matches:
                content = match.group(0)
                # Create unique token
                hash_id = hashlib.md5(content.encode()).hexdigest()[:8]
                token = f"__{prefix}_{hash_id}__"
                
                # Store mapping
                self.masks[token] = content
                
                # Replace in text
                masked = masked.replace(content, token)
        
        return masked
    
    def unmask(self, text: str) -> str:
        """Restore masked content in text."""
        unmasked = text
        
        # Restore all masks
        for token, content in self.masks.items():
            unmasked = unmasked.replace(token, content)
        
        return unmasked
    
    def clear(self):
        """Clear all masks."""
        self.masks.clear()
