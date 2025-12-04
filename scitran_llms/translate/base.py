"""Base translator class."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class TranslationResult:
    """Result from a translation."""
    translation: str
    confidence: float = 0.0
    candidates: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTranslator(ABC):
    """Abstract base class for translators."""
    
    def __init__(self, source_lang: str = "en", target_lang: str = "fr"):
        """Initialize the translator."""
        self.source_lang = source_lang
        self.target_lang = target_lang
    
    @abstractmethod
    def translate(self, text: str, **kwargs) -> Dict[str, Any]:
        """Translate text."""
        pass
    
    def batch_translate(self, texts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Translate multiple texts."""
        return [self.translate(text, **kwargs) for text in texts]
