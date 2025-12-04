"""Google Translate unofficial API translator."""

from typing import Dict, Any
from .base import BaseTranslator
import time


class GoogleTranslator(BaseTranslator):
    """Google Translate using googletrans library."""
    
    def __init__(self, source_lang: str = "en", target_lang: str = "fr"):
        """Initialize the translator."""
        super().__init__(source_lang, target_lang)
        self.translator = None
        self._init_translator()
    
    def _init_translator(self):
        """Initialize the googletrans translator."""
        try:
            from googletrans import Translator
            self.translator = Translator()
        except ImportError:
            print("Warning: googletrans not installed. Install with: pip install googletrans==4.0.0rc1")
            self.translator = None
    
    def translate(self, text: str, **kwargs) -> Dict[str, Any]:
        """Translate text using Google Translate."""
        if not text or not text.strip():
            return {"translation": ""}
        
        if not self.translator:
            # Fallback to original if googletrans not available
            return {
                "translation": text,
                "confidence": 0.0,
                "backend": "googletrans",
                "error": "googletrans not installed"
            }
        
        try:
            # Google Translate can handle longer texts
            result = self.translator.translate(
                text, 
                src=self.source_lang,
                dest=self.target_lang
            )
            
            return {
                "translation": result.text,
                "confidence": 0.9,
                "backend": "googletrans"
            }
            
        except Exception as e:
            print(f"Google Translate error: {str(e)}")
            # Fallback to original
            return {
                "translation": text,
                "confidence": 0.0,
                "backend": "googletrans",
                "error": str(e)
            }
