"""Dictionary-based translator."""

from typing import Dict, Any
from .base import BaseTranslator


class DictionaryTranslator(BaseTranslator):
    """Simple dictionary-based translator."""
    
    def __init__(self, source_lang: str = "en", target_lang: str = "fr"):
        """Initialize the translator."""
        super().__init__(source_lang, target_lang)
        self.dictionary = self._load_dictionary()
    
    def _load_dictionary(self) -> Dict[str, str]:
        """Load translation dictionary."""
        # Basic technical dictionary for demo
        if self.source_lang == "en" and self.target_lang == "fr":
            return {
                "quantum": "quantique",
                "mechanical": "mécanique",
                "wave": "onde",
                "function": "fonction",
                "particle": "particule",
                "probability": "probabilité",
                "amplitude": "amplitude",
                "state": "état",
                "energy": "énergie",
                "momentum": "impulsion",
                "equation": "équation",
                "system": "système",
                "theory": "théorie",
                "model": "modèle",
                "analysis": "analyse",
                "algorithm": "algorithme",
                "neural": "neuronal",
                "network": "réseau",
                "learning": "apprentissage",
                "machine": "machine",
            }
        return {}
    
    def translate(self, text: str, **kwargs) -> Dict[str, Any]:
        """Translate text using dictionary."""
        if not text:
            return {"translation": ""}
        
        # Simple word-by-word translation
        words = text.split()
        translated = []
        
        for word in words:
            # Preserve punctuation
            punct = ""
            clean_word = word.lower()
            if clean_word and clean_word[-1] in ".,!?;:":
                punct = clean_word[-1]
                clean_word = clean_word[:-1]
            
            # Translate or keep original
            if clean_word in self.dictionary:
                translated_word = self.dictionary[clean_word]
                # Preserve capitalization
                if word[0].isupper():
                    translated_word = translated_word.capitalize()
                translated.append(translated_word + punct)
            else:
                translated.append(word)
        
        return {
            "translation": " ".join(translated),
            "confidence": 0.5,
            "backend": "dictionary"
        }
