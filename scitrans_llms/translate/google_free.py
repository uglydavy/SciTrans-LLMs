"""
Google Free translator using googletrans library.

This translator uses the unofficial Google Translate API via googletrans.
It doesn't require an API key but has rate limits and may break if 
Google changes their interface.

For production use, prefer the official Google Cloud Translation API.
"""

from __future__ import annotations

from typing import Optional

from scitrans_llms.translate.base import (
    Translator,
    TranslationResult,
    TranslationContext,
)


class GoogleFreeTranslator(Translator):
    """Free Google Translate via googletrans library.
    
    Note: This uses an unofficial API and may be rate-limited or break.
    For production, use the official Google Cloud Translation API.
    
    Usage:
        translator = GoogleFreeTranslator()
        result = translator.translate("Hello world", context)
    """
    
    def __init__(self):
        self._translator = None
    
    @property
    def name(self) -> str:
        return "google-free"
    
    def _get_translator(self):
        """Lazy initialization of googletrans translator."""
        if self._translator is None:
            try:
                from googletrans import Translator as GoogleTranslator
            except ImportError:
                raise ImportError(
                    "googletrans library required. Install with: "
                    "pip install googletrans==4.0.0-rc1"
                )
            
            self._translator = GoogleTranslator()
        
        return self._translator
    
    def translate(
        self,
        text: str,
        context: Optional[TranslationContext] = None,
        num_candidates: int = 1,
    ) -> TranslationResult:
        """Translate text using Google Translate (free).
        
        Args:
            text: Source text to translate
            context: Optional context (used for language codes)
            num_candidates: Ignored (only one translation supported)
            
        Returns:
            TranslationResult with translation
        """
        translator = self._get_translator()
        
        # Determine source and target languages
        src_lang = "en"
        dest_lang = "fr"
        if context:
            src_lang = self._normalize_lang(context.source_lang)
            dest_lang = self._normalize_lang(context.target_lang)
        
        try:
            result = translator.translate(
                text,
                src=src_lang,
                dest=dest_lang,
            )
            translated = result.text
        except Exception as e:
            raise RuntimeError(f"Google Free translation failed: {e}")
        
        return TranslationResult(
            text=translated,
            source_text=text,
            metadata={
                "translator": self.name,
                "src_lang": src_lang,
                "dest_lang": dest_lang,
            },
        )
    
    def _normalize_lang(self, lang: str) -> str:
        """Normalize language code for googletrans.
        
        googletrans uses ISO 639-1 codes (e.g., 'en', 'fr').
        """
        # Map common variations
        lang_map = {
            "english": "en",
            "french": "fr",
            "german": "de",
            "spanish": "es",
            "italian": "it",
            "portuguese": "pt",
            "chinese": "zh-cn",
            "japanese": "ja",
            "korean": "ko",
            "russian": "ru",
            "arabic": "ar",
        }
        
        lang_lower = lang.lower().strip()
        
        # Check if it's already a valid code
        if len(lang_lower) == 2:
            return lang_lower
        
        # Try to map
        return lang_map.get(lang_lower, lang_lower[:2])

