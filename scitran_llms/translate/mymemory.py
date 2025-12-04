"""MyMemory API translator - Free translation service."""

import requests
from typing import Dict, Any, List
import time
import re
from .base import BaseTranslator


class MyMemoryTranslator(BaseTranslator):
    """MyMemory translation API - free, no key required."""
    
    API_URL = "https://api.mymemory.translated.net/get"
    
    def __init__(self, source_lang: str = "en", target_lang: str = "fr"):
        """Initialize the translator."""
        super().__init__(source_lang, target_lang)
        self.session = requests.Session()
    
    def translate(self, text: str, **kwargs) -> Dict[str, Any]:
        """Translate text using MyMemory API."""
        if not text or not text.strip():
            return {"translation": ""}
        
        # MyMemory has a 500 char limit for free tier
        # So we need to split long text into chunks
        if len(text) > 500:
            # Split into chunks
            chunks = []
            words = text.split()
            current_chunk = []
            current_length = 0
            
            for word in words:
                word_length = len(word) + 1  # +1 for space
                if current_length + word_length > 490:  # Leave some margin
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                    current_chunk = [word]
                    current_length = word_length
                else:
                    current_chunk.append(word)
                    current_length += word_length
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            # Translate each chunk
            translated_chunks = []
            for chunk in chunks:
                chunk_result = self._translate_chunk(chunk)
                translated_chunks.append(chunk_result["translation"])
                time.sleep(0.3)  # Rate limiting
            
            # Join translated chunks
            return {
                "translation": " ".join(translated_chunks),
                "confidence": 0.7,
                "backend": "mymemory"
            }
        else:
            # Text is short enough for single request
            return self._translate_chunk(text)
    
    def _translate_chunk(self, text: str) -> Dict[str, Any]:
        """Translate a single chunk (max 500 chars)."""
        params = {
            "q": text[:500],
            "langpair": f"{self.source_lang}|{self.target_lang}"
        }
        
        try:
            # Make request
            response = self.session.get(self.API_URL, params=params, timeout=10)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            if data.get("responseStatus") == 200:
                translation = data.get("responseData", {}).get("translatedText", text)
                match = data.get("responseData", {}).get("match", 0)
                
                return {
                    "translation": translation,
                    "confidence": float(match),
                    "backend": "mymemory"
                }
            else:
                # Fallback to original text
                return {
                    "translation": text,
                    "confidence": 0.0,
                    "backend": "mymemory",
                    "error": data.get("responseDetails", "Unknown error")
                }
                
        except requests.exceptions.RequestException as e:
            # Network error - fallback to original
            return {
                "translation": text,
                "confidence": 0.0,
                "backend": "mymemory",
                "error": str(e)
            }
        except Exception as e:
            # Other error
            return {
                "translation": text,
                "confidence": 0.0,
                "backend": "mymemory",
                "error": str(e)
            }
