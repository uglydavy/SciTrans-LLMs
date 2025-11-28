"""
Free API translation backends for testing without API balance.

This module provides:
- Hugging Face Inference API (free tier available)
- Ollama (local, completely free)
- Google Translate Free API (via googletrans)
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Optional
import requests

from scitrans_llms.translate.base import (
    Translator,
    TranslationResult,
    TranslationContext,
)


class HuggingFaceTranslator(Translator):
    """Hugging Face Inference API translator (free tier available).
    
    Usage:
        translator = HuggingFaceTranslator(model="facebook/mbart-large-50-many-to-many-mmt")
        result = translator.translate("Hello world")
    
    Free tier: https://huggingface.co/pricing
    - Free tier: 1000 requests/month
    - No credit card required for free tier
    """
    
    def __init__(
        self,
        model: str = "facebook/mbart-large-50-many-to-many-mmt",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
        self.api_url = f"https://api-inference.huggingface.co/models/{model}"
    
    @property
    def name(self) -> str:
        return f"huggingface-{self.model.split('/')[-1]}"
    
    def translate(
        self,
        text: str,
        context: Optional[TranslationContext] = None,
        num_candidates: int = 1,
    ) -> TranslationResult:
        """Translate using Hugging Face Inference API."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Determine language pair from context
        source_lang = context.source_lang if context else "en"
        target_lang = context.target_lang if context else "fr"
        
        # Map language codes to mbart codes if using mbart model
        lang_map = {
            "en": "en_XX",
            "fr": "fr_XX",
            "de": "de_DE",
            "es": "es_XX",
            "it": "it_IT",
            "pt": "pt_XX",
            "ru": "ru_RU",
            "zh": "zh_CN",
            "ja": "ja_XX",
            "ko": "ko_KR",
            "ar": "ar_AR",
            "hi": "hi_IN",
        }
        
        src_code = lang_map.get(source_lang, f"{source_lang}_XX")
        tgt_code = lang_map.get(target_lang, f"{target_lang}_XX")
        
        payload = {
            "inputs": text,
            "parameters": {
                "src_lang": src_code,
                "tgt_lang": tgt_code,
            }
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            
            result_data = response.json()
            
            # Handle different response formats
            if isinstance(result_data, list) and len(result_data) > 0:
                translated_text = result_data[0].get("translation_text", result_data[0].get("generated_text", text))
            elif isinstance(result_data, dict):
                translated_text = result_data.get("translation_text", result_data.get("generated_text", text))
            else:
                translated_text = str(result_data) if result_data else text
            
            return TranslationResult(
                text=translated_text,
                source_text=text,
                metadata={
                    "translator": self.name,
                    "model": self.model,
                    "api_used": "huggingface",
                },
            )
        except requests.exceptions.RequestException as e:
            return TranslationResult(
                text=text,
                source_text=text,
                metadata={
                    "translator": self.name,
                    "error": str(e),
                    "warning": "Translation failed, returning original text",
                },
            )


class OllamaTranslator(Translator):
    """Ollama translator (local, completely free).
    
    Requires Ollama to be installed and running locally.
    
    Installation:
        1. Install Ollama: https://ollama.ai
        2. Pull a translation model: ollama pull llama3
        3. Use this translator
    
    Usage:
        translator = OllamaTranslator(model="llama3", base_url="http://localhost:11434")
        result = translator.translate("Hello world")
    """
    
    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/api/generate"
    
    @property
    def name(self) -> str:
        return f"ollama-{self.model}"
    
    def translate(
        self,
        text: str,
        context: Optional[TranslationContext] = None,
        num_candidates: int = 1,
    ) -> TranslationResult:
        """Translate using Ollama."""
        source_lang = context.source_lang if context else "English"
        target_lang = context.target_lang if context else "French"
        
        # Build prompt
        prompt = f"""Translate the following text from {source_lang} to {target_lang}.
Preserve all placeholders exactly as they appear (e.g., <<MATH_001>>, <<URL_002>>).
Do not add explanations, only provide the translation.

Text to translate:
{text}

Translation:"""
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
            }
        }
        
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=120,  # Ollama can be slow
            )
            response.raise_for_status()
            
            result_data = response.json()
            translated_text = result_data.get("response", text).strip()
            
            # Clean up response (remove prompt if included)
            if "Translation:" in translated_text:
                translated_text = translated_text.split("Translation:")[-1].strip()
            
            return TranslationResult(
                text=translated_text,
                source_text=text,
                metadata={
                    "translator": self.name,
                    "model": self.model,
                    "api_used": "ollama",
                },
            )
        except requests.exceptions.RequestException as e:
            return TranslationResult(
                text=text,
                source_text=text,
                metadata={
                    "translator": self.name,
                    "error": str(e),
                    "warning": "Translation failed, returning original text. Make sure Ollama is running.",
                },
            )


class GoogleFreeTranslator(Translator):
    """Google Translate Free API (via googletrans library).
    
    This uses the unofficial googletrans library which is free but:
    - May have rate limits
    - Not officially supported by Google
    - May break if Google changes their API
    
    Usage:
        translator = GoogleFreeTranslator()
        result = translator.translate("Hello world")
    """
    
    def __init__(self):
        try:
            from googletrans import Translator as GoogleTrans
            self._translator = GoogleTrans()
        except ImportError:
            raise ImportError(
                "googletrans required. Install with: pip install googletrans==4.0.0rc1"
            )
    
    @property
    def name(self) -> str:
        return "google-free"
    
    def translate(
        self,
        text: str,
        context: Optional[TranslationContext] = None,
        num_candidates: int = 1,
    ) -> TranslationResult:
        """Translate using Google Translate Free API."""
        source_lang = context.source_lang if context else "en"
        target_lang = context.target_lang if context else "fr"
        
        try:
            result = self._translator.translate(
                text,
                src=source_lang,
                dest=target_lang,
            )
            
            return TranslationResult(
                text=result.text,
                source_text=text,
                metadata={
                    "translator": self.name,
                    "api_used": "googletrans",
                },
            )
        except Exception as e:
            return TranslationResult(
                text=text,
                source_text=text,
                metadata={
                    "translator": self.name,
                    "error": str(e),
                    "warning": "Translation failed, returning original text",
                },
            )

