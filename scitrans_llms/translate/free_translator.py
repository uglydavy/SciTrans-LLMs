"""
Comprehensive free translation backend with cascading fallbacks and caching.

This module provides a meta-translator that tries multiple free services:
1. Lingva Translate (privacy-focused Google Translate frontend)
2. LibreTranslate (open source, self-hosted option)
3. MyMemory (translation memory API)

All translations are cached locally for:
- Reduced API calls
- Faster repeated translations
- Offline capability for cached content
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

from .base import Translator, TranslationResult, TranslationContext


@dataclass
class CacheEntry:
    """A cached translation entry."""
    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    backend: str
    timestamp: float


class TranslationCache:
    """Local cache for translations."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        if cache_dir is None:
            from scitrans_llms.config import DATA_DIR
            cache_dir = DATA_DIR / "cache" / "translations"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "translation_cache.json"
        self._cache: Dict[str, CacheEntry] = {}
        self._load_cache()
    
    def _make_key(self, text: str, source_lang: str, target_lang: str) -> str:
        """Create a cache key from text and languages."""
        content = f"{source_lang}:{target_lang}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_cache(self):
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for key, entry_data in data.items():
                        self._cache[key] = CacheEntry(**entry_data)
            except Exception:
                pass
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            data = {
                key: {
                    'source_text': entry.source_text,
                    'translated_text': entry.translated_text,
                    'source_lang': entry.source_lang,
                    'target_lang': entry.target_lang,
                    'backend': entry.backend,
                    'timestamp': entry.timestamp,
                }
                for key, entry in self._cache.items()
            }
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    
    def get(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """Get cached translation if available."""
        key = self._make_key(text, source_lang, target_lang)
        entry = self._cache.get(key)
        if entry:
            return entry.translated_text
        return None
    
    def set(self, text: str, translation: str, source_lang: str, target_lang: str, backend: str):
        """Cache a translation."""
        key = self._make_key(text, source_lang, target_lang)
        self._cache[key] = CacheEntry(
            source_text=text,
            translated_text=translation,
            source_lang=source_lang,
            target_lang=target_lang,
            backend=backend,
            timestamp=time.time()
        )
        self._save_cache()
    
    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        if self.cache_file.exists():
            self.cache_file.unlink()


class FreeTranslator(Translator):
    """Meta-translator that tries multiple free services with caching.
    
    Translation cascade:
    1. Check local cache first (instant, offline)
    2. Try Lingva Translate (privacy-focused, fast)
    3. Try LibreTranslate (open source, may need self-hosting)
    4. Try MyMemory (reliable, rate limited)
    5. Fall back to basic dictionary
    
    All successful translations are cached locally.
    """
    
    # Public Lingva instances (privacy-focused Google Translate frontends)
    LINGVA_INSTANCES = [
        "https://lingva.ml",
        "https://translate.plausibility.cloud",
        "https://translate.projectsegfau.lt",
    ]
    
    # Public LibreTranslate instances
    LIBRETRANSLATE_INSTANCES = [
        "https://libretranslate.com",
        "https://translate.argosopentech.com",
    ]
    
    def __init__(self, cache_dir: Optional[Path] = None, timeout: float = 5.0):
        """Initialize free translator.
        
        Args:
            cache_dir: Directory for translation cache
            timeout: Request timeout in seconds
        """
        self.cache = TranslationCache(cache_dir)
        self.timeout = timeout
        self._stats = {
            'cache_hits': 0,
            'lingva_success': 0,
            'libretranslate_success': 0,
            'mymemory_success': 0,
            'fallback_used': 0,
        }
    
    @property
    def name(self) -> str:
        return "free-cascade"
    
    def _try_lingva(self, text: str, source: str, target: str) -> Optional[str]:
        """Try Lingva Translate instances."""
        for instance in self.LINGVA_INSTANCES:
            try:
                # Lingva API: GET /api/v1/{source}/{target}/{query}
                encoded_text = urllib.parse.quote(text)
                url = f"{instance}/api/v1/{source}/{target}/{encoded_text}"
                
                req = urllib.request.Request(url)
                req.add_header('User-Agent', 'SciTrans-LLMs/0.1.0')
                
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode())
                    translation = data.get('translation')
                    
                    if translation and translation.strip():
                        self._stats['lingva_success'] += 1
                        return translation.strip()
            except Exception:
                continue
        
        return None
    
    def _try_libretranslate(self, text: str, source: str, target: str) -> Optional[str]:
        """Try LibreTranslate instances."""
        for instance in self.LIBRETRANSLATE_INSTANCES:
            try:
                url = f"{instance}/translate"
                data = json.dumps({
                    'q': text,
                    'source': source,
                    'target': target,
                    'format': 'text'
                }).encode()
                
                req = urllib.request.Request(url, data=data)
                req.add_header('Content-Type', 'application/json')
                req.add_header('User-Agent', 'SciTrans-LLMs/0.1.0')
                
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    result = json.loads(response.read().decode())
                    translation = result.get('translatedText')
                    
                    if translation and translation.strip():
                        self._stats['libretranslate_success'] += 1
                        return translation.strip()
            except Exception:
                continue
        
        return None
    
    def _try_mymemory(self, text: str, source: str, target: str) -> Optional[str]:
        """Try MyMemory Translation API."""
        try:
            params = {
                'q': text,
                'langpair': f'{source}|{target}'
            }
            url = f"https://api.mymemory.translated.net/get?{urllib.parse.urlencode(params)}"
            
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'SciTrans-LLMs/0.1.0')
            
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                data = json.loads(response.read().decode())
                
                if data.get('responseStatus') == 200:
                    translation = data.get('responseData', {}).get('translatedText', '')
                    if translation and translation.strip():
                        self._stats['mymemory_success'] += 1
                        return translation.strip()
        except Exception:
            pass
        
        return None
    
    def _fallback_dictionary(self, text: str) -> str:
        """Basic dictionary fallback for common words."""
        # Simple word-by-word translation for most common words
        basic_dict = {
            'hello': 'bonjour', 'world': 'monde', 'the': 'le', 'a': 'un',
            'is': 'est', 'are': 'sont', 'machine': 'machine', 'learning': 'apprentissage',
            'neural': 'neuronal', 'network': 'réseau', 'deep': 'profond',
            'model': 'modèle', 'data': 'données', 'algorithm': 'algorithme',
            'how': 'comment', 'are': 'êtes', 'you': 'vous',
        }
        
        words = text.lower().split()
        translated_words = [basic_dict.get(word, word) for word in words]
        return ' '.join(translated_words)
    
    def translate(
        self,
        text: str,
        context: Optional[TranslationContext] = None
    ) -> TranslationResult:
        """Translate text using cascading free services with caching."""
        if not text or not text.strip():
            return TranslationResult(
                text=text,
                source_text=text,
                metadata={'translator': self.name, 'backend': 'none'}
            )
        
        source_lang = context.source_lang if context else 'en'
        target_lang = context.target_lang if context else 'fr'
        
        # Normalize language codes (Lingva uses 2-letter codes)
        source = source_lang[:2].lower()
        target = target_lang[:2].lower()
        
        # 1. Try cache first (instant, offline)
        cached = self.cache.get(text, source, target)
        if cached:
            self._stats['cache_hits'] += 1
            return TranslationResult(
                text=cached,
                source_text=text,
                metadata={
                    'translator': self.name,
                    'backend': 'cache',
                    'cached': True
                }
            )
        
        translation = None
        backend_used = None
        
        # 2. Try Lingva (fast, privacy-focused)
        translation = self._try_lingva(text, source, target)
        if translation:
            backend_used = 'lingva'
        
        # 3. Try LibreTranslate (open source)
        if not translation:
            translation = self._try_libretranslate(text, source, target)
            if translation:
                backend_used = 'libretranslate'
        
        # 4. Try MyMemory (reliable)
        if not translation:
            translation = self._try_mymemory(text, source, target)
            if translation:
                backend_used = 'mymemory'
        
        # 5. Fall back to basic dictionary
        if not translation:
            translation = self._fallback_dictionary(text)
            backend_used = 'dictionary_fallback'
            self._stats['fallback_used'] += 1
        
        # Cache successful translation
        if translation and backend_used != 'dictionary_fallback':
            self.cache.set(text, translation, source, target, backend_used)
        
        return TranslationResult(
            text=translation,
            source_text=text,
            metadata={
                'translator': self.name,
                'backend': backend_used,
                'cached': False
            }
        )
    
    def get_stats(self) -> Dict[str, int]:
        """Get translation statistics."""
        return self._stats.copy()
    
    def clear_cache(self):
        """Clear the translation cache."""
        self.cache.clear()
        self._stats = {k: 0 for k in self._stats}


# Convenience function for batch translation
def batch_translate_free(
    texts: List[str],
    source_lang: str = "en",
    target_lang: str = "fr",
    show_progress: bool = True
) -> List[str]:
    """Translate multiple texts using the free translator.
    
    Args:
        texts: List of texts to translate
        source_lang: Source language code
        target_lang: Target language code
        show_progress: Whether to show progress
        
    Returns:
        List of translated texts
    """
    translator = FreeTranslator()
    results = []
    
    for i, text in enumerate(texts):
        if show_progress and i % 10 == 0:
            print(f"Translating: {i}/{len(texts)}...")
        
        from scitrans_llms.translate.base import TranslationContext
        context = TranslationContext(
            source_lang=source_lang,
            target_lang=target_lang
        )
        
        result = translator.translate(text, context)
        results.append(result.text)
        
        # Rate limiting
        if i % 10 == 0:
            time.sleep(0.5)
    
    if show_progress:
        stats = translator.get_stats()
        print(f"\nTranslation complete!")
        print(f"  Cache hits: {stats['cache_hits']}")
        print(f"  Lingva: {stats['lingva_success']}")
        print(f"  LibreTranslate: {stats['libretranslate_success']}")
        print(f"  MyMemory: {stats['mymemory_success']}")
        print(f"  Fallback: {stats['fallback_used']}")
    
    return results

