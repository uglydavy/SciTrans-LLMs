from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, Optional

try:
    import requests
except Exception:  # pragma: no cover - optional dependency
    requests = None

from ..config import CACHE_DIR


class AdaptiveDictionary:
    """Hybrid local/remote dictionary with caching and simple scoring."""

    def __init__(self, cache_name: str = "adaptive_dictionary.json", ttl_hours: int = 72):
        self.cache_path = CACHE_DIR / cache_name
        self.ttl_hours = ttl_hours
        self._cache: Dict[str, Dict[str, str]] = {}
        self._load()

    def _load(self) -> None:
        if self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                self._cache = {}

    def _save(self) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache, indent=2), encoding="utf-8")
        except Exception:
            # Cache failures should never break translation
            pass

    def _expired(self, ts: float) -> bool:
        return (time.time() - ts) > self.ttl_hours * 3600

    def lookup(self, term: str, src: str, tgt: str) -> Optional[str]:
        key = term.lower().strip()
        if not key:
            return None
        cached = self._cache.get(key)
        if cached and not self._expired(cached.get("ts", 0)):
            return cached.get("translation")
        translation = self._fetch_online(key, src, tgt)
        if translation:
            self._cache[key] = {"translation": translation, "ts": time.time()}
            self._save()
        return translation

    def _fetch_online(self, term: str, src: str, tgt: str) -> Optional[str]:
        """Use a free endpoint (MyMemory) with graceful degradation."""

        if requests is None:
            return None
        params = {"q": term, "langpair": f"{src[:2]}|{tgt[:2]}", "de": "noreply@example.com"}
        try:
            resp = requests.get("https://api.mymemory.translated.net/get", params=params, timeout=5)
            if resp.status_code != 200:
                return None
            data = resp.json()
            cand = data.get("responseData", {}).get("translatedText")
            if cand and not re.search(r"</?i>", cand):
                return cand
            for match in data.get("matches", []):
                text = match.get("translation")
                if text and match.get("quality", 0) >= 50:
                    return text
        except Exception:
            return None
        return None
