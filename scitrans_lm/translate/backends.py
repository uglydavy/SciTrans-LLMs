from __future__ import annotations
import re
import warnings
from typing import Dict, List
from ..keys import get_key


class BaseTranslator:
    def translate(self, texts: List[str], src: str, tgt: str, prompt: str = "", glossary: Dict[str, str] | None = None) -> List[str]:
        raise NotImplementedError


class DictionaryTranslator(BaseTranslator):
    """Lightweight offline translator with aggressive glossary usage.

    The previous implementation performed a single pass of regex
    substitutions which often resulted in English output or empty strings
    after masking. This variant performs:

    - Phrase-first replacement (longest match wins)
    - Word-level fallbacks with a small built-in seed lexicon
    - Case-aware replacement so titles/subtitles keep capitalization
    - A safety fallback that returns the source text if everything else
      collapses, preventing blank overlays in the renderer.
    """

    _BASE_LEXICON = {
        # Common helpers and science terms to ensure visible output offline
        "the": "le",
        "a": "un",
        "and": "et",
        "of": "de",
        "for": "pour",
        "with": "avec",
        "results": "résultats",
        "methods": "méthodes",
        "introduction": "introduction",
        "conclusion": "conclusion",
        "analysis": "analyse",
        "figure": "figure",
        "table": "tableau",
        "equation": "équation",
        "data": "données",
        "model": "modèle",
        "experiment": "expérience",
        "samples": "échantillons",
        "abstract": "résumé",
        "references": "références",
    }

    def __init__(self, mapping: Dict[str, str]):
        merged = dict(self._BASE_LEXICON)
        merged.update({k.lower(): v for k, v in (mapping or {}).items()})
        self.mapping = merged

    def _patterns(self, glossary: Dict[str, str] | None):
        merged = dict(self.mapping)
        for k, v in (glossary or {}).items():
            merged[k.lower()] = v
        terms = sorted(merged.items(), key=lambda kv: len(kv[0]), reverse=True)
        return [(re.compile(rf"(?i)\\b{re.escape(src)}\\b"), tgt) for src, tgt in terms]

    def _case_preserving_replace(self, source: str, replacement: str) -> str:
        if source.isupper():
            return replacement.upper()
        if source[0].isupper():
            return replacement.capitalize()
        return replacement

    def _translate_words(self, text: str, glossary: Dict[str, str] | None) -> str:
        glossary = glossary or {}
        def convert(token: str) -> str:
            cleaned = re.sub(r"[^\w\-]", "", token)
            lower = cleaned.lower()
            if not lower:
                return token
            replacement = glossary.get(lower) or self.mapping.get(lower)
            if not replacement:
                return token
            return token.replace(cleaned, self._case_preserving_replace(cleaned, replacement))

        parts = re.split(r"(\W)", text)
        return "".join(convert(p) if i % 2 == 0 else p for i, p in enumerate(parts))

    def translate(self, texts, src, tgt, prompt: str = "", glossary=None):
        patterns = self._patterns(glossary)
        out = []
        for t in texts:
            replaced = t
            for pat, repl in patterns:
                replaced = pat.sub(lambda m, repl=repl: self._case_preserving_replace(m.group(0), repl), replaced)
            replaced = self._translate_words(replaced, glossary)
            cleaned = replaced.strip() or t
            out.append(cleaned)
        return out


class OpenAITranslator(BaseTranslator):
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.api_key = get_key("openai")
        self._client = None

    def _client_ok(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except Exception as e:
                raise RuntimeError(
                    "OpenAI SDK not installed or API key missing. Run: pip install openai && python3 -m scitrans_lm set-key openai"
                ) from e

    def translate(self, texts, src, tgt, prompt: str = "", glossary=None):
        self._client_ok()
        results = []
        for t in texts:
            content = (prompt or "") + f"\nTranslate the following from {src} to {tgt}. Preserve placeholders: [FORMULA_*] etc.\n\n{t}"
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a precise scientific translator."},
                        {"role": "user", "content": content},
                    ],
                    temperature=0.2,
                )
                results.append(resp.choices[0].message.content.strip())
            except Exception as e:
                raise RuntimeError(
                    "OpenAI translation failed. Check API key/credits or switch to offline dictionary mode."
                ) from e
        return results


class DeepLTranslator(BaseTranslator):
    def __init__(self):
        self.api_key = get_key("deepl")
        self._client = None

    def _client_ok(self):
        if self._client is None:
            try:
                import deepl
                self._client = deepl.Translator(self.api_key)
            except Exception as e:
                raise RuntimeError(
                    "deepl SDK not installed or API key missing. Run: pip install deepl && python3 -m scitrans_lm set-key deepl"
                ) from e

    def translate(self, texts, src, tgt, prompt: str = "", glossary=None):
        self._client_ok()
        tgt_code = "FR" if tgt.lower().startswith("fr") else "EN-GB"
        out = []
        for t in texts:
            res = self._client.translate_text(t, target_lang=tgt_code)
            out.append(res.text)
        return out


class GoogleTranslator(BaseTranslator):
    def translate(self, texts, src, tgt, prompt: str = "", glossary=None):
        # Placeholder - require google-cloud-translate setup
        raise RuntimeError("Google Translate backend not configured. Install google-cloud-translate and implement credentials.")


class DeepSeekTranslator(BaseTranslator):
    def translate(self, texts, src, tgt, prompt: str = "", glossary=None):
        raise RuntimeError("DeepSeek backend placeholder. Provide API and SDK.")


class PerplexityTranslator(BaseTranslator):
    def translate(self, texts, src, tgt, prompt: str = "", glossary=None):
        raise RuntimeError("Perplexity backend placeholder. Provide API and SDK.")


def get_translator(name: str, dictionary: Dict[str, str] | None = None) -> BaseTranslator:
    name = (name or "").lower()
    try:
        if name in ("openai", "gpt", "gpt4", "gpt-4o"):
            return OpenAITranslator()
        if name in ("deepl",):
            return DeepLTranslator()
        if name in ("google",):
            return GoogleTranslator()
        if name in ("deepseek",):
            return DeepSeekTranslator()
        if name in ("perplexity",):
            return PerplexityTranslator()
    except Exception as exc:
        warnings.warn(
            f"Falling back to offline dictionary translation because the requested backend failed to load: {exc}",
            RuntimeWarning,
        )
    # default offline dictionary
    return DictionaryTranslator(dictionary or {})
