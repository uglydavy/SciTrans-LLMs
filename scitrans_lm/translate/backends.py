from __future__ import annotations

import re
import warnings
from typing import Dict, List, Optional

from ..keys import get_key
from ..translate.memory import TranslationMemory
from .online_dictionary import AdaptiveDictionary


class BaseTranslator:
    def translate(
        self,
        texts: List[str],
        src: str,
        tgt: str,
        prompt: str = "",
        glossary: Dict[str, str] | None = None,
        context: Optional[TranslationMemory] = None,
    ) -> List[str]:
        raise NotImplementedError


class DictionaryTranslator(BaseTranslator):
    """Hybrid offline/online translator that aggressively enforces glossaries."""

    _BASE_LEXICON = {
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

    def __init__(self, mapping: Dict[str, str], adaptive: Optional[AdaptiveDictionary] = None):
        merged = dict(self._BASE_LEXICON)
        merged.update({k.lower(): v for k, v in (mapping or {}).items()})
        self.mapping = merged
        self.adaptive = adaptive or AdaptiveDictionary()

    def _patterns(self, glossary: Dict[str, str] | None, context: Optional[TranslationMemory]):
        merged = dict(self.mapping)
        for k, v in (glossary or {}).items():
            merged[k.lower()] = v
        if context:
            for src, tgt in context.glossary_candidates():
                merged.setdefault(src, tgt)
        terms = sorted(merged.items(), key=lambda kv: len(kv[0]), reverse=True)
        return [(re.compile(rf"(?i)\\b{re.escape(src)}\\b"), tgt) for src, tgt in terms]

    def _case_preserving_replace(self, source: str, replacement: str) -> str:
        if source.isupper():
            return replacement.upper()
        if source[:1].isupper():
            return replacement.capitalize()
        return replacement

    def _translate_words(self, text: str, glossary: Dict[str, str] | None, context: Optional[TranslationMemory], src: str, tgt: str) -> str:
        glossary = glossary or {}

        def convert(token: str) -> str:
            cleaned = re.sub(r"[^\w\-]", "", token)
            lower = cleaned.lower()
            if not lower:
                return token
            replacement = glossary.get(lower) or self.mapping.get(lower)
            if not replacement and context:
                for csrc, ctgt in context.snapshot():
                    if csrc.lower().startswith(lower):
                        replacement = ctgt
                        break
            if not replacement:
                replacement = self.adaptive.lookup(lower, src, tgt) or replacement
            if not replacement:
                return token
            return token.replace(cleaned, self._case_preserving_replace(cleaned, replacement))

        parts = re.split(r"(\W)", text)
        return "".join(convert(p) if i % 2 == 0 else p for i, p in enumerate(parts))

    def translate(self, texts, src, tgt, prompt: str = "", glossary=None, context: Optional[TranslationMemory] = None):
        patterns = self._patterns(glossary, context)
        out = []
        for t in texts:
            replaced = t
            for pat, repl in patterns:
                replaced = pat.sub(lambda m, repl=repl: self._case_preserving_replace(m.group(0), repl), replaced)
            replaced = self._translate_words(replaced, glossary, context, src, tgt)
            cleaned = replaced.strip() or t
            out.append(cleaned)
        return out


class _OpenAICompatTranslator(BaseTranslator):
    def __init__(self, model: str, api_key_name: str, base_url: Optional[str] = None, system_prompt: str = ""):
        self.model = model
        self.api_key = get_key(api_key_name)
        self.base_url = base_url
        self.system_prompt = system_prompt or "You are a precise scientific translator."
        self._client = None

    def _client_ok(self):
        if self._client is None:
            if not self.api_key:
                raise RuntimeError(
                    f"API key missing for {self.model}. Run: python3 -m scitrans_lm set-key {self.model.split('-')[0]}"
                )
            try:
                from openai import OpenAI

                kwargs = {"api_key": self.api_key}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                self._client = OpenAI(**kwargs)
            except Exception as e:
                raise RuntimeError("OpenAI-compatible SDK not installed. Run: pip install openai") from e

    def _build_messages(self, user_content: str):
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

    def translate(self, texts, src, tgt, prompt: str = "", glossary=None, context: Optional[TranslationMemory] = None):
        self._client_ok()
        results = []
        memory_hint = context.contextual_prompt() if context else ""
        for t in texts:
            content = (
                (prompt or "")
                + ("\n" + memory_hint if memory_hint else "")
                + f"\nTranslate the following from {src} to {tgt}. Preserve placeholders: [FORMULA_*] etc.\n\n{t}"
            )
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=self._build_messages(content),
                    temperature=0.2,
                )
                results.append(resp.choices[0].message.content.strip())
            except Exception as e:
                raise RuntimeError(
                    f"{self.model} translation failed. Check API key/credits or switch to offline dictionary mode."
                ) from e
        return results


class OpenAITranslator(_OpenAICompatTranslator):
    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__(model=model, api_key_name="openai", system_prompt="You are a precise scientific translator.")


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

    def translate(self, texts, src, tgt, prompt: str = "", glossary=None, context: Optional[TranslationMemory] = None):
        self._client_ok()
        tgt_code = "FR" if tgt.lower().startswith("fr") else "EN-GB"
        out = []
        for t in texts:
            res = self._client.translate_text(t, target_lang=tgt_code)
            out.append(res.text)
        return out


class GoogleTranslator(BaseTranslator):
    def __init__(self):
        self._client = None

    def _client_ok(self):
        if self._client is None:
            try:
                from google.cloud import translate_v2 as translate

                self._client = translate.Client()
            except Exception as e:
                raise RuntimeError(
                    "Google Translate backend not configured. Install google-cloud-translate and authenticate via GOOGLE_APPLICATION_CREDENTIALS."
                ) from e

    def translate(self, texts, src, tgt, prompt: str = "", glossary=None, context: Optional[TranslationMemory] = None):
        self._client_ok()
        tgt_code = "fr" if tgt.lower().startswith("fr") else "en"
        out = []
        for t in texts:
            res = self._client.translate(t, target_language=tgt_code, source_language="en" if tgt_code == "fr" else "fr")
            out.append(res["translatedText"])
        return out


class GoogleFreeTranslator(BaseTranslator):
    """Free, keyless translator built on googletrans (community API)."""

    def __init__(self):
        try:
            from googletrans import Translator

            self._client = Translator()
        except Exception as exc:
            raise RuntimeError(
                "googletrans not installed. Run: pip install googletrans==4.0.0-rc1"
            ) from exc

    def translate(self, texts, src, tgt, prompt: str = "", glossary=None, context: Optional[TranslationMemory] = None):
        tgt_code = "fr" if tgt.lower().startswith("fr") else "en"
        src_code = "en" if tgt_code == "fr" else "fr"
        results = []
        for text in texts:
            try:
                res = self._client.translate(text, src=src_code, dest=tgt_code)
                results.append(res.text)
            except Exception:
                results.append(text)
        return results


class DeepSeekTranslator(_OpenAICompatTranslator):
    def __init__(self, model: str = "deepseek-chat"):
        super().__init__(
            model=model,
            api_key_name="deepseek",
            base_url="https://api.deepseek.com",
            system_prompt="You are DeepSeek, a layout-aware translator. Use concise terminology and obey glossary hints.",
        )


class PerplexityTranslator(_OpenAICompatTranslator):
    def __init__(self, model: str = "llama-3-sonar-large-32k-chat"):
        super().__init__(
            model=model,
            api_key_name="perplexity",
            base_url="https://api.perplexity.ai",
            system_prompt="You are Perplexity, a factual translator. Cite glossary terms verbatim and avoid hallucination.",
        )


def get_translator(name: str, dictionary: Dict[str, str] | None = None) -> BaseTranslator:
    name = (name or "").lower()
    try:
        if name in ("openai", "gpt", "gpt4", "gpt-4o"):
            return OpenAITranslator()
        if name in ("deepl",):
            return DeepLTranslator()
        if name in ("google",):
            return GoogleTranslator()
        if name in ("google-free", "googlefree", "googletrans", "free"):
            return GoogleFreeTranslator()
        if name in ("deepseek",):
            return DeepSeekTranslator()
        if name in ("perplexity",):
            return PerplexityTranslator()
        if name in ("dictionary", "offline", "local", "lexicon"):
            return DictionaryTranslator(dictionary or {})
    except Exception as exc:
        warnings.warn(
            f"Falling back to offline dictionary translation because the requested backend failed to load: {exc}",
            RuntimeWarning,
        )
    return DictionaryTranslator(dictionary or {})
