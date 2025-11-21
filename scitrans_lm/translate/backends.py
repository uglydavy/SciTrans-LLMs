from __future__ import annotations
import re
import warnings
from typing import Dict, List

from ..keys import get_key


class BaseTranslator:
    def translate(self, texts: List[str], src: str, tgt: str, prompt: str = "", glossary: Dict[str, str] | None = None) -> List[str]:
        raise NotImplementedError


class DictionaryTranslator(BaseTranslator):
    def __init__(self, mapping: Dict[str, str]):
        self.mapping = {k.lower(): v for k, v in (mapping or {}).items()}

    def _patterns(self, glossary: Dict[str, str] | None):
        merged = dict(self.mapping)
        for k, v in (glossary or {}).items():
            merged[k.lower()] = v
        terms = sorted(merged.items(), key=lambda kv: len(kv[0]), reverse=True)
        return [(re.compile(rf"(?i)\\b{re.escape(src)}\\b"), tgt) for src, tgt in terms]

    def translate(self, texts, src, tgt, prompt: str = "", glossary=None):
        patterns = self._patterns(glossary)
        out = []
        for t in texts:
            replaced = t
            for pat, repl in patterns:
                replaced = pat.sub(repl, replaced)
            out.append(replaced)
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
    def __init__(self):
        try:
            from googletrans import Translator  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime import
            raise RuntimeError(
                "googletrans not installed. Run: pip install googletrans==4.0.0-rc1"
            ) from exc
        self._client = Translator()

    @staticmethod
    def _lang(code: str) -> str:
        code = (code or "en").lower()
        if code.startswith("fr"):
            return "fr"
        return "en"

    def translate(self, texts, src, tgt, prompt: str = "", glossary=None):
        src_code, tgt_code = self._lang(src), self._lang(tgt)
        out: List[str] = []
        for t in texts:
            try:
                res = self._client.translate(t, src=src_code, dest=tgt_code)
                translated = res.text
            except Exception as exc:  # pragma: no cover - network failure fallback
                raise RuntimeError(
                    "Google free translation failed (network / throttling)."
                ) from exc
            out.append(translated)
        return out


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
        if name in ("google", "google-free", "googletrans"):
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
