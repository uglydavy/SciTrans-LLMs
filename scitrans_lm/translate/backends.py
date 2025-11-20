
from __future__ import annotations
from typing import Dict, List
from ..keys import get_key

class BaseTranslator:
    def translate(self, texts: List[str], src: str, tgt: str, prompt: str = "", glossary: Dict[str,str] | None = None) -> List[str]:
        raise NotImplementedError

class DictionaryTranslator(BaseTranslator):
    def __init__(self, mapping: Dict[str,str]):
        self.mapping = mapping or {}

    def translate(self, texts, src, tgt, prompt: str = "", glossary=None):
        # extremely naive word-level replacement
        out = []
        for t in texts:
            tokens = t.split()
            mapped = [self.mapping.get(tok.lower(), tok) for tok in tokens]
            out.append(" ".join(mapped))
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
                raise RuntimeError("OpenAI SDK not installed or API key missing. Run: pip install openai && python3 -m scitrans_lm set-key openai") from e

    def translate(self, texts, src, tgt, prompt: str = "", glossary=None):
        self._client_ok()
        results = []
        for t in texts:
            content = (prompt or "") + f"\nTranslate the following from {src} to {tgt}. Preserve placeholders: [FORMULA_*] etc.\n\n{t}"
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role":"system","content":"You are a precise scientific translator."},
                              {"role":"user","content":content}],
                    temperature=0.2,
                )
                results.append(resp.choices[0].message.content.strip())
            except Exception as e:
                raise
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
                raise RuntimeError("deepl SDK not installed or API key missing. Run: pip install deepl && python3 -m scitrans_lm set-key deepl") from e

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

def get_translator(name: str, dictionary: Dict[str,str] | None = None) -> BaseTranslator:
    name = (name or "").lower()
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
    # default offline dictionary
    return DictionaryTranslator(dictionary or {})
