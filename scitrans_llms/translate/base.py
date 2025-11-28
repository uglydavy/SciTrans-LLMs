"""
Base translator interface and implementations.

This module defines:
- Abstract Translator interface that all backends implement
- DummyTranslator for testing (echo or simple transformations)
- DictionaryTranslator for offline glossary-only translation

Thesis Contribution #1 & #2: The translator interface supports:
- Glossary-aware translation (term injection in prompts)
- Document-level context (previous translations passed to translator)
- Candidate generation for reranking

Design Philosophy:
- Translators are stateless: they receive all context in each call
- All translators return TranslationResult with metadata
- Easy to add new backends (OpenAI, DeepL, etc.)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from scitrans_llms.models import Block, Document
from scitrans_llms.translate.glossary import Glossary


@dataclass
class TranslationResult:
    """Result of a translation operation.
    
    Attributes:
        text: The translated text
        source_text: Original source text
        candidates: Alternative translations (for reranking)
        metadata: Additional info (model, tokens used, etc.)
        glossary_terms_used: Which glossary terms were applied
    """
    text: str
    source_text: str
    candidates: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    glossary_terms_used: list[str] = field(default_factory=list)
    
    @property
    def has_candidates(self) -> bool:
        return len(self.candidates) > 0


@dataclass
class TranslationContext:
    """Context passed to translator for document-level awareness.
    
    Thesis Contribution #2: This enables document-level translation
    by providing previous translations as context.
    
    Attributes:
        previous_source: Previous source segments (for context)
        previous_target: Previous translated segments
        document_summary: Optional summary of the document
        glossary: Glossary to enforce
        source_lang: Source language code
        target_lang: Target language code
    """
    previous_source: list[str] = field(default_factory=list)
    previous_target: list[str] = field(default_factory=list)
    document_summary: Optional[str] = None
    glossary: Optional[Glossary] = None
    source_lang: str = "en"
    target_lang: str = "fr"
    
    def get_context_window(self, max_segments: int = 3) -> str:
        """Format recent context for prompt injection."""
        if not self.previous_source:
            return ""
        
        lines = ["Recent context:"]
        start = max(0, len(self.previous_source) - max_segments)
        for i in range(start, len(self.previous_source)):
            src = self.previous_source[i][:200] + "..." if len(self.previous_source[i]) > 200 else self.previous_source[i]
            tgt = self.previous_target[i][:200] + "..." if len(self.previous_target[i]) > 200 else self.previous_target[i]
            lines.append(f"  [{self.source_lang}] {src}")
            lines.append(f"  [{self.target_lang}] {tgt}")
        return "\n".join(lines)


class Translator(ABC):
    """Abstract base class for all translation backends.
    
    All translators must implement:
    - translate(): Translate a single text segment
    - translate_batch(): Translate multiple segments (can be overridden for efficiency)
    
    Translators receive a TranslationContext with:
    - Previous translations for document-level coherence
    - Glossary for terminology enforcement
    - Language pair information
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the translator name (e.g., 'openai', 'deepl', 'dummy')."""
        pass
    
    @property
    def supports_candidates(self) -> bool:
        """Whether this translator can generate multiple candidates."""
        return False
    
    @abstractmethod
    def translate(
        self,
        text: str,
        context: TranslationContext | None = None,
        num_candidates: int = 1,
    ) -> TranslationResult:
        """Translate a single text segment.
        
        Args:
            text: Source text to translate
            context: Optional context with previous translations and glossary
            num_candidates: Number of translation candidates to generate
            
        Returns:
            TranslationResult with translation and metadata
        """
        pass
    
    def translate_batch(
        self,
        texts: list[str],
        context: TranslationContext | None = None,
    ) -> list[TranslationResult]:
        """Translate multiple segments.
        
        Default implementation calls translate() in a loop.
        Override for backends that support batching.
        """
        results = []
        for text in texts:
            result = self.translate(text, context)
            results.append(result)
            # Update context with this translation for next iteration
            if context is not None:
                context.previous_source.append(text)
                context.previous_target.append(result.text)
        return results
    
    def translate_block(
        self,
        block: Block,
        context: TranslationContext | None = None,
    ) -> TranslationResult:
        """Translate a Block, using masked_text if available."""
        text = block.masked_text if block.masked_text else block.source_text
        return self.translate(text, context)


class DummyTranslator(Translator):
    """A dummy translator for testing.
    
    Modes:
    - 'echo': Return the input unchanged
    - 'upper': Return uppercase version
    - 'prefix': Add [TRANSLATED] prefix
    - 'reverse': Reverse the text (for debugging)
    """
    
    def __init__(self, mode: str = "prefix"):
        self.mode = mode
    
    @property
    def name(self) -> str:
        return f"dummy-{self.mode}"
    
    def translate(
        self,
        text: str,
        context: TranslationContext | None = None,
        num_candidates: int = 1,
    ) -> TranslationResult:
        if self.mode == "echo":
            translated = text
        elif self.mode == "upper":
            translated = text.upper()
        elif self.mode == "reverse":
            translated = text[::-1]
        else:  # prefix
            translated = f"[TRANSLATED] {text}"
        
        return TranslationResult(
            text=translated,
            source_text=text,
            metadata={"translator": self.name, "mode": self.mode},
        )


class DictionaryTranslator(Translator):
    """Offline translator using only glossary lookups.
    
    This translator replaces known terms from the glossary
    but leaves unknown text unchanged. Useful as:
    - A fallback when API-based translators fail
    - A baseline for ablation studies
    - Testing glossary coverage
    """
    
    def __init__(self, glossary: Glossary | None = None):
        self.glossary = glossary
    
    @property
    def name(self) -> str:
        return "dictionary"
    
    def translate(
        self,
        text: str,
        context: TranslationContext | None = None,
        num_candidates: int = 1,
    ) -> TranslationResult:
        # Use context glossary if available, else use instance glossary
        glossary = None
        if context and context.glossary:
            glossary = context.glossary
        elif self.glossary:
            glossary = self.glossary
        
        if glossary is None:
            return TranslationResult(
                text=text,
                source_text=text,
                metadata={"translator": self.name, "warning": "no glossary provided"},
            )
        
        # Find and replace glossary terms
        result = text
        terms_used = []
        
        # Sort by length (longest first) to avoid partial replacements
        entries = sorted(glossary.entries, key=lambda e: len(e.source), reverse=True)
        
        for entry in entries:
            import re
            pattern = re.compile(rf'\b{re.escape(entry.source)}\b', re.IGNORECASE)
            if pattern.search(result):
                # Preserve original case for first letter
                def replace_with_case(match):
                    matched = match.group(0)
                    target = entry.target
                    if matched[0].isupper():
                        target = target[0].upper() + target[1:]
                    return target
                
                result = pattern.sub(replace_with_case, result)
                terms_used.append(entry.source)
        
        return TranslationResult(
            text=result,
            source_text=text,
            metadata={"translator": self.name},
            glossary_terms_used=terms_used,
        )


def create_translator(backend: str, **kwargs) -> Translator:
    """Factory function to create a translator by name.
    
    Args:
        backend: Translator backend name ('dummy', 'dictionary', 'openai', etc.)
        **kwargs: Backend-specific arguments
        
    Returns:
        Configured Translator instance
    """
    backend_lower = backend.lower()
    
    if backend_lower in ("dummy", "echo", "test"):
        mode = kwargs.get("mode", "prefix")
        return DummyTranslator(mode=mode)
    
    elif backend_lower in ("dictionary", "offline", "glossary"):
        glossary = kwargs.get("glossary")
        return DictionaryTranslator(glossary=glossary)
    
    elif backend_lower in ("openai", "gpt", "gpt4", "gpt-4", "gpt5", "gpt-5", "gpt-5.1"):
        from scitrans_llms.translate.llm import OpenAITranslator, LLMConfig
        # Support GPT-5.1 and other newer models
        model = kwargs.get("model", "gpt-4o")
        if backend_lower in ("gpt5", "gpt-5", "gpt-5.1"):
            model = "gpt-5.1" if backend_lower == "gpt-5.1" else "gpt-5"
        config = kwargs.get("config") or LLMConfig(model=model)
        return OpenAITranslator(config=config, api_key=kwargs.get("api_key"))
    
    elif backend_lower in ("improved-offline", "improved", "offline-improved"):
        from scitrans_llms.translate.offline import ImprovedOfflineTranslator
        glossary = kwargs.get("glossary")
        model_path = kwargs.get("model_path")
        learned_model = kwargs.get("learned_model")
        return ImprovedOfflineTranslator(glossary=glossary, learned_model=learned_model, model_path=model_path)
    
    elif backend_lower in ("huggingface", "hf", "hugging-face"):
        from scitrans_llms.translate.free_apis import HuggingFaceTranslator
        model = kwargs.get("model", "facebook/mbart-large-50-many-to-many-mmt")
        api_key = kwargs.get("api_key")
        return HuggingFaceTranslator(model=model, api_key=api_key)
    
    elif backend_lower in ("ollama", "local-llm"):
        from scitrans_llms.translate.free_apis import OllamaTranslator
        model = kwargs.get("model", "llama3")
        base_url = kwargs.get("base_url", "http://localhost:11434")
        return OllamaTranslator(model=model, base_url=base_url)
    
    elif backend_lower in ("googlefree", "google-free", "googletrans"):
        from scitrans_llms.translate.free_apis import GoogleFreeTranslator
        return GoogleFreeTranslator()
    
    elif backend_lower in ("deepseek", "ds"):
        from scitrans_llms.translate.llm import DeepSeekTranslator, LLMConfig
        config = kwargs.get("config") or LLMConfig(model=kwargs.get("model", "deepseek-chat"))
        return DeepSeekTranslator(config=config, api_key=kwargs.get("api_key"))
    
    elif backend_lower in ("anthropic", "claude"):
        from scitrans_llms.translate.llm import AnthropicTranslator, LLMConfig
        config = kwargs.get("config") or LLMConfig(model=kwargs.get("model", "claude-3-sonnet-20240229"))
        return AnthropicTranslator(config=config, api_key=kwargs.get("api_key"))
    
    else:
        raise ValueError(f"Unknown translator backend: {backend}")

