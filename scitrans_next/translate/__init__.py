"""
Translation module for SciTrans-Next.

This module provides:
- Translator base class and interface
- Glossary loading and enforcement
- Document-level context management
- Multiple backend implementations (dummy, dictionary, LLM)
- Candidate reranking
"""

from scitrans_next.translate.glossary import (
    Glossary,
    GlossaryEntry,
    GlossaryMatch,
    load_glossary_csv,
    load_glossary_txt,
    enforce_glossary,
    check_glossary_adherence,
    get_default_glossary,
)
from scitrans_next.translate.base import (
    Translator,
    TranslationResult,
    TranslationContext,
    DummyTranslator,
    DictionaryTranslator,
    create_translator,
)
from scitrans_next.translate.context import (
    DocumentContext,
    ContextWindow,
    extract_document_summary,
    extract_entities,
)

__all__ = [
    # Glossary
    "Glossary",
    "GlossaryEntry",
    "GlossaryMatch",
    "load_glossary_csv",
    "load_glossary_txt",
    "enforce_glossary",
    "check_glossary_adherence",
    "get_default_glossary",
    # Translator
    "Translator",
    "TranslationResult",
    "TranslationContext",
    "DummyTranslator",
    "DictionaryTranslator",
    "create_translator",
    # Context
    "DocumentContext",
    "ContextWindow",
    "extract_document_summary",
    "extract_entities",
]

# Lazy imports for LLM backends (avoid import errors if openai not installed)
def __getattr__(name):
    if name in ("OpenAITranslator", "DeepSeekTranslator", "AnthropicTranslator", 
                "LLMConfig", "create_llm_translator", "MultiTurnTranslator"):
        from scitrans_next.translate import llm
        return getattr(llm, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
