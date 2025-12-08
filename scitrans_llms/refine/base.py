"""
Base refinement interface and implementations.

Refinement is a post-translation step that improves:
- Coherence across paragraphs (pronoun consistency, flow)
- Terminology adherence (glossary enforcement)
- Style and fluency
- Placeholder preservation

Thesis Contribution #2: Document-level refinement is a key
differentiator from segment-level translation.

Design Philosophy:
- Refiners are composable (can chain multiple refiners)
- Each refiner focuses on one aspect
- All refiners preserve placeholders
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from scitrans_llms.models import Document, Block
from scitrans_llms.translate.glossary import Glossary, enforce_glossary
from scitrans_llms.masking import validate_placeholders


@dataclass
class RefinementResult:
    """Result of a refinement operation.
    
    Attributes:
        original_text: Text before refinement
        refined_text: Text after refinement
        changes_made: List of changes applied
        metadata: Additional information (model, tokens, etc.)
    """
    original_text: str
    refined_text: str
    changes_made: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    
    @property
    def was_changed(self) -> bool:
        return self.original_text != self.refined_text


class Refiner(ABC):
    """Abstract base class for refinement passes.
    
    Refiners take translated text and improve it according to
    specific criteria (coherence, glossary, style, etc.).
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the refiner name."""
        pass
    
    @abstractmethod
    def refine(
        self,
        translated_text: str,
        source_text: Optional[str] = None,
        context: Optional[str] = None,
        glossary: Optional[Glossary] = None,
    ) -> RefinementResult:
        """Refine translated text.
        
        Args:
            translated_text: The translation to refine
            source_text: Original source (for reference)
            context: Surrounding context (previous paragraphs)
            glossary: Glossary for terminology enforcement
            
        Returns:
            RefinementResult with refined text
        """
        pass
    
    def refine_block(
        self,
        block: Block,
        source_text: Optional[str] = None,
        context: Optional[str] = None,
        glossary: Optional[Glossary] = None,
    ) -> RefinementResult:
        """Refine a Block's translated_text (convenience method)."""
        if not block.translated_text:
            return RefinementResult(
                original_text="",
                refined_text="",
                metadata={"warning": "no translated text to refine"},
            )
        
        result = self.refine(
            translated_text=block.translated_text,
            source_text=source_text or block.source_text,
            context=context,
            glossary=glossary,
        )
        
        # Update block in place
        block.translated_text = result.refined_text
        return result
    
    def refine_document(
        self,
        doc: Document,
        glossary: Optional[Glossary] = None,
    ) -> list[RefinementResult]:
        """Refine all blocks in a document."""
        results = []
        context_parts = []
        
        for block in doc.all_blocks:
            if not block.is_translatable or not block.translated_text:
                continue
            
            # Build context from previous translations
            context = "\n\n".join(context_parts[-3:]) if context_parts else None
            
            result = self.refine_block(
                block=block,
                context=context,
                glossary=glossary,
            )
            results.append(result)
            
            # Add to context for next block
            context_parts.append(block.translated_text)
        
        return results


class NoOpRefiner(Refiner):
    """A no-op refiner that returns text unchanged.
    
    Useful for:
    - Testing the pipeline without refinement
    - Ablation studies (comparing with/without refinement)
    - Placeholder when LLM refinement is not available
    """
    
    @property
    def name(self) -> str:
        return "noop"
    
    def refine(
        self,
        translated_text: str,
        source_text: Optional[str] = None,
        context: Optional[str] = None,
        glossary: Optional[Glossary] = None,
    ) -> RefinementResult:
        return RefinementResult(
            original_text=translated_text,
            refined_text=translated_text,
            changes_made=[],
            metadata={"refiner": self.name},
        )


class GlossaryRefiner(Refiner):
    """Refiner that enforces glossary terms in translations.
    
    This is a rule-based refiner that:
    1. Finds glossary terms that appear in source
    2. Checks if corresponding target terms appear in translation
    3. Replaces incorrect translations with glossary terms
    
    Thesis Contribution #1: Terminology enforcement as post-processing.
    """
    
    def __init__(self, default_glossary: Optional[Glossary] = None):
        self.default_glossary = default_glossary
    
    @property
    def name(self) -> str:
        return "glossary"
    
    def refine(
        self,
        translated_text: str,
        source_text: Optional[str] = None,
        context: Optional[str] = None,
        glossary: Optional[Glossary] = None,
    ) -> RefinementResult:
        # Use provided glossary or fall back to default
        active_glossary = glossary or self.default_glossary
        
        if not active_glossary or not source_text:
            return RefinementResult(
                original_text=translated_text,
                refined_text=translated_text,
                changes_made=[],
                metadata={"refiner": self.name, "warning": "no glossary or source"},
            )
        
        # Apply glossary enforcement
        refined = enforce_glossary(
            translated_text=translated_text,
            source_text=source_text,
            glossary=active_glossary,
        )
        
        # Track what changed
        changes = []
        if refined != translated_text:
            changes.append("glossary terms enforced")
        
        return RefinementResult(
            original_text=translated_text,
            refined_text=refined,
            changes_made=changes,
            metadata={"refiner": self.name},
        )


class PlaceholderValidator(Refiner):
    """Refiner that validates and fixes placeholder preservation.
    
    This checks that all placeholders from the source appear in
    the translation, and attempts to restore any that are missing.
    """
    
    @property
    def name(self) -> str:
        return "placeholder_validator"
    
    def refine(
        self,
        translated_text: str,
        source_text: Optional[str] = None,
        context: Optional[str] = None,
        glossary: Optional[Glossary] = None,
    ) -> RefinementResult:
        if not source_text:
            return RefinementResult(
                original_text=translated_text,
                refined_text=translated_text,
                changes_made=[],
                metadata={"refiner": self.name},
            )
        
        # Check for missing placeholders
        missing = validate_placeholders(source_text, translated_text)
        
        changes = []
        refined = translated_text
        
        if missing:
            # Try to restore missing placeholders
            # This is a simple heuristic: append them at the end
            # A more sophisticated approach would try to find likely positions
            for placeholder in missing:
                changes.append(f"restored missing placeholder: {placeholder}")
                refined = refined.rstrip() + " " + placeholder
        
        return RefinementResult(
            original_text=translated_text,
            refined_text=refined,
            changes_made=changes,
            metadata={
                "refiner": self.name,
                "missing_placeholders": missing,
            },
        )


class CompositeRefiner(Refiner):
    """Chains multiple refiners in sequence.
    
    Each refiner's output becomes the input for the next one.
    This allows combining different refinement strategies.
    """
    
    def __init__(self, refiners: list[Refiner]):
        self.refiners = refiners
    
    @property
    def name(self) -> str:
        names = [r.name for r in self.refiners]
        return f"composite({', '.join(names)})"
    
    def refine(
        self,
        translated_text: str,
        source_text: Optional[str] = None,
        context: Optional[str] = None,
        glossary: Optional[Glossary] = None,
    ) -> RefinementResult:
        current_text = translated_text
        all_changes = []
        all_metadata = {"refiner": self.name, "steps": []}
        
        for refiner in self.refiners:
            result = refiner.refine(
                translated_text=current_text,
                source_text=source_text,
                context=context,
                glossary=glossary,
            )
            current_text = result.refined_text
            all_changes.extend(result.changes_made)
            all_metadata["steps"].append({
                "refiner": refiner.name,
                "changed": result.was_changed,
            })
        
        return RefinementResult(
            original_text=translated_text,
            refined_text=current_text,
            changes_made=all_changes,
            metadata=all_metadata,
        )


def create_refiner(
    mode: str = "default",
    glossary: Optional[Glossary] = None,
    api_key: Optional[str] = None,
) -> Refiner:
    """Factory function to create refiners.
    
    Modes:
    - 'none' or 'noop': No refinement
    - 'glossary': Glossary enforcement only
    - 'default': Glossary + placeholder validation
    - 'llm': LLM-based coherence refinement (requires OpenAI API key)
    - 'full': Glossary + placeholder + LLM refinement (best quality)
    
    Args:
        mode: Refiner mode
        glossary: Glossary for terminology enforcement
        api_key: OpenAI API key for LLM refinement (optional, uses env var if not provided)
    """
    import os
    mode_lower = mode.lower()
    
    if mode_lower in ("none", "noop"):
        return NoOpRefiner()
    
    elif mode_lower == "glossary":
        return GlossaryRefiner(default_glossary=glossary)
    
    elif mode_lower in ("default", "standard"):
        return CompositeRefiner([
            GlossaryRefiner(default_glossary=glossary),
            PlaceholderValidator(),
        ])
    
    elif mode_lower == "llm":
        # Use LLM-based refiner if API key is available
        key = api_key or os.getenv("OPENAI_API_KEY")
        if key:
            from scitrans_llms.refine.llm import LLMRefiner
            return LLMRefiner(api_key=key)
        else:
            # Fall back to default if no API key
            import warnings
            warnings.warn("No OpenAI API key found, falling back to glossary refiner")
            return CompositeRefiner([
                GlossaryRefiner(default_glossary=glossary),
                PlaceholderValidator(),
            ])
    
    elif mode_lower == "full":
        # Full refinement: glossary + placeholder + LLM
        key = api_key or os.getenv("OPENAI_API_KEY")
        refiners = [
            GlossaryRefiner(default_glossary=glossary),
            PlaceholderValidator(),
        ]
        if key:
            from scitrans_llms.refine.llm import LLMRefiner
            refiners.append(LLMRefiner(api_key=key))
        return CompositeRefiner(refiners)
    
    else:
        raise ValueError(f"Unknown refiner mode: {mode}")

