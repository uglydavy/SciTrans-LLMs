"""
Main translation pipeline for SciTrans-Next.

This module orchestrates the complete translation workflow:
1. Parse document (from text or PDF)
2. Apply masking to protect formulas, code, etc.
3. Translate with document-level context
4. Refine translations (glossary, coherence)
5. Unmask placeholders
6. Render output (text or PDF)

Thesis Contributions:
- #1: Terminology-constrained translation via glossary + masking
- #2: Document-level LLM context and refinement
- #3: Pluggable components for ablation studies

Design Philosophy:
- Pipeline is configurable via PipelineConfig
- Each stage is independent and testable
- Progress callbacks for GUI/CLI integration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional
from pathlib import Path

from scitrans_next.models import Document, Block
from scitrans_next.masking import (
    MaskRegistry,
    MaskConfig,
    mask_document,
    unmask_document,
)
from scitrans_next.translate.base import (
    Translator,
    TranslationContext,
    DummyTranslator,
    create_translator,
)
from scitrans_next.translate.glossary import Glossary, get_default_glossary
from scitrans_next.translate.context import DocumentContext
from scitrans_next.refine.base import Refiner, create_refiner


# Type alias for progress callbacks
ProgressCallback = Callable[[str, float], None]


@dataclass
class PipelineConfig:
    """Configuration for the translation pipeline.
    
    This controls all aspects of the translation process
    and enables ablation studies by toggling components.
    """
    # Translation settings
    source_lang: str = "en"
    target_lang: str = "fr"
    translator_backend: str = "dummy"  # 'dummy', 'dictionary', 'openai', etc.
    translator_kwargs: dict = field(default_factory=dict)
    
    # Masking settings
    enable_masking: bool = True
    mask_config: MaskConfig = field(default_factory=MaskConfig)
    
    # Glossary settings
    enable_glossary: bool = True
    glossary: Optional[Glossary] = None  # Will use default if None
    glossary_in_prompt: bool = True  # Include glossary in translator prompt
    glossary_post_process: bool = True  # Enforce glossary after translation
    
    # Document-level context
    enable_context: bool = True
    context_window_size: int = 5  # Number of previous segments
    
    # Refinement settings
    enable_refinement: bool = True
    refiner_mode: str = "default"  # 'none', 'glossary', 'default', 'llm'
    
    # Candidate generation (for reranking)
    num_candidates: int = 1  # >1 enables candidate generation
    
    def to_dict(self) -> dict:
        """Serialize config for logging/debugging."""
        return {
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "translator_backend": self.translator_backend,
            "enable_masking": self.enable_masking,
            "enable_glossary": self.enable_glossary,
            "enable_context": self.enable_context,
            "enable_refinement": self.enable_refinement,
            "refiner_mode": self.refiner_mode,
            "num_candidates": self.num_candidates,
        }


@dataclass
class PipelineResult:
    """Result of running the translation pipeline.
    
    Contains the translated document plus metadata about
    the translation process (useful for evaluation/debugging).
    """
    document: Document
    config: PipelineConfig
    mask_registry: Optional[MaskRegistry] = None
    stats: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    
    @property
    def success(self) -> bool:
        return len(self.errors) == 0
    
    @property
    def translated_text(self) -> str:
        return self.document.translated_text


class TranslationPipeline:
    """Main translation pipeline orchestrating all components.
    
    Usage:
        config = PipelineConfig(translator_backend="openai")
        pipeline = TranslationPipeline(config)
        
        doc = Document.from_text("Hello world")
        result = pipeline.translate(doc)
        
        print(result.translated_text)
    """
    
    def __init__(
        self,
        config: PipelineConfig | None = None,
        progress_callback: ProgressCallback | None = None,
    ):
        self.config = config or PipelineConfig()
        self.progress_callback = progress_callback or (lambda msg, pct: None)
        
        # Initialize components
        self._init_components()
    
    def _init_components(self) -> None:
        """Initialize pipeline components based on config."""
        # Translator
        self.translator = create_translator(
            backend=self.config.translator_backend,
            **self.config.translator_kwargs,
        )
        
        # Glossary
        if self.config.enable_glossary:
            self.glossary = self.config.glossary or get_default_glossary()
        else:
            self.glossary = None
        
        # Refiner
        if self.config.enable_refinement:
            self.refiner = create_refiner(
                mode=self.config.refiner_mode,
                glossary=self.glossary,
            )
        else:
            self.refiner = None
    
    def translate(self, document: Document) -> PipelineResult:
        """Run the complete translation pipeline on a document.
        
        Steps:
        1. Masking: Protect formulas, code, URLs, etc.
        2. Translation: Translate each block with context
        3. Refinement: Enforce glossary, check coherence
        4. Unmasking: Restore protected content
        
        Args:
            document: Input Document to translate
            
        Returns:
            PipelineResult with translated document and stats
        """
        errors = []
        stats = {
            "total_blocks": len(document.all_blocks),
            "translated_blocks": 0,
            "skipped_blocks": 0,
            "refined_blocks": 0,
        }
        
        # Step 1: Masking
        self.progress_callback("Applying masks...", 0.1)
        mask_registry = None
        if self.config.enable_masking:
            mask_registry = mask_document(document, self.config.mask_config)
            stats["masks_applied"] = len(mask_registry.mappings)
        
        # Attach glossary to document
        if self.glossary:
            document.glossary_terms = self.glossary.to_dict()
        
        # Step 2: Translation
        self.progress_callback("Translating...", 0.2)
        
        # Build document context
        doc_context = DocumentContext.from_document(
            doc=document,
            glossary=self.glossary,
            window_size=self.config.context_window_size,
        )
        
        # Translate blocks
        translatable_blocks = [b for b in document.all_blocks if b.is_translatable]
        total_translatable = len(translatable_blocks)
        
        for i, block in enumerate(translatable_blocks):
            # Progress update
            progress = 0.2 + (0.5 * (i / max(total_translatable, 1)))
            self.progress_callback(f"Translating block {i+1}/{total_translatable}...", progress)
            
            try:
                # Build context for this translation
                context = TranslationContext(
                    previous_source=doc_context.context_window.source_segments.copy(),
                    previous_target=doc_context.context_window.target_segments.copy(),
                    glossary=self.glossary if self.config.glossary_in_prompt else None,
                    source_lang=document.source_lang,
                    target_lang=document.target_lang,
                )
                
                # Translate
                result = self.translator.translate_block(block, context)
                block.translated_text = result.text
                
                # Update context window
                source = block.masked_text or block.source_text
                doc_context.add_translation(source, result.text)
                
                stats["translated_blocks"] += 1
                
            except Exception as e:
                errors.append(f"Translation failed for block {block.block_id}: {e}")
                block.translated_text = block.source_text  # Fallback
        
        # Handle non-translatable blocks (copy source or leave empty)
        for block in document.all_blocks:
            if block.is_protected:
                block.translated_text = block.source_text
                stats["skipped_blocks"] += 1
        
        # Step 3: Refinement
        self.progress_callback("Refining translations...", 0.75)
        if self.refiner:
            refine_results = self.refiner.refine_document(document, self.glossary)
            stats["refined_blocks"] = sum(1 for r in refine_results if r.was_changed)
        
        # Step 4: Unmasking
        self.progress_callback("Restoring protected content...", 0.9)
        if mask_registry:
            unmask_document(document, mask_registry)
        
        self.progress_callback("Complete!", 1.0)
        
        return PipelineResult(
            document=document,
            config=self.config,
            mask_registry=mask_registry,
            stats=stats,
            errors=errors,
        )
    
    def translate_text(self, text: str) -> str:
        """Convenience method to translate plain text.
        
        Creates a Document from text, translates, and returns the result.
        """
        doc = Document.from_text(
            text,
            source_lang=self.config.source_lang,
            target_lang=self.config.target_lang,
        )
        result = self.translate(doc)
        return result.translated_text


# ============================================================================
# Convenience Functions
# ============================================================================

def translate_text(
    text: str,
    source_lang: str = "en",
    target_lang: str = "fr",
    backend: str = "dummy",
    enable_glossary: bool = True,
    enable_refinement: bool = True,
) -> str:
    """Quick translation of plain text.
    
    This is the simplest API for translating text:
    
        result = translate_text("Hello world", backend="openai")
    
    For more control, use TranslationPipeline directly.
    """
    config = PipelineConfig(
        source_lang=source_lang,
        target_lang=target_lang,
        translator_backend=backend,
        enable_glossary=enable_glossary,
        enable_refinement=enable_refinement,
    )
    pipeline = TranslationPipeline(config)
    return pipeline.translate_text(text)


def translate_document(
    document: Document,
    config: PipelineConfig | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PipelineResult:
    """Translate a Document with optional config and progress callback."""
    pipeline = TranslationPipeline(config, progress_callback)
    return pipeline.translate(document)

