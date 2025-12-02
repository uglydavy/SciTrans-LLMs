"""
Main translation pipeline for SciTrans-LLMs.

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

from scitrans_llms.models import Document, Block
from scitrans_llms.masking import (
    MaskRegistry,
    MaskConfig,
    mask_document,
    unmask_document,
)
from scitrans_llms.translate.base import (
    Translator,
    TranslationContext,
    DummyTranslator,
    create_translator,
)
from scitrans_llms.translate.glossary import Glossary, get_default_glossary
from scitrans_llms.translate.context import DocumentContext
from scitrans_llms.refine.base import Refiner, create_refiner
from scitrans_llms.refine.postprocess import preserve_list_structure


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
        translator_kwargs = dict(self.config.translator_kwargs)
        # Always pass language pair to backends that care (e.g., dictionary)
        translator_kwargs.setdefault("source_lang", self.config.source_lang)
        translator_kwargs.setdefault("target_lang", self.config.target_lang)
        base_translator = create_translator(
            backend=self.config.translator_backend,
            **translator_kwargs,
        )
        
        # Wrap with reranking if num_candidates > 1
        if self.config.num_candidates > 1:
            from scitrans_llms.refine.rerank import RerankedTranslator
            self.translator = RerankedTranslator(
                base_translator=base_translator,
                num_candidates=self.config.num_candidates,
            )
        else:
            self.translator = base_translator
        
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
        first_translation_metadata: dict | None = None
        
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
                
                # Translate with candidates if reranking enabled
                num_cands = self.config.num_candidates if self.config.num_candidates > 1 else 1
                
                # Use translate() directly to get candidates
                text_to_translate = block.masked_text if block.masked_text else block.source_text
                result = self.translator.translate(text_to_translate, context, num_candidates=num_cands)

                # Capture metadata for the very first translation (for GUI prompt inspection)
                if first_translation_metadata is None:
                    try:
                        first_translation_metadata = dict(result.metadata or {})
                    except Exception:
                        first_translation_metadata = {}
                
                # If we have multiple candidates, use reranking
                if num_cands > 1 and (result.candidates or num_cands > 1):
                    from scitrans_llms.refine.rerank import CandidateReranker
                    reranker = CandidateReranker(use_llm_scoring=False)
                    
                    # Collect all candidates
                    candidates = [result.text]
                    if result.candidates:
                        candidates.extend(result.candidates[:num_cands-1])
                    elif num_cands > 1:
                        # Generate more candidates if translator doesn't support it
                        for _ in range(num_cands - 1):
                            alt_result = self.translator.translate(text_to_translate, context, num_candidates=1)
                            if alt_result.text not in candidates:
                                candidates.append(alt_result.text)
                    
                    if len(candidates) > 1:
                        rerank_result = reranker.rerank(
                            source_text=text_to_translate,
                            candidates=candidates,
                            context=context,
                            glossary=self.glossary,
                            source_masked=block.masked_text,
                        )
                        translated_text = rerank_result.best_candidate
                    else:
                        translated_text = result.text
                else:
                    translated_text = result.text
                
                # Preserve list structure (1., 1.1, 2., a), etc.)
                source_text = block.masked_text or block.source_text
                translated_text = preserve_list_structure(source_text, translated_text)
                
                block.translated_text = translated_text
                
                # Update context window
                doc_context.add_translation(source_text, translated_text)
                
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
        
        # Expose metadata from the first translation call (if any) for tooling/GUI
        if first_translation_metadata:
            stats["first_translation_metadata"] = first_translation_metadata
        
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
    input_path: str,
    output_path: str,
    engine: str = "dictionary",
    direction: str = "en-fr",
    pages: str = "all",
    preserve_figures: bool = True,
    quality_loops: int = 3,
    enable_rerank: bool = True,
    use_mineru: bool = True,  # MinerU enforced by default
    num_candidates: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> PipelineResult:
    """Translate a PDF document and save the result.
    
    This is the main entry point for the GUI.
    
    Args:
        input_path: Path to input PDF
        output_path: Path to save translated PDF
        engine: Translation backend (dictionary, openai, deepseek, etc.)
        direction: Translation direction (en-fr or fr-en)
        pages: Page range (all, or 1-5 format)
        preserve_figures: Whether to preserve figures/formulas
        quality_loops: Number of refinement loops
        enable_rerank: Whether to enable candidate reranking
        use_mineru: Whether to use MinerU for extraction (default True)
        progress: Optional progress callback
        
    Returns:
        PipelineResult with translated document
    """
    from scitrans_llms.ingest import parse_pdf
    from scitrans_llms.render.pdf import render_pdf as render_pdf_output
    
    # Parse direction
    if direction == "en-fr":
        source_lang, target_lang = "en", "fr"
    else:
        source_lang, target_lang = "fr", "en"
    
    # Progress wrapper
    def prog(msg: str, pct: float = 0):
        if progress:
            progress(msg)
    
    # Parse PDF
    prog("Parsing layout from PDF...", 0.1)
    
    # Parse page range
    page_list = None
    if pages and pages.lower() != "all":
        try:
            if "-" in pages:
                start, end = pages.split("-")
                page_list = list(range(int(start) - 1, int(end)))
            elif "," in pages:
                page_list = [int(p.strip()) - 1 for p in pages.split(",")]
            else:
                page_list = [int(pages) - 1]
        except ValueError:
            pass  # Use all pages
    
    document = parse_pdf(
        input_path,
        pages=page_list,
        source_lang=source_lang,
        target_lang=target_lang,
        use_mineru=use_mineru,
        use_pdfminer=True,  # Use PDFMiner for better block extraction
    )
    
    # Configure pipeline
    prog("Translating blocks...", 0.3)
    # Determine candidate count for reranking
    if num_candidates is not None:
        candidates_per_block = max(1, num_candidates)
    else:
        candidates_per_block = 3 if enable_rerank else 1
    config = PipelineConfig(
        source_lang=source_lang,
        target_lang=target_lang,
        translator_backend=engine,
        enable_glossary=True,
        enable_refinement=quality_loops > 0,
        num_candidates=candidates_per_block,  # Enable reranking by generating multiple candidates
    )
    
    pipeline = TranslationPipeline(config, progress_callback=prog)
    result = pipeline.translate(document)
    
    # Render translated PDF with actual text overlay
    prog("Rendering translated PDF...", 0.9)
    try:
        # CRITICAL: Use result.document which contains translated blocks
        # Verify blocks have translated_text
        translated_blocks = [b for b in result.document.all_blocks if b.translated_text]
        if not translated_blocks:
            raise ValueError("No translated blocks found - translation may have failed")
        
        render_pdf_output(
            document=result.document,  # Use translated document
            source_pdf=input_path,
            output_path=output_path,
            mode="replace"  # Replace mode for better visibility
        )
        prog(f"Saved translated PDF to {output_path}", 1.0)
    except Exception as e:
        # Fallback: save as text if PDF rendering fails
        prog(f"PDF rendering failed ({e}), saving as text...", 0.95)
        text_path = Path(output_path).with_suffix('.txt')
        text_path.write_text(result.translated_text, encoding='utf-8')
        result.errors.append(f"PDF rendering failed: {e}. Saved as text: {text_path}")
        prog(f"Saved translated text to {text_path}", 1.0)
    
    return result


def _translate_document_internal(
    document: Document,
    config: PipelineConfig | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PipelineResult:
    """Translate a Document with optional config and progress callback.
    
    Internal function for direct document translation.
    """
    pipeline = TranslationPipeline(config, progress_callback)
    return pipeline.translate(document)


def _collect_layout_detections(
    pdf_path: str,
    page_indices: list[int],
) -> dict:
    """Collect layout detections from a PDF for debugging.
    
    Args:
        pdf_path: Path to PDF file
        page_indices: List of page indices to analyze
        
    Returns:
        Dict mapping page index to list of detections
    """
    try:
        from scitrans_llms.ingest.pdf import YOLOLayoutDetector, PageContent, TextSpan, BoundingBox
        import fitz
    except ImportError:
        return {}
    
    detector = YOLOLayoutDetector()
    if not detector.is_available:
        return {}
    
    detections = {}
    
    try:
        doc = fitz.open(pdf_path)
        
        for page_idx in page_indices:
            if 0 <= page_idx < len(doc):
                page = doc[page_idx]
                
                # Create minimal PageContent for detection
                content = PageContent(
                    page_num=page_idx,
                    width=page.rect.width,
                    height=page.rect.height,
                )
                
                # Run detection
                page_detections = detector.detect(content)
                
                if page_detections:
                    detections[page_idx] = [
                        type("Detection", (), {"label": dt[1].name, "bbox": dt[0]})()
                        for dt in page_detections
                    ]
        
        doc.close()
        
    except Exception:
        pass
    
    return detections

