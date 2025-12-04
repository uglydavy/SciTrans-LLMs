"""
Translation pipeline for SciTrans-LLMs.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time
import re
from pathlib import Path

from .models import Document, Block, BlockType, Glossary
from .config import DEFAULT_BACKEND, CHUNK_SIZE


@dataclass
class PipelineConfig:
    """Configuration for the translation pipeline."""
    backend: str = DEFAULT_BACKEND
    source_lang: str = "en"
    target_lang: str = "fr"
    enable_masking: bool = True
    enable_glossary: bool = True
    enable_reranking: bool = False
    num_candidates: int = 1
    context_window: int = 3
    chunk_size: int = CHUNK_SIZE
    prompt_template: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backend": self.backend,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "enable_masking": self.enable_masking,
            "enable_glossary": self.enable_glossary,
            "enable_reranking": self.enable_reranking,
            "num_candidates": self.num_candidates,
            "context_window": self.context_window,
            "chunk_size": self.chunk_size,
            "prompt_template": self.prompt_template,
        }


@dataclass
class PipelineResult:
    """Result of a translation pipeline."""
    document: Document
    stats: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    time_taken: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = len(self.document.blocks)
        if total == 0:
            return 0.0
        translated = sum(1 for b in self.document.blocks if b.translation)
        return translated / total


class TranslationPipeline:
    """Main translation pipeline."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize the pipeline."""
        self.config = config
        self.glossary = Glossary()
        self.translator = None
        self._init_translator()
    
    def _init_translator(self):
        """Initialize the translator backend."""
        backend = self.config.backend
        
        if backend == "dictionary":
            from .translate.dictionary import DictionaryTranslator
            self.translator = DictionaryTranslator(
                self.config.source_lang,
                self.config.target_lang
            )
        elif backend == "mymemory":
            from .translate.mymemory import MyMemoryTranslator
            self.translator = MyMemoryTranslator(
                self.config.source_lang,
                self.config.target_lang
            )
        elif backend == "googletrans":
            from .translate.googletrans import GoogleTranslator
            self.translator = GoogleTranslator(
                self.config.source_lang,
                self.config.target_lang
            )
        elif backend == "openai":
            from .translate.openai import OpenAITranslator
            self.translator = OpenAITranslator(
                self.config.source_lang,
                self.config.target_lang
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def translate_text(self, text: str) -> str:
        """Translate plain text."""
        if not text or not text.strip():
            return ""
        
        # Apply masking if enabled
        if self.config.enable_masking:
            from .masking import Masker
            masker = Masker()
            masked_text = masker.mask(text)
            
            # Translate
            if self.translator:
                result = self.translator.translate(masked_text)
                translation = result.get("translation", masked_text)
            else:
                translation = masked_text
            
            # Restore masks
            translation = masker.unmask(translation)
        else:
            # Direct translation
            if self.translator:
                result = self.translator.translate(text)
                translation = result.get("translation", text)
            else:
                translation = text
        
        # Apply glossary if enabled
        if self.config.enable_glossary and self.glossary:
            for entry in self.glossary.entries:
                translation = translation.replace(
                    entry.source, 
                    entry.target
                )
        
        return translation
    
    def translate_document(self, document: Document) -> PipelineResult:
        """Translate a document."""
        start_time = time.time()
        result = PipelineResult(document=document)
        
        print(f"Starting translation of {len(document.blocks)} blocks...")
        
        # Translate each block
        for i, block in enumerate(document.blocks):
            try:
                # Progress indicator
                if i % 10 == 0:
                    print(f"Translating block {i+1}/{len(document.blocks)}...")
                
                # Skip if already translated
                if block.translation:
                    continue
                
                # Translate based on block type
                if block.block_type in [BlockType.EQUATION, BlockType.CODE]:
                    # Don't translate equations or code
                    block.translation = block.text
                    print(f"  Block {i}: Skipping {block.block_type.value}")
                else:
                    # Translate text - ensure we're actually translating
                    original_text = block.text
                    
                    # For long text, split into chunks
                    if len(original_text) > self.config.chunk_size:
                        # Split by sentences
                        sentences = re.split(r'(?<=[.!?])\s+', original_text)
                        translated_parts = []
                        
                        for sentence in sentences:
                            if sentence.strip():
                                trans = self.translate_text(sentence.strip())
                                translated_parts.append(trans)
                        
                        block.translation = ' '.join(translated_parts)
                    else:
                        # Translate whole block
                        block.translation = self.translate_text(original_text)
                    
                    # Verify translation happened
                    if block.translation == original_text and self.config.backend != "dictionary":
                        print(f"  Warning: Block {i} not translated (same as original)")
                    else:
                        print(f"  Block {i}: Translated {len(original_text)} -> {len(block.translation)} chars")
                
            except Exception as e:
                print(f"  Error translating block {i}: {str(e)}")
                result.errors.append(f"Block {i}: {str(e)}")
                block.translation = block.text  # Fallback to original
        
        # Calculate stats
        result.time_taken = time.time() - start_time
        translated_count = sum(1 for b in document.blocks if b.translation and b.translation != b.text)
        
        result.stats = {
            "total_blocks": len(document.blocks),
            "translated_blocks": translated_count,
            "skipped_blocks": sum(1 for b in document.blocks if b.block_type in [BlockType.EQUATION, BlockType.CODE]),
            "errors": len(result.errors),
            "time_taken": result.time_taken,
            "success_rate": result.success_rate,
        }
        
        print(f"\nTranslation complete:")
        print(f"  Total blocks: {result.stats['total_blocks']}")
        print(f"  Translated: {result.stats['translated_blocks']}")  
        print(f"  Skipped: {result.stats['skipped_blocks']}")
        print(f"  Errors: {result.stats['errors']}")
        print(f"  Time: {result.time_taken:.2f}s")
        
        return result
    
    def set_glossary(self, glossary: Glossary):
        """Set the glossary for translation."""
        self.glossary = glossary
    
    def load_glossary(self, path: Path):
        """Load glossary from file."""
        # Implementation depends on file format
        pass
