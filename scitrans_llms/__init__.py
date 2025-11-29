"""
SciTrans-LLMs: Adaptive Document Translation Enhanced by Technology based on LLMs

A research-grade, layout-preserving translation system for scientific PDFs.

Core Research Contributions:
1. Terminology-constrained, layout-preserving translation
2. Document-level LLM context and refinement  
3. Research-grade evaluation & ablation framework

Author: SciTrans-LLMs Team
License: MIT
"""

__version__ = "0.1.0"

from scitrans_llms.models import Document, Segment, Block, BlockType
from scitrans_llms.pipeline import TranslationPipeline

__all__ = [
    "Document",
    "Segment", 
    "Block",
    "BlockType",
    "TranslationPipeline",
]

