"""
SciTrans-LLMs: Scientific Document Translation Enhanced by Large Language Models
"""

__version__ = "0.2.0"
__author__ = "SciTrans Team"

from .pipeline import TranslationPipeline, PipelineConfig, PipelineResult
from .models import Document, Block, Glossary

__all__ = [
    "TranslationPipeline",
    "PipelineConfig", 
    "PipelineResult",
    "Document",
    "Block",
    "Glossary",
    "__version__",
]
