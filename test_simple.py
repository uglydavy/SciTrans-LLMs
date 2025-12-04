#!/usr/bin/env python3
"""
Simple test to verify basic translation works.
"""

from scitran_llms.pipeline import TranslationPipeline, PipelineConfig
from scitran_llms.models import Document

# Test basic translation
text = "Machine learning is a subset of artificial intelligence."
print(f"Input text: {text}")

# Create document
doc = Document.from_text(text)

# Create pipeline with dictionary backend (offline)
config = PipelineConfig(
    backend="dictionary",
    source_lang="en", 
    target_lang="fr"
)
pipeline = TranslationPipeline(config)

# Translate
result = pipeline.translate(doc)

if result.success:
    print(f"Translation: {result.document.translated_text}")
    print(f"Success rate: {result.translation_rate:.1%}")
else:
    print(f"Translation failed: {result.errors}")
