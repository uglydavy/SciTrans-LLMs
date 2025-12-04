#!/usr/bin/env python3
"""
Test script to verify translation functionality.
"""

from scitran_llms.pipeline import TranslationPipeline, PipelineConfig
from scitran_llms.models import Document, Block, BlockType

# Test sentences
test_sentences = [
    "The quantum mechanical wave function describes the probability amplitude.",
    "Machine learning models can learn complex patterns from data.",
    "This paper presents a novel approach to neural machine translation.",
]

print("=" * 60)
print("Translation Test - English to French")
print("=" * 60)

# Test with GoogleTranslate (default)
print("\n1. Testing with GoogleTranslate:")
config = PipelineConfig(
    backend="googletrans",
    source_lang="en",
    target_lang="fr"
)
pipeline = TranslationPipeline(config)

for sentence in test_sentences:
    result = pipeline.translate_text(sentence)
    print(f"\nEN: {sentence}")
    print(f"FR: {result}")
    
# Test with dictionary backend
print("\n\n2. Testing with Dictionary backend:")
config = PipelineConfig(
    backend="dictionary",
    source_lang="en", 
    target_lang="fr"
)
pipeline = TranslationPipeline(config)

for sentence in test_sentences[:1]:  # Just test one with dictionary
    result = pipeline.translate_text(sentence)
    print(f"\nEN: {sentence}")
    print(f"FR: {result}")

# Test document translation
print("\n\n3. Testing Document Translation:")
doc = Document()
for sentence in test_sentences:
    doc.add_block(sentence, BlockType.PARAGRAPH)

config = PipelineConfig(backend="googletrans", source_lang="en", target_lang="fr")
pipeline = TranslationPipeline(config)
result = pipeline.translate_document(doc)

print(f"\nDocument translation stats:")
print(f"  Success rate: {result.success_rate:.1%}")
print(f"  Time taken: {result.time_taken:.2f}s")
print(f"  Errors: {len(result.errors)}")

print("\n" + "=" * 60)
print("✅ All tests completed!")
print("=" * 60)
