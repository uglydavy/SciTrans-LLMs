#!/usr/bin/env python3
"""Simple script to run translation without CLI issues"""

import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from scitran_llms.pipeline import TranslationPipeline, PipelineConfig
from scitran_llms.models import Document, Block, BlockType

def main():
    print("\n" + "="*50)
    print("SciTrans-LLMs - Quick Translation Test")
    print("="*50 + "\n")
    
    # Test sentences
    test_text = "The quantum mechanical wave function describes the probability amplitude."
    
    # Setup pipeline
    config = PipelineConfig(
        backend="googletrans",
        source_lang="en",
        target_lang="fr"
    )
    
    print(f"Backend: {config.backend}")
    print(f"Direction: {config.source_lang} → {config.target_lang}\n")
    
    # Create pipeline
    pipeline = TranslationPipeline(config)
    
    # Translate
    print(f"Original: {test_text}")
    result = pipeline.translate_text(test_text)
    print(f"Translation: {result}\n")
    
    print("✅ Translation working!")
    print("\nTo translate a PDF, use:")
    print("  scitran translate --input paper.pdf --output translated.pdf --target fr")
    print("\nOr use the GUI:")
    print("  python3 run_gui_simple.py")

if __name__ == "__main__":
    main()
