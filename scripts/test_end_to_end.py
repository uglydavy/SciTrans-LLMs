#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end translation test with real PDF."""

import tempfile
from pathlib import Path


def create_test_pdf():
    """Create a simple test PDF for translation testing."""
    try:
        import fitz
    except ImportError:
        print("ERROR: PyMuPDF required")
        return None
    
    # Create a simple PDF with English text
    doc = fitz.open()
    page = doc.new_page()
    
    # Add title
    page.insert_text(
        (72, 72),
        "Machine Learning in Scientific Translation",
        fontsize=18,
        fontname="helv"
    )
    
    # Add paragraph
    text = """
This paper presents a novel approach to scientific document translation 
using large language models. Our system combines terminology-constrained 
translation with document-level context awareness to improve translation 
quality for technical texts.

The methodology includes:
1. Glossary-based masking for technical terms
2. Context-aware neural machine translation
3. Post-translation refinement with LLM feedback
4. Layout-preserving PDF rendering

Results show a 15% improvement in BLEU scores compared to baseline systems.
"""
    
    rect = fitz.Rect(72, 100, 540, 700)
    page.insert_textbox(rect, text, fontsize=11, fontname="helv")
    
    # Save to temp file
    tmp_path = Path(tempfile.mkdtemp()) / "test_paper.pdf"
    doc.save(str(tmp_path))
    doc.close()
    
    return tmp_path


def test_extraction(pdf_path):
    """Test PDF extraction."""
    print("\n[1] Testing PDF Extraction...")
    
    from scitrans_llms.ingest.pdf import parse_pdf
    
    try:
        doc = parse_pdf(pdf_path)
        print(f"    Extraction method: {doc.metadata.get('extraction_method', 'unknown')}")
        print(f"    Segments: {len(doc.segments)}")
        print(f"    Total blocks: {len(doc.all_blocks)}")
        
        for i, block in enumerate(doc.all_blocks[:3]):
            text_preview = block.source_text[:60].replace('\n', ' ')
            print(f"    Block {i+1}: [{block.block_type.name}] {text_preview}...")
        
        return doc
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def test_translation(doc):
    """Test translation pipeline."""
    print("\n[2] Testing Translation Pipeline...")
    
    from scitrans_llms.pipeline import TranslationPipeline, PipelineConfig
    
    try:
        config = PipelineConfig(
            source_lang="en",
            target_lang="fr",
            translator_backend="dictionary",  # Use dictionary for offline test
            enable_masking=True,
            enable_glossary=True,
            enable_context=True,
            enable_refinement=True,
        )
        
        pipeline = TranslationPipeline(config)
        result = pipeline.translate(doc)
        
        print(f"    Translated blocks: {len(result.document.all_blocks)}")
        print(f"    Pipeline stats: {result.stats}")
        
        # Show sample translation
        for i, block in enumerate(result.document.all_blocks[:2]):
            if block.translated_text:
                src = block.source_text[:40].replace('\n', ' ')
                tgt = block.translated_text[:40].replace('\n', ' ')
                print(f"    Sample {i+1}:")
                print(f"      EN: {src}...")
                print(f"      FR: {tgt}...")
        
        return result
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_rendering(result, source_pdf, output_path):
    """Test PDF rendering."""
    print("\n[3] Testing PDF Rendering...")
    
    from scitrans_llms.render.pdf import render_pdf
    
    try:
        output = render_pdf(
            result.document,
            source_pdf,
            output_path,
            mode="hybrid"
        )
        
        print(f"    Output PDF: {output}")
        print(f"    File size: {output.stat().st_size / 1024:.1f} KB")
        
        return output
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("=" * 60)
    print("SciTrans-LLMs End-to-End Test")
    print("=" * 60)
    
    # Create test PDF
    print("\n[0] Creating test PDF...")
    pdf_path = create_test_pdf()
    if not pdf_path:
        return
    print(f"    Created: {pdf_path}")
    
    # Test extraction
    doc = test_extraction(pdf_path)
    if not doc:
        return
    
    # Test translation
    result = test_translation(doc)
    if not result:
        return
    
    # Test rendering
    output_path = pdf_path.parent / "test_paper_translated.pdf"
    rendered = test_rendering(result, pdf_path, output_path)
    
    print("\n" + "=" * 60)
    if rendered:
        print("SUCCESS: End-to-end test completed!")
        print(f"Output: {rendered}")
    else:
        print("PARTIAL: Translation worked but rendering failed")
    print("=" * 60)


if __name__ == "__main__":
    main()

