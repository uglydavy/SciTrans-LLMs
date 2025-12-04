#!/usr/bin/env python3
"""
System test script for SciTrans-LLMs.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        import scitran_llms
        print("✓ scitran_llms")
        
        from scitran_llms import Document, Block, TranslationPipeline
        print("✓ Core models")
        
        from scitran_llms.translate import DictionaryTranslator, FreeTranslator
        print("✓ Translators")
        
        from scitran_llms.ingest import parse_pdf
        print("✓ PDF ingestion")
        
        from scitran_llms.render import render_pdf
        print("✓ PDF rendering")
        
        from scitran_llms.cli import app
        print("✓ CLI")
        
        from scitran_llms.gui import TranslationGUI
        print("✓ GUI")
        
        print("\n✅ All imports successful!")
        return True
    except Exception as e:
        print(f"\n❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_translation():
    """Test basic translation functionality."""
    print("\nTesting basic translation...")
    try:
        from scitran_llms.pipeline import TranslationPipeline, PipelineConfig
        from scitran_llms.models import Document
        
        # Create simple document
        text = "Machine learning is amazing."
        doc = Document.from_text(text)
        
        # Create pipeline
        config = PipelineConfig(backend="dictionary")
        pipeline = TranslationPipeline(config)
        
        # Translate
        result = pipeline.translate(doc)
        
        if result.success:
            print(f"✓ Translation successful")
            print(f"  Source: {text}")
            print(f"  Result: {result.document.translated_text}")
        else:
            print(f"✗ Translation failed: {result.errors}")
            
        return result.success
        
    except Exception as e:
        print(f"\n❌ Translation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cli_commands():
    """Test CLI commands."""
    print("\nTesting CLI commands...")
    try:
        import subprocess
        
        # Test help command
        result = subprocess.run(
            ["python", "-m", "scitran_llms", "--help"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✓ CLI help works")
        else:
            print(f"✗ CLI help failed: {result.stderr}")
            return False
        
        # Test demo command
        result = subprocess.run(
            ["python", "-m", "scitran_llms", "demo"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✓ CLI demo works")
        else:
            print(f"✗ CLI demo failed: {result.stderr}")
            return False
        
        # Test info command
        result = subprocess.run(
            ["python", "-m", "scitran_llms", "info"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✓ CLI info works")
        else:
            print(f"✗ CLI info failed: {result.stderr}")
            return False
        
        return True
        
    except Exception as e:
        print(f"\n❌ CLI test error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("SciTrans-LLMs System Test")
    print("=" * 50)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test basic translation
    if not test_basic_translation():
        all_passed = False
    
    # Test CLI
    if not test_cli_commands():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All tests passed! System is ready.")
        print("\nTo start the GUI, run:")
        print("  python -m scitran_llms gui")
        print("\nTo translate a file:")
        print("  python -m scitran_llms translate --input paper.pdf --output paper_fr.pdf")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
