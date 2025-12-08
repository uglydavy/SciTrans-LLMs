#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick check of extraction capabilities."""

import sys
from pathlib import Path

def main():
    print("=" * 60)
    print("SciTrans-LLMs Extraction Capabilities Check")
    print("=" * 60)
    
    # Check YOLO
    print("\n[1] DocLayout-YOLO:")
    try:
        from scitrans_llms.ingest.pdf import YOLOLayoutDetector
        yolo = YOLOLayoutDetector()
        if yolo.is_available:
            # Test actual inference
            try:
                from scitrans_llms.ingest.pdf import PageContent, BoundingBox
                import fitz
                import tempfile
                
                # Create minimal test
                doc = fitz.open()
                page = doc.new_page()
                page.insert_text((72, 72), "Test", fontsize=12)
                pix = page.get_pixmap()
                img_bytes = pix.tobytes("png")
                doc.close()
                
                test_page = PageContent(page_num=0, width=612, height=792, image_bytes=img_bytes)
                yolo.detect(test_page)
                print("    [OK] YOLO model working correctly")
            except Exception as e:
                print("    [WARN] YOLO model loaded but inference failed: %s" % str(e)[:50])
                print("    [INFO] System will use PDFMiner fallback")
        else:
            print("    [WARN] YOLO model not available (check weights/ultralytics)")
    except Exception as e:
        print("    [ERR] YOLO error: %s" % e)
    
    # Check MinerU
    print("\n[2] MinerU (magic-pdf):")
    try:
        import magic_pdf
        print("    [OK] MinerU available")
    except ImportError:
        print("    [WARN] MinerU not installed (pip install magic-pdf)")
    
    # Check PDFMiner
    print("\n[3] PDFMiner:")
    try:
        from pdfminer.high_level import extract_pages
        print("    [OK] PDFMiner available")
    except ImportError:
        print("    [WARN] PDFMiner not installed (pip install pdfminer.six)")
    
    # Check PyMuPDF
    print("\n[4] PyMuPDF:")
    try:
        import fitz
        print("    [OK] PyMuPDF %s available" % fitz.version[0])
    except ImportError:
        print("    [ERR] PyMuPDF not installed (required)")
    
    # Check PDF Rendering
    print("\n[5] PDF Rendering:")
    try:
        from scitrans_llms.render.pdf import PDFRenderer, RenderConfig
        print("    [OK] PDF renderer available")
    except Exception as e:
        print("    [ERR] Renderer error: %s" % e)
    
    # Check LLM Refiner
    print("\n[6] LLM Refiner:")
    try:
        from scitrans_llms.refine.base import create_refiner
        refiner = create_refiner("llm")
        print("    [OK] LLM refiner: %s" % refiner.name)
    except Exception as e:
        print("    [WARN] LLM refiner: %s" % e)
    
    # Test basic extraction
    print("\n[7] Basic Extraction Test:")
    try:
        from scitrans_llms.ingest.pdf import HeuristicLayoutDetector, PDFParser
        parser = PDFParser(layout_detector=HeuristicLayoutDetector())
        print("    [OK] Heuristic parser ready (always available fallback)")
    except Exception as e:
        print("    [ERR] Parser error: %s" % e)
    
    print("\n" + "=" * 60)
    print("Extraction priority: YOLO -> MinerU -> PDFMiner -> Heuristic")
    print("=" * 60)

if __name__ == "__main__":
    main()
