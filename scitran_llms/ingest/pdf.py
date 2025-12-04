"""PDF parsing module."""

import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any
import re
from ..models import Document, Block, BlockType


def parse_pdf(pdf_path: Path) -> Document:
    """Parse a PDF file into a Document."""
    doc = Document()
    
    try:
        # Open PDF
        pdf = fitz.open(str(pdf_path))
        doc.metadata["pages"] = len(pdf)
        doc.metadata["filename"] = pdf_path.name
        
        # Extract text from each page
        for page_num, page in enumerate(pdf):
            # Get ALL text from page, not just blocks
            # This ensures we don't miss any content
            full_text = page.get_text()
            
            # Split into paragraphs (by double newline or significant spacing)
            paragraphs = re.split(r'\n\s*\n', full_text)
            
            for para in paragraphs:
                text = para.strip()
                if not text or len(text) < 10:  # Skip very short text
                    continue
                
                # Better block type detection
                block_type = BlockType.PARAGRAPH
                
                # Check for headings - typically shorter and may be numbered
                if (len(text) < 100 and 
                    (text[0].isupper() or 
                     re.match(r'^\d+\.?\s+[A-Z]', text) or
                     re.match(r'^[IVX]+\.?\s+[A-Z]', text) or
                     text.isupper())):
                    block_type = BlockType.HEADING
                
                # Check for equations (LaTeX)
                elif '\\begin{equation' in text or '\\[' in text or re.search(r'\$.*\$', text):
                    block_type = BlockType.EQUATION
                
                # Check for code blocks
                elif text.startswith('```') or text.startswith('    '):
                    block_type = BlockType.CODE
                
                # Check for lists
                elif re.match(r'^[\*\-\•\d]+[\.\)]\s', text):
                    block_type = BlockType.LIST
                
                # Add block with metadata
                doc.add_block(
                    text=text,
                    block_type=block_type,
                    page=page_num + 1,
                    char_count=len(text)
                )
        
        pdf.close()
        
        # Log extraction stats
        print(f"Extracted {len(doc.blocks)} blocks from {doc.metadata['pages']} pages")
        print(f"Block types: {dict((t.value, sum(1 for b in doc.blocks if b.block_type == t)) for t in BlockType)}")
        
    except Exception as e:
        # More informative error
        import traceback
        error_msg = f"Error parsing PDF: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        doc.add_block(
            text=error_msg,
            block_type=BlockType.PARAGRAPH
        )
    
    return doc
