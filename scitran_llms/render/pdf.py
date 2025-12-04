"""PDF rendering module."""

import fitz  # PyMuPDF
from pathlib import Path
from typing import Optional
import textwrap
from ..models import Document, BlockType


def render_pdf(document: Document, output_path: Path, original_pdf: Optional[Path] = None):
    """Render a translated document to PDF."""
    
    # Create new PDF
    pdf = fitz.open()
    
    # Page settings
    page_width = 595  # A4 width in points
    page_height = 842  # A4 height in points
    margin = 72  # 1 inch margin
    text_width = page_width - (2 * margin)
    
    # Add first page
    page = pdf.new_page(width=page_width, height=page_height)
    
    # Font settings
    fontsize_normal = 11
    fontsize_heading = 14
    line_height = 14
    para_spacing = 6
    
    # Starting position
    y_position = margin
    
    # Add translated content
    for i, block in enumerate(document.blocks):
        text = block.translation or block.text
        if not text:
            continue
        
        # Determine font size based on block type
        if block.block_type == BlockType.HEADING:
            fontsize = fontsize_heading
            current_line_height = line_height * 1.5
        else:
            fontsize = fontsize_normal
            current_line_height = line_height
        
        # Wrap text to fit page width
        # Estimate characters per line (roughly 80 chars for 11pt font on A4)
        chars_per_line = int(text_width / (fontsize * 0.5))
        wrapped_lines = textwrap.wrap(text, width=chars_per_line)
        
        # Check if we need a new page
        text_height = len(wrapped_lines) * current_line_height + para_spacing
        if y_position + text_height > page_height - margin:
            page = pdf.new_page(width=page_width, height=page_height)
            y_position = margin
        
        # Insert each wrapped line
        for line in wrapped_lines:
            if y_position + current_line_height > page_height - margin:
                page = pdf.new_page(width=page_width, height=page_height)
                y_position = margin
            
            # Insert text line
            page.insert_text(
                (margin, y_position),
                line,
                fontsize=fontsize,
                fontname="helv"
            )
            
            y_position += current_line_height
        
        # Add paragraph spacing
        y_position += para_spacing
        
        # Progress indicator
        if i % 10 == 0:
            print(f"Rendering block {i+1}/{len(document.blocks)}")
    
    # Save PDF
    pdf.save(str(output_path))
    pdf.close()
    print(f"PDF saved to {output_path}")
