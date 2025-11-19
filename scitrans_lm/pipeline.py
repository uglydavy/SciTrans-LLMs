
from __future__ import annotations
from typing import List, Tuple
from .ingest.pdf import extract_blocks, Block
from .translate.glossary import merge_glossaries, inject_prompt_instructions, enforce_post
from .translate.backends import get_translator
from .refine.postprocess import normalize
from .render.pdf import render_overlay, BlockOut

def translate_document(
    input_pdf: str,
    output_pdf: str,
    engine: str = "dictionary",
    direction: str = "en-fr",
    pages: str = "all",
    preserve_figures: bool = True,
) -> str:
    import fitz
    from .utils import parse_page_range
    # Open doc to get page count
    doc = fitz.open(input_pdf)
    total = doc.page_count
    doc.close()
    s, e = parse_page_range(pages, total)
    page_indices = list(range(s, e+1))

    # 1) Ingest
    blocks, _size = extract_blocks(input_pdf, page_indices)

    # 2) (YOLO detection would go here to find figures/tables if needed)

    # 3) Glossary
    glossary = merge_glossaries()
    src, tgt = ("English", "French") if direction.lower() == "en-fr" else ("French", "English")

    # 4) Backend
    translator = get_translator(engine, dictionary=glossary)
    prompt = inject_prompt_instructions(glossary, src, tgt)

    # 5) Translate per block
    texts = [b.text for b in blocks]
    translated = translator.translate(texts, src=src, tgt=tgt, prompt=prompt, glossary=glossary)
    translated = [normalize(enforce_post(t, glossary)) for t in translated]

    # 6) Render overlay
    out_blocks = [BlockOut(page_index=b.page_index, bbox=b.bbox, text=t) for b, t in zip(blocks, translated)]
    render_overlay(input_pdf, output_pdf, out_blocks, page_indices)
    return output_pdf
