from __future__ import annotations
from typing import Dict, List, Optional, Tuple

from .bootstrap import ensure_default_glossary, ensure_layout_model
from .ingest.pdf import Block, extract_blocks
from .ingest.analyzer import classify_block
from .mask import (
    looks_like_formula_block,
    looks_like_numeric_table,
    mask_protected_segments,
    unmask,
)
from .refine.postprocess import normalize
from .refine.prompting import build_prompt, evaluate_translation, refine_prompt
from .refine.rerank import rerank_candidates
from .render.pdf import BlockOut, render_overlay
from .translate.backends import get_translator
from .translate.glossary import (
    enforce_post,
    merge_glossaries,
)
from .translate.memory import TranslationMemory
from .utils import boxes_intersect, parse_page_range


def translate_document(
    input_pdf: str,
    output_pdf: str,
    engine: str = "dictionary",
    direction: str = "en-fr",
    pages: str = "all",
    preserve_figures: bool = True,
    quality_loops: int = 3,
    enable_rerank: bool = True,
) -> str:
    import fitz

    ensure_layout_model()
    ensure_default_glossary(refresh_remote=True)

    doc = fitz.open(input_pdf)
    total = doc.page_count
    doc.close()
    s, e = parse_page_range(pages, total)
    page_indices = list(range(s, e + 1))

    blocks, _size = extract_blocks(input_pdf, page_indices)

    layout_detections = (
        _collect_layout_detections(input_pdf, page_indices)
        if preserve_figures
        else {}
    )
    _label_blocks_with_layout(blocks, layout_detections)

    glossary = merge_glossaries()
    src, tgt = ("English", "French") if direction.lower() == "en-fr" else ("French", "English")

    translator = get_translator(engine, dictionary=glossary)
    memory = TranslationMemory(max_entries=64)
    base_prompt = build_prompt(src, tgt, glossary, memory=memory)

    translated_texts: List[Optional[str]] = [None] * len(blocks)
    masked_payloads: List[Tuple[int, str, List[Tuple[str, str]]]] = []
    skip_overlay: set[int] = set()

    for idx, block in enumerate(blocks):
        kind = (block.kind or "paragraph").lower()
        if preserve_figures and (
            kind in {"figure", "chart", "image", "formula", "equation", "table", "figure_caption", "table_caption"}
            or looks_like_formula_block(block.text)
        ):
            skip_overlay.add(idx)
            continue
        if not block.kind:
            block.kind = classify_block(block)
        if looks_like_numeric_table(block.text):
            translated_texts[idx] = normalize(block.text)
            continue
        masked_text, placeholders = mask_protected_segments(block.text)
        masked_payloads.append((idx, masked_text, placeholders))

    if masked_payloads:
        for idx, masked_text, placeholders in masked_payloads:
            translated = _iterative_translate(
                translator=translator,
                text=masked_text,
                src=src,
                tgt=tgt,
                base_prompt=base_prompt,
                glossary=glossary,
                memory=memory,
                max_loops=quality_loops,
                enable_rerank=enable_rerank,
            )
            restored = unmask(normalize(enforce_post(translated, glossary)), placeholders)
            translated_texts[idx] = restored
            memory.add(masked_text, restored)

    out_blocks = []
    for idx, (block, text) in enumerate(zip(blocks, translated_texts)):
        if idx in skip_overlay:
            continue
        if text is None:
            continue
        if not text.strip():
            continue
        out_blocks.append(
            BlockOut(page_index=block.page_index, bbox=block.bbox, text=text)
        )

    render_overlay(input_pdf, output_pdf, out_blocks, page_indices)
    return output_pdf


def _collect_layout_detections(pdf_path: str, page_indices: List[int]) -> Dict[int, List["Detection"]]:
    import os
    import tempfile
    import fitz

    try:
        from .yolo.predictor import Detection, LayoutPredictor
    except Exception:
        return {}

    predictor = None
    try:
        predictor = LayoutPredictor()
    except Exception:
        return {}

    doc = fitz.open(pdf_path)
    layout: Dict[int, List[Detection]] = {}
    for pi in page_indices:
        page = doc.load_page(pi)
        matrix = fitz.Matrix(2, 2)
        pix = page.get_pixmap(alpha=False, matrix=matrix)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            pix.save(tmp.name)
            dets = predictor.detect(tmp.name)
        finally:
            tmp.close()
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
        scale_x = pix.width / page.rect.width if page.rect.width else 1.0
        scale_y = pix.height / page.rect.height if page.rect.height else 1.0
        adjusted: List[Detection] = []
        for det in dets:
            x0, y0, x1, y1 = det.bbox
            converted = (
                x0 / scale_x,
                y0 / scale_y,
                x1 / scale_x,
                y1 / scale_y,
            )
            adjusted.append(
                Detection(label=det.label.lower(), score=det.score, bbox=converted)
            )
        layout[pi] = adjusted
    doc.close()
    return layout


def _label_blocks_with_layout(blocks: List[Block], layout: Dict[int, List["Detection"]]) -> None:
    if not layout:
        for block in blocks:
            if not block.kind:
                block.kind = classify_block(block)
        return
    label_alias = {
        "math": "formula",
        "equation": "formula",
        "table cell": "table",
        "figure": "figure",
        "image": "image",
    }
    for block in blocks:
        best_label = classify_block(block)
        best_score = -1.0
        for det in layout.get(block.page_index, []):
            label = label_alias.get(det.label, det.label)
            if boxes_intersect(block.bbox, det.bbox, padding=4.0) and det.score > best_score:
                best_label = label
                best_score = det.score
        block.kind = best_label


def _iterative_translate(
    translator,
    text: str,
    src: str,
    tgt: str,
    base_prompt: str,
    glossary: Dict[str, str],
    memory: TranslationMemory,
    max_loops: int = 4,
    enable_rerank: bool = True,
) -> str:
    prompt = base_prompt
    attempts: List[str] = []
    for i in range(max_loops):
        attempt = translator.translate([text], src=src, tgt=tgt, prompt=prompt, glossary=glossary, context=memory)[0]
        evaluation = evaluate_translation(text, attempt)
        attempts.append(attempt)
        if evaluation.acceptable:
            break
        prompt = refine_prompt(base_prompt, evaluation, iteration=i)
    if enable_rerank:
        return rerank_candidates(text, attempts, glossary).text
    return (attempts[-1] if attempts else text) or text
