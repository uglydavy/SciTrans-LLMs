from __future__ import annotations
import os
import shutil
import tempfile
import warnings
from collections import Counter
from pathlib import Path
from textwrap import shorten

import fitz
import requests
import huggingface_hub as hfh


def _ensure_hf_folder():
    """Backport huggingface_hub.HfFolder for versions where it was removed."""

    if hasattr(hfh, "HfFolder"):
        return

    class _HfFolder:
        @staticmethod
        def get_token():
            getter = getattr(hfh, "get_token", None)
            return getter() if callable(getter) else None

        @staticmethod
        def save_token(token: str):
            setter = getattr(hfh, "set_access_token", None)
            if callable(setter):
                setter(token)
            else:
                warnings.warn(
                    "Cannot persist Hugging Face token; set HUGGINGFACEHUB_TOKEN "
                    "or HF_TOKEN in your environment instead.",
                    RuntimeWarning,
                )

    hfh.HfFolder = _HfFolder


_ensure_hf_folder()
import gradio as gr
import gradio_client.utils as grc_utils

from .ingest.analyzer import analyze_document
from .pipeline import translate_document, _collect_layout_detections
from .bootstrap import ensure_layout_model, ensure_default_glossary
from .config import GLOSSARY_DIR
from .mask import mask_protected_segments, unmask
from .refine.rerank import rerank_candidates
from .refine.scoring import bleu
from .translate.glossary import merge_glossaries
from .diagnostics import collect_diagnostics, summarize_checks
from .utils import parse_page_range


def _patch_gradio_json_schema():
    """Work around gradio_client assuming JSON Schema objects are dicts.

    Some gradio builds emit boolean ``additionalProperties`` flags. Newer
    gradio_client releases handle these, but older ones raise ``TypeError``
    when they try to iterate over the boolean. We coerce those booleans to
    permissive ``any``/""never"" strings so API schema generation succeeds.
    """

    original_get_type = grc_utils.get_type

    def safe_get_type(schema):  # type: ignore[override]
        if isinstance(schema, bool):
            return "boolean" if schema else "null"
        return original_get_type(schema)

    grc_utils.get_type = safe_get_type


def launch():
    _patch_gradio_json_schema()
    ensure_layout_model()
    ensure_default_glossary()

    progress = gr.Progress(track_tqdm=True)

    def do_translate(pdf_file, engine, direction, pages, preserve_figures, quality_loops, enable_rerank):
        if not pdf_file:
            return None, "Please upload a PDF before starting a run.", "", "", ""
        tmp = tempfile.NamedTemporaryFile(prefix="scitranslm_out_", suffix=".pdf", delete=False)
        out_path = tmp.name
        tmp.close()
        events: list[str] = []
        user_events: list[str] = []
        stage_events: list[str] = []

        def log(msg: str):
            events.append(msg)
            print(f"[SciTrans-LM] {msg}")
            if any(
                key in msg
                for key in (
                    "Parsing layout",
                    "Translating block",
                    "Rendering translated overlay",
                    "Saved translated PDF",
                    "Reranking",
                )
            ):
                if not user_events or user_events[-1] != msg:
                    user_events.append(msg)
            if any(key in msg for key in ("Parsing layout", "Translating block", "Reranking", "Rendering")):
                if not stage_events or stage_events[-1] != msg:
                    stage_events.append(msg)
            progress(0, desc=msg)

        try:
            translate_document(
                pdf_file.name,
                out_path,
                engine=engine,
                direction=direction,
                pages=pages,
                preserve_figures=preserve_figures,
                quality_loops=quality_loops,
                enable_rerank=enable_rerank,
                progress=log,
            )
            size_mb = os.path.getsize(out_path) / (1024 * 1024)
            note = " (compressed)" if size_mb > 0 else ""
            status = f"Done. File size: {size_mb:.1f} MB{note}."
            summary = "Quality pipeline: YOLO layout ➜ masking ➜ memory-aware prompting ➜ rerank." if enable_rerank else "Translated with direct prompting and glossary enforcement."
            status_lines = [status]
            if user_events:
                status_lines.append(" • ".join(user_events))
            preview_text = _preview_pdf_text(out_path, max_chars=1400)
            log_text = "\n".join(events[-120:])
            timeline = "\n".join(f"• {m}" for m in stage_events[-12:]) or "Awaiting next run."
            return out_path, "\n".join(status_lines), summary, preview_text, timeline
        except Exception as e:
            try:
                os.unlink(out_path)
            except OSError:
                pass
            log_text = "\n".join(events[-120:]) if events else ""
            timeline = "\n".join(f"• {m}" for m in stage_events[-12:]) if stage_events else ""
            return None, f"Error: {e}", "", "", timeline or log_text

    def fetch_remote_pdf(url_text: str):
        if not url_text or not url_text.strip():
            return gr.File.update(value=None), "Enter a direct PDF link to download."
        try:
            resp = requests.get(url_text.strip(), timeout=20)
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            return gr.File.update(value=None), f"Download failed: {exc}"

        content_type = resp.headers.get("content-type", "").lower()
        note = ""
        if "pdf" not in content_type:
            note = " (warning: response is not labeled as a PDF)"

        tmp = tempfile.NamedTemporaryFile(prefix="scitranslm_remote_", suffix=".pdf", delete=False)
        tmp.write(resp.content)
        tmp.flush()
        tmp.close()
        size_mb = len(resp.content) / (1024 * 1024)
        return gr.File.update(value=tmp.name), f"Fetched {size_mb:.2f} MB from URL.{note}"

    def inspect_layout(pdf_file, pages, preserve_figures):
        if not pdf_file:
            return "Upload or fetch a PDF, then click Analyze."
        try:
            doc = fitz.open(pdf_file.name)
        except Exception as exc:  # noqa: BLE001
            return f"Could not open PDF: {exc}"

        total = doc.page_count
        doc.close()
        s, e = parse_page_range(pages, total)
        page_indices = list(range(s, e + 1))

        notes: list[str] = []
        summaries = analyze_document(pdf_file.name, page_indices, notes=notes)
        counts = Counter(s.kind for s in summaries)
        count_lines = [f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda kv: kv[0])]
        samples = [f"p{b.page_index+1} [{b.kind}] {shorten(b.text_preview, width=110)}" for b in summaries[:12]]

        layout_info = ""
        if preserve_figures:
            detections = _collect_layout_detections(pdf_file.name, page_indices)
            if detections:
                label_counts = Counter(det.label for dets in detections.values() for det in dets)
                layout_info = "YOLO detections: " + ", ".join(
                    f"{k}={v}" for k, v in sorted(label_counts.items(), key=lambda kv: kv[0])
                )

        lines = [f"Analyzed pages {s+1}-{e+1} (total blocks: {len(summaries)})."]
        if count_lines:
            lines.append("Kinds → " + ", ".join(count_lines))
        if layout_info:
            lines.append(layout_info)
        if notes:
            lines.append("Notes:")
            lines.extend([f"- {n}" for n in notes])
        lines.append("Samples:")
        if samples:
            lines.extend([" • " + s for s in samples])
        else:
            lines.append(" • No text blocks detected.")
        return "\n".join(lines)

    def mask_and_restore(sample_text: str):
        if not sample_text:
            return "", "No text provided.", ""
        masked, placeholders = mask_protected_segments(sample_text)
        restored = unmask(masked, placeholders)
        placeholder_list = ", ".join(t for _, t in placeholders) or "(none)"
        return masked, f"Placeholders: {placeholder_list}", restored

    def rerank_sandbox(source_text: str, candidates_text: str):
        glossary = merge_glossaries()
        source_text = source_text or ""
        candidates = [c.strip() for c in (candidates_text or "").split("\n") if c.strip()]
        if not candidates:
            return "", "Provide at least one candidate translation (one per line)."
        best = rerank_candidates(source_text, candidates, glossary)
        detail = ", ".join(f"{k}={v:.3f}" for k, v in best.detail.items())
        return best.text, f"Best score: {best.score:.3f} ({detail})"

    def quick_bleu(reference: str, hypothesis: str):
        reference = (reference or "").strip()
        hypothesis = (hypothesis or "").strip()
        if not reference or not hypothesis:
            return "Enter reference and hypothesis text to score."
        try:
            score = bleu(hypothesis, reference)
            return f"BLEU: {score:.2f}"
        except Exception as exc:  # noqa: BLE001
            return f"Scoring failed: {exc}"

    def run_diagnostics_ui():
        checks = collect_diagnostics()
        icons = {"ok": "✅", "warn": "⚠️", "error": "❌"}
        lines = ["Environment + assets"]
        for check in checks:
            prefix = icons.get(check.status, "•")
            lines.append(f"{prefix} **{check.name}:** {check.detail}")
        summary = summarize_checks(checks)
        lines.append(
            f"\nSummary: {summary['ok']} OK, {summary['warn']} warning(s), {summary['error']} error(s)."
        )
        return "\n".join(lines)

    def upload_glossary(file_obj):
        if not file_obj:
            return "Upload a CSV with 'source,target' columns to enforce terminology."
        dest = GLOSSARY_DIR / Path(file_obj.name).name
        shutil.copy(file_obj.name, dest)
        return f"Saved glossary as {dest.name}. It will be merged automatically on next translation."

    css = """
    #status-text {text-align:center;}
    #summary-text {text-align:center; color:#1f2937;}
    #preview-box textarea {min-height:220px;}
    .compact-row {gap: 0.6rem;}
    .log-box textarea {min-height: 180px;}
    """

    with gr.Blocks(title="SciTrans-LM – EN↔FR PDF Translator", css=css, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "<div style='text-align:center'><h2>SciTrans-LM – English ↔ French PDF Translator</h2>"
            "<p>Layout-preserving translation with glossary enforcement, refinement, and offline fallbacks.</p></div>"
        )

        with gr.Tabs():
            with gr.Tab("Translate"):
                with gr.Row(equal_height=True, elem_classes="compact-row"):
                    with gr.Column(scale=6, variant="panel"):
                        pdf = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
                        with gr.Row():
                            url_box = gr.Textbox(label="Or fetch via URL", placeholder="https://example.com/paper.pdf")
                            fetch_btn = gr.Button("Download", variant="secondary")
                        link_status = gr.Markdown("Upload or download a PDF to begin.")
                        with gr.Row():
                            engine = gr.Dropdown(
                                choices=["dictionary", "google-free", "openai", "deepl", "google", "deepseek", "perplexity"],
                                value="dictionary",
                                label="Engine",
                                info="Pick a backend. Dictionary and Google-free do not require API keys.",
                            )
                            direction = gr.Radio(["en-fr", "fr-en"], value="en-fr", label="Direction")
                        with gr.Row():
                            pages = gr.Textbox(value="all", label="Pages", placeholder="all or 1-5")
                            preserve = gr.Checkbox(value=True, label="Preserve figures/formulas")
                        with gr.Row():
                            quality_loops = gr.Slider(1, 5, value=3, step=1, label="Refinement loops")
                            enable_rerank = gr.Checkbox(value=True, label="Rerank candidates")
                        go = gr.Button("Translate", variant="primary")
                        status = gr.Markdown("<div id='status-text'><strong>Ready.</strong></div>")
                        summary = gr.Markdown("", elem_id="summary-text")
                        pipeline_log = gr.Textbox(
                            label="Pipeline timeline (latest run)",
                            lines=8,
                            interactive=False,
                            elem_classes=["log-box"],
                            placeholder="Parsing layout → translating blocks → reranking → rendering.",
                        )
                    with gr.Column(scale=5, variant="panel"):
                        preview = gr.Textbox(label="Preview (first pages)", lines=10, interactive=False, elem_id="preview-box")
                        out_pdf = gr.File(label="Download translated PDF", interactive=False)
                        with gr.Accordion("Glossary options", open=False):
                            glossary_upload = gr.File(label="Upload custom glossary (.csv)")
                            glossary_status = gr.Markdown("Glossary: built-in 50+ terms. Upload to extend.")

                fetch_btn.click(fetch_remote_pdf, inputs=[url_box], outputs=[pdf, link_status])
                go.click(
                    do_translate,
                    inputs=[pdf, engine, direction, pages, preserve, quality_loops, enable_rerank],
                    outputs=[out_pdf, status, summary, preview, pipeline_log],
                )
                glossary_upload.change(upload_glossary, inputs=[glossary_upload], outputs=[glossary_status])

            with gr.Tab("Debug / QA"):
                gr.Markdown(
                    "View how the PDF is segmented and when OCR fallbacks trigger. Use this tab to verify headers, lists, and captions before translating."
                )
                inspect_btn = gr.Button("Analyze layout & extraction", variant="secondary")
                layout_summary = gr.Textbox(label="Layout summary", lines=14, interactive=False)

                inspect_btn.click(inspect_layout, inputs=[pdf, pages, preserve], outputs=[layout_summary])

            with gr.Tab("Pipeline Lab"):
                gr.Markdown(
                    "Experiment with individual modules (masking, reranking, BLEU scoring) without running a full translation."
                )
                with gr.Row(equal_height=True, elem_classes="compact-row"):
                    with gr.Column(scale=5, variant="panel"):
                        gr.Markdown("**Masking sandbox** – see how protected segments are tokenized and restored.")
                        sample_text = gr.Textbox(label="Sample source text", lines=5)
                        mask_btn = gr.Button("Mask & restore", variant="secondary")
                        masked = gr.Textbox(label="Masked text", lines=4, interactive=False)
                        placeholder_status = gr.Markdown()
                        restored = gr.Textbox(label="Restored text", lines=4, interactive=False)
                        mask_btn.click(mask_and_restore, inputs=[sample_text], outputs=[masked, placeholder_status, restored])

                    with gr.Column(scale=5, variant="panel"):
                        gr.Markdown("**Rerank sandbox** – provide candidates to see glossary-aware reranking.")
                        rerank_source = gr.Textbox(label="Source", lines=4)
                        rerank_candidates_box = gr.Textbox(
                            label="Candidate translations (one per line)", lines=6, placeholder="Cand 1\nCand 2"
                        )
                        rerank_btn = gr.Button("Rerank", variant="secondary")
                        rerank_choice = gr.Textbox(label="Selected translation", lines=3, interactive=False)
                        rerank_detail = gr.Markdown()
                        rerank_btn.click(
                            rerank_sandbox,
                            inputs=[rerank_source, rerank_candidates_box],
                            outputs=[rerank_choice, rerank_detail],
                        )

                with gr.Row(equal_height=True, elem_classes="compact-row"):
                    with gr.Column(scale=5, variant="panel"):
                        gr.Markdown("**BLEU quick check** – compare a hypothesis against a reference.")
                        ref_text = gr.Textbox(label="Reference", lines=4)
                        hyp_text = gr.Textbox(label="Hypothesis", lines=4)
                        bleu_btn = gr.Button("Compute BLEU", variant="secondary")
                        bleu_score = gr.Markdown()
                        bleu_btn.click(quick_bleu, inputs=[ref_text, hyp_text], outputs=[bleu_score])

                    with gr.Column(scale=5, variant="panel"):
                        gr.Markdown("**Layout snapshot** – reuse the analyzer to spot headings/tables quickly.")
                        quick_layout_btn = gr.Button("Analyze first page", variant="secondary")
                        quick_layout_out = gr.Textbox(label="Snapshot", lines=8, interactive=False)

                        def _quick_layout(pdf_file):
                            if not pdf_file:
                                return "Upload a PDF first."
                            notes: list[str] = []
                            summaries = analyze_document(pdf_file.name, [0], notes=notes)
                            if not summaries:
                                return "No blocks detected on first page."
                            top = summaries[:6]
                            lines = [f"p{b.page_index+1} [{b.kind}] {shorten(b.text_preview, width=100)}" for b in top]
                            if notes:
                                lines.append("Notes:")
                                lines.extend(["- " + n for n in notes])
                            return "\n".join(lines)

                        quick_layout_btn.click(_quick_layout, inputs=[pdf], outputs=[quick_layout_out])

            with gr.Tab("System Check"):
                gr.Markdown(
                    "Verify dependencies, models, and keys before launching a long translation run."
                )
                diag_btn = gr.Button("Run diagnostics", variant="secondary")
                diag_out = gr.Markdown("Status will appear here.")
                diag_btn.click(run_diagnostics_ui, outputs=[diag_out])

    try:
        demo.launch()
    except ValueError as e:
        if "shareable link must be created" in str(e):
            demo.launch(share=True, show_api=False)
        else:
            raise


def _preview_pdf_text(pdf_path: str, pages: int = 2, max_chars: int = 1400) -> str:
    """Extract a short preview from the translated PDF without downloading."""

    try:
        import fitz
    except Exception:
        return ""
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return ""
    snippets = []
    try:
        for idx in range(min(pages, doc.page_count)):
            page = doc.load_page(idx)
            snippets.append(page.get_text("text"))
    finally:
        doc.close()
    preview = "\n".join(snippets).strip()
    return preview[:max_chars]


if __name__ == "__main__":
    launch()
