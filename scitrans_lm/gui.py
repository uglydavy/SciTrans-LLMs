from __future__ import annotations
import os
import shutil
import tempfile
import warnings
from pathlib import Path

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

from .pipeline import translate_document
from .bootstrap import ensure_layout_model, ensure_default_glossary
from .config import GLOSSARY_DIR


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
            return None, "Please upload a PDF.", "", ""
        tmp = tempfile.NamedTemporaryFile(prefix="scitranslm_out_", suffix=".pdf", delete=False)
        out_path = tmp.name
        tmp.close()
        events: list[str] = []
        user_events: list[str] = []

        def log(msg: str):
            events.append(msg)
            print(f"[SciTrans-LM] {msg}")
            if any(key in msg for key in ("Parsing layout", "Translating block", "Rendering translated overlay", "Saved translated PDF")):
                if not user_events or user_events[-1] != msg:
                    user_events.append(msg)
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
            return out_path, "\n".join(status_lines), summary, preview_text
        except Exception as e:
            try:
                os.unlink(out_path)
            except OSError:
                pass
            return None, f"Error: {e}", "", ""

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
    """

    with gr.Blocks(title="SciTrans-LM – EN↔FR PDF Translator", css=css, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "<div style='text-align:center'><h2>SciTrans-LM – English ↔ French PDF Translator</h2>"
            "<p>Layout-preserving translation with glossary enforcement, refinement, and offline fallbacks.</p></div>"
        )
        with gr.Row(equal_height=True):
            with gr.Column(scale=6):
                pdf = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
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
            with gr.Column(scale=5):
                preview = gr.Textbox(label="Preview (first pages)", lines=10, interactive=False, elem_id="preview-box")
                out_pdf = gr.File(label="Download translated PDF", interactive=False)
                with gr.Accordion("Glossary options", open=False):
                    glossary_upload = gr.File(label="Upload custom glossary (.csv)")
                    glossary_status = gr.Markdown("Glossary: built-in 50+ terms. Upload to extend.")

        go.click(
            do_translate,
            inputs=[pdf, engine, direction, pages, preserve, quality_loops, enable_rerank],
            outputs=[out_pdf, status, summary, preview],
        )
        glossary_upload.change(upload_glossary, inputs=[glossary_upload], outputs=[glossary_status])

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
