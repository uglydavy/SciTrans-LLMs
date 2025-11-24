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

from .pipeline import translate_document
from .bootstrap import ensure_layout_model, ensure_default_glossary
from .config import GLOSSARY_DIR


def launch():
    ensure_layout_model()
    ensure_default_glossary()

    progress = gr.Progress(track_tqdm=True)

    def do_translate(pdf_file, engine, direction, pages, preserve_figures, quality_loops, enable_rerank):
        if not pdf_file:
            return None, "Please upload a PDF.", None
        tmp = tempfile.NamedTemporaryFile(prefix="scitranslm_out_", suffix=".pdf", delete=False)
        out_path = tmp.name
        tmp.close()
        events = []

        def log(msg: str):
            events.append(msg)
            progress(0.02, desc=msg)
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
            if size_mb > 18:
                status = f"Done. File size: {size_mb:.1f} MB{note}. Large files are compressed to stay downloadable."
            else:
                status = f"Done. File size: {size_mb:.1f} MB{note}."
            summary = "Quality pipeline: YOLO layout ➜ masking ➜ memory-aware prompting ➜ rerank." if enable_rerank else "Translated with direct prompting and glossary enforcement."
            timeline = "\n".join(f"- {msg}" for msg in events)
            return out_path, f"{status}\n{timeline}", summary
        except Exception as e:
            try:
                os.unlink(out_path)
            except OSError:
                pass
            return None, f"Error: {e}", None

    def upload_glossary(file_obj):
        if not file_obj:
            return "Upload a CSV with 'source,target' columns to enforce terminology."
        dest = GLOSSARY_DIR / Path(file_obj.name).name
        shutil.copy(file_obj.name, dest)
        return f"Saved glossary as {dest.name}. It will be merged automatically on next translation."

    with gr.Blocks(title="SciTrans-LM – EN↔FR PDF Translator") as demo:
        gr.Markdown(
            """
            ## SciTrans-LM – English ↔ French PDF Translator
            *Layout-preserving EN↔FR for research PDFs with glossary enforcement and iterative refinement.*

            - YOLO layout detection keeps figures/formulas untouched.
            - Masking + glossary enforcement + translation memory for consistent terminology.
            - Offline dictionary + optional online engines (OpenAI, DeepSeek, Perplexity, DeepL, Google).
            - Reranking and self-checking prompts to reduce hallucinations.
            - Upload your own glossary to preserve lab- or domain-specific terms.
            """
        )
        with gr.Row():
            with gr.Column():
                pdf = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
                engine = gr.Dropdown(
                    choices=["openai", "deepl", "google", "google-free", "deepseek", "perplexity", "dictionary"],
                    value="dictionary",
                    label="Engine",
                    info="Online engines use secure keys; dictionary is offline with glossary + adaptive lookup.",
                )
                direction = gr.Radio(["en-fr", "fr-en"], value="en-fr", label="Direction (EN↔FR)")
                pages = gr.Textbox(value="all", label="Pages (e.g., all or 1-5)")
                preserve = gr.Checkbox(value=True, label="Preserve figures & formulas (recommended)")
                quality_loops = gr.Slider(1, 5, value=3, step=1, label="Refinement loops (re-prompt if weak)")
                enable_rerank = gr.Checkbox(value=True, label="Enable reranking with glossary/fluency scoring")
                go = gr.Button("Translate", variant="primary")
                status = gr.Markdown("Ready.")
                summary = gr.Markdown(visible=True)
            with gr.Column():
                out_pdf = gr.File(label="Translated PDF", interactive=False)
                glossary_upload = gr.File(label="Upload custom glossary (.csv)")
                glossary_status = gr.Markdown("Glossary: built-in 50+ terms. Upload to extend.")

        go.click(do_translate, inputs=[pdf, engine, direction, pages, preserve, quality_loops, enable_rerank], outputs=[out_pdf, status, summary])
        glossary_upload.change(upload_glossary, inputs=[glossary_upload], outputs=[glossary_status])

    try:
        demo.launch()
    except ValueError as e:
        if "shareable link must be created" in str(e):
            demo.launch(share=True, show_api=False)
        else:
            raise


if __name__ == "__main__":
    launch()
