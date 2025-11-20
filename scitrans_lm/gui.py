
from __future__ import annotations
import os
import tempfile
import warnings

import huggingface_hub as hfh


def _ensure_hf_folder():
    """Backport huggingface_hub.HfFolder for versions where it was removed.

    Gradio still imports ``HfFolder`` directly. Newer versions of
    ``huggingface_hub`` (>1.0) dropped this symbol, which triggers an
    ImportError before our GUI even starts. We recreate a minimal compatible
    wrapper so Gradio's import succeeds. Token persistence is best-effort:
    if ``set_access_token`` is unavailable we emit a warning instructing users
    to rely on environment variables.
    """

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

def launch():
    ensure_layout_model()
    ensure_default_glossary()

    def do_translate(pdf_file, engine, direction, pages, preserve_figures):
        if not pdf_file:
            return None, "Please upload a PDF."
        out_path = os.path.join(tempfile.gettempdir(), "scitranslm_out.pdf")
        try:
            translate_document(pdf_file.name, out_path, engine=engine, direction=direction, pages=pages, preserve_figures=preserve_figures)
            return out_path, "Done."
        except Exception as e:
            return None, f"Error: {e}"

    with gr.Blocks(title="SciTrans-LM – EN↔FR PDF Translator") as demo:
        gr.Markdown("## SciTrans-LM – English ↔ French PDF Translator\nUpload a PDF, choose engine and options, and translate. Figures/formulas are preserved by default.")
        with gr.Row():
            with gr.Column():
                pdf = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
                engine = gr.Dropdown(choices=["openai", "deepl", "google", "deepseek", "perplexity", "dictionary"], value="dictionary", label="Engine")
                direction = gr.Radio(["en-fr", "fr-en"], value="en-fr", label="Direction (EN↔FR)")
                pages = gr.Textbox(value="all", label="Pages (e.g., all or 1-5)")
                preserve = gr.Checkbox(value=True, label="Preserve figures & formulas (recommended)")
                go = gr.Button("Translate", variant="primary")
                status = gr.Markdown("Ready.")
            with gr.Column():
                out_pdf = gr.File(label="Translated PDF", interactive=False)

        go.click(do_translate, inputs=[pdf, engine, direction, pages, preserve], outputs=[out_pdf, status])

    demo.launch()

if __name__ == "__main__":
    launch()
