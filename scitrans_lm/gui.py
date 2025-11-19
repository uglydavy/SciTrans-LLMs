
from __future__ import annotations
import gradio as gr
import tempfile, os
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
