"""
SciTrans-LLMs Web GUI - Modern Translation Interface
"""

from __future__ import annotations

import os
import tempfile
import warnings
from pathlib import Path
from typing import Optional, Tuple


def launch(port: int = 7860, share: bool = False):
    """Launch the SciTrans-LLMs web GUI."""
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("Gradio not installed. Run: pip install gradio>=4.0.0")
    
    try:
        import requests
    except ImportError:
        requests = None
    
    # Initialize components with suppressed warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            from .bootstrap import ensure_layout_model, ensure_default_glossary
            ensure_layout_model()
            ensure_default_glossary()
        except Exception:
            pass
    
    from .pipeline import TranslationPipeline, PipelineConfig
    from .models import Document
    from .keys import KeyManager
    
    km = KeyManager()
    
    # Backend configuration
    BACKENDS = ["free", "dictionary", "ollama", "huggingface"]
    if km.get_key("openai"):
        BACKENDS.append("openai")
    if km.get_key("deepseek"):
        BACKENDS.append("deepseek")
    if km.get_key("anthropic"):
        BACKENDS.append("anthropic")
    
    def translate_text(text, backend, source_lang, target_lang, use_glossary):
        """Translate text."""
        if not text or not text.strip():
            return "Please enter some text to translate."
        
        try:
            config = PipelineConfig(
                source_lang=source_lang,
                target_lang=target_lang,
                translator_backend=backend,
                enable_glossary=use_glossary,
                enable_masking=True,
                enable_refinement=True,
            )
            
            doc = Document.from_text(text, source_lang, target_lang)
            pipeline = TranslationPipeline(config)
            result = pipeline.translate(doc)
            
            if result.success:
                return result.translated_text
            else:
                return f"Error: {'; '.join(result.errors)}"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def translate_file(file, backend, source_lang, target_lang, use_glossary):
        """Translate a file."""
        if file is None:
            return None, "Please upload a file."
        
        try:
            file_path = Path(file.name if hasattr(file, 'name') else file)
            
            if file_path.suffix.lower() == '.pdf':
                from .ingest import parse_pdf
                doc = parse_pdf(file_path, source_lang=source_lang, target_lang=target_lang)
            else:
                text = file_path.read_text(encoding='utf-8')
                doc = Document.from_text(text, source_lang, target_lang)
            
            config = PipelineConfig(
                source_lang=source_lang,
                target_lang=target_lang,
                translator_backend=backend,
                enable_glossary=use_glossary,
                enable_masking=True,
                enable_refinement=True,
            )
            
            pipeline = TranslationPipeline(config)
            result = pipeline.translate(doc)
            
            if result.success:
                out = tempfile.NamedTemporaryFile(
                    mode='w', suffix='.txt',
                    prefix=f'translated_{file_path.stem}_',
                    delete=False, encoding='utf-8'
                )
                out.write(result.translated_text)
                out.close()
                
                preview = result.translated_text[:3000]
                if len(result.translated_text) > 3000:
                    preview += "\n\n... (truncated)"
                
                return out.name, preview
            else:
                return None, f"Error: {'; '.join(result.errors)}"
                
        except Exception as e:
            import traceback
            return None, f"Error: {str(e)}\n\n{traceback.format_exc()}"
    
    def get_system_status():
        """Get system status."""
        lines = ["## System Status\n"]
        
        try:
            if requests:
                resp = requests.get("http://localhost:11434/api/tags", timeout=2)
                if resp.status_code == 200:
                    models = resp.json().get("models", [])
                    if models:
                        model_names = ", ".join(m.get("name", "?") for m in models[:3])
                        lines.append(f"Ollama: Running ({model_names})")
                    else:
                        lines.append("Ollama: Running but no models")
                else:
                    lines.append("Ollama: Not running")
            else:
                lines.append("Ollama: Unable to check")
        except Exception:
            lines.append("Ollama: Not running")
        
        lines.append("\n### API Keys")
        for service in ["openai", "deepseek", "anthropic", "huggingface", "deepl"]:
            key = km.get_key(service)
            if key:
                lines.append(f"{service.title()}: Configured")
            else:
                lines.append(f"{service.title()}: Not set")
        
        return "\n".join(lines)

    # Build interface - disable analytics to avoid API schema bugs
    with gr.Blocks(
        title="SciTrans-LLMs", 
        theme=gr.themes.Soft(primary_hue="indigo"),
        analytics_enabled=False
    ) as demo:
        
        gr.Markdown("# SciTrans-LLMs\n## Scientific Document Translation")

        with gr.Tabs():
            with gr.Tab("Text"):
                gr.Markdown("Paste text below to translate.")
                
                with gr.Row():
                    source_lang = gr.Dropdown(choices=["en", "fr", "de", "es"], value="en", label="From")
                    target_lang = gr.Dropdown(choices=["fr", "en", "de", "es"], value="fr", label="To")
                
                with gr.Row():
                    input_text = gr.Textbox(label="Source Text", placeholder="Enter text...", lines=8)
                    output_text = gr.Textbox(label="Translation", lines=8, interactive=False)
                
                with gr.Row():
                    backend = gr.Dropdown(choices=BACKENDS, value="free", label="Engine", info="free = no API key")
                    glossary = gr.Checkbox(label="Use Glossary", value=True)
                
                translate_btn = gr.Button("Translate", variant="primary")
                translate_btn.click(
                    fn=translate_text,
                    inputs=[input_text, backend, source_lang, target_lang, glossary],
                    outputs=[output_text],
                    api_name=False  # Disable API to avoid schema bug
                )
            
            with gr.Tab("Document"):
                gr.Markdown("Upload a PDF or text file.")
                
                with gr.Row():
                    with gr.Column():
                        file_input = gr.File(label="Upload Document", file_types=[".pdf", ".txt"])
                        with gr.Row():
                            doc_source = gr.Dropdown(choices=["en", "fr"], value="en", label="From")
                            doc_target = gr.Dropdown(choices=["fr", "en"], value="fr", label="To")
                        doc_backend = gr.Dropdown(choices=BACKENDS, value="free", label="Engine")
                        doc_glossary = gr.Checkbox(label="Use Glossary", value=True)
                        doc_btn = gr.Button("Translate Document", variant="primary")
                    
                    with gr.Column():
                        output_file = gr.File(label="Download Translation")
                        preview = gr.Textbox(label="Preview", lines=15, interactive=False)
                
                doc_btn.click(
                    fn=translate_file,
                    inputs=[file_input, doc_backend, doc_source, doc_target, doc_glossary],
                    outputs=[output_file, preview],
                    api_name=False  # Disable API to avoid schema bug
                )
            
            with gr.Tab("Status"):
                status_output = gr.Markdown(get_system_status())
                refresh_btn = gr.Button("Refresh")
                refresh_btn.click(fn=get_system_status, outputs=[status_output], api_name=False)
                
                gr.Markdown("### Quick Start\n\n- **Free**: Select 'free' and translate\n- **Ollama**: Run `ollama serve` and `ollama pull llama3`\n- **API keys**: Run `scitrans keys set openai`")
            
            with gr.Tab("Help"):
                gr.Markdown("# Help\n\n| Engine | Quality | Cost |\n|--------|---------|------|\n| free | Good | Free |\n| ollama | Best | Free |\n| openai | Best | Paid |\n\nCLI: `scitrans translate --text 'Hello' --backend free`")
        
        gr.Markdown("---\nSciTrans-LLMs v0.1.0")
    
    # Launch
    print(f"Starting GUI on port {port}...")
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=share,
        show_error=True,
        show_api=False  # Disable API docs to avoid schema bug
    )


if __name__ == "__main__":
    launch()
