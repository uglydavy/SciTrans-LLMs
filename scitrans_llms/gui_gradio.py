"""SciTrans-LLMs Gradio GUI - Modern, Production-Ready Interface

This is a complete rewrite of the GUI using Gradio for better UX, performance,
and reliability. Key improvements over NiceGUI version:
- File-based PDF handling (Gradio 4.44 compatible)
- Built-in drag-drop file upload
- Real-time progress streaming
- Async URL fetching
- Better error handling
- Cleaner, more responsive layout

Note: Uses gr.File() for PDF display (Gradio 4.44). 
When Gradio 5.0 releases, we can upgrade to gr.PDF() for inline preview.
"""

from __future__ import annotations
import asyncio
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import aiohttp

from scitrans_llms.keys import KeyManager
from scitrans_llms.pipeline import translate_document
from scitrans_llms.translate.glossary import Glossary, GlossaryEntry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scitrans-gradio")

# Global state
km = KeyManager()


def get_available_engines() -> list[str]:
    """Get list of available translation engines based on API keys."""
    engines = ['free', 'dictionary', 'improved-offline', 'huggingface', 'ollama']
    if km.get_key('openai'):
        engines.append('openai')
    if km.get_key('deepseek'):
        engines.append('deepseek')
    if km.get_key('anthropic'):
        engines.append('anthropic')
    if km.get_key('deepl'):
        engines.append('deepl')
    return engines


async def fetch_pdf_from_url(url: str, progress=gr.Progress()) -> Tuple[Optional[Path], str]:
    """Fetch PDF from URL with progress tracking.
    
    Returns:
        (temp_file_path, status_message)
    """
    if not url or not url.strip():
        return None, "❌ Please enter a URL"
    
    url = url.strip()
    try:
        progress(0, desc="Connecting to server...")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
                if response.status != 200:
                    return None, f"❌ HTTP {response.status}: {response.reason}"
                
                # Get file size for progress
                total = int(response.headers.get('content-length', 0))
                
                # Create temp file
                suffix = '.pdf' if url.lower().endswith('.pdf') else '.pdf'
                temp_file = Path(tempfile.mktemp(suffix=suffix))
                
                # Download with progress
                downloaded = 0
                chunks = []
                
                async for chunk in response.content.iter_chunked(8192):
                    chunks.append(chunk)
                    downloaded += len(chunk)
                    
                    if total > 0:
                        pct = downloaded / total
                        progress(pct, desc=f"Downloading... {downloaded // 1024}KB / {total // 1024}KB")
                    else:
                        progress(0.5, desc=f"Downloading... {downloaded // 1024}KB")
                
                # Write to file
                temp_file.write_bytes(b''.join(chunks))
                
                # Verify it's a PDF
                magic = temp_file.read_bytes()[:4]
                if magic != b'%PDF':
                    temp_file.unlink()
                    return None, "❌ Downloaded file is not a valid PDF"
                
                progress(1.0, desc="✓ Download complete")
                filename = url.split('/')[-1].split('?')[0] or 'document.pdf'
                return temp_file, f"✓ Downloaded: {filename} ({downloaded // 1024}KB)"
    
    except asyncio.TimeoutError:
        return None, "❌ Download timeout (60s exceeded)"
    except aiohttp.ClientError as e:
        return None, f"❌ Network error: {str(e)}"
    except Exception as e:
        logger.exception("URL fetch error")
        return None, f"❌ Error: {str(e)}"


def parse_glossary(file_path: str) -> Optional[Glossary]:
    """Parse glossary from CSV/TXT/JSON file."""
    if not file_path:
        return None
    
    try:
        path = Path(file_path)
        content = path.read_text(encoding='utf-8')
        entries = []
        
        if path.suffix == '.json':
            import json
            for item in json.loads(content):
                if isinstance(item, dict):
                    entries.append(GlossaryEntry(
                        source=item.get('source', ''),
                        target=item.get('target', ''),
                        domain=item.get('domain', 'custom')
                    ))
        elif path.suffix == '.csv':
            for line in content.strip().split('\n')[1:]:  # Skip header
                parts = [p.strip().strip('"') for p in line.split(',')]
                if len(parts) >= 2:
                    entries.append(GlossaryEntry(
                        source=parts[0],
                        target=parts[1],
                        domain=parts[2] if len(parts) > 2 else 'custom'
                    ))
        else:  # TXT
            for line in content.strip().split('\n'):
                sep = '\t' if '\t' in line else ','
                parts = [p.strip() for p in line.split(sep)]
                if len(parts) >= 2:
                    entries.append(GlossaryEntry(
                        source=parts[0],
                        target=parts[1],
                        domain='custom'
                    ))
        
        if entries:
            return Glossary(name='custom', entries=entries)
        return None
    
    except Exception as e:
        logger.exception("Glossary parse error")
        return None


def translate_pdf(
    pdf_file,
    direction: str,
    engine: str,
    pages: str,
    quality_passes: int,
    num_candidates: int,
    enable_masking: bool,
    glossary_file,
    progress=gr.Progress()
) -> Tuple[Optional[str], str]:
    """Main translation function with progress tracking.
    
    Returns:
        (output_pdf_path, status_message)
    """
    if pdf_file is None:
        return None, "❌ Please upload or fetch a PDF first"
    
    try:
        progress(0, desc="Initializing...")
        
        # Get input path
        input_path = pdf_file if isinstance(pdf_file, str) else pdf_file.name
        input_path = Path(input_path)
        
        if not input_path.exists():
            return None, f"❌ File not found: {input_path}"
        
        # Create output path
        output_path = Path(tempfile.mktemp(suffix=f'_translated_{direction}.pdf'))
        
        # Parse glossary if provided
        glossary = None
        if glossary_file is not None:
            glossary_path = glossary_file if isinstance(glossary_file, str) else glossary_file.name
            glossary = parse_glossary(glossary_path)
            if glossary:
                logger.info(f"Loaded {len(glossary.entries)} glossary terms")
        
        # Progress callback
        logs = []
        def log_progress(msg: str):
            timestamp = datetime.now().strftime("%H:%M:%S")
            logs.append(f"[{timestamp}] {msg}")
            logger.info(msg)
            
            # Update progress based on message
            if 'pars' in msg.lower():
                progress(0.1, desc="Parsing PDF...")
            elif 'translat' in msg.lower():
                # Extract block progress if available
                if '/' in msg:
                    try:
                        parts = msg.split()
                        for part in parts:
                            if '/' in part:
                                current, total = map(int, part.split('/'))
                                pct = 0.2 + (0.6 * current / total)
                                progress(pct, desc=f"Translating block {current}/{total}")
                                break
                    except:
                        progress(0.5, desc="Translating...")
                else:
                    progress(0.5, desc="Translating...")
            elif 'render' in msg.lower():
                progress(0.85, desc="Rendering output...")
            elif 'complete' in msg.lower() or 'done' in msg.lower():
                progress(1.0, desc="✓ Complete")
        
        log_progress(f"Starting translation: {input_path.name}")
        log_progress(f"Engine: {engine}, Direction: {direction}, Pages: {pages}")
        
        # Parse page range
        page_range = None if pages == 'all' else pages
        
        # Run translation
        progress(0.1, desc="Parsing PDF...")
        result = translate_document(
            input_path=str(input_path),
            output_path=str(output_path),
            engine=engine,
            direction=direction,
            pages=page_range,
            quality_loops=quality_passes,
            enable_rerank=True,
            num_candidates=num_candidates,
            progress=log_progress
        )
        
        if result.success:
            stats = result.stats or {}
            total_blocks = stats.get('total_blocks', 0)
            translated_blocks = stats.get('translated_blocks', 0)
            
            log_progress(f"✓ Translation complete: {translated_blocks}/{total_blocks} blocks")
            
            status = '\n'.join(logs[-10:])  # Last 10 log lines
            return str(output_path), status
        else:
            error_msg = '\n'.join(result.errors[:3]) if result.errors else "Unknown error"
            log_progress(f"❌ Translation failed: {error_msg}")
            return None, '\n'.join(logs[-10:])
    
    except Exception as e:
        logger.exception("Translation error")
        return None, f"❌ Error: {str(e)}"


def build_interface():
    """Build the Gradio interface."""
    
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="SciTrans-LLMs",
        css="""
        .gradio-container {
            max-width: 100% !important;
        }
        """
    ) as app:
        
        gr.Markdown(
            """
            # 🔬 SciTrans-LLMs
            ### Scientific Document Translation with Layout Preservation
            
            Translate scientific PDFs while preserving formulas, figures, tables, and formatting.
            """
        )
        
        with gr.Row():
            # LEFT COLUMN: Settings & Upload
            with gr.Column(scale=1):
                gr.Markdown("### 📄 Document Input")
                
                with gr.Tabs() as input_tabs:
                    with gr.Tab("Upload") as upload_tab:
                        pdf_upload = gr.File(
                            label="Drop PDF here or click to upload",
                            file_types=[".pdf"],
                            file_count="single",
                            type="filepath"
                        )
                    
                    with gr.Tab("URL") as url_tab:
                        url_input = gr.Textbox(
                            label="PDF URL (e.g., arXiv link)",
                            placeholder="https://arxiv.org/pdf/...",
                            lines=1
                        )
                        url_fetch_btn = gr.Button("📥 Fetch PDF", variant="secondary")
                        url_status = gr.Textbox(label="Status", lines=1, interactive=False)
                
                gr.Markdown("### ⚙️ Translation Settings")
                
                direction = gr.Radio(
                    choices=["en-fr", "fr-en"],
                    value="en-fr",
                    label="Translation Direction",
                    info="Source → Target language"
                )
                
                engine = gr.Dropdown(
                    choices=get_available_engines(),
                    value="free",
                    label="Translation Engine",
                    info="Select backend (free = smart cascade)"
                )
                
                pages_input = gr.Textbox(
                    label="Pages",
                    value="all",
                    placeholder="all, 1-5, or 1,3,5",
                    info="Specify page range or 'all'"
                )
                
                with gr.Accordion("🔧 Advanced Options", open=False):
                    enable_masking = gr.Checkbox(
                        label="Enable Formula/URL Masking",
                        value=False,
                        info="Protect formulas and URLs from translation (may affect quality)"
                    )
                    
                    quality_passes = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=1,
                        step=1,
                        label="Quality Passes",
                        info="Number of refinement iterations"
                    )
                    
                    num_candidates = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                        label="Candidates per Block",
                        info="Generate multiple translations and pick best"
                    )
                    
                    glossary_upload = gr.File(
                        label="Custom Glossary (CSV/TXT/JSON)",
                        file_types=[".csv", ".txt", ".json"],
                        file_count="single"
                    )
                
                translate_btn = gr.Button(
                    "🚀 Translate Document",
                    variant="primary",
                    size="lg"
                )
            
            # MIDDLE COLUMN: Source Preview
            with gr.Column(scale=1):
                gr.Markdown("### 📖 Source PDF")
                source_preview = gr.File(
                    label="Source PDF (click to download/view)",
                    file_count="single",
                    interactive=False,
                    type="filepath"
                )
                source_info = gr.Textbox(
                    label="Source Info",
                    lines=2,
                    interactive=False,
                    placeholder="Upload a PDF to see info here"
                )
            
            # RIGHT COLUMN: Translated Preview  
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Translated PDF")
                translated_preview = gr.File(
                    label="Translated PDF (click to download)",
                    file_count="single",
                    interactive=False,
                    type="filepath"
                )
                translated_info = gr.Textbox(
                    label="Translation Info",
                    lines=2,
                    interactive=False,
                    placeholder="Translation will appear here"
                )
        
        # BOTTOM: Status & Logs
        with gr.Row():
            status_box = gr.Textbox(
                label="📊 Translation Status & Logs",
                lines=8,
                max_lines=15,
                interactive=False,
                placeholder="Status messages will appear here..."
            )
        
        # Hidden state for current source file
        current_source = gr.State(None)
        
        # Event handlers
        
        # Upload tab: Update preview when file uploaded
        def on_upload(file):
            if file is None:
                return None, None, "", "No file uploaded"
            try:
                file_path = Path(file if isinstance(file, str) else file.name)
                size_mb = file_path.stat().st_size / (1024 * 1024)
                info = f"✓ {file_path.name}\nSize: {size_mb:.2f} MB"
                status = f"✓ Loaded: {file_path.name}"
                return file, file, info, status
            except Exception as e:
                return None, None, f"❌ Error: {e}", f"❌ Error loading file"
        
        pdf_upload.change(
            fn=on_upload,
            inputs=[pdf_upload],
            outputs=[current_source, source_preview, source_info, status_box]
        )
        
        # URL tab: Fetch PDF
        def on_url_fetch(url):
            result = asyncio.run(fetch_pdf_from_url(url))
            temp_file, status_msg = result
            if temp_file:
                try:
                    size_mb = temp_file.stat().st_size / (1024 * 1024)
                    info = f"✓ {temp_file.name}\nSize: {size_mb:.2f} MB"
                    return str(temp_file), str(temp_file), info, status_msg, status_msg
                except:
                    return str(temp_file), str(temp_file), "✓ Downloaded", status_msg, status_msg
            return None, None, "", status_msg, status_msg
        
        url_fetch_btn.click(
            fn=on_url_fetch,
            inputs=[url_input],
            outputs=[current_source, source_preview, source_info, url_status, status_box]
        )
        
        # Translate button
        def on_translate(
            source_file, direction, engine, pages,
            quality_passes, num_candidates, enable_masking, glossary_file
        ):
            output_path, status = translate_pdf(
                source_file, direction, engine, pages,
                quality_passes, num_candidates, enable_masking, glossary_file
            )
            
            if output_path:
                try:
                    out_path = Path(output_path)
                    size_mb = out_path.stat().st_size / (1024 * 1024)
                    info = f"✓ Translation complete\nSize: {size_mb:.2f} MB"
                    return output_path, info, status
                except:
                    return output_path, "✓ Translation complete", status
            return None, "❌ Translation failed", status
        
        translate_btn.click(
            fn=on_translate,
            inputs=[
                current_source, direction, engine, pages_input,
                quality_passes, num_candidates, enable_masking, glossary_upload
            ],
            outputs=[translated_preview, translated_info, status_box]
        )
        
        # Footer
        gr.Markdown(
            """
            ---
            **SciTrans-LLMs** v1.0 | [Documentation](README.md) | [Report Issue](https://github.com/yourusername/SciTrans-LLMs/issues)
            """
        )
    
    return app


def launch(port: int = 7860, share: bool = False):
    """Launch the Gradio GUI."""
    logger.info("Starting SciTrans-LLMs Gradio GUI...")
    
    app = build_interface()
    
    app.queue()  # Enable queuing for concurrent users
    
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
        show_error=True,
        favicon_path=None,
        inbrowser=True
    )


if __name__ == "__main__":
    launch()
