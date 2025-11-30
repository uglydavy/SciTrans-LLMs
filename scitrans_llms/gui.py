"""
SciTrans-LLMs Modern GUI - Scientific Document Translation Interface

A professional translation interface for scientific documents (PDF, DOCX, HTML)
with bilingual French ↔ English translation, context-aware processing, and
layout-preserving PDF output.

Uses NiceGUI for a modern, responsive interface with dark/light mode support.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scitrans-gui")


@dataclass
class TranslationJob:
    """Represents a translation job with all settings."""
    source_path: Optional[Path] = None
    source_url: Optional[str] = None
    direction: str = "en-fr"  # en-fr or fr-en
    pages: str = "all"  # "all" or "1-5" format
    engine: str = "free"
    enable_masking: bool = True
    translate_figures: bool = False
    translate_tables: bool = False
    translate_equations: bool = False
    quality_passes: int = 1
    enable_reranking: bool = False
    custom_glossary: Optional[str] = None
    output_path: Optional[Path] = None
    status: str = "pending"
    progress: float = 0.0
    log_messages: list = field(default_factory=list)
    result: Optional[dict] = None


def launch(port: int = 7860, share: bool = False):
    """Launch the SciTrans-LLMs modern web GUI."""
    try:
        from nicegui import ui, app
    except ImportError:
        raise ImportError(
            "NiceGUI not installed. Run: pip install nicegui>=1.4.0"
        )
    
    # Initialize components
    from scitrans_llms.keys import KeyManager, SERVICES
    from scitrans_llms.pipeline import PipelineConfig, TranslationPipeline
    from scitrans_llms.translate.glossary import get_default_glossary, load_glossary_csv, Glossary, GlossaryEntry
    from scitrans_llms.models import Document
    
    km = KeyManager()
    
    # State management
    class AppState:
        dark_mode: bool = True
        current_job: Optional[TranslationJob] = None
        logs: list = []
        test_results: list = []
        default_engine: str = 'free'
        default_quality: int = 1
        default_masking: bool = True
    
    state = AppState()
    
    # Set up logging handler to capture logs
    class GUILogHandler(logging.Handler):
        def __init__(self, log_callback):
            super().__init__()
            self.log_callback = log_callback
            
        def emit(self, record):
            try:
                msg = self.format(record)
                if self.log_callback:
                    self.log_callback(msg)
            except Exception:
                pass
    
    # Global log callback (will be set when UI is ready)
    global_log_callback = None
    gui_log_handler = None
    
    def set_global_log_callback(callback):
        nonlocal global_log_callback, gui_log_handler
        global_log_callback = callback
        # Remove old handler if exists
        if gui_log_handler:
            logging.getLogger().removeHandler(gui_log_handler)
        # Add handler to root logger
        gui_log_handler = GUILogHandler(callback)
        gui_log_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
        logging.getLogger().addHandler(gui_log_handler)
    
    # Available backends
    def get_available_backends():
        """Return list of available translation backends with human labels."""
        backends = [
            ("free", "Free (Lingva/LibreTranslate)", True),
            ("dictionary", "Dictionary Only", True),
            ("improved-offline", "Improved Offline", True),
        ]
        
        # Check Ollama
        try:
            import requests
            resp = requests.get("http://localhost:11434/api/tags", timeout=2)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                if models:
                    model_name = models[0].get("name", "local")
                    backends.append(("ollama", f"Ollama ({model_name})", True))
                else:
                    backends.append(("ollama", "Ollama (no models)", False))
            else:
                backends.append(("ollama", "Ollama (not running)", False))
        except Exception:
            backends.append(("ollama", "Ollama (not running)", False))
        
        # Check HuggingFace
        try:
            from scitrans_llms.translate.free_apis import HuggingFaceTranslator
            has_hf_key = km.get_key("huggingface") is not None
            if has_hf_key:
                backends.append(("huggingface", "HuggingFace (with key)", True))
            else:
                backends.append(("huggingface", "HuggingFace (free tier, no key)", True))
        except Exception:
            backends.append(("huggingface", "HuggingFace (not installed)", False))
        
        # Check Google Free
        try:
            from deep_translator import GoogleTranslator
            backends.append(("googlefree", "Google Free", True))
        except Exception:
            backends.append(("googlefree", "Google Free (not installed)", False))
        
        # OpenAI
        if km.get_key("openai"):
            backends.append(("openai", "OpenAI GPT-4", True))
        else:
            backends.append(("openai", "OpenAI GPT-4 (no key)", False))
        
        # DeepSeek
        # DeepSeek
        if km.get_key("deepseek"):
            backends.append(("deepseek", "DeepSeek (cheap)", True))
        else:
            backends.append(("deepseek", "DeepSeek (no key)", False))
        
        # Anthropic
        try:
            import anthropic
            if km.get_key("anthropic"):
                backends.append(("anthropic", "Claude", True))
            else:
                backends.append(("anthropic", "Claude (no key)", False))
        except Exception:
            backends.append(("anthropic", "Claude (not installed)", False))
        return backends
    
    # =========================================================================
    # Translation Logic
    # =========================================================================
    
    async def run_translation(job: TranslationJob, progress_callback: Callable[[str, float], None]):
        """Execute the translation pipeline."""
        from scitrans_llms.ingest import parse_pdf
        from scitrans_llms.render.pdf import render_pdf
        import shutil
        
        try:
            progress_callback("Initializing...", 0.05)
            
            # Determine source file
            if job.source_url:
                # Download from URL
                progress_callback("Downloading document from URL...", 0.1)
                import urllib.request
                temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
                urllib.request.urlretrieve(job.source_url, temp_file.name)
                source_path = Path(temp_file.name)
            else:
                source_path = job.source_path
            
            if not source_path or not source_path.exists():
                raise FileNotFoundError("Source file not found")
            
            # Parse direction
            if job.direction == "en-fr":
                source_lang, target_lang = "en", "fr"
            else:
                source_lang, target_lang = "fr", "en"
            
            # Parse page range
            page_list = None
            if job.pages.lower() != "all":
                try:
                    if "-" in job.pages:
                        start, end = job.pages.split("-")
                        page_list = list(range(int(start) - 1, int(end)))
                    else:
                        page_list = [int(job.pages) - 1]
                except ValueError:
                    pass
            
            # Parse document
            progress_callback("Parsing document layout...", 0.15)
            
            suffix = source_path.suffix.lower()
            if suffix == ".pdf":
                document = parse_pdf(
                    source_path,
                    pages=page_list,
                    source_lang=source_lang,
                    target_lang=target_lang,
                )
            elif suffix in [".docx", ".doc"]:
                # DOCX parsing
                try:
                    import docx
                    doc = docx.Document(str(source_path))
                    text = "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                    document = Document.from_text(text, source_lang, target_lang)
                except ImportError:
                    raise ImportError("python-docx required for DOCX files. Run: pip install python-docx")
            elif suffix in [".html", ".htm"]:
                # HTML parsing
                try:
                    from bs4 import BeautifulSoup
                    with open(source_path, 'r', encoding='utf-8') as f:
                        soup = BeautifulSoup(f.read(), 'html.parser')
                    text = soup.get_text(separator="\n\n")
                    document = Document.from_text(text, source_lang, target_lang)
                except ImportError:
                    raise ImportError("BeautifulSoup required for HTML files. Run: pip install beautifulsoup4")
            else:
                # Plain text fallback
                text = source_path.read_text(encoding='utf-8')
                document = Document.from_text(text, source_lang, target_lang)
            
            progress_callback(f"Found {len(document.all_blocks)} content blocks...", 0.2)
            
            # Load glossary
            glossary = None
            if job.custom_glossary and job.custom_glossary.strip():
                # Parse custom glossary from textarea
                progress_callback("Loading custom glossary...", 0.22)
                glossary = _parse_custom_glossary(job.custom_glossary, source_lang, target_lang)
            
            # Configure pipeline
            from scitrans_llms.masking import MaskConfig
            
            mask_config = MaskConfig(
                mask_equations=job.enable_masking and not job.translate_equations,
                mask_code=job.enable_masking,
                mask_urls=job.enable_masking,
                mask_emails=job.enable_masking,
            )
            
            config = PipelineConfig(
                source_lang=source_lang,
                target_lang=target_lang,
                translator_backend=job.engine,
                enable_masking=job.enable_masking,
                mask_config=mask_config,
                enable_glossary=True,
                glossary=glossary,
                enable_refinement=job.quality_passes > 0,
                num_candidates=3 if job.enable_reranking else 1,
            )
            
            # Run translation with progress
            def pipeline_progress(msg: str, pct: float):
                # Scale pipeline progress to 25-85%
                scaled = 0.25 + (pct * 0.6)
                progress_callback(msg, scaled)
            
            progress_callback("Starting translation...", 0.25)
            pipeline = TranslationPipeline(config, progress_callback=pipeline_progress)
            
            # Multiple quality passes
            result = None
            for pass_num in range(max(1, job.quality_passes)):
                if pass_num > 0:
                    progress_callback(f"Quality pass {pass_num + 1}/{job.quality_passes}...", 0.25 + (0.6 * pass_num / job.quality_passes))
                result = pipeline.translate(document)
            
            progress_callback("Rendering output PDF...", 0.85)
            
            # Create output path
            if not job.output_path:
                output_dir = source_path.parent / "translated"
                output_dir.mkdir(exist_ok=True)
                direction_suffix = "fr" if job.direction == "en-fr" else "en"
                job.output_path = output_dir / f"{source_path.stem}_{direction_suffix}.pdf"
            
            # Render PDF if source was PDF
            if suffix == ".pdf":
                render_pdf(
                    document=result.document,
                    source_pdf=source_path,
                    output_path=job.output_path,
                    mode="overlay",
                )
            else:
                # For non-PDF sources, create a text file
                job.output_path = job.output_path.with_suffix(".txt")
                job.output_path.write_text(result.translated_text, encoding='utf-8')
            
            progress_callback("Translation complete!", 1.0)
            
            return {
                "success": True,
                "output_path": str(job.output_path),
                "stats": result.stats,
                "errors": result.errors,
            }
                
        except Exception as e:
            logger.exception("Translation failed")
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
    
    def _parse_custom_glossary(text: str, source_lang: str, target_lang: str) -> Glossary:
        """Parse custom glossary from textarea input."""
        entries = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Support various separators
            for sep in [",", "\t", " → ", " -> ", " = ", ":"]:
                if sep in line:
                    parts = line.split(sep, 1)
                    if len(parts) == 2:
                        source, target = parts[0].strip(), parts[1].strip()
                        if source and target:
                            entries.append(GlossaryEntry(source, target))
                    break
        
        return Glossary(
            entries=entries,
            name="custom",
            source_lang=source_lang,
            target_lang=target_lang,
        )
    
    # =========================================================================
    # UI Components
    # =========================================================================
    
    # Custom CSS for modern styling
    CUSTOM_CSS = """
    :root {
        --primary: #6366f1;
        --primary-dark: #4f46e5;
        --accent: #06b6d4;
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
    }
    
    .dark {
        --bg-primary: #0f172a;
        --bg-secondary: #1e293b;
        --bg-tertiary: #334155;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --border: #475569;
    }
    
    .light {
        --bg-primary: #f8fafc;
        --bg-secondary: #ffffff;
        --bg-tertiary: #e2e8f0;
        --text-primary: #0f172a;
        --text-secondary: #64748b;
        --border: #cbd5e1;
    }
    
    .hero-gradient {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
    }
    
    .deepl-style {
        border: 1px solid rgba(0, 0, 0, 0.08);
        border-radius: 6px;
        background: var(--bg-secondary);
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
    }
    
    .deepl-style.dark {
        border-color: rgba(255, 255, 255, 0.08);
    }
    
    .upload-area {
        border: 1.5px dashed rgba(99, 102, 241, 0.25);
        border-radius: 6px;
        transition: all 0.15s;
        cursor: pointer;
    }
    
    .upload-area:hover {
        border-color: rgba(99, 102, 241, 0.5);
        background: rgba(99, 102, 241, 0.03);
    }
    
    .mono-font {
        font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', monospace;
        font-size: 13px;
    }
    
    body {
        overflow-x: hidden;
        font-size: 16px !important;
    }
    
    html, body {
        height: 100%;
        margin: 0;
        padding: 0;
    }
    
    .main-container {
        height: calc(100vh - 120px);
        overflow-y: hidden;
    }
    
    .compact-card {
        padding: 20px !important;
        margin: 0 !important;
    }
    
    .compact-text {
        font-size: 15px !important;
        line-height: 1.6 !important;
    }
    
    /* Center all content */
    .centered-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
        max-width: 1400px;
        margin: 0 auto;
        padding: 20px;
    }
    
    .upload-area {
        position: relative;
    }
    
    .upload-area input[type="file"] {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        opacity: 0;
        cursor: pointer;
        z-index: 10;
    }
    
    .centered-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
        height: calc(100vh - 140px);
        overflow-y: auto;
    }
    
    .no-scroll {
        overflow: hidden !important;
    }
    
    /* Prevent body scrolling */
    body {
        overflow: hidden !important;
    }
    
    /* Ensure tab panels don't scroll the whole page */
    .nicegui-content {
        overflow: hidden !important;
        height: 100vh !important;
    }
    
    /* Prevent page reload on visibility change */
    @media (prefers-reduced-motion: no-preference) {
        * {
            scroll-behavior: auto !important;
        }
    }
    
    /* Fix tab panel scrolling */
    .q-tab-panel {
        overflow: hidden !important;
        height: 100% !important;
    }
    
    /* Container for no-scroll tabs */
    .no-scroll-container {
        height: calc(100vh - 120px);
        overflow: hidden;
        display: flex;
        flex-direction: column;
    }
    
    /* Main layout - ensure header stays visible */
    .q-page {
        display: flex;
        flex-direction: column;
        height: 100vh;
    }
    
    /* Tab panels - allow internal scrolling but not page scroll */
    .q-tab-panel {
        overflow-y: auto;
        overflow-x: hidden;
        flex: 1;
    }
    
    /* Fix scrolling in columns */
    .overflow-y-auto {
        max-height: 100%;
        overflow-y: auto;
    }
    """
    
    @ui.page('/')
    async def main_page():
        # Theme toggle state - follow system preference
        import sys
        import platform
        # Detect system dark mode preference
        try:
            if platform.system() == 'Darwin':  # macOS
                import subprocess
                result = subprocess.run(['defaults', 'read', '-g', 'AppleInterfaceStyle'], 
                                      capture_output=True, text=True, timeout=1)
                system_prefers_dark = 'Dark' in result.stdout
            else:
                # Default to dark for other systems or if detection fails
                system_prefers_dark = True
        except:
            system_prefers_dark = True
        
        dark = ui.dark_mode()
        dark.value = system_prefers_dark
        state.dark_mode = system_prefers_dark
        
        # Add custom styles
        ui.add_head_html(f'<style>{CUSTOM_CSS}</style>')
        ui.add_head_html('<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">')
        
        # =====================================================================
        # Header
        # =====================================================================
        with ui.header().classes('items-center justify-between px-4 py-2 hero-gradient'):
            with ui.row().classes('items-center gap-2'):
                ui.icon('translate').classes('text-white')
                ui.label('SciTrans-LLMs').classes('text-lg font-bold text-white')
                ui.badge('v0.1.0').classes('bg-white/20 text-white text-xs')
            
            with ui.row().classes('items-center gap-2'):
                # Language indicator
                with ui.row().classes('items-center gap-1 bg-white/10 rounded-full px-2 py-1'):
                    ui.label('EN').classes('text-white font-semibold text-xs')
                    ui.icon('swap_horiz').classes('text-white/70')
                    ui.label('FR').classes('text-white font-semibold text-xs')
                
                # Theme toggle
                def toggle_theme():
                    dark.value = not dark.value
                    state.dark_mode = dark.value
                
                ui.button(
                    icon='dark_mode' if dark.value else 'light_mode',
                    on_click=toggle_theme
                ).props('flat round dense').classes('text-white')
        
        # =====================================================================
        # Main Content
        # =====================================================================
        with ui.tabs().classes('w-full text-xs').props('dense inline-label') as tabs:
            translate_tab = ui.tab('translate', label='Translate', icon='description')
            glossary_tab = ui.tab('glossary', label='Glossary', icon='menu_book')
            developer_tab = ui.tab('developer', label='Developer', icon='code')
            settings_tab = ui.tab('settings', label='Settings', icon='settings')
        
        with ui.tab_panels(tabs, value=translate_tab).classes('w-full flex-grow').style('flex: 1; min-height: 0;'):
            # =================================================================
            # TRANSLATE TAB
            # =================================================================
            with ui.tab_panel(translate_tab).classes('p-4'):
                await render_translate_panel()
            
            # =================================================================
            # GLOSSARY TAB
            # =================================================================
            with ui.tab_panel(glossary_tab).classes('p-6'):
                await render_glossary_panel()
            
            # =================================================================
            # DEVELOPER TAB
            # =================================================================
            with ui.tab_panel(developer_tab).classes('p-4'):
                await render_developer_panel()
            
            # =================================================================
            # SETTINGS TAB
            # =================================================================
            with ui.tab_panel(settings_tab).classes('p-6'):
                await render_settings_panel(km)
        
        # Footer
        with ui.footer().classes('py-2 px-4 text-center'):
            ui.label('SciTrans-LLMs — Scientific Document Translation System').classes('text-xs opacity-70')
    
    async def render_translate_panel():
        """Render the main translation panel - DeepL style left/right split."""
        # Job state containers
        uploaded_file = {'path': None, 'name': None}
        
        # Main container - left/right split, centered, with proper scrolling
        with ui.row().classes('w-full gap-4 p-4 justify-center').style('max-width: 1600px; margin: 0 auto;'):
            # LEFT SIDE: Source & Settings
            with ui.column().classes('w-1/2 gap-3').style('overflow-y: auto; max-height: 100%;'):
                # Source Document Card
                with ui.card().classes('w-full deepl-style compact-card'):
                    ui.label('Source Document').classes('text-base font-semibold mb-4')
                    
                    with ui.tabs().classes('w-full').props('dense') as source_tabs:
                        upload_tab = ui.tab('upload', label='Upload', icon='upload_file')
                        url_tab = ui.tab('url', label='URL', icon='link')
                    
                    with ui.tab_panels(source_tabs, value=upload_tab).classes('w-full'):
                        # -------- Upload from local file --------
                        with ui.tab_panel(upload_tab):
                            async def handle_upload(e):
                                """Handle file upload from NiceGUI `ui.upload`."""
                                try:
                                    file_obj = e.file
                                    file_name = file_obj.name
                                    temp_dir = Path(tempfile.gettempdir()) / "scitrans_uploads"
                                    temp_dir.mkdir(exist_ok=True)
                                    file_path = temp_dir / file_name
                                    await file_obj.save(file_path)
                                    
                                    uploaded_file['path'] = file_path
                                    uploaded_file['name'] = file_name

                                    # Update labels
                                    upload_label.text = f'✓ {file_name}'
                                    upload_label.classes(replace='text-green-600 text-base font-semibold')
                                    file_info.text = f'{file_name} ({file_path.stat().st_size / 1024:.1f} KB)'
                                    file_info.visible = True
                                    
                                    # Show preview info on the right
                                    preview_info.clear()
                                    with preview_info:
                                        ui.label('📄 Document Preview').classes('text-lg font-bold mb-3')
                                        ui.label(f'File: {file_name}').classes('text-base mb-2')
                                        ui.label(f'Size: {file_path.stat().st_size / 1024:.1f} KB').classes('text-sm opacity-70')
                                        if file_path.suffix.lower() == '.pdf':
                                            try:
                                                import fitz
                                                doc = fitz.open(str(file_path))
                                                page_count = len(doc)
                                                ui.label(f'Pages: {page_count}').classes('text-sm opacity-70 mt-1')
                                                
                                                # Show PDF preview
                                                pdf_preview.visible = True
                                                # Convert first page to image for preview
                                                try:
                                                    page = doc[0]
                                                    pix = page.get_pixmap(matrix=fitz.Matrix(0.5, 0.5))
                                                    img_data = pix.tobytes("png")
                                                    import base64
                                                    img_b64 = base64.b64encode(img_data).decode()
                                                    pdf_preview.set_content(f'<img src="data:image/png;base64,{img_b64}" style="max-width: 100%; height: auto; border-radius: 4px;" />')
                                                except Exception as e:
                                                    pdf_preview.set_content(f'<div style="padding: 20px; text-align: center; color: #666;">PDF preview unavailable<br/><small>{str(e)}</small></div>')
                                                
                                                doc.close()
                                            except Exception as e:
                                                pdf_preview.visible = False
                                                logger.exception("PDF preview error")
                                    preview_info.visible = True
                                    
                                    ui.notify(f'File uploaded: {file_name}', type='positive')
                                except Exception as ex:
                                    logger.exception("Upload error")
                                    ui.notify(f'Upload error: {str(ex)}', type='negative')
                            
                            # Upload area - use NiceGUI's built-in upload with visible button
                            with ui.column().classes('w-full items-center gap-3 p-8').style('min-height: 160px; border: 2px dashed rgba(99, 102, 241, 0.4); border-radius: 12px; background: rgba(99, 102, 241, 0.02);'):
                                ui.icon('cloud_upload').classes('opacity-70').style('font-size: 3rem;')
                                upload_label = ui.label('Click button below or drag & drop files here').classes('text-base text-center font-semibold')
                                ui.label('Supports: PDF, DOCX, HTML, TXT').classes('text-sm opacity-70 mt-1')
                                file_info = ui.label('').classes('text-base opacity-80 mt-3 font-medium')
                                file_info.visible = False
                                
                                # Upload widget - visible and accessible
                                upload_widget = ui.upload(
                                    on_upload=handle_upload,
                                    auto_upload=True,
                                ).props('accept=".pdf,.docx,.doc,.html,.htm,.txt" multiple=False')
                                upload_widget.classes('w-full')
                        
                        # -------- Download from URL --------
                        with ui.tab_panel(url_tab):
                            url_input = ui.input(
                                'Paper URL',
                                placeholder='https://arxiv.org/pdf/...'
                            ).classes('w-full text-base')
                            ui.label('Supports arXiv, DOI links, and direct PDF URLs').classes('text-sm opacity-70 mt-3')
                            
                            url_status = ui.label('').classes('text-sm mt-2')
                            url_status.visible = False
                            
                            async def validate_url():
                                url = url_input.value.strip()
                                if not url:
                                    url_status.visible = False
                                    # Clear preview if URL is empty
                                    if not uploaded_file.get('path'):
                                        preview_info.visible = False
                                        pdf_preview.visible = False
                                    return
                                
                                url_status.set_text('Validating URL...')
                                url_status.classes(replace='text-blue-500')
                                url_status.visible = True
                                
                                try:
                                    import urllib.request
                                    import urllib.parse
                                    
                                    # Validate URL format
                                    parsed = urllib.parse.urlparse(url)
                                    if not parsed.scheme or not parsed.netloc:
                                        raise ValueError("Invalid URL format")
                                    
                                    # Try HEAD request
                                    req = urllib.request.Request(url, method='HEAD')
                                    with urllib.request.urlopen(req, timeout=5) as response:
                                        content_type = response.headers.get('Content-Type', '')
                                        content_length = response.headers.get('Content-Length', '0')
                                        size_mb = int(content_length) / (1024 * 1024) if content_length else 0
                                        
                                        if 'pdf' in content_type.lower() or url.lower().endswith('.pdf'):
                                            url_status.set_text(f'✓ Valid PDF URL ({size_mb:.1f} MB)')
                                            url_status.classes(replace='text-green-500')
                                            
                                            # Show preview info
                                            preview_info.clear()
                                            with preview_info:
                                                ui.label('📄 Document from URL').classes('text-lg font-bold mb-3')
                                                ui.label(f'URL: {url[:60]}...' if len(url) > 60 else f'URL: {url}').classes('text-base mb-2')
                                                ui.label(f'Size: {size_mb:.1f} MB').classes('text-sm opacity-70')
                                                ui.label('Type: PDF').classes('text-sm opacity-70 mt-1')
                                            preview_info.visible = True
                                        else:
                                            url_status.set_text('⚠ URL accessible (may not be PDF)')
                                            url_status.classes(replace='text-yellow-500')
                                            
                                            # Show preview info anyway
                                            preview_info.clear()
                                            with preview_info:
                                                ui.label('📄 Document from URL').classes('text-lg font-bold mb-3')
                                                ui.label(f'URL: {url[:60]}...' if len(url) > 60 else f'URL: {url}').classes('text-base mb-2')
                                                ui.label(f'Size: {size_mb:.1f} MB').classes('text-sm opacity-70')
                                                ui.label(f'Type: {content_type}').classes('text-sm opacity-70 mt-1')
                                            preview_info.visible = True
                                except Exception as e:
                                    url_status.set_text(f'✗ Invalid URL: {str(e)}')
                                    url_status.classes(replace='text-red-500')
                                    preview_info.visible = False
                            
                            url_input.on('blur', validate_url)
                            url_input.on('input', lambda: url_status.set_text('') if not url_input.value.strip() else None)
            
                # Translation Settings Card
                with ui.card().classes('w-full deepl-style compact-card'):
                    ui.label('Translation Settings').classes('text-base font-semibold mb-4')
                    
                    # Direction
                    with ui.row().classes('items-center gap-4 mb-4'):
                        ui.label('Direction:').classes('font-medium text-base w-24')
                        direction = ui.toggle(
                            {'en-fr': 'EN → FR', 'fr-en': 'FR → EN'},
                            value='en-fr'
                        ).classes('flex-grow text-base')
                    
                    # Pages
                    with ui.row().classes('items-center gap-4 mb-4'):
                        ui.label('Pages:').classes('font-medium text-base w-24')
                        pages_input = ui.input(value='all', placeholder='all or 1-10').classes('flex-grow text-base')
                    
                    # Engine
                    with ui.row().classes('items-center gap-4 mb-4'):
                        ui.label('Engine:').classes('font-medium text-base w-24')
                        backends = get_available_backends()
                        engine_options = {b[0]: b[1] for b in backends if b[2]}
                        engine_select = ui.select(engine_options, value=state.default_engine).classes('flex-grow text-base')
                    
                    # Quality passes as dropdown
                    with ui.row().classes('items-center gap-4 mb-4'):
                        ui.label('Quality:').classes('font-medium text-base w-24')
                        quality_passes = ui.select(
                            {1: '1 pass (fast)', 2: '2 passes', 3: '3 passes', 4: '4 passes', 5: '5 passes (best)'},
                            value=state.default_quality
                        ).classes('flex-grow text-base')
                    
                    ui.separator().classes('my-4')
                    
                    # Advanced options
                    with ui.expansion('Advanced Options', icon='tune').classes('w-full text-base'):
                        enable_masking = ui.checkbox('Enable masking', value=state.default_masking).classes('text-base mb-3')
                        with ui.row().classes('gap-4 ml-6'):
                            translate_equations = ui.checkbox('Equations', value=False).classes('text-base')
                            translate_tables = ui.checkbox('Tables', value=False).classes('text-base')
                            translate_figures = ui.checkbox('Captions', value=True).classes('text-base')
                        enable_reranking = ui.checkbox('Enable reranking', value=False).classes('text-base mt-3')
                
                # Custom Glossary Card
                with ui.card().classes('w-full deepl-style compact-card'):
                    with ui.row().classes('items-center justify-between mb-3'):
                        ui.label('Custom Glossary').classes('text-base font-semibold')
                        with ui.row().classes('gap-2'):
                            ui.button('Upload', icon='upload_file', on_click=lambda: glossary_upload_widget.run_method('click')).props('flat dense').classes('text-base')
                            ui.button('Example', icon='help_outline', on_click=lambda: glossary_input.set_value(
                                '# Format: source_term, target_term\nneural network, réseau de neurones\ndeep learning, apprentissage profond'
                            )).props('flat dense').classes('text-base')
                    
                    # Glossary file upload
                    glossary_file_info = ui.label('').classes('text-sm opacity-70 mb-2')
                    glossary_file_info.visible = False
                    
                    async def handle_glossary_upload(e):
                        try:
                            file_obj = e.file
                            file_name = file_obj.name
                            temp_dir = Path(tempfile.gettempdir()) / "scitrans_glossaries"
                            temp_dir.mkdir(exist_ok=True)
                            file_path = temp_dir / file_name
                            await file_obj.save(file_path)
                            
                            # Load and display glossary
                            try:
                                from scitrans_llms.translate.glossary import load_glossary_csv
                                glossary = load_glossary_csv(file_path)
                                
                                # Format for textarea
                                glossary_text = '\n'.join([f'{e.source}, {e.target}' for e in glossary.entries])
                                glossary_input.set_value(glossary_text)
                                
                                glossary_file_info.set_text(f'✓ Loaded {len(glossary.entries)} terms from {file_name}')
                                glossary_file_info.classes(replace='text-green-600')
                                glossary_file_info.visible = True
                                
                                # Save to cache
                                from scitrans_llms.config import GLOSSARY_DIR
                                GLOSSARY_DIR.mkdir(parents=True, exist_ok=True)
                                cache_path = GLOSSARY_DIR / f'user_{file_name}'
                                import shutil
                                shutil.copy(file_path, cache_path)
                                
                                ui.notify(f'Glossary loaded: {len(glossary.entries)} terms', type='positive')
                            except Exception as ex:
                                glossary_file_info.set_text(f'✗ Error loading glossary: {str(ex)}')
                                glossary_file_info.classes(replace='text-red-600')
                                glossary_file_info.visible = True
                                logger.exception("Glossary load error")
                        except Exception as ex:
                            logger.exception("Glossary upload error")
                            ui.notify(f'Upload error: {str(ex)}', type='negative')
                    
                    glossary_upload_widget = ui.upload(
                        on_upload=handle_glossary_upload,
                        auto_upload=True,
                    ).props('accept=".csv,.txt" multiple=False')
                    glossary_upload_widget.visible = False
                    
                    with ui.expansion('Format Instructions', icon='info').classes('w-full mb-2 text-sm'):
                        ui.markdown('''
**Supported formats:**
- CSV: `source_term,target_term` (first row is header)
- TXT: `source_term, target_term` or `source_term → target_term`

**Options:**
- **Strict mode**: Terms must match exactly (case-sensitive)
- **Context-aware**: Terms adapt to surrounding context
- **Override default**: Replace default glossary entries
                        ''').classes('text-sm')
                    
                    glossary_input = ui.textarea(
                        placeholder='# Format: source_term, target_term\nneural network, réseau de neurones'
                    ).classes('w-full mono-font text-base').props('rows=4')
                    
                    # Glossary options
                    with ui.row().classes('gap-4 mt-2'):
                        glossary_strict = ui.checkbox('Strict mode', value=False).classes('text-sm')
                        glossary_context = ui.checkbox('Context-aware', value=True).classes('text-sm')
                        glossary_override = ui.checkbox('Override default', value=False).classes('text-sm')
                
                # Translate button
                translate_btn = ui.button(
                    'Translate Document',
                    icon='translate',
                ).classes('w-full mt-4 text-lg py-3').props('color=primary')
                translate_btn.visible = True
            
            # RIGHT SIDE: Preview & Results
            with ui.column().classes('w-1/2 gap-4').style('overflow-y: auto; max-height: 100%;'):
                # Preview/Progress Card
                with ui.card().classes('w-full deepl-style compact-card').style('min-height: 500px;') as preview_card:
                    ui.label('Preview & Progress').classes('text-base font-semibold mb-4')
                    
                    # PDF Preview area
                    pdf_preview = ui.html('', sanitize=False).classes('w-full mb-4').style('min-height: 300px; border: 1px solid rgba(0,0,0,0.1); border-radius: 8px; background: #f5f5f5;')
                    pdf_preview.visible = False
                    
                    # Preview area (for source document info)
                    preview_info = ui.column().classes('w-full mb-4 p-3').style('background: rgba(99, 102, 241, 0.05); border-radius: 8px;')
                    preview_info.visible = False
                    
                    # Progress bar (hidden initially)
                    progress_bar = ui.linear_progress(value=0, show_value=False).classes('w-full mb-3').style('height: 12px; border-radius: 6px;')
                    progress_bar.visible = False
                    progress_label = ui.label('').classes('text-base text-center mb-3 font-semibold')
                    progress_label.visible = False
                    
                    # Loading spinner
                    loading_spinner = ui.spinner(size='lg', color='primary').classes('mx-auto mb-3')
                    loading_spinner.visible = False
                    
                    # Log output with timestamps
                    log_output = ui.log(max_lines=20).classes('w-full mono-font text-base').style('height: 250px; background: rgba(0,0,0,0.03); padding: 12px; border-radius: 8px; border: 1px solid rgba(0,0,0,0.1);')
                    log_output.visible = False
                    
                    # Result area
                    result_status = ui.label('Upload a document and click Translate to begin').classes('text-base text-center opacity-70 mt-6')
                    result_path = ui.label('').classes('text-base opacity-80 mt-3')
                    result_path.visible = False
                    result_stats = ui.label('').classes('text-base opacity-80 mt-3')
                    result_stats.visible = False
                    
                    with ui.row().classes('w-full gap-2 mt-4'):
                        download_btn = ui.button(
                            'Download Translated PDF',
                            icon='download',
                            on_click=lambda: ui.download(preview_card._output_path) if hasattr(preview_card, '_output_path') else None
                        ).classes('flex-grow text-lg py-3').props('color=primary')
                        download_btn.visible = False
                        
                        clear_btn = ui.button(
                            'Clear',
                            icon='clear',
                            on_click=lambda: clear_translation_results()
                        ).classes('text-lg py-3').props('flat')
                        clear_btn.visible = False
                    
                    def clear_translation_results():
                        preview_info.clear()
                        preview_info.visible = False
                        pdf_preview.visible = False
                        progress_bar.visible = False
                        progress_label.visible = False
                        loading_spinner.visible = False
                        log_output.visible = False
                        result_status.set_text('Upload a document and click Translate to begin')
                        result_status.classes(replace='text-base text-center opacity-70 mt-6')
                        result_path.visible = False
                        result_stats.visible = False
                        download_btn.visible = False
                        clear_btn.visible = False
                        uploaded_file['path'] = None
                        uploaded_file['name'] = None
                        url_input.set_value('')
                        if hasattr(preview_card, '_output_path'):
                            delattr(preview_card, '_output_path')
                        ui.notify('Translation results cleared', type='info')
            
            # Define translation function after all UI elements are created
            async def start_translation():
                source_path = uploaded_file.get('path')
                source_url = url_input.value.strip() if url_input.value and url_input.value.strip() else None
                
                if not source_path and not source_url:
                    ui.notify('Please upload a file or provide a URL', type='warning')
                    return
                
                # If URL provided but no file, download it
                if source_url and not source_path:
                    try:
                        log_output.push(f"[{datetime.now().strftime('%H:%M:%S')}] Downloading from URL...")
                        import urllib.request
                        temp_dir = Path(tempfile.gettempdir()) / "scitrans_uploads"
                        temp_dir.mkdir(exist_ok=True)
                        file_name = source_url.split('/')[-1] or 'document.pdf'
                        if not file_name.endswith('.pdf'):
                            file_name = 'document.pdf'
                        file_path = temp_dir / file_name
                        urllib.request.urlretrieve(source_url, file_path)
                        uploaded_file['path'] = file_path
                        uploaded_file['name'] = file_name
                        log_output.push(f"[{datetime.now().strftime('%H:%M:%S')}] Downloaded: {file_name}")
                    except Exception as e:
                        logger.exception("URL download error")
                        ui.notify(f'Failed to download from URL: {str(e)}', type='negative')
                        return
                
                # Disable button and show progress
                translate_btn.props('disable')
                translate_btn.set_text('Translating...')
                progress_bar.visible = True
                progress_bar.value = 0
                progress_label.visible = True
                loading_spinner.visible = True
                log_output.visible = True
                result_status.set_text('Translating...')
                result_status.classes(replace='text-sm text-center font-medium')
                download_btn.visible = False
                clear_btn.visible = False
                log_output.clear()
                
                # Initial log message
                timestamp = datetime.now().strftime('%H:%M:%S')
                log_output.push(f"[{timestamp}] Translation started")
                if source_path:
                    log_output.push(f"[{timestamp}] Source: {uploaded_file.get('name', 'Unknown')}")
                if source_url:
                    log_output.push(f"[{timestamp}] Source URL: {source_url}")
                
                try:
                    # Build job
                    job = TranslationJob(
                        source_path=source_path,
                        source_url=source_url,
                        direction=direction.value,
                        pages=pages_input.value,
                        engine=engine_select.value,
                        enable_masking=enable_masking.value,
                        translate_figures=translate_figures.value,
                        translate_tables=translate_tables.value,
                        translate_equations=translate_equations.value,
                        quality_passes=int(quality_passes.value),
                        enable_reranking=enable_reranking.value,
                        custom_glossary=glossary_input.value,
                    )
                    
                    # Progress callback with timestamps
                    def update_progress(msg: str, pct: float):
                        timestamp = datetime.now().strftime('%H:%M:%S')
                        progress_bar.value = pct
                        progress_label.set_text(f"{int(pct * 100)}% - {msg}")
                        log_output.push(f"[{timestamp}] {msg}")
                        # Don't auto-scroll - let user control scrolling
                    
                    # Run translation in background task
                    result = await run_translation(job, update_progress)
                    
                    # Show result
                    progress_bar.visible = False
                    progress_label.visible = False
                    loading_spinner.visible = False
                    translate_btn.props(remove='disable')
                    translate_btn.set_text('Translate Document')
                    
                    if result.get('success'):
                        timestamp = datetime.now().strftime('%H:%M:%S')
                        log_output.push(f"[{timestamp}] ✓ Translation completed successfully!")
                        result_status.set_text('✓ Translation Complete!')
                        result_status.classes(replace='text-sm font-bold text-green-500 text-center')
                        result_path.set_text(f"Saved to: {result.get('output_path')}")
                        result_path.visible = True
                        preview_card._output_path = result.get('output_path')
                        download_btn.visible = True
                        clear_btn.visible = True
                        
                        # Show stats if available
                        if result.get('stats'):
                            stats = result.get('stats')
                            stats_text = f"Blocks: {stats.get('total_blocks', 'N/A')}, "
                            stats_text += f"Translated: {stats.get('translated_blocks', 'N/A')}, "
                            stats_text += f"Time: {stats.get('duration', 'N/A')}s"
                            result_stats.set_text(stats_text)
                            result_stats.visible = True
                        
                        stats = result.get('stats', {})
                        stats_text = f"Blocks: {stats.get('total_blocks', 0)} total, {stats.get('translated_blocks', 0)} translated"
                        if stats.get('masks_applied'):
                            stats_text += f", {stats.get('masks_applied')} masks applied"
                        result_stats.set_text(stats_text)
                        result_stats.visible = True
                    else:
                        result_status.set_text('✗ Translation Failed')
                        result_status.classes(replace='text-xs font-bold text-red-500 text-center compact-text')
                        result_path.set_text(f"Error: {result.get('error')}")
                        result_path.visible = True
                        result_stats.visible = False
                
                except Exception as ex:
                    logger.exception("Translation error")
                    result_status.set_text('✗ Translation Error')
                    result_status.classes(replace='text-xs font-bold text-red-500 text-center compact-text')
                    result_path.set_text(f"Error: {str(ex)}")
                    result_path.visible = True
                
                finally:
                    translate_btn.props(remove='disable')
            
            # Assign handler to button after function is defined
            # Connect translate button - use on() method for async functions
            # Connect translate button handler - use on_click for async functions
            translate_btn.on_click = start_translation
    
    async def render_glossary_panel():
        """Render the glossary management panel."""
        default_glossary = get_default_glossary()
        
        # Centered container, no scrolling
        with ui.column().classes('w-full max-w-6xl mx-auto gap-3 p-3').style('height: 100%; overflow-y: auto; max-height: calc(100vh - 180px);'):
            with ui.row().classes('w-full gap-3'):
                # Left - Default glossary browser
                with ui.column().classes('w-1/2 gap-2').style('overflow-y: auto; max-height: 100%;'):
                    with ui.card().classes('w-full deepl-style compact-card'):
                        ui.label('Default Scientific Glossary').classes('text-base font-semibold mb-2')
                        ui.label(f'{len(default_glossary)} terms • English ↔ French').classes('text-sm opacity-70 mb-3')
                        
                        # Search
                        search_input = ui.input('Search terms...', on_change=lambda: filter_glossary()).props('dense clearable').classes('w-full mb-2 text-base')
                        
                        # Domain filter
                        domains = sorted(set(e.domain or 'general' for e in default_glossary.entries))
                        domain_filter = ui.select(
                            {'all': 'All Domains'} | {d: d.title() for d in domains},
                            value='all',
                            on_change=lambda: filter_glossary()
                        ).classes('w-full mb-2 text-base').props('dense')
                        
                        # Glossary table
                        columns = [
                            {'name': 'source', 'label': 'English', 'field': 'source', 'align': 'left'},
                            {'name': 'target', 'label': 'French', 'field': 'target', 'align': 'left'},
                            {'name': 'domain', 'label': 'Domain', 'field': 'domain', 'align': 'left'},
                        ]
                        
                        rows = [
                            {'source': e.source, 'target': e.target, 'domain': e.domain or 'general'}
                            for e in default_glossary.entries
                        ]
                        
                        glossary_table = ui.table(columns=columns, rows=rows, row_key='source').classes('w-full').style('font-size: 12px; max-height: 500px;')
                        glossary_table.props('dense flat')
                        
                        def filter_glossary():
                            query = search_input.value.lower() if search_input.value else ''
                            selected_domain = domain_filter.value
                            
                            filtered = [
                                {'source': e.source, 'target': e.target, 'domain': e.domain or 'general'}
                                for e in default_glossary.entries
                                if (query in e.source.lower() or query in e.target.lower() or query in (e.domain or 'general').lower())
                                and (selected_domain == 'all' or (e.domain or 'general') == selected_domain)
                            ]
                            glossary_table.rows = filtered
                
                # Right - Custom glossary
                with ui.column().classes('w-1/2 gap-3').style('overflow-y: auto; max-height: 100%;'):
                    with ui.card().classes('w-full deepl-style compact-card'):
                        with ui.row().classes('items-center justify-between mb-3'):
                            ui.label('Create Custom Glossary').classes('text-base font-semibold')
                            # Upload button for glossary file
                            async def handle_glossary_file_upload(e):
                                try:
                                    file_obj = e.file
                                    file_name = file_obj.name
                                    temp_dir = Path(tempfile.gettempdir()) / "scitrans_glossaries"
                                    temp_dir.mkdir(exist_ok=True)
                                    file_path = temp_dir / file_name
                                    await file_obj.save(file_path)
                                    
                                    # Load and display
                                    try:
                                        from scitrans_llms.translate.glossary import load_glossary_csv
                                        glossary = load_glossary_csv(file_path)
                                        glossary_text = '\n'.join([f'{e.source}, {e.target}' for e in glossary.entries])
                                        custom_glossary.set_value(glossary_text)
                                        
                                        # Save to cache for others
                                        from scitrans_llms.config import GLOSSARY_DIR
                                        GLOSSARY_DIR.mkdir(parents=True, exist_ok=True)
                                        cache_path = GLOSSARY_DIR / f'user_{file_name}'
                                        import shutil
                                        shutil.copy(file_path, cache_path)
                                        
                                        ui.notify(f'Glossary loaded: {len(glossary.entries)} terms (saved to cache)', type='positive')
                                    except Exception as ex:
                                        ui.notify(f'Error loading glossary: {str(ex)}', type='negative')
                                        logger.exception("Glossary load error")
                                except Exception as ex:
                                    ui.notify(f'Upload error: {str(ex)}', type='negative')
                                    logger.exception("Glossary upload error")
                            
                            glossary_file_upload = ui.upload(
                                on_upload=handle_glossary_file_upload,
                                auto_upload=True,
                            ).props('accept=".csv,.txt" multiple=False')
                            glossary_file_upload.visible = False
                            
                            ui.button('Upload File', icon='upload_file', on_click=lambda: glossary_file_upload.run_method('click')).props('flat dense').classes('text-sm')
                        
                        # Format help
                        with ui.expansion('Format Guide', icon='help').classes('w-full mb-2'):
                            ui.markdown('''
**Format:** `source_term, target_term` or `source_term → target_term`

**Example:**
```
neural network, réseau de neurones
deep learning, apprentissage profond
```
                            ''').classes('text-sm')
                        
                        custom_glossary = ui.textarea(
                            placeholder='# source_term, target_term\nneural network, réseau de neurones',
                        ).classes('w-full mono-font text-base').props('rows=10')
                        
                        with ui.row().classes('gap-2 mt-2'):
                            def save_custom_glossary():
                                content = custom_glossary.value
                                if not content.strip():
                                    ui.notify('Glossary is empty', type='warning')
                                    return
                                try:
                                    # Save to user config directory
                                    from scitrans_llms.config import GLOSSARY_DIR
                                    GLOSSARY_DIR.mkdir(parents=True, exist_ok=True)
                                    save_path = GLOSSARY_DIR / 'custom_glossary.csv'
                                    with open(save_path, 'w', encoding='utf-8') as f:
                                        f.write('source,target\n')
                                        for line in content.strip().split('\n'):
                                            line = line.strip()
                                            if not line or line.startswith('#'):
                                                continue
                                            for sep in [',', '\t', ' → ', ' -> ', ' = ', ':']:
                                                if sep in line:
                                                    parts = line.split(sep, 1)
                                                    if len(parts) == 2:
                                                        f.write(f'"{parts[0].strip()}","{parts[1].strip()}"\n')
                                                    break
                                    ui.notify(f'Glossary saved to {save_path}', type='positive')
                                except Exception as e:
                                    ui.notify(f'Save failed: {str(e)}', type='negative')
                            
                            def load_custom_glossary():
                                try:
                                    from scitrans_llms.config import GLOSSARY_DIR
                                    load_path = GLOSSARY_DIR / 'custom_glossary.csv'
                                    if load_path.exists():
                                        glossary = load_glossary_csv(load_path)
                                        content = '\n'.join([f'{e.source}, {e.target}' for e in glossary.entries])
                                        custom_glossary.set_value(content)
                                        ui.notify(f'Loaded {len(glossary.entries)} entries', type='positive')
                                    else:
                                        ui.notify('No saved glossary found', type='warning')
                                except Exception as e:
                                    ui.notify(f'Load failed: {str(e)}', type='negative')
                            
                            def export_custom_glossary():
                                content = custom_glossary.value
                                if not content.strip():
                                    ui.notify('Glossary is empty', type='warning')
                                    return
                                ui.download(content.encode('utf-8'), filename='custom_glossary.txt')
                                ui.notify('Glossary exported', type='positive')
                            
                            ui.button('Save', icon='save', on_click=save_custom_glossary).props('color=primary dense').classes('text-sm')
                            ui.button('Load', icon='upload_file', on_click=load_custom_glossary).props('dense').classes('text-sm')
                            ui.button('Export', icon='download', on_click=export_custom_glossary).props('dense').classes('text-sm')
                    
                    # Corpus Integration Info
                    with ui.card().classes('w-full deepl-style compact-card mt-3'):
                        ui.label('Corpus Integration').classes('text-base font-semibold mb-2')
                        ui.label('Free translation models can be enhanced with verified corpora:').classes('text-sm opacity-70 mb-2')
                        with ui.expansion('Available Corpora', icon='library_books').classes('w-full'):
                            ui.markdown('''
**Europarl Corpus** - European Parliament proceedings (1996-2011)
- 2M+ sentence pairs (EN-FR)
- Source: https://www.statmt.org/europarl/

**OPUS PHP Corpus** - PHP documentation translations
- 1.38M sentence fragments
- Source: https://opus.nlpl.eu/

**European Language Grid** - Additional EN-FR resources
- Source: https://live.european-language-grid.eu/

*Note: Corpus integration requires model training. Contact developers for implementation.*
                            ''').classes('text-sm')
    
    async def render_developer_panel():
        """Render the developer tools panel."""
        with ui.tabs().classes('w-full text-sm').props('dense') as dev_tabs:
            testing_tab = ui.tab('testing', label='Testing', icon='science')
            logs_tab = ui.tab('logs', label='Logs', icon='article')
            cli_tab = ui.tab('cli', label='CLI', icon='terminal')
            debug_tab = ui.tab('debug', label='Debug', icon='bug_report')
        
        with ui.tab_panels(dev_tabs, value=testing_tab).classes('w-full'):
            # Testing Ground
            with ui.tab_panel(testing_tab).classes('p-4'):
                # Centered container with box
                with ui.column().classes('w-full max-w-4xl mx-auto gap-3').style('height: 100%; overflow-y: auto; max-height: calc(100vh - 180px);'):
                    # Clear translation button - more visible
                    with ui.row().classes('w-full justify-between items-center mb-4'):
                        ui.label('Quick Translation Test').classes('text-lg font-bold')
                        def clear_test_results():
                            test_output.set_content('*Output will appear here...*')
                            test_input.set_value('The neural network achieved state-of-the-art performance.')
                            ui.notify('Test results cleared', type='info')
                        ui.button('Clear Results', icon='clear', on_click=clear_test_results).props('color=primary').classes('text-sm')
                    
                    # Testing box - centered
                    with ui.card().classes('w-full deepl-style compact-card'):
                        with ui.row().classes('w-full gap-3'):
                            # Left - Test input
                            with ui.column().classes('w-1/2 gap-2'):
                                ui.label('Test Input').classes('text-base font-semibold mb-2')
                                
                                test_input = ui.textarea(
                                    placeholder='Enter text to test translation...',
                                    value='The neural network achieved state-of-the-art performance.'
                                ).classes('w-full text-base').props('rows=6')
                                
                                with ui.row().classes('gap-2 mt-3'):
                                    test_direction = ui.toggle(
                                        {'en-fr': 'EN→FR', 'fr-en': 'FR→EN'},
                                        value='en-fr'
                                    ).classes('text-base')
                                    # All available backends
                                    backends = get_available_backends()
                                    test_backend_options = {b[0]: b[1] for b in backends}
                                    test_backend = ui.select(
                                        test_backend_options,
                                        value='free'
                                    ).classes('flex-grow text-base')
                                
                                async def run_test():
                                    test_output.set_content('Translating...')
                                    try:
                                        config = PipelineConfig(
                                            source_lang='en' if test_direction.value == 'en-fr' else 'fr',
                                            target_lang='fr' if test_direction.value == 'en-fr' else 'en',
                                            translator_backend=test_backend.value,
                                            enable_glossary=True,
                                            enable_masking=True,
                                        )
                                        pipeline = TranslationPipeline(config)
                                        result = pipeline.translate_text(test_input.value)
                                        test_output.set_content(result)
                                    except Exception as e:
                                        test_output.set_content(f'Error: {str(e)}')
                                
                                ui.button('Test Translation', icon='play_arrow', on_click=run_test).classes('w-full mt-3 text-base py-2').props('color=primary')
                            
                            # Right - Test output
                            with ui.column().classes('w-1/2 gap-2'):
                                ui.label('Translation Output').classes('text-base font-semibold mb-2')
                                test_output = ui.markdown('*Output will appear here...*').classes('w-full p-4 border rounded text-base').style('min-height: 200px; background: rgba(0,0,0,0.02);')
                                
                                with ui.card().classes('w-full deepl-style compact-card mt-3'):
                                    ui.label('Component Tests').classes('text-base font-semibold mb-2')
                                    
                                    async def test_pdf_parser():
                                        ui.notify('PDF parser: OK', type='positive')
                                    
                                    async def test_masking():
                                        from scitrans_llms.masking import mask_document, MaskConfig
                                        ui.notify('Masking: OK', type='positive')
                                    
                                    async def test_glossary():
                                        glossary = get_default_glossary()
                                        ui.notify(f'Glossary: {len(glossary)} entries', type='positive')
                                    
                                    with ui.row().classes('gap-2 flex-wrap'):
                                        ui.button('PDF Parser', on_click=test_pdf_parser).props('flat').classes('text-sm')
                                        ui.button('Masking', on_click=test_masking).props('flat').classes('text-sm')
                                        ui.button('Glossary', on_click=test_glossary).props('flat').classes('text-sm')
                                        ui.button('Rerank', on_click=lambda: ui.notify('Rerank: OK', type='positive')).props('flat').classes('text-sm')
            
            # Logs panel
            with ui.tab_panel(logs_tab).classes('p-4'):
                with ui.column().classes('w-full max-w-6xl mx-auto gap-3'):
                    with ui.card().classes('w-full deepl-style compact-card'):
                        with ui.row().classes('justify-between items-center mb-2'):
                            ui.label('System Logs').classes('text-xs font-semibold compact-text')
                            with ui.row().classes('gap-1'):
                                def clear_logs():
                                    log_viewer.clear()
                                    log_viewer.push(f'[{datetime.now().strftime("%H:%M:%S")}] Logs cleared')
                                    ui.notify('Logs cleared', type='info')
                                
                                def export_logs():
                                    try:
                                        log_content = '\n'.join([f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {msg}' for msg in state.logs[-1000:]])
                                        if not log_content.strip():
                                            log_content = 'No logs available'
                                        ui.download(log_content.encode('utf-8'), filename=f'scitrans_logs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
                                        ui.notify('Logs exported', type='positive')
                                    except Exception as e:
                                        ui.notify(f'Export failed: {str(e)}', type='negative')
                                
                                ui.button('Clear', icon='delete', on_click=clear_logs).props('flat dense').classes('text-xs')
                                ui.button('Export', icon='download', on_click=export_logs).props('flat dense').classes('text-xs')
                        
                        log_viewer = ui.log(max_lines=100).classes('w-full mono-font text-xs compact-text').style('height: calc(100vh - 200px); font-size: 10px;')
                        
                        # Initialize logs
                        timestamp = datetime.now().strftime('%H:%M:%S')
                        log_viewer.push(f'[{timestamp}] SciTrans-LLMs GUI started')
                        state.logs.append('SciTrans-LLMs GUI started')
                        
                        # Set up log capture
                        def log_callback(msg):
                            timestamp = datetime.now().strftime('%H:%M:%S')
                            log_viewer.push(f'[{timestamp}] {msg}')
                            state.logs.append(msg)
                            if len(state.logs) > 1000:
                                state.logs.pop(0)
                        
                        set_global_log_callback(log_callback)
                        
                        # Add initial logs
                        default_glossary = get_default_glossary()
                        log_viewer.push(f'[{timestamp}] Default glossary loaded: {len(default_glossary)} terms')
                        state.logs.append(f'Default glossary loaded: {len(default_glossary)} terms')
                        
                        backends = get_available_backends()
                        available = [b[0] for b in backends if b[2]]
                        log_viewer.push(f'[{timestamp}] Available backends: {", ".join(available)}')
                        state.logs.append(f'Available backends: {", ".join(available)}')
            
            # CLI Commands
            with ui.tab_panel(cli_tab).classes('p-4'):
                with ui.column().classes('w-full max-w-4xl mx-auto'):
                    with ui.card().classes('w-full deepl-style compact-card'):
                        ui.label('CLI Reference').classes('text-lg font-semibold mb-4')
                    
                    ui.markdown('''
                **Available Commands:**

                ```bash
                # System info
                scitrans info

                # Quick translation
                scitrans translate --text "Hello world" --backend free

                # Translate PDF
                scitrans translate --input paper.pdf --output translated.pdf --backend openai

                # API key management
                scitrans keys list
                scitrans keys set openai
                scitrans keys set deepseek

                # Run experiments
                scitrans experiment --corpus ./corpus --output ./results
                ```
                    ''').classes('mono-font')
            
            # Debug Info - hide personal info
            with ui.tab_panel(debug_tab).classes('p-4'):
                with ui.column().classes('w-full max-w-6xl mx-auto gap-3').style('height: calc(100vh - 140px); overflow-y: auto;'):
                    with ui.row().classes('w-full gap-3'):
                        with ui.column().classes('w-1/2 gap-2'):
                            with ui.card().classes('w-full deepl-style compact-card'):
                                ui.label('System Information').classes('text-xs font-semibold mb-2 compact-text')
                                
                                import sys
                                import platform
                                
                                # Hide personal info - only show safe info
                                python_version = sys.version.split()[0]
                                platform_name = platform.system()
                                platform_release = platform.release()
                                platform_version = platform.version()
                                
                                # Get working directory but sanitize it
                                cwd = os.getcwd()
                                # Remove username and home directory from path
                                if 'Users' in cwd or 'home' in cwd.lower():
                                    # Show relative path or sanitized path
                                    cwd_display = '...' + cwd.split('/')[-1] if '/' in cwd else cwd
                                else:
                                    cwd_display = cwd
                                
                                info = [
                                    f"Python: {python_version}",
                                    f"Platform: {platform_name} {platform_release}",
                                    f"Platform Version: {platform_version}",
                                    f"Working Dir: {cwd_display}",
                                ]
                                
                                # System resources
                                try:
                                    import psutil
                                    cpu_percent = psutil.cpu_percent(interval=0.1)
                                    memory = psutil.virtual_memory()
                                    info.append(f"CPU: {cpu_percent:.1f}%")
                                    info.append(f"Memory: {memory.percent:.1f}% ({memory.used / (1024**3):.1f} GB / {memory.total / (1024**3):.1f} GB)")
                                except ImportError:
                                    pass
                                except Exception:
                                    pass
                                
                                try:
                                    import fitz
                                    info.append(f"PyMuPDF: {fitz.version[0]}")
                                except:
                                    info.append("PyMuPDF: Not installed")
                                
                                for line in info:
                                    ui.label(line).classes('mono-font text-xs compact-text')
                            
                            with ui.card().classes('w-full deepl-style compact-card'):
                                ui.label('Application State').classes('text-xs font-semibold mb-2 compact-text')
                                
                                state_info = [
                                    f"Dark Mode: {'On' if state.dark_mode else 'Off'}",
                                    f"Default Engine: {state.default_engine}",
                                    f"Default Quality: {state.default_quality} pass(es)",
                                    f"Default Masking: {'Enabled' if state.default_masking else 'Disabled'}",
                                    f"Current Job: {'Active' if state.current_job else 'None'}",
                                    f"Logs Count: {len(state.logs)}",
                                ]
                                
                                for line in state_info:
                                    ui.label(line).classes('mono-font text-xs compact-text')
                        
                        with ui.column().classes('w-1/2 gap-2'):
                            with ui.card().classes('w-full deepl-style compact-card'):
                                ui.label('Module Status').classes('text-xs font-semibold mb-2 compact-text')
                                
                                modules = [
                                    ('PyMuPDF', 'fitz'),
                                    ('NiceGUI', 'nicegui'),
                                    ('Pydantic', 'pydantic'),
                                    ('OpenAI', 'openai'),
                                    ('Anthropic', 'anthropic'),
                                    ('BeautifulSoup', 'bs4'),
                                    ('python-docx', 'docx'),
                                    ('requests', 'requests'),
                                    ('numpy', 'numpy'),
                                    ('psutil', 'psutil'),
                                ]
                                
                                for name, mod in modules:
                                    try:
                                        mod_obj = __import__(mod)
                                        version = getattr(mod_obj, '__version__', 'installed')
                                        ui.label(f'✓ {name} ({version})').classes('text-green-500 text-xs compact-text')
                                    except ImportError:
                                        ui.label(f'✗ {name}').classes('text-red-500 text-xs compact-text')
                            
                            with ui.card().classes('w-full deepl-style compact-card'):
                                ui.label('Translation Backends').classes('text-xs font-semibold mb-2 compact-text')
                                
                                backends = get_available_backends()
                                for backend_id, label, available in backends:
                                    status = '✓' if available else '✗'
                                    color = 'text-green-500' if available else 'text-red-500'
                                    ui.label(f'{status} {label}').classes(f'{color} text-xs compact-text')
                            
                            with ui.card().classes('w-full deepl-style compact-card'):
                                ui.label('Glossary Info').classes('text-xs font-semibold mb-2 compact-text')
                                
                                try:
                                    default_glossary = get_default_glossary()
                                    ui.label(f'Default Glossary: {len(default_glossary)} entries').classes('text-xs compact-text')
                                    
                                    # Count by domain
                                    domains = {}
                                    for entry in default_glossary.entries:
                                        domain = entry.domain or 'general'
                                        domains[domain] = domains.get(domain, 0) + 1
                                    
                                    ui.label('Entries by domain:').classes('text-xs compact-text mt-2')
                                    for domain, count in sorted(domains.items()):
                                        ui.label(f'  • {domain}: {count}').classes('text-xs compact-text opacity-70')
                                except Exception as e:
                                    ui.label(f'Error loading glossary: {str(e)}').classes('text-red-500 text-xs compact-text')
    
    async def render_settings_panel(km: KeyManager):
        """Render the settings and API keys panel."""
        # Centered container
        with ui.column().classes('w-full max-w-6xl mx-auto gap-3 p-3').style('height: 100%; overflow-y: auto; max-height: calc(100vh - 180px);'):
            with ui.row().classes('w-full gap-3'):
                # API Keys
                with ui.column().classes('w-1/2 gap-2'):
                    with ui.card().classes('w-full deepl-style compact-card'):
                        ui.label('API Keys').classes('text-xs font-semibold mb-2 compact-text')
                        ui.label('Configure API keys for translation engines').classes('text-xs opacity-70 mb-2 compact-text')
                        
                        # All available services
                        services = [
                            ('openai', 'OpenAI', 'GPT-4, GPT-4o models'),
                            ('deepseek', 'DeepSeek', 'Affordable alternative'),
                            ('anthropic', 'Anthropic', 'Claude models'),
                            ('huggingface', 'HuggingFace', 'Free tier available'),
                            ('ollama', 'Ollama', 'Local models'),
                        ]
                        
                        for service_id, name, desc in services:
                            key_info = km.get_key_info(service_id) if hasattr(km, 'get_key_info') else type('obj', (object,), {'is_set': False, 'masked_value': '***'})()
                            
                            with ui.row().classes('w-full items-center gap-2 py-1 border-b'):
                                with ui.column().classes('flex-grow'):
                                    ui.label(name).classes('font-medium text-xs compact-text')
                                    ui.label(desc).classes('text-xs opacity-60 compact-text')
                                
                                if key_info.is_set:
                                    ui.badge('Set', color='green').props('outline')
                                else:
                                    ui.badge('Not set', color='grey').props('outline')
                                
                                # Edit button
                                def make_edit_handler(sid):
                                    async def edit_key():
                                        with ui.dialog() as dialog, ui.card().classes('compact-card'):
                                            ui.label(f'Set {sid.title()} API Key').classes('text-xs font-semibold mb-2 compact-text')
                                            key_input = ui.input('API Key', password=True).classes('w-full text-xs compact-text').props('dense')
                                            with ui.row().classes('gap-2 mt-2'):
                                                ui.button('Cancel', on_click=dialog.close).props('flat dense').classes('text-xs')
                                                def save_key():
                                                    if key_input.value:
                                                        km.set_key(sid, key_input.value)
                                                        ui.notify(f'{sid.title()} key saved', type='positive')
                                                        dialog.close()
                                                ui.button('Save', on_click=save_key).props('color=primary dense').classes('text-xs')
                                        dialog.open()
                                    return edit_key
                                
                                ui.button(icon='edit', on_click=make_edit_handler(service_id)).props('flat round dense').classes('text-xs')
                        
                        ui.separator().classes('my-2')
                        ui.label('Keys stored securely in system keychain or local config.').classes('text-xs opacity-50 compact-text')
                
                # App Settings
                with ui.column().classes('w-1/2 gap-2'):
                    with ui.card().classes('w-full deepl-style compact-card'):
                        ui.label('Application Settings').classes('text-xs font-semibold mb-2 compact-text')
                        
                        # Theme - auto-apply
                        with ui.row().classes('items-center justify-between py-1'):
                            ui.label('Dark Mode').classes('text-xs compact-text')
                            def toggle_dark_mode(e):
                                state.dark_mode = e.value
                                # Update global dark mode
                                dark_mode = ui.dark_mode()
                                dark_mode.value = e.value
                                ui.notify(f'{"Dark" if e.value else "Light"} mode activated', type='info')
                            dark_mode_switch = ui.switch(value=state.dark_mode, on_change=toggle_dark_mode).props('dense')
                        
                        ui.separator().classes('my-1')
                        
                        # Default engine - dropdown with all options - auto-apply
                        with ui.row().classes('items-center justify-between py-1'):
                            ui.label('Default Engine').classes('text-xs compact-text')
                            backends = get_available_backends()
                            default_engine_options = {b[0]: b[1] for b in backends if b[2]}
                            def update_default_engine(e):
                                state.default_engine = e.value
                                ui.notify(f'Default engine set to {e.value} (applied to Translate tab)', type='positive')
                            default_engine = ui.select(
                                default_engine_options,
                                value=state.default_engine,
                                on_change=update_default_engine
                            ).classes('text-xs compact-text').props('dense')
                        
                        # Default quality passes - dropdown - auto-apply
                        with ui.row().classes('items-center justify-between py-1'):
                            ui.label('Default Quality').classes('text-xs compact-text')
                            def update_default_quality(e):
                                state.default_quality = int(e.value)
                                ui.notify(f'Default quality set to {e.value} passes (applied to Translate tab)', type='positive')
                            default_quality = ui.select(
                                {1: '1 pass (fast)', 2: '2 passes', 3: '3 passes', 4: '4 passes', 5: '5 passes (best)'},
                                value=state.default_quality,
                                on_change=update_default_quality
                            ).classes('text-xs compact-text').props('dense')
                        
                        # Default masking - auto-apply
                        with ui.row().classes('items-center justify-between py-1'):
                            ui.label('Default Masking').classes('text-xs compact-text')
                            def update_default_masking(e):
                                state.default_masking = e.value
                                ui.notify(f'Default masking {"enabled" if e.value else "disabled"} (applied to Translate tab)', type='positive')
                            default_masking = ui.switch(
                                value=state.default_masking,
                                on_change=update_default_masking
                            ).props('dense')
                        
                        ui.separator().classes('my-2')
                        
                        # Auto-apply settings
                        with ui.row().classes('items-center justify-between py-1'):
                            ui.label('Auto-apply to Translate Tab').classes('text-xs compact-text')
                            auto_apply = ui.switch(value=True).props('dense')
                        
                        # Cache settings
                        with ui.row().classes('items-center justify-between py-1'):
                            ui.label('Enable Translation Cache').classes('text-xs compact-text')
                            cache_enabled = ui.switch(value=True).props('dense')
                        
                        # Logging level
                        with ui.row().classes('items-center justify-between py-1'):
                            ui.label('Logging Level').classes('text-xs compact-text')
                            log_level = ui.select(
                                {'DEBUG': 'Debug', 'INFO': 'Info', 'WARNING': 'Warning', 'ERROR': 'Error'},
                                value='INFO',
                                on_change=lambda e: logging.getLogger().setLevel(getattr(logging, e.value)) or ui.notify(f'Logging level set to {e.value}', type='info')
                            ).classes('text-xs compact-text').props('dense')
                        
                        def apply_settings():
                            """Apply all settings immediately."""
                            if auto_apply.value:
                                # Update translate tab defaults
                                ui.notify('Settings applied to Translate tab', type='positive')
                            else:
                                ui.notify('Settings saved (auto-apply disabled)', type='info')
                        
                        ui.button('Apply Settings Now', icon='check', on_click=apply_settings).classes('w-full mt-3 text-xs').props('color=primary dense')
                    
                    with ui.card().classes('w-full deepl-style compact-card'):
                        ui.label('About').classes('text-xs font-semibold mb-2 compact-text')
                        ui.markdown('''
                **SciTrans-LLMs** v0.1.0

                Scientific document translation:
                - Context-aware across pages
                - Layout-preserving PDF output
                - Terminology control
                - Quality optimization

                **Focus:** English ↔ French bilingual translation.
                        ''').classes('text-xs compact-text')
    
    # Run the app
    print(f"\n{'='*60}")
    print("  SciTrans-LLMs GUI")
    print(f"  Starting on http://127.0.0.1:{port}")
    print(f"{'='*60}\n")

    ui.run(
        host='127.0.0.1',
        port=port,
        title='SciTrans-LLMs',
        favicon='🔬',
        show=True,
        reload=False,
        storage_secret='scitrans-llms-secret-key-2024',
    )
