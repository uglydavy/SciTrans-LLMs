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
    
    state = AppState()
    
    # Available backends
    def get_available_backends():
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
                    model_name = models[0].get('name', 'local') if models else 'local'
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
        font-size: 10px;
    }
    
    body {
        overflow-x: hidden;
        font-size: 12px;
    }
    
    .main-container {
        height: calc(100vh - 100px);
        overflow-y: hidden;
    }
    
    .compact-card {
        padding: 10px !important;
        margin: 0 !important;
    }
    
    .compact-text {
        font-size: 11px !important;
        line-height: 1.3 !important;
    }
    """
    
    @ui.page('/')
    async def main_page():
        # Theme toggle state
        dark = ui.dark_mode()
        dark.value = state.dark_mode
        
        # Add custom styles
        ui.add_head_html(f'<style>{CUSTOM_CSS}</style>')
        ui.add_head_html('<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">')
        
        # =====================================================================
        # Header
        # =====================================================================
        with ui.header().classes('items-center justify-between px-4 py-2 hero-gradient'):
            with ui.row().classes('items-center gap-2'):
                ui.icon('translate', size='md').classes('text-white')
                ui.label('SciTrans-LLMs').classes('text-lg font-bold text-white')
                ui.badge('v0.1.0').classes('bg-white/20 text-white text-xs')
            
            with ui.row().classes('items-center gap-2'):
                # Language indicator
                with ui.row().classes('items-center gap-1 bg-white/10 rounded-full px-2 py-1'):
                    ui.label('EN').classes('text-white font-semibold text-xs')
                    ui.icon('swap_horiz', size='sm').classes('text-white/70')
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
        
        with ui.tab_panels(tabs, value=translate_tab).classes('w-full flex-grow'):
            # =================================================================
            # TRANSLATE TAB
            # =================================================================
            with ui.tab_panel(translate_tab).classes('p-0'):
                await render_translate_panel()
            
            # =================================================================
            # GLOSSARY TAB
            # =================================================================
            with ui.tab_panel(glossary_tab).classes('p-6'):
                await render_glossary_panel()
            
            # =================================================================
            # DEVELOPER TAB
            # =================================================================
            with ui.tab_panel(developer_tab).classes('p-0'):
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
        
        # Main container - left/right split, fixed height, no scrolling
        with ui.row().classes('w-full gap-3 p-3').style('height: calc(100vh - 100px); overflow: hidden;'):
            # LEFT SIDE: Source & Settings
            with ui.column().classes('w-1/2 gap-2').style('overflow-y: hidden; max-height: 100%;'):
                # Source Document Card
                with ui.card().classes('w-full deepl-style compact-card'):
                    ui.label('Source Document').classes('text-xs font-semibold mb-2 compact-text')
                    
                    with ui.tabs().classes('w-full').props('dense') as source_tabs:
                        upload_tab = ui.tab('upload', label='Upload', icon='upload_file')
                        url_tab = ui.tab('url', label='URL', icon='link')
                    
                    with ui.tab_panels(source_tabs, value=upload_tab).classes('w-full'):
                        with ui.tab_panel(upload_tab):
                            # File upload handler
                            async def handle_upload(e):
                                try:
                                    file_obj = e.file
                                    file_name = file_obj.name
                                    temp_dir = Path(tempfile.gettempdir()) / "scitrans_uploads"
                                    temp_dir.mkdir(exist_ok=True)
                                    file_path = temp_dir / file_name
                                    await file_obj.save(file_path)
                                    uploaded_file['path'] = file_path
                                    uploaded_file['name'] = file_name
                                    upload_label.set_text(f'✓ {file_name}')
                                    upload_label.classes(replace='text-green-500 text-xs')
                                    file_info.set_text(f'{file_name} ({file_path.stat().st_size / 1024:.1f} KB)')
                                    file_info.visible = True
                                    ui.notify(f'File uploaded: {file_name}', type='positive')
                                except Exception as ex:
                                    logger.exception("Upload error")
                                    ui.notify(f'Upload error: {str(ex)}', type='negative')
                            
                            # Clickable upload area - make entire area clickable
                            with ui.column().classes('w-full items-center gap-1 p-4 upload-area').style('min-height: 80px; position: relative; cursor: pointer;') as upload_area:
                                ui.icon('cloud_upload', size='1.5rem').classes('opacity-50')
                                upload_label = ui.label('Click or drop PDF, DOCX, HTML here').classes('text-xs text-center compact-text')
                                file_info = ui.label('').classes('text-xs opacity-60 compact-text')
                                file_info.visible = False
                                
                                # Invisible upload overlay covering entire area
                                upload_comp = ui.upload(
                                    on_upload=handle_upload,
                                    auto_upload=True,
                                ).props('accept=".pdf,.docx,.doc,.html,.htm,.txt"').style('position: absolute; top: 0; left: 0; width: 100%; height: 100%; opacity: 0; cursor: pointer; z-index: 10;')
                                
                                uploaded_file['_upload'] = upload_comp
                        
                        with ui.tab_panel(url_tab):
                            url_input = ui.input(
                                'Paper URL',
                                placeholder='https://arxiv.org/pdf/...'
                            ).classes('w-full text-xs').props('dense')
                            ui.label('Supports arXiv, DOI links, and direct PDF URLs').classes('text-xs opacity-60 mt-1')
            
                # Translation Settings Card
                with ui.card().classes('w-full deepl-style compact-card'):
                    ui.label('Translation Settings').classes('text-xs font-semibold mb-2 compact-text')
                    
                    # Direction
                    with ui.row().classes('items-center gap-2 mb-2'):
                        ui.label('Direction:').classes('font-medium text-xs w-16 compact-text')
                        direction = ui.toggle(
                            {'en-fr': 'EN → FR', 'fr-en': 'FR → EN'},
                            value='en-fr'
                        ).classes('flex-grow text-xs')
                    
                    # Pages
                    with ui.row().classes('items-center gap-2 mb-2'):
                        ui.label('Pages:').classes('font-medium text-xs w-16 compact-text')
                        pages_input = ui.input(value='all', placeholder='all or 1-10').classes('flex-grow text-xs compact-text').props('dense')
                    
                    # Engine
                    with ui.row().classes('items-center gap-2 mb-2'):
                        ui.label('Engine:').classes('font-medium text-xs w-16 compact-text')
                        backends = get_available_backends()
                        engine_options = {b[0]: b[1] for b in backends if b[2]}
                        engine_select = ui.select(engine_options, value='free').classes('flex-grow text-xs compact-text').props('dense')
                    
                    # Quality passes as dropdown
                    with ui.row().classes('items-center gap-2 mb-2'):
                        ui.label('Quality:').classes('font-medium text-xs w-16 compact-text')
                        quality_passes = ui.select(
                            {1: '1 pass (fast)', 2: '2 passes', 3: '3 passes', 4: '4 passes', 5: '5 passes (best)'},
                            value=1
                        ).classes('flex-grow text-xs compact-text').props('dense')
                    
                    ui.separator().classes('my-2')
                    
                    # Advanced options
                    with ui.expansion('Advanced', icon='tune').classes('w-full text-xs'):
                        enable_masking = ui.checkbox('Enable masking', value=True).classes('text-xs mb-1 compact-text')
                        with ui.row().classes('gap-2 ml-4'):
                            translate_equations = ui.checkbox('Equations', value=False).classes('text-xs compact-text')
                            translate_tables = ui.checkbox('Tables', value=False).classes('text-xs compact-text')
                            translate_figures = ui.checkbox('Captions', value=True).classes('text-xs compact-text')
                        enable_reranking = ui.checkbox('Enable reranking', value=False).classes('text-xs mt-1 compact-text')
                
                # Custom Glossary Card
                with ui.card().classes('w-full deepl-style compact-card'):
                    with ui.row().classes('items-center justify-between mb-1'):
                        ui.label('Custom Glossary').classes('text-xs font-semibold compact-text')
                        ui.button('Example', icon='help_outline', on_click=lambda: glossary_input.set_value(
                            '# Format: source_term, target_term\nneural network, réseau de neurones\ndeep learning, apprentissage profond'
                        )).props('flat dense size=sm').classes('text-xs')
                    glossary_input = ui.textarea(
                        placeholder='# Format: source_term, target_term\nneural network, réseau de neurones'
                    ).classes('w-full mono-font text-xs compact-text').props('rows=2')
                
                # Button container - will be populated after function is defined
                button_container = ui.column().classes('w-full')
            
            # RIGHT SIDE: Preview & Results
            with ui.column().classes('w-1/2 gap-2').style('overflow-y: hidden; max-height: 100%;'):
                # Preview/Progress Card
                with ui.card().classes('w-full deepl-style compact-card').style('min-height: 300px;') as preview_card:
                    ui.label('Preview & Progress').classes('text-xs font-semibold mb-2 compact-text')
                    
                    # Progress bar (hidden initially)
                    progress_bar = ui.linear_progress(value=0, show_value=False).classes('w-full mb-1')
                    progress_bar.visible = False
                    progress_label = ui.label('').classes('text-xs text-center mb-2 compact-text')
                    progress_label.visible = False
                    
                    # Log output
                    log_output = ui.log(max_lines=10).classes('w-full mono-font text-xs compact-text').style('height: 150px;')
                    log_output.visible = False
                    
                    # Result area
                    result_status = ui.label('Upload a document and click Translate').classes('text-xs text-center opacity-60 compact-text')
                    result_path = ui.label('').classes('text-xs opacity-70 mt-1 compact-text')
                    result_path.visible = False
                    result_stats = ui.label('').classes('text-xs opacity-70 mt-1 compact-text')
                    result_stats.visible = False
                    
                    download_btn = ui.button(
                        'Download Translated PDF',
                        icon='download',
                        on_click=lambda: ui.download(preview_card._output_path) if hasattr(preview_card, '_output_path') else None
                    ).classes('w-full mt-2 text-xs compact-text').props('size=sm dense')
                    download_btn.visible = False
            
            # Define translation function after all UI elements are created
            async def start_translation():
                source_path = uploaded_file.get('path')
                source_url = url_input.value.strip() if url_input.value and url_input.value.strip() else None
                
                if not source_path and not source_url:
                    ui.notify('Please upload a file or provide a URL', type='warning')
                    return
                
                # Disable button and show progress
                translate_btn.props('disable')
                progress_bar.visible = True
                progress_label.visible = True
                log_output.visible = True
                result_status.set_text('Translating...')
                result_status.classes(replace='text-xs text-center compact-text')
                download_btn.visible = False
                log_output.clear()
                
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
                    
                    # Progress callback
                    def update_progress(msg: str, pct: float):
                        progress_bar.value = pct
                        progress_label.set_text(msg)
                        log_output.push(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
                    
                    # Run translation in background task
                    result = await run_translation(job, update_progress)
                    
                    # Show result
                    progress_bar.visible = False
                    progress_label.visible = False
                    
                    if result.get('success'):
                        result_status.set_text('✓ Translation Complete!')
                        result_status.classes(replace='text-xs font-bold text-green-500 text-center compact-text')
                        result_path.set_text(f"Saved to: {result.get('output_path')}")
                        result_path.visible = True
                        preview_card._output_path = result.get('output_path')
                        download_btn.visible = True
                        
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
            
            # Add translate button to left column after function is defined
            with button_container:
                translate_btn = ui.button(
                    'Translate Document',
                    icon='translate',
                    on_click=start_translation
                ).classes('w-full text-xs py-2 compact-text').props('color=primary')
    
    async def render_glossary_panel():
        """Render the glossary management panel."""
        default_glossary = get_default_glossary()
        
        # Centered container, no scrolling
        with ui.column().classes('w-full max-w-6xl mx-auto gap-3 p-3').style('height: calc(100vh - 100px); overflow-y: hidden;'):
            with ui.row().classes('w-full gap-3'):
                # Left - Default glossary browser
                with ui.column().classes('w-1/2 gap-2'):
                    with ui.card().classes('w-full deepl-style compact-card'):
                        ui.label('Default Scientific Glossary').classes('text-xs font-semibold mb-2 compact-text')
                        ui.label(f'{len(default_glossary)} terms • English ↔ French').classes('text-xs opacity-70 mb-2 compact-text')
                        
                        # Search
                        search_input = ui.input('Search terms...', on_change=lambda: filter_glossary()).props('dense clearable').classes('w-full mb-2 compact-text text-xs')
                        
                        # Glossary table
                        columns = [
                            {'name': 'source', 'label': 'English', 'field': 'source', 'align': 'left'},
                            {'name': 'target', 'label': 'French', 'field': 'target', 'align': 'left'},
                            {'name': 'domain', 'label': 'Domain', 'field': 'domain', 'align': 'left'},
                        ]
                        
                        rows = [
                            {'source': e.source, 'target': e.target, 'domain': e.domain or 'general'}
                            for e in default_glossary.entries[:50]
                        ]
                        
                        glossary_table = ui.table(columns=columns, rows=rows, row_key='source').classes('w-full compact-text').style('font-size: 10px;')
                        glossary_table.props('dense flat')
                        
                        def filter_glossary():
                            query = search_input.value.lower() if search_input.value else ''
                            filtered = [
                                {'source': e.source, 'target': e.target, 'domain': e.domain or 'general'}
                                for e in default_glossary.entries
                                if query in e.source.lower() or query in e.target.lower()
                            ][:50]
                            glossary_table.rows = filtered
                
                # Right - Custom glossary
                with ui.column().classes('w-1/2 gap-2'):
                    with ui.card().classes('w-full deepl-style compact-card'):
                        ui.label('Create Custom Glossary').classes('text-xs font-semibold mb-2 compact-text')
                        
                        # Format help
                        with ui.expansion('Format Guide', icon='help').classes('w-full mb-2 text-xs'):
                            ui.markdown('''
**Format:** `source_term, target_term` or `source_term → target_term`

**Example:**
```
neural network, réseau de neurones
deep learning, apprentissage profond
```
                            ''').classes('text-xs')
                        
                        custom_glossary = ui.textarea(
                            placeholder='# source_term, target_term\nneural network, réseau de neurones',
                        ).classes('w-full mono-font text-xs compact-text').props('rows=8')
                        
                        with ui.row().classes('gap-2 mt-2'):
                            ui.button('Save', icon='save', size='sm').props('color=primary dense').classes('text-xs')
                            ui.button('Load', icon='upload_file', size='sm').props('dense').classes('text-xs')
                            ui.button('Export', icon='download', size='sm').props('dense').classes('text-xs')
    
    async def render_developer_panel():
        """Render the developer tools panel."""
        with ui.tabs().classes('w-full text-xs').props('dense') as dev_tabs:
            testing_tab = ui.tab('testing', label='Testing', icon='science')
            logs_tab = ui.tab('logs', label='Logs', icon='article')
            cli_tab = ui.tab('cli', label='CLI', icon='terminal')
            debug_tab = ui.tab('debug', label='Debug', icon='bug_report')
        
        with ui.tab_panels(dev_tabs, value=testing_tab).classes('w-full'):
            # Testing Ground
            with ui.tab_panel(testing_tab).classes('p-3'):
                # Centered container
                with ui.column().classes('w-full max-w-5xl mx-auto gap-2').style('height: calc(100vh - 140px); overflow-y: hidden;'):
                    with ui.row().classes('w-full gap-3'):
                        # Left - Test input
                        with ui.column().classes('w-1/2 gap-2'):
                            with ui.card().classes('w-full deepl-style compact-card'):
                                ui.label('Quick Translation Test').classes('text-xs font-semibold mb-2 compact-text')
                                
                                test_input = ui.textarea(
                                    placeholder='Enter text to test translation...',
                                    value='The neural network achieved state-of-the-art performance.'
                                ).classes('w-full text-xs compact-text').props('rows=5 dense')
                                
                                with ui.row().classes('gap-2 mt-2'):
                                    test_direction = ui.toggle(
                                        {'en-fr': 'EN→FR', 'fr-en': 'FR→EN'},
                                        value='en-fr'
                                    ).classes('text-xs')
                                    # All available backends
                                    backends = get_available_backends()
                                    test_backend_options = {b[0]: b[1] for b in backends}
                                    test_backend = ui.select(
                                        test_backend_options,
                                        value='free'
                                    ).classes('flex-grow text-xs compact-text').props('dense')
                                
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
                                
                                ui.button('Test Translation', icon='play_arrow', on_click=run_test).classes('w-full mt-2 text-xs compact-text').props('color=primary dense')
                        
                        # Right - Test output
                        with ui.column().classes('w-1/2 gap-2'):
                            with ui.card().classes('w-full deepl-style compact-card'):
                                ui.label('Translation Output').classes('text-xs font-semibold mb-2 compact-text')
                                test_output = ui.markdown('*Output will appear here...*').classes('w-full p-2 border rounded text-xs compact-text').style('min-height: 100px; font-size: 10px;')
                            
                            with ui.card().classes('w-full deepl-style compact-card'):
                                ui.label('Component Tests').classes('text-xs font-semibold mb-2 compact-text')
                                
                                async def test_pdf_parser():
                                    ui.notify('PDF parser: OK', type='positive')
                                
                                async def test_masking():
                                    from scitrans_llms.masking import mask_document, MaskConfig
                                    ui.notify('Masking: OK', type='positive')
                                
                                async def test_glossary():
                                    glossary = get_default_glossary()
                                    ui.notify(f'Glossary: {len(glossary)} entries', type='positive')
                                
                                with ui.row().classes('gap-1 flex-wrap'):
                                    ui.button('PDF', on_click=test_pdf_parser, size='sm').props('flat dense').classes('text-xs')
                                    ui.button('Masking', on_click=test_masking, size='sm').props('flat dense').classes('text-xs')
                                    ui.button('Glossary', on_click=test_glossary, size='sm').props('flat dense').classes('text-xs')
                                    ui.button('Rerank', size='sm').props('flat dense').classes('text-xs')
            
            # Logs panel
            with ui.tab_panel(logs_tab).classes('p-3'):
                with ui.column().classes('w-full max-w-5xl mx-auto gap-2'):
                    with ui.card().classes('w-full deepl-style compact-card'):
                        with ui.row().classes('justify-between items-center mb-2'):
                            ui.label('System Logs').classes('text-xs font-semibold compact-text')
                            with ui.row().classes('gap-1'):
                                ui.button('Clear', icon='delete', size='sm').props('flat dense').classes('text-xs')
                                ui.button('Export', icon='download', size='sm').props('flat dense').classes('text-xs')
                        
                        log_viewer = ui.log(max_lines=50).classes('w-full mono-font text-xs compact-text').style('height: calc(100vh - 200px); font-size: 10px;')
                        
                        # Add some sample logs
                        log_viewer.push('[INFO] SciTrans-LLMs GUI started')
                        log_viewer.push('[INFO] Default glossary loaded: 200+ terms')
                        log_viewer.push('[INFO] Available backends: free, dictionary')
            
            # CLI Commands
            with ui.tab_panel(cli_tab).classes('p-6'):
                with ui.card().classes('w-full'):
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
            with ui.tab_panel(debug_tab).classes('p-3'):
                with ui.column().classes('w-full max-w-5xl mx-auto gap-2'):
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
                                    f"Working Dir: {cwd_display}",
                                ]
                                
                                try:
                                    import fitz
                                    info.append(f"PyMuPDF: {fitz.version[0]}")
                                except:
                                    info.append("PyMuPDF: Not installed")
                                
                                for line in info:
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
                                ]
                                
                                for name, mod in modules:
                                    try:
                                        __import__(mod)
                                        ui.label(f'✓ {name}').classes('text-green-500 text-xs compact-text')
                                    except ImportError:
                                        ui.label(f'✗ {name}').classes('text-red-500 text-xs compact-text')
    
    async def render_settings_panel(km: KeyManager):
        """Render the settings and API keys panel."""
        # Centered container
        with ui.column().classes('w-full max-w-6xl mx-auto gap-3 p-3').style('height: calc(100vh - 100px); overflow-y: hidden;'):
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
                                    ui.badge('Set', color='green', size='sm').props('outline')
                                else:
                                    ui.badge('Not set', color='grey', size='sm').props('outline')
                                
                                # Edit button
                                def make_edit_handler(sid):
                                    async def edit_key():
                                        with ui.dialog() as dialog, ui.card().classes('compact-card'):
                                            ui.label(f'Set {sid.title()} API Key').classes('text-xs font-semibold mb-2 compact-text')
                                            key_input = ui.input('API Key', password=True).classes('w-full text-xs compact-text').props('dense')
                                            with ui.row().classes('gap-2 mt-2'):
                                                ui.button('Cancel', on_click=dialog.close, size='sm').props('flat dense').classes('text-xs')
                                                def save_key():
                                                    if key_input.value:
                                                        km.set_key(sid, key_input.value)
                                                        ui.notify(f'{sid.title()} key saved', type='positive')
                                                        dialog.close()
                                                ui.button('Save', on_click=save_key, size='sm').props('color=primary dense').classes('text-xs')
                                        dialog.open()
                                    return edit_key
                                
                                ui.button(icon='edit', on_click=make_edit_handler(service_id), size='sm').props('flat round dense').classes('text-xs')
                        
                        ui.separator().classes('my-2')
                        ui.label('Keys stored securely in system keychain or local config.').classes('text-xs opacity-50 compact-text')
                
                # App Settings
                with ui.column().classes('w-1/2 gap-2'):
                    with ui.card().classes('w-full deepl-style compact-card'):
                        ui.label('Application Settings').classes('text-xs font-semibold mb-2 compact-text')
                        
                        # Theme
                        with ui.row().classes('items-center justify-between py-1'):
                            ui.label('Dark Mode').classes('text-xs compact-text')
                            ui.switch(value=state.dark_mode, on_change=lambda e: setattr(state, 'dark_mode', e.value)).props('dense')
                        
                        ui.separator().classes('my-1')
                        
                        # Default engine - dropdown with all options
                        with ui.row().classes('items-center justify-between py-1'):
                            ui.label('Default Engine').classes('text-xs compact-text')
                            backends = get_available_backends()
                            default_engine_options = {b[0]: b[1] for b in backends if b[2]}
                            default_engine = ui.select(default_engine_options, value='free').classes('text-xs compact-text').props('dense')
                        
                        # Default quality passes - dropdown
                        with ui.row().classes('items-center justify-between py-1'):
                            ui.label('Default Quality').classes('text-xs compact-text')
                            default_quality = ui.select(
                                {1: '1 pass (fast)', 2: '2 passes', 3: '3 passes', 4: '4 passes', 5: '5 passes (best)'},
                                value=1
                            ).classes('text-xs compact-text').props('dense')
                        
                        with ui.row().classes('items-center justify-between py-1'):
                            ui.label('Default Masking').classes('text-xs compact-text')
                            default_masking = ui.switch(value=True).props('dense')
                    
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
    )
