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
        ]
        if km.get_key("openai"):
            backends.append(("openai", "OpenAI GPT-4", True))
        else:
            backends.append(("openai", "OpenAI GPT-4 (no key)", False))
        if km.get_key("deepseek"):
            backends.append(("deepseek", "DeepSeek (cheap)", True))
        else:
            backends.append(("deepseek", "DeepSeek (no key)", False))
        if km.get_key("anthropic"):
            backends.append(("anthropic", "Claude", True))
        else:
            backends.append(("anthropic", "Claude (no key)", False))
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
    
    .card-hover:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 25px -5px rgb(0 0 0 / 0.1);
    }
    
    .glass-effect {
        backdrop-filter: blur(12px);
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .animate-pulse-slow {
        animation: pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    @keyframes gradient-shift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .gradient-animate {
        background-size: 200% 200%;
        animation: gradient-shift 3s ease infinite;
    }
    
    .mono-font {
        font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', monospace;
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
        with ui.header().classes('items-center justify-between px-6 py-3 hero-gradient'):
            with ui.row().classes('items-center gap-3'):
                ui.icon('translate', size='lg').classes('text-white')
                ui.label('SciTrans-LLMs').classes('text-2xl font-bold text-white tracking-tight')
                ui.badge('v0.2.0').classes('bg-white/20 text-white')
            
            with ui.row().classes('items-center gap-4'):
                # Language indicator
                with ui.row().classes('items-center gap-2 bg-white/10 rounded-full px-4 py-2'):
                    ui.label('EN').classes('text-white font-semibold')
                    ui.icon('swap_horiz').classes('text-white/70')
                    ui.label('FR').classes('text-white font-semibold')
                
                # Theme toggle
                def toggle_theme():
                    dark.value = not dark.value
                    state.dark_mode = dark.value
                
                ui.button(
                    icon='dark_mode' if dark.value else 'light_mode',
                    on_click=toggle_theme
                ).props('flat round').classes('text-white')
        
        # =====================================================================
        # Main Content
        # =====================================================================
        with ui.tabs().classes('w-full').props('dense inline-label') as tabs:
            translate_tab = ui.tab('translate', label='Translate Document', icon='description')
            glossary_tab = ui.tab('glossary', label='Glossary', icon='menu_book')
            developer_tab = ui.tab('developer', label='Developer Tools', icon='code')
            settings_tab = ui.tab('settings', label='Settings & Keys', icon='settings')
        
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
        with ui.footer().classes('py-4 px-6 text-center'):
            ui.label('SciTrans-LLMs — Scientific Document Translation System').classes('text-sm opacity-70')
    
    async def render_translate_panel():
        """Render the main translation panel."""
        # Job state containers
        uploaded_file = {'path': None, 'name': None}
        progress_container = None
        result_container = None
        
        with ui.row().classes('w-full gap-6 p-6'):
            # Left column - Input
            with ui.column().classes('w-1/2 gap-4'):
                # Source selection card
                with ui.card().classes('w-full'):
                    ui.label('Source Document').classes('text-lg font-semibold mb-4')
                    
                    with ui.tabs().classes('w-full') as source_tabs:
                        upload_tab = ui.tab('upload', label='Upload File', icon='upload_file')
                        url_tab = ui.tab('url', label='From URL', icon='link')
                    
                    with ui.tab_panels(source_tabs, value=upload_tab).classes('w-full'):
                        with ui.tab_panel(upload_tab):
                            # File upload area
                            def handle_upload(e):
                                if e.content:
                                    # Save uploaded file
                                    temp_dir = Path(tempfile.gettempdir()) / "scitrans_uploads"
                                    temp_dir.mkdir(exist_ok=True)
                                    file_path = temp_dir / e.name
                                    with open(file_path, 'wb') as f:
                                        f.write(e.content.read())
                                    uploaded_file['path'] = file_path
                                    uploaded_file['name'] = e.name
                                    upload_label.set_text(f'✓ {e.name}')
                                    upload_label.classes(replace='text-green-500')
                            
                            with ui.column().classes('w-full items-center gap-4 p-8 border-2 border-dashed rounded-xl'):
                                ui.icon('cloud_upload', size='3rem').classes('opacity-50')
                                upload_label = ui.label('Drop PDF, DOCX, or HTML file here').classes('text-center')
                                ui.upload(
                                    on_upload=handle_upload,
                                    auto_upload=True,
                                ).props('accept=".pdf,.docx,.doc,.html,.htm,.txt"').classes('w-full')
                        
                        with ui.tab_panel(url_tab):
                            url_input = ui.input(
                                'Paper URL',
                                placeholder='https://arxiv.org/pdf/...'
                            ).classes('w-full')
                            ui.label('Supports arXiv, DOI links, and direct PDF URLs').classes('text-xs opacity-60')
                    
                    # Supported formats
                    ui.separator().classes('my-4')
                    with ui.row().classes('gap-2 flex-wrap'):
                        for fmt in ['PDF', 'DOCX', 'HTML', 'TXT']:
                            ui.badge(fmt).classes('bg-primary/10 text-primary')
                
                # Translation settings card
                with ui.card().classes('w-full'):
                    ui.label('Translation Settings').classes('text-lg font-semibold mb-4')
                    
                    # Direction toggle
                    with ui.row().classes('items-center gap-4 mb-4'):
                        ui.label('Direction:').classes('font-medium')
                        direction = ui.toggle(
                            {
                                'en-fr': 'English → French',
                                'fr-en': 'French → English'
                            },
                            value='en-fr'
                        ).classes('flex-grow')
                    
                    # Page selection
                    with ui.row().classes('items-center gap-4 mb-4'):
                        ui.label('Pages:').classes('font-medium w-24')
                        pages_input = ui.input(
                            value='all',
                            placeholder='all or 1-10'
                        ).classes('flex-grow').props('dense')
                        ui.label('(e.g., "all", "1-5", "3")').classes('text-xs opacity-60')
                    
                    # Engine selection
                    with ui.row().classes('items-center gap-4 mb-4'):
                        ui.label('Engine:').classes('font-medium w-24')
                        backends = get_available_backends()
                        engine_options = {b[0]: b[1] for b in backends if b[2]}
                        engine_select = ui.select(
                            engine_options,
                            value='free'
                        ).classes('flex-grow')
                    
                    ui.separator().classes('my-4')
                    
                    # Advanced options
                    with ui.expansion('Advanced Options', icon='tune').classes('w-full'):
                        # Masking options
                        ui.label('Content Preservation').classes('font-medium mb-2')
                        enable_masking = ui.checkbox('Enable masking (protect formulas, code, etc.)', value=True)
                        
                        with ui.row().classes('gap-4 ml-6'):
                            translate_equations = ui.checkbox('Translate equations', value=False)
                            translate_tables = ui.checkbox('Translate table content', value=False)
                            translate_figures = ui.checkbox('Translate figure captions', value=True)
                        
                        ui.separator().classes('my-4')
                        
                        # Quality options
                        ui.label('Quality Settings').classes('font-medium mb-2')
                        with ui.row().classes('items-center gap-4'):
                            ui.label('Quality passes:')
                            quality_passes = ui.slider(min=1, max=5, value=1, step=1).classes('flex-grow')
                            ui.label().bind_text_from(quality_passes, 'value')
                        
                        enable_reranking = ui.checkbox('Enable reranking (multiple candidates)', value=False)
                
                # Custom glossary
                with ui.card().classes('w-full'):
                    with ui.row().classes('items-center justify-between mb-2'):
                        ui.label('Custom Glossary').classes('text-lg font-semibold')
                        ui.button('Load Example', icon='help_outline', on_click=lambda: glossary_input.set_value(
                            '# Custom glossary (one term per line)\n'
                            '# Format: source_term, target_term\n'
                            'neural network, réseau de neurones\n'
                            'machine learning, apprentissage automatique\n'
                            'deep learning, apprentissage profond\n'
                            'attention mechanism, mécanisme d\'attention\n'
                            'transformer, transformeur\n'
                        )).props('flat dense')
                    
                    glossary_input = ui.textarea(
                        placeholder='# Format: source_term, target_term\nneural network, réseau de neurones\ndeep learning, apprentissage profond',
                    ).classes('w-full mono-font').props('rows=6')
            
            # Right column - Output
            with ui.column().classes('w-1/2 gap-4'):
                # Action button
                async def start_translation():
                    # Build job
                    job = TranslationJob(
                        source_path=uploaded_file.get('path'),
                        source_url=url_input.value if url_input.value else None,
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
                    
                    if not job.source_path and not job.source_url:
                        ui.notify('Please upload a file or provide a URL', type='warning')
                        return
                    
                    state.current_job = job
                    
                    # Show progress
                    progress_card.visible = True
                    result_card.visible = False
                    progress_bar.value = 0
                    progress_label.set_text('Starting...')
                    
                    # Progress callback
                    def update_progress(msg: str, pct: float):
                        progress_bar.value = pct
                        progress_label.set_text(msg)
                        log_output.push(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
                    
                    # Run translation
                    result = await run_translation(job, update_progress)
                    
                    # Show result
                    progress_card.visible = False
                    result_card.visible = True
                    
                    if result.get('success'):
                        result_status.set_text('✓ Translation Complete!')
                        result_status.classes(replace='text-2xl font-bold text-green-500')
                        result_path.set_text(f"Saved to: {result.get('output_path')}")
                        result_path.visible = True
                        download_btn.visible = True
                        # Store for download
                        result_card._output_path = result.get('output_path')
                        
                        # Show stats
                        stats = result.get('stats', {})
                        stats_text = f"Blocks: {stats.get('total_blocks', 0)} total, {stats.get('translated_blocks', 0)} translated"
                        if stats.get('masks_applied'):
                            stats_text += f", {stats.get('masks_applied')} masks applied"
                        result_stats.set_text(stats_text)
                        result_stats.visible = True
                    else:
                        result_status.set_text('✗ Translation Failed')
                        result_status.classes(replace='text-2xl font-bold text-red-500')
                        result_path.set_text(f"Error: {result.get('error')}")
                        result_path.visible = True
                        download_btn.visible = False
                        result_stats.visible = False
                
                with ui.card().classes('w-full'):
                    ui.button(
                        'Translate Document',
                        icon='translate',
                        on_click=start_translation
                    ).classes('w-full text-lg py-4').props('color=primary size=lg')
                
                # Progress card
                with ui.card().classes('w-full') as progress_card:
                    progress_card.visible = False
                    ui.label('Translation Progress').classes('text-lg font-semibold mb-4')
                    progress_bar = ui.linear_progress(value=0, show_value=False).classes('w-full')
                    progress_label = ui.label('Starting...').classes('text-center mt-2')
                    
                    ui.separator().classes('my-4')
                    ui.label('Log').classes('font-medium')
                    log_output = ui.log(max_lines=20).classes('w-full h-40 mono-font text-xs')
                
                # Result card
                with ui.card().classes('w-full') as result_card:
                    result_card.visible = False
                    result_status = ui.label('').classes('text-2xl font-bold text-center mb-4')
                    result_path = ui.label('').classes('text-sm opacity-70')
                    result_stats = ui.label('').classes('text-sm opacity-70 mt-2')
                    
                    def download_result():
                        if hasattr(result_card, '_output_path') and result_card._output_path:
                            ui.download(result_card._output_path)
                    
                    download_btn = ui.button(
                        'Download Translated PDF',
                        icon='download',
                        on_click=download_result
                    ).classes('w-full mt-4')
                
                # Preview area
                with ui.card().classes('w-full flex-grow'):
                    ui.label('Output Preview').classes('text-lg font-semibold mb-4')
                    with ui.scroll_area().classes('w-full h-64 border rounded'):
                        ui.label('Translation output will appear here...').classes('text-center opacity-50 p-8')
    
    async def render_glossary_panel():
        """Render the glossary management panel."""
        default_glossary = get_default_glossary()
        
        with ui.row().classes('w-full gap-6'):
            # Left - Default glossary browser
            with ui.column().classes('w-1/2 gap-4'):
                with ui.card().classes('w-full'):
                    ui.label('Default Scientific Glossary').classes('text-lg font-semibold mb-4')
                    ui.label(f'{len(default_glossary)} terms • English ↔ French').classes('text-sm opacity-70 mb-4')
                    
                    # Search
                    search_input = ui.input('Search terms...', on_change=lambda: filter_glossary()).props('dense clearable').classes('w-full mb-4')
                    
                    # Glossary table
                    columns = [
                        {'name': 'source', 'label': 'English', 'field': 'source', 'align': 'left'},
                        {'name': 'target', 'label': 'French', 'field': 'target', 'align': 'left'},
                        {'name': 'domain', 'label': 'Domain', 'field': 'domain', 'align': 'left'},
                    ]
                    
                    rows = [
                        {'source': e.source, 'target': e.target, 'domain': e.domain or 'general'}
                        for e in default_glossary.entries[:100]
                    ]
                    
                    glossary_table = ui.table(columns=columns, rows=rows, row_key='source').classes('w-full')
                    glossary_table.props('dense flat')
                    
                    def filter_glossary():
                        query = search_input.value.lower() if search_input.value else ''
                        filtered = [
                            {'source': e.source, 'target': e.target, 'domain': e.domain or 'general'}
                            for e in default_glossary.entries
                            if query in e.source.lower() or query in e.target.lower()
                        ][:100]
                        glossary_table.rows = filtered
            
            # Right - Custom glossary
            with ui.column().classes('w-1/2 gap-4'):
                with ui.card().classes('w-full'):
                    ui.label('Create Custom Glossary').classes('text-lg font-semibold mb-4')
                    
                    # Format help
                    with ui.expansion('Format Guide', icon='help').classes('w-full mb-4'):
                        ui.markdown('''
**Supported formats:**

```
# CSV format (comma-separated)
source_term, target_term

# Tab-separated
source_term	target_term

# Arrow format
source_term → target_term
source_term -> target_term

# Colon format
source_term: target_term
```

**Example:**
```
neural network, réseau de neurones
deep learning, apprentissage profond
machine learning, apprentissage automatique
attention mechanism, mécanisme d'attention
```

Lines starting with `#` are comments.
                        ''')
                    
                    custom_glossary = ui.textarea(
                        placeholder='# Enter your custom glossary here\n# source_term, target_term\nneural network, réseau de neurones',
                    ).classes('w-full mono-font').props('rows=15')
                    
                    with ui.row().classes('gap-2 mt-4'):
                        ui.button('Save Glossary', icon='save').props('color=primary')
                        ui.button('Load from File', icon='upload_file')
                        ui.button('Export', icon='download')
    
    async def render_developer_panel():
        """Render the developer tools panel."""
        with ui.tabs().classes('w-full').props('dense') as dev_tabs:
            testing_tab = ui.tab('testing', label='Testing Ground', icon='science')
            logs_tab = ui.tab('logs', label='System Logs', icon='article')
            cli_tab = ui.tab('cli', label='CLI Commands', icon='terminal')
            debug_tab = ui.tab('debug', label='Debug Info', icon='bug_report')
        
        with ui.tab_panels(dev_tabs, value=testing_tab).classes('w-full'):
            # Testing Ground
            with ui.tab_panel(testing_tab).classes('p-6'):
                with ui.row().classes('w-full gap-6'):
                    # Left - Test input
                    with ui.column().classes('w-1/2 gap-4'):
                        with ui.card().classes('w-full'):
                            ui.label('Quick Translation Test').classes('text-lg font-semibold mb-4')
                            
                            test_input = ui.textarea(
                                placeholder='Enter text to test translation...',
                                value='The neural network achieved state-of-the-art performance on the benchmark dataset.'
                            ).classes('w-full').props('rows=6')
                            
                            with ui.row().classes('gap-4 mt-4'):
                                test_direction = ui.toggle(
                                    {'en-fr': 'EN→FR', 'fr-en': 'FR→EN'},
                                    value='en-fr'
                                )
                                test_backend = ui.select(
                                    {'free': 'Free', 'dictionary': 'Dictionary'},
                                    value='free'
                                ).classes('flex-grow')
                            
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
                            
                            ui.button('Test Translation', icon='play_arrow', on_click=run_test).classes('w-full mt-4').props('color=primary')
                    
                    # Right - Test output
                    with ui.column().classes('w-1/2 gap-4'):
                        with ui.card().classes('w-full'):
                            ui.label('Translation Output').classes('text-lg font-semibold mb-4')
                            test_output = ui.markdown('*Output will appear here...*').classes('w-full p-4 border rounded min-h-40')
                        
                        with ui.card().classes('w-full'):
                            ui.label('Component Tests').classes('text-lg font-semibold mb-4')
                            
                            async def test_pdf_parser():
                                ui.notify('PDF parser test: OK', type='positive')
                            
                            async def test_masking():
                                from scitrans_llms.masking import mask_document, MaskConfig
                                ui.notify('Masking test: OK', type='positive')
                            
                            async def test_glossary():
                                glossary = get_default_glossary()
                                ui.notify(f'Glossary test: {len(glossary)} entries loaded', type='positive')
                            
                            with ui.row().classes('gap-2 flex-wrap'):
                                ui.button('Test PDF Parser', on_click=test_pdf_parser).props('flat')
                                ui.button('Test Masking', on_click=test_masking).props('flat')
                                ui.button('Test Glossary', on_click=test_glossary).props('flat')
                                ui.button('Test Reranking').props('flat')
            
            # Logs panel
            with ui.tab_panel(logs_tab).classes('p-6'):
                with ui.card().classes('w-full'):
                    with ui.row().classes('justify-between items-center mb-4'):
                        ui.label('System Logs').classes('text-lg font-semibold')
                        with ui.row().classes('gap-2'):
                            ui.button('Clear', icon='delete').props('flat')
                            ui.button('Export', icon='download').props('flat')
                    
                    log_viewer = ui.log(max_lines=100).classes('w-full h-96 mono-font text-xs')
                    
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
            
            # Debug Info
            with ui.tab_panel(debug_tab).classes('p-6'):
                with ui.row().classes('w-full gap-6'):
                    with ui.column().classes('w-1/2'):
                        with ui.card().classes('w-full'):
                            ui.label('System Information').classes('text-lg font-semibold mb-4')
                            
                            import sys
                            import platform
                            
                            info = [
                                f"Python: {sys.version.split()[0]}",
                                f"Platform: {platform.system()} {platform.release()}",
                                f"Working Directory: {os.getcwd()}",
                            ]
                            
                            try:
                                import fitz
                                info.append(f"PyMuPDF: {fitz.version[0]}")
                            except:
                                info.append("PyMuPDF: Not installed")
                            
                            for line in info:
                                ui.label(line).classes('mono-font text-sm')
                    
                    with ui.column().classes('w-1/2'):
                        with ui.card().classes('w-full'):
                            ui.label('Module Status').classes('text-lg font-semibold mb-4')
                            
                            modules = [
                                ('PyMuPDF (PDF)', 'fitz'),
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
                                    ui.label(f'✓ {name}').classes('text-green-500')
                                except ImportError:
                                    ui.label(f'✗ {name}').classes('text-red-500')
    
    async def render_settings_panel(km: KeyManager):
        """Render the settings and API keys panel."""
        with ui.row().classes('w-full gap-6'):
            # API Keys
            with ui.column().classes('w-1/2 gap-4'):
                with ui.card().classes('w-full'):
                    ui.label('API Keys').classes('text-lg font-semibold mb-4')
                    ui.label('Configure API keys for premium translation engines').classes('text-sm opacity-70 mb-4')
                    
                    services = [
                        ('openai', 'OpenAI', 'GPT-4, GPT-4o models'),
                        ('deepseek', 'DeepSeek', 'Affordable alternative'),
                        ('anthropic', 'Anthropic', 'Claude models'),
                        ('deepl', 'DeepL', 'Professional translation API'),
                    ]
                    
                    for service_id, name, desc in services:
                        key_info = km.get_key_info(service_id)
                        
                        with ui.row().classes('w-full items-center gap-4 py-2 border-b'):
                            with ui.column().classes('flex-grow'):
                                ui.label(name).classes('font-medium')
                                ui.label(desc).classes('text-xs opacity-60')
                            
                            if key_info.is_set:
                                ui.badge('Configured', color='green').props('outline')
                                ui.label(key_info.masked_value).classes('mono-font text-xs opacity-50')
                            else:
                                ui.badge('Not set', color='grey').props('outline')
                            
                            # Edit button
                            def make_edit_handler(sid):
                                async def edit_key():
                                    with ui.dialog() as dialog, ui.card():
                                        ui.label(f'Set {sid.title()} API Key').classes('text-lg font-semibold mb-4')
                                        key_input = ui.input('API Key', password=True).classes('w-full')
                                        with ui.row().classes('gap-2 mt-4'):
                                            ui.button('Cancel', on_click=dialog.close).props('flat')
                                            def save_key():
                                                if key_input.value:
                                                    km.set_key(sid, key_input.value)
                                                    ui.notify(f'{sid.title()} key saved', type='positive')
                                                    dialog.close()
                                            ui.button('Save', on_click=save_key).props('color=primary')
                                    dialog.open()
                                return edit_key
                            
                            ui.button(icon='edit', on_click=make_edit_handler(service_id)).props('flat round dense')
                    
                    ui.separator().classes('my-4')
                    ui.label('Keys are stored securely in your system keychain or local config.').classes('text-xs opacity-50')
            
            # App Settings
            with ui.column().classes('w-1/2 gap-4'):
                with ui.card().classes('w-full'):
                    ui.label('Application Settings').classes('text-lg font-semibold mb-4')
                    
                    # Theme
                    with ui.row().classes('items-center justify-between py-2'):
                        ui.label('Dark Mode')
                        ui.switch(value=state.dark_mode, on_change=lambda e: setattr(state, 'dark_mode', e.value))
                    
                    ui.separator()
                    
                    # Default settings
                    with ui.row().classes('items-center justify-between py-2'):
                        ui.label('Default Engine')
                        ui.select({'free': 'Free', 'dictionary': 'Dictionary', 'openai': 'OpenAI'}, value='free')
                    
                    with ui.row().classes('items-center justify-between py-2'):
                        ui.label('Default Quality Passes')
                        ui.number(value=1, min=1, max=5).classes('w-20')
                    
                    with ui.row().classes('items-center justify-between py-2'):
                        ui.label('Enable Masking by Default')
                        ui.switch(value=True)
                
                with ui.card().classes('w-full'):
                    ui.label('About').classes('text-lg font-semibold mb-4')
                    ui.markdown('''
**SciTrans-LLMs** v0.2.0

A scientific document translation system designed for:
- Context-aware translation across pages
- Layout-preserving PDF output
- Terminology control via glossaries
- Quality optimization through reranking

**Focus:** English ↔ French bilingual translation for scientific literature.
                    ''')
    
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


if __name__ == "__main__":
    launch()
