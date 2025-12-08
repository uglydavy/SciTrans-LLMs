"""SciTrans-LLMs GUI - Scientific Document Translation Interface

A clean, user-friendly interface for translating scientific PDFs.
Built with NiceGUI for a modern responsive experience.
"""

from __future__ import annotations
import base64
import tempfile
import asyncio
import io
from datetime import datetime
from pathlib import Path
from typing import Optional

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scitrans-gui")


def launch(port: int = 7860, share: bool = False):
    """Launch the SciTrans-LLMs GUI."""
    from nicegui import ui, app
    from scitrans_llms.keys import KeyManager
    from scitrans_llms.pipeline import translate_document
    from scitrans_llms.translate.glossary import get_default_glossary
    from scitrans_llms.models import Document
    
    km = KeyManager()
    
    def get_engines():
        """Get available translation engines based on API keys."""
        engines = ['dictionary', 'free']
        if km.get_key('openai'):
            engines.append('openai')
        if km.get_key('deepseek'):
            engines.append('deepseek')
        if km.get_key('anthropic'):
            engines.append('anthropic')
        return engines
    
    def render_pdf_preview(pdf_path: str, page: int = 0) -> str:
        """Render a PDF page as base64 image HTML."""
        try:
            import fitz
            doc = fitz.open(pdf_path)
            if page < len(doc):
                # Higher resolution for better quality
                pix = doc[page].get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                b64 = base64.b64encode(pix.tobytes("png")).decode()
                doc.close()
                return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;max-height:100%;object-fit:contain;"/>'
            doc.close()
        except Exception as e:
            logger.error(f"Preview error: {e}")
        return '<div class="placeholder">No preview available</div>'
    
    def get_pdf_page_count(pdf_path: str) -> int:
        """Get total pages in PDF."""
        try:
            import fitz
            doc = fitz.open(pdf_path)
            count = len(doc)
            doc.close()
            return count
        except:
            return 0

    # Clean CSS
    CSS = """
    <style>
    :root {
        --primary: #4f46e5;
        --primary-hover: #4338ca;
        --bg-dark: #0f172a;
        --bg-card: #1e293b;
        --border: #334155;
        --text: #f1f5f9;
        --text-muted: #94a3b8;
    }
    
    html, body {
        margin: 0;
        padding: 0;
        background: var(--bg-dark);
        color: var(--text);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .app-header {
        background: linear-gradient(135deg, var(--primary), #7c3aed);
        padding: 16px 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .app-title {
        font-size: 20px;
        font-weight: 700;
        color: white;
    }
    
    .app-subtitle {
        font-size: 13px;
        color: rgba(255,255,255,0.8);
    }
    
    .main-content {
        display: grid;
        grid-template-columns: 400px 1fr;
        gap: 20px;
        padding: 20px;
        height: calc(100vh - 80px);
        box-sizing: border-box;
    }
    
    .sidebar {
        display: flex;
        flex-direction: column;
        gap: 16px;
        overflow-y: auto;
    }
    
    .card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 20px;
    }
    
    .card-title {
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 16px;
        color: var(--text);
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .card-title::before {
        content: '';
        width: 4px;
        height: 16px;
        background: var(--primary);
        border-radius: 2px;
    }
    
    .preview-container {
        display: flex;
        flex-direction: column;
        gap: 16px;
        height: 100%;
    }
    
    .preview-tabs {
        display: flex;
        gap: 8px;
    }
    
    .preview-tab {
        padding: 10px 20px;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.2s;
        color: var(--text-muted);
    }
    
    .preview-tab.active {
        background: var(--primary);
        border-color: var(--primary);
        color: white;
    }
    
    .preview-area {
        flex: 1;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        min-height: 400px;
    }
    
    .preview-area img {
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
    }
    
    .placeholder {
        color: var(--text-muted);
        font-size: 14px;
        text-align: center;
        padding: 40px;
    }
    
    .page-nav {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        padding: 12px;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
    }
    
    .btn-primary {
        background: var(--primary) !important;
        color: white !important;
    }
    
    .btn-primary:hover {
        background: var(--primary-hover) !important;
    }
    
    .upload-area {
        border: 2px dashed var(--border);
        border-radius: 8px;
        padding: 24px;
        text-align: center;
        transition: all 0.2s;
        cursor: pointer;
    }
    
    .upload-area:hover {
        border-color: var(--primary);
        background: rgba(79, 70, 229, 0.1);
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 500;
    }
    
    .status-success {
        background: rgba(34, 197, 94, 0.2);
        color: #22c55e;
    }
    
    .status-pending {
        background: rgba(234, 179, 8, 0.2);
        color: #eab308;
    }
    
    .progress-container {
        margin-top: 16px;
    }
    
    .log-output {
        background: #0f172a;
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 12px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: var(--text-muted);
        max-height: 150px;
        overflow-y: auto;
    }
    
    /* Light mode */
    body.body--light {
        --bg-dark: #f8fafc;
        --bg-card: #ffffff;
        --border: #e2e8f0;
        --text: #1e293b;
        --text-muted: #64748b;
    }
    
    body.body--light .app-header {
        background: linear-gradient(135deg, var(--primary), #7c3aed);
    }
    </style>
    """

    @ui.page('/')
    async def main_page():
        ui.add_head_html(CSS)
        
        # State
        state = {
            'source_path': None,
            'source_name': None,
            'translated_path': None,
            'source_page': 0,
            'trans_page': 0,
            'logs': [],
            'is_translating': False,
        }
        
        # Storage for session persistence
        storage = app.storage.user
        if storage.get('source_path') and Path(storage.get('source_path', '')).exists():
            state['source_path'] = storage.get('source_path')
            state['source_name'] = storage.get('source_name')
        
        def log(msg: str):
            ts = datetime.now().strftime("%H:%M:%S")
            state['logs'].append(f"[{ts}] {msg}")
            if len(state['logs']) > 50:
                state['logs'] = state['logs'][-50:]
            try:
                log_area.value = '\n'.join(state['logs'][-10:])
            except:
                pass
            logger.info(msg)
        
        # Header
        with ui.element('div').classes('app-header'):
            with ui.column().classes('gap-0'):
                ui.label('SciTrans-LLMs').classes('app-title')
                ui.label('Scientific Document Translation').classes('app-subtitle')
            with ui.row().classes('gap-2'):
                dark = ui.dark_mode()
                dark.enable()
                ui.button(icon='dark_mode', on_click=lambda: dark.toggle()).props('flat round color=white')
        
        # Main content
        with ui.element('div').classes('main-content'):
            
            # LEFT SIDEBAR
            with ui.element('div').classes('sidebar'):
                
                # Upload Card
                with ui.element('div').classes('card'):
                    ui.label('📄 Source Document').classes('card-title')
                    
                    file_status = ui.label('No file selected').classes('text-sm opacity-70 mb-3')
                    if state['source_name']:
                        file_status.text = f"✓ {state['source_name']}"
                    
                    async def handle_upload(e):
                        """Handle file upload from NiceGUI."""
                        try:
                            log("Processing upload...")
                            
                            # NiceGUI upload provides different attributes based on version
                            content = None
                            filename = 'document.pdf'
                            
                            # Try to get file content
                            if hasattr(e, 'content'):
                                content_obj = e.content
                                if hasattr(content_obj, 'read'):
                                    # It's a file-like object
                                    content = content_obj.read()
                                elif isinstance(content_obj, bytes):
                                    content = content_obj
                                else:
                                    # Try reading as stream
                                    try:
                                        content = await content_obj.read()
                                    except:
                                        content = bytes(content_obj) if content_obj else None
                            
                            if hasattr(e, 'name') and e.name:
                                filename = e.name
                            
                            if not content:
                                ui.notify('Could not read file content', type='warning')
                                log("Upload failed: empty content")
                                return
                            
                            # Validate PDF
                            if not content[:5].startswith(b'%PDF'):
                                ui.notify('Invalid PDF file', type='warning')
                                log("Upload failed: not a PDF")
                                return
                            
                            # Save to temp file
                            tmp_dir = Path(tempfile.mkdtemp())
                            tmp_path = tmp_dir / filename
                            tmp_path.write_bytes(content)
                            
                            state['source_path'] = str(tmp_path)
                            state['source_name'] = filename
                            state['translated_path'] = None
                            state['source_page'] = 0
                            
                            # Persist to storage
                            storage['source_path'] = str(tmp_path)
                            storage['source_name'] = filename
                            
                            file_status.text = f"✓ {filename}"
                            source_preview.set_content(render_pdf_preview(str(tmp_path), 0))
                            update_page_nav()
                            trans_preview.set_content('<div class="placeholder">Upload a document and click Translate</div>')
                            
                            log(f"Loaded: {filename} ({len(content):,} bytes)")
                            ui.notify(f'Loaded: {filename}', type='positive')
                            
                        except Exception as ex:
                            log(f"Upload error: {ex}")
                            ui.notify(f'Upload error: {str(ex)[:50]}', type='negative')
                    
                    ui.upload(
                        on_upload=handle_upload,
                        auto_upload=True,
                        max_files=1,
                    ).props('accept=".pdf" flat bordered label="Drop PDF here or click to browse"').classes('w-full')
                
                # Settings Card
                with ui.element('div').classes('card'):
                    ui.label('⚙️ Translation Settings').classes('card-title')
                    
                    with ui.row().classes('w-full gap-2 mb-3'):
                        dir_btn_en = ui.button('EN → FR', on_click=lambda: set_direction('en-fr')).props('outline size=sm').classes('flex-grow')
                        dir_btn_fr = ui.button('FR → EN', on_click=lambda: set_direction('fr-en')).props('outline size=sm').classes('flex-grow')
                    
                    direction = {'value': 'en-fr'}
                    def set_direction(d):
                        direction['value'] = d
                        if d == 'en-fr':
                            dir_btn_en.props('color=primary')
                            dir_btn_fr.props(remove='color')
                        else:
                            dir_btn_fr.props('color=primary')
                            dir_btn_en.props(remove='color')
                    set_direction('en-fr')
                    
                    engine_select = ui.select(
                        options=get_engines(),
                        value='dictionary',
                        label='Translation Engine'
                    ).classes('w-full mb-2')
                    
                    pages_input = ui.input(
                        label='Pages',
                        placeholder='all, or 1-5, or 1,3,5',
                        value='all'
                    ).classes('w-full mb-2').props('dense')
                    
                    with ui.expansion('Advanced Options', icon='tune').classes('w-full'):
                        quality_select = ui.select(
                            options={1: '1 pass', 2: '2 passes', 3: '3 passes'},
                            value=1,
                            label='Quality Passes'
                        ).classes('w-full mb-2').props('dense')
                        
                        candidates_select = ui.select(
                            options={1: '1 candidate', 3: '3 candidates', 5: '5 candidates'},
                            value=1,
                            label='Candidates'
                        ).classes('w-full').props('dense')
                
                # Translate Button
                translate_btn = ui.button(
                    'Translate',
                    icon='translate',
                    on_click=lambda: do_translate()
                ).props('color=primary size=lg').classes('w-full')
                
                # Progress Card
                with ui.element('div').classes('card'):
                    ui.label('📊 Progress').classes('card-title')
                    progress_bar = ui.linear_progress(value=0, show_value=False).classes('w-full')
                    progress_text = ui.label('Ready').classes('text-sm opacity-70 mt-2')
                    log_area = ui.textarea(value='', label='').props('readonly outlined').classes('w-full log-output mt-2')
                    log_area.style('min-height: 100px; font-family: monospace; font-size: 11px;')
                
                # Download Card
                with ui.element('div').classes('card'):
                    ui.label('📥 Download').classes('card-title')
                    download_btn = ui.button(
                        'Download Translated PDF',
                        icon='download',
                        on_click=lambda: ui.download(state['translated_path']) if state['translated_path'] else ui.notify('No translation available', type='warning')
                    ).props('outline').classes('w-full')
                    download_btn.set_enabled(False)
            
            # RIGHT PREVIEW AREA
            with ui.element('div').classes('preview-container'):
                
                # Tab buttons
                with ui.element('div').classes('preview-tabs'):
                    source_tab = ui.button('Source', on_click=lambda: show_tab('source')).props('outline').classes('preview-tab active')
                    trans_tab = ui.button('Translated', on_click=lambda: show_tab('translated')).props('outline').classes('preview-tab')
                
                current_tab = {'value': 'source'}
                
                def show_tab(tab):
                    current_tab['value'] = tab
                    if tab == 'source':
                        source_tab.classes(add='active')
                        trans_tab.classes(remove='active')
                        source_preview_container.set_visibility(True)
                        trans_preview_container.set_visibility(False)
                    else:
                        trans_tab.classes(add='active')
                        source_tab.classes(remove='active')
                        source_preview_container.set_visibility(False)
                        trans_preview_container.set_visibility(True)
                
                # Source preview
                with ui.element('div').classes('preview-area') as source_preview_container:
                    initial_content = '<div class="placeholder">Upload a PDF to preview</div>'
                    if state['source_path'] and Path(state['source_path']).exists():
                        initial_content = render_pdf_preview(state['source_path'], 0)
                    source_preview = ui.html(initial_content)
                
                # Translated preview (hidden initially)
                with ui.element('div').classes('preview-area') as trans_preview_container:
                    trans_preview = ui.html('<div class="placeholder">Translation will appear here</div>')
                trans_preview_container.set_visibility(False)
                
                # Page navigation
                with ui.element('div').classes('page-nav'):
                    ui.button(icon='chevron_left', on_click=lambda: prev_page()).props('flat')
                    page_label = ui.label('Page 1 / 1').classes('text-sm')
                    ui.button(icon='chevron_right', on_click=lambda: next_page()).props('flat')
                
                def update_page_nav():
                    if current_tab['value'] == 'source' and state['source_path']:
                        total = get_pdf_page_count(state['source_path'])
                        page_label.text = f"Page {state['source_page'] + 1} / {total}"
                    elif current_tab['value'] == 'translated' and state['translated_path']:
                        total = get_pdf_page_count(state['translated_path'])
                        page_label.text = f"Page {state['trans_page'] + 1} / {total}"
                    else:
                        page_label.text = "Page 1 / 1"
                
                def prev_page():
                    if current_tab['value'] == 'source' and state['source_path']:
                        if state['source_page'] > 0:
                            state['source_page'] -= 1
                            source_preview.set_content(render_pdf_preview(state['source_path'], state['source_page']))
                            update_page_nav()
                    elif current_tab['value'] == 'translated' and state['translated_path']:
                        if state['trans_page'] > 0:
                            state['trans_page'] -= 1
                            trans_preview.set_content(render_pdf_preview(state['translated_path'], state['trans_page']))
                            update_page_nav()
                
                def next_page():
                    if current_tab['value'] == 'source' and state['source_path']:
                        total = get_pdf_page_count(state['source_path'])
                        if state['source_page'] < total - 1:
                            state['source_page'] += 1
                            source_preview.set_content(render_pdf_preview(state['source_path'], state['source_page']))
                            update_page_nav()
                    elif current_tab['value'] == 'translated' and state['translated_path']:
                        total = get_pdf_page_count(state['translated_path'])
                        if state['trans_page'] < total - 1:
                            state['trans_page'] += 1
                            trans_preview.set_content(render_pdf_preview(state['translated_path'], state['trans_page']))
                            update_page_nav()
        
        async def do_translate():
            """Execute translation."""
            if not state['source_path']:
                ui.notify('Please upload a PDF first', type='warning')
                return
            
            if state['is_translating']:
                ui.notify('Translation in progress...', type='info')
                return
            
            state['is_translating'] = True
            translate_btn.disable()
            progress_bar.value = 0
            progress_text.text = 'Starting...'
            log('Translation started')
            
            try:
                # Prepare output path
                source_path = Path(state['source_path'])
                output_path = source_path.parent / f"{source_path.stem}_translated.pdf"
                
                def progress_callback(msg: str):
                    log(msg)
                    if 'pars' in msg.lower():
                        progress_bar.value = 0.2
                        progress_text.text = 'Parsing PDF...'
                    elif 'translat' in msg.lower():
                        progress_bar.value = 0.5
                        progress_text.text = 'Translating...'
                    elif 'render' in msg.lower():
                        progress_bar.value = 0.8
                        progress_text.text = 'Rendering...'
                
                # Run translation in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: translate_document(
                        input_path=str(source_path),
                        output_path=str(output_path),
                        engine=engine_select.value,
                        direction=direction['value'],
                        pages=pages_input.value or 'all',
                        quality_loops=int(quality_select.value),
                        enable_rerank=int(candidates_select.value) > 1,
                        num_candidates=int(candidates_select.value),
                        progress=progress_callback,
                    )
                )
                
                progress_bar.value = 1.0
                
                if result.success and output_path.exists():
                    state['translated_path'] = str(output_path)
                    state['trans_page'] = 0
                    
                    trans_preview.set_content(render_pdf_preview(str(output_path), 0))
                    download_btn.set_enabled(True)
                    
                    stats = result.stats or {}
                    log(f"Complete! {stats.get('translated_blocks', 0)} blocks translated")
                    progress_text.text = 'Complete!'
                    
                    # Switch to translated tab
                    show_tab('translated')
                    update_page_nav()
                    
                    ui.notify('Translation complete!', type='positive')
                else:
                    errors = result.errors[:2] if result.errors else ['Unknown error']
                    log(f"Failed: {errors}")
                    progress_text.text = 'Failed'
                    ui.notify(f'Translation failed: {errors[0][:50]}', type='negative')
                    
            except Exception as ex:
                log(f"Error: {ex}")
                progress_text.text = 'Error'
                ui.notify(f'Error: {str(ex)[:50]}', type='negative')
            finally:
                state['is_translating'] = False
                translate_btn.enable()
    
    # Run the app
    ui.run(
        port=port,
        title='SciTrans-LLMs',
        favicon='🌐',
        dark=True,
        storage_secret='scitrans_secret_key_12345',
        show=False,
    )


if __name__ == '__main__':
    launch()
