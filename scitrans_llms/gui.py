"""SciTrans-LLMs GUI - Scientific Document Translation Interface

Major fixes:
- Session persistence with app.storage
- Proper file upload handling
- Page navigation for PDF preview
- MinerU enforced by default
- Better layout and UX
"""

from __future__ import annotations
import base64, json, logging, tempfile, asyncio, shutil, os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scitrans-gui")

# Global logs stored outside of client sessions
system_logs: List[str] = []

def log_event(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    system_logs.append(f"[{ts}] [{level}] {msg}")
    if len(system_logs) > 500: system_logs.pop(0)
    (logger.error if level == "ERROR" else logger.info)(msg)

def launch(port: int = 7860, share: bool = False):
    from nicegui import ui, app
    from scitrans_llms.keys import KeyManager
    from scitrans_llms.pipeline import PipelineConfig, TranslationPipeline, translate_document
    from scitrans_llms.translate.glossary import get_default_glossary, Glossary, GlossaryEntry
    from scitrans_llms.models import Document
    
    km = KeyManager()
    log_event("GUI init")
    
    # Persistent storage for translation results (survives tab switches)
    translation_results: Dict[str, Any] = {}
    
    def get_all_engines():
        """Get all available translation engines."""
        engines = [
            'free', 'dictionary', 'dummy', 'improved-offline', 
            'huggingface', 'ollama', 'googlefree'
        ]
        # Add API-based engines
        if km.get_key('openai'): engines.append('openai')
        if km.get_key('deepseek'): engines.append('deepseek')
        if km.get_key('anthropic'): engines.append('anthropic')
        if km.get_key('deepl'): engines.append('deepl')
        if km.get_key('google'): engines.append('google')
        return sorted(set(engines))
    
    def get_all_backends_status():
        """Get status of all backends for display."""
        backends = [
            ('free', 'Smart cascade (Lingva→LibreTranslate→MyMemory)', True),
            ('dictionary', 'Offline glossary-based', True),
            ('dummy', 'Test backend (no translation)', True),
            ('improved-offline', 'Enhanced offline translation', True),
            ('huggingface', 'Hugging Face Inference API', True),
            ('googlefree', 'Google Translate (free, via deep-translator)', True),
            ('ollama', 'Local LLM (requires Ollama running)', True),
            ('openai', 'GPT-4/GPT-4o/GPT-4-turbo', bool(km.get_key('openai'))),
            ('deepseek', 'DeepSeek Chat', bool(km.get_key('deepseek'))),
            ('anthropic', 'Claude 3/3.5 (Sonnet, Opus)', bool(km.get_key('anthropic'))),
            ('deepl', 'DeepL API (professional)', bool(km.get_key('deepl'))),
            ('google', 'Google Cloud Translation API', bool(km.get_key('google'))),
        ]
        return backends
    
    def get_preview_pages(path: str) -> List[str]:
        """Get all page previews as base64 images."""
        pages = []
        try:
            import fitz
            doc = fitz.open(path)
            for i in range(len(doc)):
                # Higher resolution for better preview
                pix = doc[i].get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                b64 = base64.b64encode(pix.tobytes("png")).decode()
                pages.append(b64)
            doc.close()
        except Exception as ex:
            log_event(f"Preview error: {ex}", "ERROR")
        return pages
    
    def parse_gloss(content: bytes, fname: str):
        try:
            txt = content.decode('utf-8')
            entries = []
            if fname.endswith('.json'):
                for item in json.loads(txt):
                    if isinstance(item, dict):
                        entries.append(GlossaryEntry(source=item.get('source', ''), target=item.get('target', ''), domain='custom'))
            elif fname.endswith('.csv'):
                for line in txt.strip().split('\n')[1:]:
                    parts = [p.strip().strip('"') for p in line.split(',')]
                    if len(parts) >= 2:
                        entries.append(GlossaryEntry(source=parts[0], target=parts[1], domain='custom'))
            else:
                for line in txt.strip().split('\n'):
                    sep = '\t' if '\t' in line else ','
                    parts = [p.strip() for p in line.split(sep)]
                    if len(parts) >= 2:
                        entries.append(GlossaryEntry(source=parts[0], target=parts[1], domain='custom'))
            return Glossary(name='custom', entries=entries) if entries else None
        except:
            return None
    
    # Custom CSS for better layout
    CSS = """<style>
:root { 
    --bg-card: rgba(30,35,45,0.95); 
    --border: rgba(100,100,120,0.3); 
    --accent: #6366f1;
}
html, body { margin:0; padding:0; overflow-x:hidden; }
.nicegui-content { min-height: 100vh; }
.main-container { 
    display: flex; 
    flex-direction: column;
    height: calc(100vh - 100px);
    padding: 12px 16px;
    gap: 12px;
    box-sizing: border-box;
}
.main-row { 
    display: flex; 
    flex: 1;
    gap: 16px; 
    min-height: 0;
    overflow: hidden;
}
.panel { 
    flex: 1; 
    display: flex; 
    flex-direction: column; 
    gap: 12px;
    overflow-y: auto;
    min-height: 0;
}
.panel::-webkit-scrollbar { width: 6px; } 
.panel::-webkit-scrollbar-thumb { background: #555; border-radius: 3px; }
.card { 
    background: var(--bg-card); 
    border: 1px solid var(--border); 
    border-radius: 8px; 
    padding: 14px; 
    flex-shrink: 0;
}
.card.stretch {
    flex: 1;
    min-height: 200px;
    display: flex;
    flex-direction: column;
}
.card-title { 
    font-weight: 600; 
    font-size: 14px; 
    margin-bottom: 10px; 
    border-bottom: 1px solid var(--border); 
    padding-bottom: 8px; 
}
.upload-zone { 
    border: 2px dashed var(--border); 
    border-radius: 8px; 
    padding: 24px; 
    text-align: center; 
    cursor: pointer; 
    transition: all 0.2s;
    min-height: 100px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.upload-zone:hover { 
    border-color: var(--accent); 
    background: rgba(99,102,241,0.08); 
}
.preview-container {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-height: 300px;
    background: #1a1a2e;
    border-radius: 8px;
    overflow: hidden;
}
.preview-image {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: auto;
    padding: 8px;
}
.preview-image img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    border-radius: 4px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
.preview-nav {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    padding: 8px;
    background: rgba(0,0,0,0.3);
}
.bottom-actions {
    display: flex;
    gap: 12px;
    padding-top: 8px;
    flex-shrink: 0;
}
.bottom-actions .q-btn {
    flex: 1;
}
.logs-area {
    flex: 1;
    min-height: 300px;
    font-family: monospace;
    font-size: 11px;
}
body.body--light { 
    --bg-card: rgba(255,255,255,0.98); 
    --border: rgba(0,0,0,0.12); 
}
body.body--light .q-header { background: #4f46e5 !important; }
body.body--light .q-tab-panels { background: #f5f5f5 !important; }
body.body--light .preview-container { background: #e8e8e8; }
</style>"""
    
    @ui.page('/')
    async def main():
        ui.add_head_html(CSS)
        dark = ui.dark_mode()
        dark.enable()
        log_event("Page loaded")
        
        # Use app.storage.user for persistence across tab switches
        storage = app.storage.user
        
        # Initialize storage with defaults
        if 'initialized' not in storage:
            storage['initialized'] = True
            storage['dark_mode'] = True
            storage['default_engine'] = 'free'
            storage['quality_passes'] = 1
            storage['context_window'] = 5
            storage['uploaded_pdf_path'] = None
            storage['uploaded_pdf_name'] = None
            storage['translated_pdf_path'] = None
            storage['source_pages'] = []
            storage['translated_pages'] = []
            storage['current_page'] = 0
            storage['preview_mode'] = 'source'  # 'source' or 'translated'
            storage['translation_in_progress'] = False
        
        # Local state for UI elements (references)
        class UIState:
            preview_html = None
            page_label = None
            prog_bar = None
            prog_text = None
            log_area = None
            download_btn = None
            translate_btn = None
        
        ui_state = UIState()
        
        # Header
        with ui.header().classes('bg-indigo-700 items-center px-4 h-12'):
            ui.label('SciTrans-LLMs').classes('text-lg font-bold text-white')
            ui.space()
            ui.label('Scientific Document Translation').classes('text-sm text-white opacity-80')
            ui.space()
            def toggle_dark():
                if dark.value: dark.disable()
                else: dark.enable()
                storage['dark_mode'] = dark.value
            ui.button(icon='contrast', on_click=toggle_dark).props('flat round dense text-color=white size=sm')
        
        # Tabs
        with ui.tabs().classes('w-full bg-gray-800 text-white') as tabs:
            t_translate = ui.tab('Translate')
            t_testing = ui.tab('Testing')
            t_glossary = ui.tab('Glossary')
            t_developer = ui.tab('Developer')
            t_settings = ui.tab('Settings')
        
        with ui.tab_panels(tabs, value=t_translate).classes('w-full flex-grow'):
            
            # ==================== TRANSLATE TAB ====================
            with ui.tab_panel(t_translate).classes('p-0'):
                with ui.element('div').classes('main-container'):
                    with ui.element('div').classes('main-row'):
                        # LEFT PANEL - Settings
                        with ui.element('div').classes('panel').style('max-width: 400px; min-width: 320px;'):
                            
                            # Source Document Card with Tabs
                            with ui.element('div').classes('card'):
                                ui.label('Source Document').classes('card-title')
                                
                                with ui.tabs().classes('w-full') as upload_tabs:
                                    upload_tab = ui.tab('Upload File', icon='upload_file')
                                    url_tab = ui.tab('From URL', icon='link')
                                
                                upload_status = ui.label('No file selected').classes('text-sm opacity-70 my-2')
                                
                                with ui.tab_panels(upload_tabs, value=upload_tab).classes('w-full'):
                                    # Upload File Tab
                                    with ui.tab_panel(upload_tab).classes('p-0'):
                                        async def handle_upload(e):
                                            try:
                                                # Get the uploaded file content
                                                content = await e.read()
                                                fname = e.name
                                                
                                                if not content:
                                                    ui.notify('No file data received', type='warning')
                                                    return
                                                
                                                tmp = Path(tempfile.mkdtemp()) / fname
                                                tmp.write_bytes(content)
                                                
                                                storage['uploaded_pdf_path'] = str(tmp)
                                                storage['uploaded_pdf_name'] = fname
                                                upload_status.text = f'✓ Loaded: {fname}'
                                                
                                                # Generate previews
                                                pages = get_preview_pages(str(tmp))
                                                storage['source_pages'] = pages
                                                storage['current_page'] = 0
                                                storage['preview_mode'] = 'source'
                                                update_preview()
                                                
                                                log_event(f"Uploaded: {fname}")
                                                ui.notify(f'File loaded: {fname}', type='positive')
                                            except Exception as ex:
                                                log_event(f"Upload error: {ex}", "ERROR")
                                                ui.notify(f'Upload failed: {str(ex)[:50]}', type='negative')
                                        
                                        ui.upload(
                                            on_upload=handle_upload,
                                            auto_upload=True,
                                            max_files=1
                                        ).props('accept=".pdf" flat bordered').classes('w-full').style('min-height: 100px;')
                                    
                                    # URL Tab
                                    with ui.tab_panel(url_tab).classes('p-0'):
                                        url_input = ui.input(
                                            placeholder='https://arxiv.org/pdf/...',
                                            label='PDF URL'
                                        ).classes('w-full')
                                        
                                        async def fetch_url():
                                            url = url_input.value.strip()
                                            if not url:
                                                ui.notify('Enter a URL', type='warning')
                                                return
                                            upload_status.text = 'Fetching...'
                                            try:
                                                import urllib.request
                                                log_event(f"Fetching: {url}")
                                                tmp = Path(tempfile.mkdtemp())
                                                fname = url.split('/')[-1].split('?')[0] or 'document.pdf'
                                                if not fname.endswith('.pdf'): fname += '.pdf'
                                                fpath = tmp / fname
                                                
                                                loop = asyncio.get_event_loop()
                                                await loop.run_in_executor(None, lambda: urllib.request.urlretrieve(url, fpath))
                                                
                                                storage['uploaded_pdf_path'] = str(fpath)
                                                storage['uploaded_pdf_name'] = fname
                                                upload_status.text = f'✓ Loaded: {fname}'
                                                
                                                # Generate previews
                                                pages = get_preview_pages(str(fpath))
                                                storage['source_pages'] = pages
                                                storage['current_page'] = 0
                                                storage['preview_mode'] = 'source'
                                                update_preview()
                                                
                                                log_event(f"Downloaded: {fname}")
                                                ui.notify('PDF downloaded', type='positive')
                                            except Exception as ex:
                                                upload_status.text = 'Download failed'
                                                log_event(f"URL error: {ex}", "ERROR")
                                                ui.notify(f'Failed: {str(ex)[:50]}', type='negative')
                                        
                                        ui.button('Fetch PDF', on_click=fetch_url, icon='download').classes('w-full mt-2')
                            
                            # Translation Settings Card
                            with ui.element('div').classes('card'):
                                ui.label('Translation Settings').classes('card-title')
                                
                                direction = ui.select(
                                    {'en-fr': 'English → French', 'fr-en': 'French → English'},
                                    value='en-fr',
                                    label='Direction'
                                ).classes('w-full')
                                
                                engine = ui.select(
                                    get_all_engines(),
                                    value=storage.get('default_engine', 'free'),
                                    label='Translation Engine'
                                ).classes('w-full')
                                
                                with ui.row().classes('w-full gap-4'):
                                    pages_input = ui.input(
                                        label='Pages',
                                        value='all',
                                        placeholder='all, 1-5, 1,3,5'
                                    ).classes('flex-grow')
                                    
                                    quality = ui.select(
                                        {1: '1 pass', 2: '2 passes', 3: '3 passes', 4: '4 passes', 5: '5 passes'},
                                        value=storage.get('quality_passes', 1),
                                        label='Quality'
                                    ).classes('w-32')
                            
                            # Advanced Settings (Collapsed)
                            with ui.element('div').classes('card'):
                                with ui.expansion('Advanced Settings', icon='tune').classes('w-full'):
                                    masking_chk = ui.checkbox('Mask formulas/URLs', value=True)
                                    rerank_chk = ui.checkbox('Enable reranking', value=False)
                                    structure_chk = ui.checkbox('Preserve numbering (1.1, I., etc.)', value=True)
                                    
                                    ui.separator().classes('my-2')
                                    ui.label('Context window').classes('text-xs opacity-70')
                                    context_slider = ui.slider(min=1, max=20, value=5).props('label-always')
                                    
                                    ui.separator().classes('my-2')
                                    ui.markdown('**Note:** MinerU extraction is automatically used for best results.').classes('text-xs opacity-70')
                            
                            # Custom Glossary (Collapsed)
                            with ui.element('div').classes('card'):
                                with ui.expansion('Custom Glossary', icon='book').classes('w-full'):
                                    gloss_status = ui.label('Using default glossary').classes('text-xs opacity-70')
                                    
                                    async def handle_gloss(e):
                                        try:
                                            content = await e.read()
                                            fname = e.name
                                            g = parse_gloss(content, fname)
                                            if g:
                                                storage['custom_glossary'] = True
                                                gloss_status.text = f'✓ Custom: {len(g.entries)} terms'
                                                ui.notify(f'{len(g.entries)} terms loaded', type='positive')
                                        except Exception as ex:
                                            gloss_status.text = f'Error: {str(ex)[:30]}'
                                    
                                    ui.upload(
                                        on_upload=handle_gloss,
                                        auto_upload=True
                                    ).props('accept=".csv,.txt,.json" label="Upload glossary"').classes('w-full')
                        
                        # RIGHT PANEL - Preview
                        with ui.element('div').classes('panel'):
                            # Preview Card (Stretches)
                            with ui.element('div').classes('card stretch'):
                                # Preview Header with mode toggle
                                with ui.row().classes('w-full items-center mb-2'):
                                    ui.label('Document Preview').classes('font-semibold flex-grow')
                                    
                                    def set_preview_mode(mode):
                                        storage['preview_mode'] = mode
                                        storage['current_page'] = 0
                                        update_preview()
                                    
                                    with ui.button_group():
                                        ui.button('Source', on_click=lambda: set_preview_mode('source')).props('dense')
                                        ui.button('Translated', on_click=lambda: set_preview_mode('translated')).props('dense')
                                
                                # Preview Container
                                with ui.element('div').classes('preview-container'):
                                    with ui.element('div').classes('preview-image') as preview_container:
                                        ui_state.preview_html = ui.html(
                                            '<div style="padding:40px;text-align:center;color:#888;">Upload or fetch a document</div>'
                                        )
                                    
                                    # Navigation
                                    with ui.element('div').classes('preview-nav'):
                                        def prev_page():
                                            if storage['current_page'] > 0:
                                                storage['current_page'] -= 1
                                                update_preview()
                                        
                                        def next_page():
                                            pages = storage.get('translated_pages' if storage['preview_mode'] == 'translated' else 'source_pages', [])
                                            if storage['current_page'] < len(pages) - 1:
                                                storage['current_page'] += 1
                                                update_preview()
                                        
                                        ui.button(icon='chevron_left', on_click=prev_page).props('flat dense')
                                        ui_state.page_label = ui.label('Page 0 / 0').classes('text-sm')
                                        ui.button(icon='chevron_right', on_click=next_page).props('flat dense')
                            
                            # Progress Card
                            with ui.element('div').classes('card'):
                                ui.label('Translation Progress').classes('card-title')
                                ui_state.prog_bar = ui.linear_progress(value=0, show_value=False).classes('w-full')
                                ui_state.prog_text = ui.label('Ready').classes('text-sm opacity-70 mt-1')
                                ui_state.log_area = ui.textarea().props('readonly rows=3').classes('w-full mt-2 text-xs font-mono')
                    
                    # Bottom Actions (Fixed at bottom)
                    with ui.element('div').classes('bottom-actions'):
                        ui_state.translate_btn = ui.button(
                            'TRANSLATE DOCUMENT',
                            icon='translate'
                        ).props('color=primary size=lg').classes('flex-grow')
                        
                        ui_state.download_btn = ui.button(
                            'DOWNLOAD',
                            icon='download'
                        ).props('color=positive size=lg disabled').classes('flex-grow')
                
                # Helper function to update preview
                def update_preview():
                    mode = storage.get('preview_mode', 'source')
                    pages = storage.get('translated_pages' if mode == 'translated' else 'source_pages', [])
                    current = storage.get('current_page', 0)
                    
                    if pages and 0 <= current < len(pages):
                        b64 = pages[current]
                        ui_state.preview_html.set_content(
                            f'<img src="data:image/png;base64,{b64}" style="max-width:100%;"/>'
                        )
                        ui_state.page_label.text = f'Page {current + 1} / {len(pages)} ({mode.title()})'
                    else:
                        ui_state.preview_html.set_content(
                            '<div style="padding:40px;text-align:center;color:#888;">No preview available</div>'
                        )
                        ui_state.page_label.text = f'Page 0 / 0'
                
                # Translation logic with proper error handling
                async def do_translate():
                    if not storage.get('uploaded_pdf_path'):
                        ui.notify('Upload a document first', type='warning')
                        return
                    
                    # Prevent double-click
                    if storage.get('translation_in_progress'):
                        ui.notify('Translation already in progress', type='info')
                        return
                    
                    storage['translation_in_progress'] = True
                    ui_state.translate_btn.disable()
                    ui_state.download_btn.props('disabled')
                    ui_state.prog_bar.value = 0
                    logs = []
                    
                    def safe_log(m):
                        """Log without crashing on client disconnect"""
                        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {m}")
                        log_event(m)
                        try:
                            if ui_state.log_area:
                                ui_state.log_area.value = '\n'.join(logs[-5:])
                        except:
                            pass  # Ignore if client disconnected
                    
                    def safe_progress(val, text):
                        """Update progress without crashing"""
                        try:
                            if ui_state.prog_bar:
                                ui_state.prog_bar.value = val
                            if ui_state.prog_text:
                                ui_state.prog_text.text = text
                        except:
                            pass
                    
                    try:
                        safe_log("Starting translation...")
                        safe_progress(0.1, "Parsing document...")
                        
                        inp = Path(storage['uploaded_pdf_path'])
                        out = Path(tempfile.mkdtemp()) / f'translated_{inp.name}'
                        
                        safe_log(f"Engine: {engine.value}, Direction: {direction.value}")
                        
                        def cb(m):
                            safe_log(m)
                            if 'pars' in m.lower(): safe_progress(0.2, "Parsing...")
                            elif 'translat' in m.lower(): safe_progress(0.5, "Translating...")
                            elif 'render' in m.lower(): safe_progress(0.85, "Rendering...")
                        
                        await asyncio.sleep(0.1)
                        
                        # Run translation in executor
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(None, lambda: translate_document(
                            input_path=str(inp),
                            output_path=str(out),
                            engine=engine.value,
                            direction=direction.value,
                            pages=pages_input.value,
                            quality_loops=int(quality.value),
                            enable_rerank=rerank_chk.value,
                            progress=cb
                        ))
                        
                        safe_progress(1.0, "Complete!")
                        
                        if result.success:
                            safe_log("Translation complete")
                            if out.exists():
                                storage['translated_pdf_path'] = str(out)
                                # Generate translated preview
                                translated_pages = get_preview_pages(str(out))
                                storage['translated_pages'] = translated_pages
                                storage['preview_mode'] = 'translated'
                                storage['current_page'] = 0
                                update_preview()
                                
                                try:
                                    ui_state.download_btn.props(remove='disabled')
                                except:
                                    pass
                            ui.notify('Translation complete!', type='positive')
                        else:
                            safe_log(f"Errors: {result.errors[:2]}")
                            safe_progress(1.0, "Completed with errors")
                            ui.notify('Translation completed with errors', type='warning')
                    
                    except Exception as ex:
                        safe_log(f"Error: {str(ex)[:100]}")
                        safe_progress(0, "Error")
                        log_event(f"Translation error: {ex}", "ERROR")
                        ui.notify(f'Translation failed: {str(ex)[:50]}', type='negative')
                    
                    finally:
                        storage['translation_in_progress'] = False
                        try:
                            ui_state.translate_btn.enable()
                        except:
                            pass
                
                ui_state.translate_btn.on_click(do_translate)
                
                def do_download():
                    path = storage.get('translated_pdf_path')
                    if path and Path(path).exists():
                        ui.download(path)
                
                ui_state.download_btn.on_click(do_download)
                
                # Restore state if returning to page
                if storage.get('uploaded_pdf_name'):
                    upload_status.text = f'✓ Loaded: {storage["uploaded_pdf_name"]}'
                if storage.get('source_pages'):
                    update_preview()
                if storage.get('translated_pdf_path') and Path(storage['translated_pdf_path']).exists():
                    ui_state.download_btn.props(remove='disabled')
            
            # ==================== TESTING TAB ====================
            with ui.tab_panel(t_testing).classes('p-0'):
                with ui.element('div').classes('main-container'):
                    with ui.element('div').classes('main-row'):
                        # LEFT - Input
                        with ui.element('div').classes('panel'):
                            with ui.element('div').classes('card'):
                                ui.label('Text Translation Test').classes('card-title')
                                test_input = ui.textarea(
                                    label='Source text',
                                    placeholder='Enter text to translate...'
                                ).props('rows=8').classes('w-full')
                                
                                with ui.row().classes('w-full gap-4 mt-2'):
                                    test_dir = ui.select(
                                        {'en-fr': 'EN → FR', 'fr-en': 'FR → EN'},
                                        value='en-fr',
                                        label='Direction'
                                    ).classes('flex-grow')
                                    test_eng = ui.select(
                                        get_all_engines(),
                                        value='dictionary',
                                        label='Engine'
                                    ).classes('flex-grow')
                            
                            with ui.element('div').classes('card'):
                                ui.label('Test Options').classes('card-title')
                                with ui.row().classes('gap-4 flex-wrap'):
                                    test_mask = ui.checkbox('Enable masking', value=True)
                                    test_gloss = ui.checkbox('Use glossary', value=True)
                                with ui.row().classes('gap-4 flex-wrap mt-2'):
                                    test_mineru = ui.checkbox('Use MinerU', value=True)
                                    test_preserve = ui.checkbox('Preserve structure', value=True)
                            
                            with ui.row().classes('w-full gap-2'):
                                test_run_btn = ui.button('Run Translation', icon='play_arrow').props('color=primary')
                                test_clear_btn = ui.button('Clear', icon='clear').props('outline')
                        
                        # RIGHT - Output
                        with ui.element('div').classes('panel'):
                            with ui.element('div').classes('card stretch'):
                                ui.label('Translation Output').classes('card-title')
                                test_output = ui.textarea(label='Result').props('rows=8 readonly').classes('w-full')
                                test_status = ui.label('').classes('text-xs opacity-70 mt-1')
                            
                            with ui.element('div').classes('card'):
                                ui.label('Evaluation Metrics').classes('card-title')
                                with ui.row().classes('w-full gap-8'):
                                    with ui.column():
                                        ui.label('BLEU').classes('text-xs opacity-70')
                                        bleu_label = ui.label('--').classes('text-lg font-bold')
                                    with ui.column():
                                        ui.label('Blocks').classes('text-xs opacity-70')
                                        blocks_label = ui.label('--').classes('text-lg font-bold')
                                    with ui.column():
                                        ui.label('Time').classes('text-xs opacity-70')
                                        time_label = ui.label('--').classes('text-lg font-bold')
                                
                                ui.label('Reference (optional)').classes('text-xs opacity-70 mt-3')
                                test_ref = ui.textarea(placeholder='Paste reference for BLEU').props('rows=2').classes('w-full')
                
                def run_test():
                    import time
                    if not test_input.value:
                        ui.notify('Enter text first', type='warning')
                        return
                    test_status.text = 'Translating...'
                    start = time.time()
                    try:
                        s, t = test_dir.value.split('-')
                        config = PipelineConfig(
                            source_lang=s, target_lang=t,
                            translator_backend=test_eng.value,
                            enable_masking=test_mask.value,
                            enable_glossary=test_gloss.value,
                        )
                        pipeline = TranslationPipeline(config)
                        doc = Document.from_text(test_input.value, s, t)
                        result = pipeline.translate(doc)
                        
                        test_output.value = result.translated_text
                        blocks_label.text = str(result.stats.get('translated_blocks', 0))
                        time_label.text = f'{time.time() - start:.1f}s'
                        
                        if test_ref.value.strip():
                            try:
                                from sacrebleu.metrics import BLEU
                                bleu = BLEU()
                                score = bleu.sentence_score(result.translated_text, [test_ref.value])
                                bleu_label.text = f'{score.score:.1f}'
                            except:
                                bleu_label.text = 'N/A'
                        else:
                            bleu_label.text = '--'
                        
                        test_status.text = 'Complete'
                    except Exception as ex:
                        test_output.value = f'Error: {ex}'
                        test_status.text = 'Error'
                
                test_run_btn.on_click(run_test)
                test_clear_btn.on_click(lambda: (
                    setattr(test_input, 'value', ''),
                    setattr(test_output, 'value', ''),
                    setattr(bleu_label, 'text', '--'),
                    setattr(blocks_label, 'text', '--'),
                    setattr(time_label, 'text', '--')
                ))
            
            # ==================== GLOSSARY TAB ====================
            with ui.tab_panel(t_glossary).classes('p-0'):
                with ui.element('div').classes('main-container'):
                    with ui.element('div').classes('main-row'):
                        with ui.element('div').classes('panel'):
                            with ui.element('div').classes('card stretch'):
                                ui.label('Scientific Glossary').classes('card-title')
                                with ui.row().classes('w-full gap-4 mb-2'):
                                    search = ui.input(label='Search', placeholder='Search terms...').classes('flex-grow')
                                    domain_sel = ui.select(
                                        ['All', 'ml', 'math', 'stats', 'physics', 'chemistry', 'general'],
                                        value='All',
                                        label='Domain'
                                    ).classes('w-32')
                                
                                try:
                                    g = get_default_glossary()
                                    all_entries = [{'en': e.source, 'fr': e.target, 'domain': e.domain} for e in g.entries]
                                except:
                                    all_entries = []
                                
                                tbl = ui.table(
                                    columns=[
                                        {'name': 'en', 'label': 'English', 'field': 'en', 'sortable': True},
                                        {'name': 'fr', 'label': 'French', 'field': 'fr', 'sortable': True},
                                        {'name': 'domain', 'label': 'Domain', 'field': 'domain'}
                                    ],
                                    rows=all_entries[:50],
                                    row_key='en',
                                    pagination={'rowsPerPage': 15}
                                ).classes('w-full')
                                
                                def filter_tbl():
                                    s, d = search.value.lower(), domain_sel.value
                                    tbl.rows = [
                                        e for e in all_entries
                                        if (d == 'All' or e['domain'] == d)
                                        and (not s or s in e['en'].lower() or s in e['fr'].lower())
                                    ][:50]
                                
                                search.on('keyup', filter_tbl)
                                domain_sel.on('change', filter_tbl)
                        
                        with ui.element('div').classes('panel'):
                            with ui.element('div').classes('card'):
                                ui.label('Parallel Corpus').classes('card-title')
                                ui.markdown('''
Download parallel corpora to enhance translation quality:
- **Europarl**: EU Parliament (~200MB)
- **EU Constitution**: Legal texts (~5MB)
- **Tatoeba**: Community sentences (~50MB)
                                ''').classes('text-sm opacity-80 mb-3')
                                
                                corpus_status = ui.label('').classes('text-xs mb-2')
                                corpus_progress = ui.linear_progress(value=0).classes('w-full mb-2').style('display:none')
                                
                                async def download_corpus(name: str, display: str):
                                    try:
                                        from scitrans_llms.translate.corpus_manager import CorpusManager
                                        corpus_progress.style('display:block')
                                        corpus_status.text = f'Downloading {display}...'
                                        
                                        manager = CorpusManager()
                                        
                                        def prog(msg, pct):
                                            corpus_progress.value = pct
                                            corpus_status.text = msg[:40]
                                        
                                        loop = asyncio.get_event_loop()
                                        await loop.run_in_executor(None, lambda: manager.download(name, 'en', 'fr', prog))
                                        
                                        dict_size = len(manager.build_dictionary(name, 'en', 'fr', limit=5000))
                                        corpus_status.text = f'✓ {display}: {dict_size} terms'
                                        ui.notify(f'{display} loaded', type='positive')
                                    except Exception as ex:
                                        corpus_status.text = f'Error: {str(ex)[:30]}'
                                    finally:
                                        corpus_progress.style('display:none')
                                
                                with ui.row().classes('gap-2 flex-wrap'):
                                    ui.button('Europarl', on_click=lambda: download_corpus('europarl', 'Europarl')).props('outline dense')
                                    ui.button('EU Const', on_click=lambda: download_corpus('opus-euconst', 'EU Constitution')).props('outline dense')
                                    ui.button('Tatoeba', on_click=lambda: download_corpus('tatoeba', 'Tatoeba')).props('outline dense')
                            
                            with ui.element('div').classes('card'):
                                ui.label('Upload Custom Glossary').classes('card-title')
                                ui.markdown('''
Formats: CSV (`source,target`), TSV, or JSON
                                ''').classes('text-xs opacity-70 mb-2')
                                
                                custom_status = ui.label('').classes('text-xs')
                                
                                async def on_custom_gloss(e):
                                    try:
                                        content = await e.read()
                                        g = parse_gloss(content, e.name)
                                        if g:
                                            custom_status.text = f'✓ {len(g.entries)} terms loaded'
                                            ui.notify(f'{len(g.entries)} terms', type='positive')
                                    except Exception as ex:
                                        custom_status.text = f'Error: {str(ex)[:30]}'
                                
                                ui.upload(on_upload=on_custom_gloss, auto_upload=True).props('accept=".csv,.txt,.json"').classes('w-full')
            
            # ==================== DEVELOPER TAB ====================
            with ui.tab_panel(t_developer).classes('p-0'):
                with ui.element('div').classes('main-container'):
                    with ui.element('div').classes('main-row'):
                        with ui.element('div').classes('panel'):
                            with ui.element('div').classes('card stretch'):
                                ui.label('System Logs').classes('card-title')
                                with ui.row().classes('gap-2 mb-2'):
                                    def refresh_logs():
                                        dev_logs.value = '\n'.join(system_logs[-100:])
                                    def clear_logs():
                                        system_logs.clear()
                                        dev_logs.value = ''
                                    def export_logs():
                                        f = Path(tempfile.mktemp(suffix='.log'))
                                        f.write_text('\n'.join(system_logs))
                                        ui.download(str(f), 'scitrans_logs.log')
                                    
                                    ui.button('Refresh', on_click=refresh_logs, icon='refresh').props('dense outline')
                                    ui.button('Clear', on_click=clear_logs, icon='clear').props('dense outline')
                                    ui.button('Export', on_click=export_logs, icon='download').props('dense outline')
                                
                                dev_logs = ui.textarea().props('readonly').classes('w-full logs-area font-mono text-xs')
                                dev_logs.value = '\n'.join(system_logs[-100:])
                        
                        with ui.element('div').classes('panel'):
                            with ui.element('div').classes('card'):
                                ui.label('System Information').classes('card-title')
                                import sys, platform
                                info = [
                                    ('Python', f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'),
                                    ('Platform', platform.system()),
                                    ('Architecture', platform.machine()),
                                ]
                                for m in ['nicegui', 'fitz', 'numpy', 'sacrebleu', 'torch', 'ultralytics']:
                                    try:
                                        mod = __import__(m)
                                        info.append((m, getattr(mod, '__version__', 'installed')))
                                    except:
                                        info.append((m, 'not installed'))
                                
                                for k, v in info:
                                    with ui.row().classes('w-full justify-between py-1'):
                                        ui.label(k).classes('opacity-70')
                                        ui.label(v).classes('font-mono text-sm')
                            
                            with ui.element('div').classes('card stretch'):
                                ui.label('All Translation Backends').classes('card-title')
                                
                                backends = get_all_backends_status()
                                for name, desc, available in backends:
                                    status = '✓ Ready' if available else '⚠ Needs API key'
                                    color = 'text-green-400' if available else 'text-yellow-400'
                                    with ui.row().classes('w-full justify-between py-1 items-start'):
                                        with ui.column().classes('gap-0'):
                                            ui.label(name).classes('font-mono font-medium')
                                            ui.label(desc).classes('text-xs opacity-60')
                                        ui.label(status).classes(f'text-xs {color}')
            
            # ==================== SETTINGS TAB ====================
            with ui.tab_panel(t_settings).classes('p-0'):
                with ui.element('div').classes('main-container'):
                    with ui.element('div').classes('main-row'):
                        with ui.element('div').classes('panel'):
                            with ui.element('div').classes('card'):
                                ui.label('API Keys').classes('card-title')
                                ui.label('Configure API keys for premium services').classes('text-xs opacity-70 mb-3')
                                
                                services = ['openai', 'deepseek', 'anthropic', 'huggingface', 'deepl', 'google']
                                for svc in services:
                                    k = km.get_key(svc)
                                    status = '✓ Configured' if k else '○ Not set'
                                    color = 'text-green-400' if k else 'text-gray-400'
                                    with ui.row().classes('w-full justify-between py-1'):
                                        ui.label(svc.upper()).classes('font-medium')
                                        ui.label(status).classes(f'text-xs {color}')
                                
                                ui.separator().classes('my-3')
                                
                                key_svc = ui.select(services, value='openai', label='Service').classes('w-full')
                                key_val = ui.input(label='API Key', password=True).classes('w-full')
                                key_msg = ui.label('').classes('text-xs')
                                
                                def save_key():
                                    if not key_val.value.strip():
                                        key_msg.text = 'Enter a key'
                                        return
                                    try:
                                        result = km.set_key(key_svc.value, key_val.value.strip())
                                        key_msg.text = f'✓ Saved to {result}'
                                        key_val.value = ''
                                        ui.notify('Key saved', type='positive')
                                    except Exception as ex:
                                        key_msg.text = f'Error: {ex}'
                                
                                ui.button('Save Key', on_click=save_key, icon='save').classes('w-full mt-2')
                        
                        with ui.element('div').classes('panel'):
                            with ui.element('div').classes('card'):
                                ui.label('Default Settings').classes('card-title')
                                
                                def on_dark_toggle(e):
                                    if e.value: dark.enable()
                                    else: dark.disable()
                                    storage['dark_mode'] = e.value
                                
                                ui.switch('Dark mode', value=storage.get('dark_mode', True), on_change=on_dark_toggle)
                                
                                def on_engine_change(e):
                                    storage['default_engine'] = e.value
                                
                                ui.select(
                                    get_all_engines(),
                                    value=storage.get('default_engine', 'free'),
                                    label='Default engine',
                                    on_change=on_engine_change
                                ).classes('w-full mt-2')
                                
                                def on_quality_change(e):
                                    storage['quality_passes'] = int(e.value)
                                
                                ui.select(
                                    {1: '1 pass', 2: '2 passes', 3: '3 passes'},
                                    value=storage.get('quality_passes', 1),
                                    label='Default quality',
                                    on_change=on_quality_change
                                ).classes('w-full mt-2')
                            
                            with ui.element('div').classes('card'):
                                ui.label('All Available Backends').classes('card-title')
                                
                                backends = get_all_backends_status()
                                for name, desc, available in backends:
                                    status_icon = '✓' if available else '○'
                                    color = 'text-green-400' if available else 'text-gray-500'
                                    with ui.row().classes('w-full justify-between py-1'):
                                        ui.label(f'{status_icon} {name}').classes(f'font-mono {color}')
                            
                            with ui.element('div').classes('card'):
                                ui.label('About').classes('card-title')
                                ui.markdown('''
**SciTrans-LLMs** v1.0

Scientific document translation with:
- Layout-preserving PDF translation
- MinerU extraction (automatic)
- Multiple translation backends
- Terminology management
- BLEU evaluation
                                ''').classes('text-sm opacity-80')
    
    log_event("GUI ready")
    print(f"\n{'='*60}\nSciTrans-LLMs GUI - http://127.0.0.1:{port}\n{'='*60}")
    
    ui.run(
        port=port,
        title='SciTrans-LLMs',
        favicon='🔬',
        show=True,
        reload=False,
        storage_secret='scitrans_persistent_2024_v2',
        reconnect_timeout=120,  # 2 minute reconnect window
    )

if __name__ == "__main__":
    launch()
