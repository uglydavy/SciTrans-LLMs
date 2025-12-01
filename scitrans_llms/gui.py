"""SciTrans-LLMs GUI - Scientific Document Translation Interface"""

from __future__ import annotations
import base64, json, logging, tempfile, asyncio, shutil, os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scitrans-gui")
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
    
    class State:
        dark_mode = True
        default_engine = 'free'
        default_masking = True
        default_reranking = True  # Reranking mandatory
        quality_passes = 1
        context_window = 5
        translate_tables = True  # Preserve by default
        translate_figures = True  # Preserve by default
        preserve_structure = True
        uploaded_pdf_path = None
        uploaded_pdf_name = None
        translated_pdf_path = None
        custom_glossary = None
    
    state = State()
    
    def get_engines():
        e = ['free', 'dictionary', 'dummy', 'improved-offline', 'huggingface', 'ollama']
        if km.get_key('openai'): e.append('openai')
        if km.get_key('deepseek'): e.append('deepseek')
        if km.get_key('anthropic'): e.append('anthropic')
        return list(set(e))
    
    def get_preview(path: str, page_num: int = 0) -> str:
        """Get PDF preview as base64 image, fit to screen."""
        try:
            import fitz
            doc = fitz.open(path)
            if page_num < len(doc):
                # Higher resolution and fit to screen
                pix = doc[page_num].get_pixmap(matrix=fitz.Matrix(2.5, 2.5))
                b64 = base64.b64encode(pix.tobytes("png")).decode()
                doc.close()
                return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;max-height:100%;object-fit:contain;"/>'
            doc.close()
        except Exception as ex:
            log_event(f"Preview error: {ex}", "ERROR")
        return '<div style="padding:40px;text-align:center;color:#888;">No preview available</div>'
    
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
    
    CSS = """<style>
:root { --bg-card: rgba(30,35,45,0.95); --border: rgba(100,100,120,0.3); }
html, body { margin:0; padding:0; overflow:hidden!important; height:100vh!important; }
.nicegui-content, .q-page-container { height:calc(100vh - 52px)!important; overflow:hidden!important; }
.q-tab-panels, .q-tab-panel { height:100%!important; overflow:hidden!important; padding:0!important; }
.main-row { display:flex; width:100%; height:calc(100vh - 100px); padding:8px 16px; gap:16px; box-sizing:border-box; }
.panel { flex:1; height:100%; overflow-y:auto; display:flex; flex-direction:column; gap:12px; }
.panel::-webkit-scrollbar { width:6px; } .panel::-webkit-scrollbar-thumb { background:#555; border-radius:3px; }
.card { background:var(--bg-card); border:1px solid var(--border); border-radius:8px; padding:14px; }
.card-title { font-weight:600; font-size:14px; margin-bottom:10px; border-bottom:1px solid var(--border); padding-bottom:8px; }
.upload-zone { border:2px dashed var(--border); border-radius:6px; padding:20px; text-align:center; cursor:pointer; transition:all 0.2s; min-height:120px; display:flex; flex-direction:column; align-items:center; justify-content:center; }
.upload-zone:hover { border-color:#6366f1; background:rgba(99,102,241,0.05); }
.preview-fit { width:100%; height:100%; display:flex; align-items:center; justify-content:center; overflow:auto; background:#1a1a2e; }
.preview-fit img { max-width:100%; max-height:100%; object-fit:contain; }
body.body--light { --bg-card: rgba(255,255,255,0.98); --border: rgba(0,0,0,0.12); }
body.body--light .q-header { background:#4f46e5!important; }
body.body--light .q-tab-panels { background:#f5f5f5!important; }
body.body--light .preview-fit { background:#e8e8e8; }
</style>"""
    
    @ui.page('/')
    async def main():
        ui.add_head_html(CSS)
        dark = ui.dark_mode()
        dark.enable()
        log_event("Page loaded")
        
        # Header
        with ui.header().classes('bg-indigo-700 items-center px-4 h-12'):
            ui.label('SciTrans-LLMs').classes('text-lg font-bold text-white')
            ui.space()
            ui.label('Scientific Document Translation').classes('text-sm text-white opacity-80')
            ui.space()
            def toggle_dark():
                if dark.value: dark.disable()
                else: dark.enable()
                state.dark_mode = dark.value
                log_event(f"Theme: {'dark' if state.dark_mode else 'light'}")
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
                with ui.element('div').classes('main-row'):
                    # LEFT PANEL
                    with ui.element('div').classes('panel'):
                        # Source Document
                        with ui.element('div').classes('card'):
                            ui.label('Source Document').classes('card-title')
                            upload_status = ui.label('No file selected').classes('text-sm opacity-70 mb-2')
                            
                            # Upload zone with proper drag & drop
                            with ui.element('div').classes('upload-zone') as upload_zone:
                                ui.label('📄 Drag & Drop PDF here').classes('text-sm font-medium mb-1')
                                ui.label('or click to browse').classes('text-xs opacity-70')
                                
                                async def handle_upload(e):
                                    try:
                                        # Handle NiceGUI upload event
                                        if hasattr(e, 'content'):
                                            content = await e.content.read() if hasattr(e.content, 'read') else e.content
                                            fname = getattr(e, 'name', 'document.pdf')
                                        elif hasattr(e, 'files'):
                                            files = e.files if isinstance(e.files, list) else [e.files]
                                            if files:
                                                f = files[0]
                                                content = await f.read() if hasattr(f, 'read') else f
                                                fname = getattr(f, 'name', 'document.pdf')
                                            else:
                                                raise ValueError("No file in upload event")
                                        else:
                                            # Try reading from sender
                                            content = await e.sender.read() if hasattr(e.sender, 'read') else e.sender
                                            fname = getattr(e.sender, 'name', 'document.pdf')
                                        
                                        if not content or len(content) == 0:
                                            ui.notify('Empty file received', type='warning')
                                            return
                                        
                                        tmp = Path(tempfile.mkdtemp()) / fname
                                        if isinstance(content, bytes):
                                            tmp.write_bytes(content)
                                        else:
                                            shutil.copy(content, tmp)
                                        
                                        state.uploaded_pdf_path = str(tmp)
                                        state.uploaded_pdf_name = fname
                                        upload_status.text = f'✓ Loaded: {fname}'
                                        
                                        # Reset translated preview
                                        state.translated_pdf_path = None
                                        translated_preview_html.set_content('<div style="padding:40px;text-align:center;color:#888;">No translation yet</div>')
                                        
                                        # Update source preview
                                        preview_html.set_content(get_preview(str(tmp), 0))
                                        current_page_num.value = 0
                                        update_page_count()
                                        
                                        log_event(f"Uploaded: {fname}")
                                        ui.notify(f'File loaded: {fname}', type='positive')
                                    except Exception as ex:
                                        log_event(f"Upload error: {ex}", "ERROR")
                                        import traceback
                                        log_event(traceback.format_exc(), "ERROR")
                                        ui.notify(f'Upload failed: {str(ex)[:80]}', type='negative')
                                
                                # Proper upload component with drag & drop support
                                upload_comp = ui.upload(
                                    on_upload=handle_upload,
                                    auto_upload=True,
                                    max_files=1
                                ).props('accept=".pdf" multiple=false')
                                
                                # Make entire zone clickable
                                upload_zone.on('click', lambda: upload_comp.run_method('pickFiles'))
                            
                            ui.label('Or enter URL:').classes('text-xs mt-3 opacity-70')
                            with ui.row().classes('w-full gap-2 items-center'):
                                url_input = ui.input(placeholder='https://arxiv.org/pdf/...').classes('flex-grow')
                                
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
                                        
                                        state.uploaded_pdf_path = str(fpath)
                                        state.uploaded_pdf_name = fname
                                        upload_status.text = f'✓ Loaded: {fname}'
                                        
                                        # Reset translated preview
                                        state.translated_pdf_path = None
                                        translated_preview_html.set_content('<div style="padding:40px;text-align:center;color:#888;">No translation yet</div>')
                                        
                                        preview_html.set_content(get_preview(str(fpath), 0))
                                        current_page_num.value = 0
                                        update_page_count()
                                        
                                        log_event(f"Downloaded: {fname}")
                                        ui.notify('PDF downloaded', type='positive')
                                    except Exception as ex:
                                        upload_status.text = 'Download failed'
                                        log_event(f"URL error: {ex}", "ERROR")
                                        ui.notify(f'Failed: {str(ex)[:50]}', type='negative')
                                
                                ui.button('Fetch', on_click=fetch_url).props('dense')
                        
                        # Translation Settings
                        with ui.element('div').classes('card'):
                            ui.label('Translation Settings').classes('card-title')
                            
                            # Direction as toggle buttons
                            with ui.row().classes('w-full gap-2 mb-2'):
                                direction_en_fr = ui.button('EN → FR', on_click=lambda: set_direction('en-fr')).props('toggle')
                                direction_fr_en = ui.button('FR → EN', on_click=lambda: set_direction('fr-en')).props('toggle')
                                direction_en_fr.props('color=primary')
                                current_direction = {'value': 'en-fr'}
                                
                                def set_direction(d):
                                    current_direction['value'] = d
                                    direction_en_fr.props(remove='color=primary' if d != 'en-fr' else '')
                                    direction_fr_en.props(remove='color=primary' if d != 'fr-en' else '')
                                    if d == 'en-fr':
                                        direction_en_fr.props('color=primary')
                                    else:
                                        direction_fr_en.props('color=primary')
                            
                            engine = ui.select(get_engines(), value=state.default_engine, label='Engine').classes('w-full')
                            
                            # Pages selection: dropdown + input
                            with ui.row().classes('w-full gap-2'):
                                pages_dropdown = ui.select(
                                    {'all': 'All pages', **{str(i): f'Pages 1-{i}' for i in range(1, 21)}},
                                    value='all',
                                    label='Pages'
                                ).classes('flex-grow')
                                
                                pages_custom = ui.input(placeholder='e.g., 1-5, 1,3,5', label='Custom').classes('w-32')
                            
                            quality = ui.select(
                                {1: '1 pass', 2: '2 passes', 3: '3 passes', 4: '4 passes', 5: '5 passes'},
                                value=state.quality_passes,
                                label='Quality passes'
                            ).classes('w-full')
                        
                        # Advanced Settings
                        with ui.element('div').classes('card'):
                            with ui.expansion('Advanced Settings', icon='tune').classes('w-full'):
                                ui.markdown('**Note:** Reranking is enabled by default for best quality.').classes('text-xs opacity-70 mb-2')
                                masking_chk = ui.checkbox('Mask formulas/URLs', value=state.default_masking)
                                structure_chk = ui.checkbox('Preserve numbering (1.1, I., etc.)', value=state.preserve_structure)
                                tables_chk = ui.checkbox('Preserve tables', value=state.translate_tables)
                                figures_chk = ui.checkbox('Preserve figures', value=state.translate_figures)
                                ui.number(label='Context window (segments)', value=state.context_window, min=1, max=20).classes('w-full mt-2')
                                ui.markdown('**Note:** MinerU extraction is automatically used for best results.').classes('text-xs opacity-70 mt-2')
                        
                        # Glossary
                        with ui.element('div').classes('card'):
                            ui.label('Custom Glossary (optional)').classes('card-title')
                            gloss_status = ui.label('Using default glossary').classes('text-xs opacity-70')
                            
                            async def handle_gloss(e):
                                try:
                                    if hasattr(e, 'content'):
                                        content = await e.content.read() if hasattr(e.content, 'read') else e.content
                                    else:
                                        content = b''
                                    fname = getattr(e, 'name', 'glossary.csv')
                                    g = parse_gloss(content, fname)
                                    if g:
                                        state.custom_glossary = g
                                        gloss_status.text = f'Custom: {len(g.entries)} terms loaded'
                                        ui.notify(f'{len(g.entries)} terms loaded', type='positive')
                                except Exception as ex:
                                    gloss_status.text = f'Error: {str(ex)[:30]}'
                            
                            ui.upload(on_upload=handle_gloss, auto_upload=True).props('accept=".csv,.txt,.json" label="Upload glossary"').classes('w-full')
                        
                        # Translate Button
                        translate_btn = ui.button('Translate Document').classes('w-full').props('color=primary size=lg')
                    
                    # RIGHT PANEL
                    with ui.element('div').classes('panel'):
                        # Preview with tabs
                        with ui.element('div').classes('card').style('flex:1; display:flex; flex-direction:column;'):
                            ui.label('Document Preview').classes('card-title')
                            
                            # Tabs for Source/Translated
                            with ui.tabs().classes('w-full mb-2') as preview_tabs:
                                source_tab = ui.tab('Source')
                                translated_tab = ui.tab('Translated')
                            
                            with ui.tab_panels(preview_tabs, value=source_tab).classes('flex-1').style('min-height:0;'):
                                # Source preview
                                with ui.tab_panel(source_tab).classes('p-0').style('height:100%;'):
                                    with ui.element('div').classes('preview-fit'):
                                        preview_html = ui.html('<div style="padding:40px;text-align:center;color:#666;">Upload a document to preview</div>', sanitize=False).style('width:100%;height:100%;')
                                    
                                    # Page navigation for source
                                    with ui.row().classes('w-full justify-center gap-2 mt-2'):
                                        def prev_source():
                                            if current_page_num.value > 0:
                                                current_page_num.value -= 1
                                                if state.uploaded_pdf_path:
                                                    preview_html.set_content(get_preview(state.uploaded_pdf_path, current_page_num.value))
                                                update_page_count()
                                        
                                        def next_source():
                                            if state.uploaded_pdf_path:
                                                import fitz
                                                doc = fitz.open(state.uploaded_pdf_path)
                                                max_pages = len(doc)
                                                doc.close()
                                                if current_page_num.value < max_pages - 1:
                                                    current_page_num.value += 1
                                                    preview_html.set_content(get_preview(state.uploaded_pdf_path, current_page_num.value))
                                                update_page_count()
                                        
                                        ui.button(icon='chevron_left', on_click=prev_source).props('flat dense')
                                        current_page_num = ui.number(value=0, min=0, format=lambda v: f'Page {int(v)+1}').props('dense readonly').style('width:100px;')
                                        ui.button(icon='chevron_right', on_click=next_source).props('flat dense')
                                        
                                        def update_page_count():
                                            if state.uploaded_pdf_path:
                                                try:
                                                    import fitz
                                                    doc = fitz.open(state.uploaded_pdf_path)
                                                    max_pages = len(doc)
                                                    doc.close()
                                                    current_page_num.props(f'label="Page {int(current_page_num.value)+1} / {max_pages}"')
                                                except:
                                                    pass
                                
                                # Translated preview
                                with ui.tab_panel(translated_tab).classes('p-0').style('height:100%;'):
                                    with ui.element('div').classes('preview-fit'):
                                        translated_preview_html = ui.html('<div style="padding:40px;text-align:center;color:#666;">No translation yet</div>', sanitize=False).style('width:100%;height:100%;')
                                    
                                    # Page navigation for translated
                                    with ui.row().classes('w-full justify-center gap-2 mt-2'):
                                        def prev_translated():
                                            if translated_page_num.value > 0:
                                                translated_page_num.value -= 1
                                                if state.translated_pdf_path:
                                                    translated_preview_html.set_content(get_preview(state.translated_pdf_path, translated_page_num.value))
                                                update_translated_page_count()
                                        
                                        def next_translated():
                                            if state.translated_pdf_path:
                                                import fitz
                                                doc = fitz.open(state.translated_pdf_path)
                                                max_pages = len(doc)
                                                doc.close()
                                                if translated_page_num.value < max_pages - 1:
                                                    translated_page_num.value += 1
                                                    translated_preview_html.set_content(get_preview(state.translated_pdf_path, translated_page_num.value))
                                                update_translated_page_count()
                                        
                                        ui.button(icon='chevron_left', on_click=prev_translated).props('flat dense')
                                        translated_page_num = ui.number(value=0, min=0, format=lambda v: f'Page {int(v)+1}').props('dense readonly').style('width:100px;')
                                        ui.button(icon='chevron_right', on_click=next_translated).props('flat dense')
                                        
                                        def update_translated_page_count():
                                            if state.translated_pdf_path:
                                                try:
                                                    import fitz
                                                    doc = fitz.open(state.translated_pdf_path)
                                                    max_pages = len(doc)
                                                    doc.close()
                                                    translated_page_num.props(f'label="Page {int(translated_page_num.value)+1} / {max_pages}"')
                                                except:
                                                    pass
                        
                        # Progress
                        with ui.element('div').classes('card'):
                            ui.label('Translation Progress').classes('card-title')
                            prog_bar = ui.linear_progress(value=0, show_value=False).classes('w-full')
                            prog_text = ui.label('Ready').classes('text-sm opacity-70 mt-1')
                            log_area = ui.textarea().props('readonly rows=4').classes('w-full mt-2 text-xs font-mono')
                        
                        # Download
                        with ui.element('div').classes('card'):
                            download_btn = ui.button('Download Translated Document').classes('w-full').props('color=positive size=lg disabled')
                
                # Translate logic
                async def do_translate():
                    if not state.uploaded_pdf_path:
                        ui.notify('Upload a document first', type='warning')
                        return
                    translate_btn.disable()
                    download_btn.props('disabled')
                    prog_bar.value = 0
                    logs = []
                    
                    # Reset translated preview
                    state.translated_pdf_path = None
                    translated_preview_html.set_content('<div style="padding:40px;text-align:center;color:#888;">Translating...</div>')
                    
                    def log(m):
                        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {m}")
                        log_area.value = '\n'.join(logs[-6:])
                        log_event(m)
                    
                    try:
                        log("Starting translation...")
                        prog_bar.value = 0.1
                        prog_text.text = "Parsing document..."
                        
                        inp = Path(state.uploaded_pdf_path)
                        out = Path(tempfile.mkdtemp()) / f'translated_{inp.name}'
                        
                        # Get pages setting
                        pages_val = pages_dropdown.value
                        if pages_custom.value.strip():
                            pages_val = pages_custom.value.strip()
                        
                        log(f"Engine: {engine.value}, Direction: {current_direction['value']}")
                        
                        def cb(m):
                            log(m)
                            if 'pars' in m.lower(): prog_bar.value = 0.2
                            elif 'translat' in m.lower(): prog_bar.value = 0.5
                            elif 'render' in m.lower(): prog_bar.value = 0.85
                        
                        await asyncio.sleep(0.1)
                        prog_bar.value = 0.25
                        prog_text.text = "Translating..."
                        
                        # Run translation with reranking enabled
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(None, lambda: translate_document(
                            input_path=str(inp),
                            output_path=str(out),
                            engine=engine.value,
                            direction=current_direction['value'],
                            pages=pages_val,
                            quality_loops=int(quality.value),
                            enable_rerank=True,  # Reranking mandatory
                            use_mineru=True,  # MinerU mandatory
                            progress=cb
                        ))
                        
                        prog_bar.value = 1.0
                        if result.success:
                            log("Translation complete")
                            prog_text.text = "Complete"
                            if out.exists():
                                state.translated_pdf_path = str(out)
                                download_btn.props(remove='disabled')
                                translated_preview_html.set_content(get_preview(str(out), 0))
                                translated_page_num.value = 0
                                update_translated_page_count()
                            ui.notify('Translation complete', type='positive')
                        else:
                            log(f"Errors: {result.errors[:2]}")
                            prog_text.text = "Completed with errors"
                    except Exception as ex:
                        log(f"Error: {ex}")
                        prog_text.text = "Error"
                        log_event(f"Translation error: {ex}", "ERROR")
                        import traceback
                        log_event(traceback.format_exc(), "ERROR")
                    finally:
                        translate_btn.enable()
                
                translate_btn.on_click(do_translate)
                download_btn.on_click(lambda: ui.download(state.translated_pdf_path) if state.translated_pdf_path else None)
            
            # ==================== TESTING TAB ====================
            with ui.tab_panel(t_testing).classes('p-0'):
                with ui.element('div').classes('main-row'):
                    # LEFT - Input & Settings
                    with ui.element('div').classes('panel'):
                        with ui.element('div').classes('card'):
                            ui.label('Text Translation Test').classes('card-title')
                            test_input = ui.textarea(label='Source text', placeholder='Enter text to translate...').props('rows=6').classes('w-full')
                            with ui.row().classes('w-full gap-4 mt-2'):
                                test_dir = ui.select({'en-fr': 'EN → FR', 'fr-en': 'FR → EN'}, value='en-fr', label='Direction').classes('w-1/3')
                                test_eng = ui.select(get_engines(), value='dictionary', label='Engine').classes('w-1/3')
                                test_passes = ui.number(label='Passes', value=1, min=1, max=5).classes('w-1/4')
                        
                        with ui.element('div').classes('card'):
                            ui.label('Test Options').classes('card-title')
                            with ui.row().classes('gap-4 flex-wrap'):
                                test_mask = ui.checkbox('Enable masking', value=True)
                                test_rerank = ui.checkbox('Enable reranking', value=False)  # Optional in testing
                                test_gloss = ui.checkbox('Use glossary', value=True)
                            with ui.row().classes('gap-4 flex-wrap mt-2'):
                                test_tables = ui.checkbox('Translate tables', value=False)
                                test_formulas = ui.checkbox('Translate formulas', value=False)
                        
                        with ui.row().classes('w-full gap-2 mt-2'):
                            test_run_btn = ui.button('Run Translation').props('color=primary')
                            test_clear_btn = ui.button('Clear').props('outline')
                    
                    # RIGHT - Output & Metrics
                    with ui.element('div').classes('panel'):
                        with ui.element('div').classes('card'):
                            ui.label('Translation Output').classes('card-title')
                            test_output = ui.textarea(label='Result').props('rows=6 readonly').classes('w-full')
                            test_status = ui.label('').classes('text-xs opacity-70 mt-1')
                        
                        with ui.element('div').classes('card'):
                            ui.label('Evaluation Metrics').classes('card-title')
                            with ui.row().classes('w-full gap-8'):
                                with ui.column():
                                    ui.label('BLEU Score').classes('text-xs opacity-70')
                                    bleu_label = ui.label('--').classes('text-lg font-bold')
                                with ui.column():
                                    ui.label('Terms Used').classes('text-xs opacity-70')
                                    terms_label = ui.label('--').classes('text-lg font-bold')
                                with ui.column():
                                    ui.label('Blocks').classes('text-xs opacity-70')
                                    blocks_label = ui.label('--').classes('text-lg font-bold')
                            
                            ui.label('Reference (for BLEU)').classes('text-xs opacity-70 mt-3')
                            test_ref = ui.textarea(placeholder='Optional: paste reference translation for BLEU score').props('rows=3').classes('w-full')
                        
                        with ui.element('div').classes('card'):
                            ui.label('Ablation Study').classes('card-title')
                            ui.label('Compare configurations to measure component contributions').classes('text-xs opacity-70 mb-2')
                            ablation_btn = ui.button('Run Ablation (coming soon)').props('outline disabled')
                
                def run_test():
                    if not test_input.value:
                        ui.notify('Enter text first', type='warning')
                        return
                    test_status.text = 'Translating...'
                    try:
                        s, t = test_dir.value.split('-')
                        config = PipelineConfig(
                            source_lang=s, target_lang=t, translator_backend=test_eng.value,
                            enable_masking=test_mask.value, enable_glossary=test_gloss.value,
                            enable_refinement=test_rerank.value, num_candidates=int(test_passes.value)
                        )
                        pipeline = TranslationPipeline(config)
                        doc = Document.from_text(test_input.value, s, t)
                        result = pipeline.translate(doc)
                        test_output.value = result.translated_text
                        blocks_label.text = str(result.stats.get('translated_blocks', 0))
                        terms_label.text = str(len(result.stats.get('glossary_terms_used', [])) if 'glossary_terms_used' in result.stats else result.stats.get('refined_blocks', 0))
                        
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
                        log_event(f"Test completed: {test_eng.value}")
                    except Exception as ex:
                        test_output.value = f'Error: {ex}'
                        test_status.text = 'Error'
                        log_event(f"Test error: {ex}", "ERROR")
                
                test_run_btn.on_click(run_test)
                test_clear_btn.on_click(lambda: (setattr(test_input, 'value', ''), setattr(test_output, 'value', ''), setattr(bleu_label, 'text', '--'), setattr(terms_label, 'text', '--'), setattr(blocks_label, 'text', '--')))
            
            # ==================== GLOSSARY TAB ====================
            with ui.tab_panel(t_glossary).classes('p-0'):
                with ui.element('div').classes('main-row'):
                    with ui.element('div').classes('panel'):
                        with ui.element('div').classes('card'):
                            ui.label('Default Scientific Glossary').classes('card-title')
                            with ui.row().classes('w-full gap-4 mb-2'):
                                search = ui.input(label='Search terms').classes('flex-grow')
                                domain_sel = ui.select(['All', 'ml', 'math', 'stats', 'physics', 'chemistry', 'general'], value='All', label='Domain').classes('w-32')
                            try:
                                g = get_default_glossary()
                                all_entries = [{'en': e.source, 'fr': e.target, 'domain': e.domain} for e in g.entries]
                            except:
                                all_entries = []
                            tbl = ui.table(
                                columns=[{'name': 'en', 'label': 'English', 'field': 'en', 'sortable': True}, {'name': 'fr', 'label': 'French', 'field': 'fr', 'sortable': True}, {'name': 'domain', 'label': 'Domain', 'field': 'domain'}],
                                rows=all_entries[:30], row_key='en', pagination=10
                            ).classes('w-full')
                            
                            def filter_tbl():
                                s, d = search.value.lower(), domain_sel.value
                                tbl.rows = [e for e in all_entries if (d == 'All' or e['domain'] == d) and (not s or s in e['en'].lower() or s in e['fr'].lower())][:30]
                            search.on('keyup', filter_tbl)
                            domain_sel.on('change', filter_tbl)
                    
                    with ui.element('div').classes('panel'):
                        with ui.element('div').classes('card'):
                            ui.label('Parallel Corpus Integration').classes('card-title')
                            ui.markdown('''
Parallel corpora provide verified translation pairs from official sources. 
Loading a corpus enhances translation quality by providing domain-specific terminology.

- **Europarl**: European Parliament proceedings (political, legal)
- **OPUS Scientific**: Academic publications and journals
- **ELG Terminology**: European Language Grid scientific terms
                            ''').classes('text-sm opacity-80 mb-3')
                            
                            corpus_status = ui.label('No corpus loaded').classes('text-xs opacity-70 mb-2')
                            with ui.row().classes('gap-2'):
                                for cid, nm in [('europarl', 'Europarl EN-FR'), ('opus', 'OPUS Scientific'), ('elg', 'ELG Terms')]:
                                    def load(n=nm):
                                        corpus_status.text = f'{n}: Ready (simulated)'
                                        log_event(f"Corpus loaded: {n}")
                                        ui.notify(f'{n} corpus activated', type='positive')
                                    ui.button(nm, on_click=load).props('outline dense')
                        
                        with ui.element('div').classes('card'):
                            ui.label('Upload Custom Glossary').classes('card-title')
                            ui.markdown('''
**Supported formats:**
- CSV: `source,target,domain` (header optional)
- TXT: tab-separated pairs
- JSON: `[{"source": "...", "target": "..."}]`
                            ''').classes('text-xs opacity-70 mb-2')
                            
                            custom_status = ui.label('').classes('text-xs')
                            async def on_custom_gloss(e):
                                try:
                                    if hasattr(e, 'content'):
                                        content = await e.content.read() if hasattr(e.content, 'read') else e.content
                                    else:
                                        content = b''
                                    fname = getattr(e, 'name', 'glossary.csv')
                                    g = parse_gloss(content, fname)
                                    if g:
                                        state.custom_glossary = g
                                        custom_status.text = f'Loaded {len(g.entries)} custom terms'
                                        ui.notify(f'{len(g.entries)} terms loaded', type='positive')
                                except Exception as ex:
                                    custom_status.text = f'Error: {str(ex)[:30]}'
                            ui.upload(on_upload=on_custom_gloss, auto_upload=True).props('accept=".csv,.txt,.json"').classes('w-full')
            
            # ==================== DEVELOPER TAB ====================
            with ui.tab_panel(t_developer).classes('p-0'):
                with ui.element('div').classes('main-row'):
                    with ui.element('div').classes('panel'):
                        with ui.element('div').classes('card'):
                            ui.label('System Logs').classes('card-title')
                            with ui.row().classes('gap-2 mb-2'):
                                def refresh(): dev_logs.value = '\n'.join(system_logs[-40:])
                                def clear(): system_logs.clear(); dev_logs.value = ''
                                def export():
                                    f = Path(tempfile.mktemp(suffix='.log'))
                                    f.write_text('\n'.join(system_logs))
                                    ui.download(str(f), 'scitrans_logs.log')
                                ui.button('Refresh', on_click=refresh).props('dense outline')
                                ui.button('Clear', on_click=clear).props('dense outline')
                                ui.button('Export', on_click=export).props('dense outline')
                            dev_logs = ui.textarea().props('readonly rows=12').classes('w-full font-mono text-xs')
                            dev_logs.value = '\n'.join(system_logs[-40:])
                    
                    with ui.element('div').classes('panel'):
                        with ui.element('div').classes('card'):
                            ui.label('System Information').classes('card-title')
                            import sys, platform
                            info = [('Python', f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'), ('Platform', platform.system()), ('Architecture', platform.machine())]
                            for m in ['nicegui', 'fitz', 'numpy', 'sacrebleu']:
                                try:
                                    mod = __import__(m)
                                    info.append((m, getattr(mod, '__version__', 'installed')))
                                except:
                                    info.append((m, 'not installed'))
                            for k, v in info:
                                with ui.row().classes('w-full justify-between py-1'):
                                    ui.label(k).classes('opacity-70')
                                    ui.label(v).classes('font-mono text-sm')
                        
                        with ui.element('div').classes('card'):
                            ui.label('Translation Backends').classes('card-title')
                            backends = [
                                ('free', 'Smart cascade (Lingva→LibreTranslate→MyMemory)', True),
                                ('dictionary', 'Offline glossary-based', True),
                                ('openai', 'GPT-4/GPT-4o', bool(km.get_key('openai'))),
                                ('deepseek', 'DeepSeek Chat', bool(km.get_key('deepseek'))),
                                ('anthropic', 'Claude 3', bool(km.get_key('anthropic'))),
                                ('ollama', 'Local LLM (requires Ollama)', True),
                            ]
                            for name, desc, available in backends:
                                status = 'Ready' if available else 'Needs API key'
                                color = 'text-green-400' if available else 'text-yellow-400'
                                with ui.row().classes('w-full justify-between py-1'):
                                    ui.label(name).classes('font-mono')
                                    ui.label(f'{status}').classes(f'text-xs {color}')
            
            # ==================== SETTINGS TAB ====================
            with ui.tab_panel(t_settings).classes('p-0'):
                with ui.element('div').classes('main-row'):
                    with ui.element('div').classes('panel'):
                        with ui.element('div').classes('card'):
                            ui.label('API Keys').classes('card-title')
                            ui.label('Configure API keys for premium translation services').classes('text-xs opacity-70 mb-3')
                            
                            services = ['openai', 'deepseek', 'anthropic', 'huggingface', 'deepl', 'google']
                            for svc in services:
                                k = km.get_key(svc)
                                status = 'Configured' if k else 'Not set'
                                color = 'text-green-400' if k else 'text-gray-400'
                                with ui.row().classes('w-full justify-between py-1'):
                                    ui.label(svc.capitalize()).classes('font-medium')
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
                                    storage = km.set_key(key_svc.value, key_val.value.strip())
                                    key_msg.text = f'Saved to {storage}'
                                    key_val.value = ''
                                    log_event(f"API key saved: {key_svc.value}")
                                    ui.notify('Key saved', type='positive')
                                except Exception as ex:
                                    key_msg.text = f'Error: {ex}'
                            ui.button('Save Key', on_click=save_key).classes('w-full mt-2')
                    
                    with ui.element('div').classes('panel'):
                        with ui.element('div').classes('card'):
                            ui.label('Default Settings').classes('card-title')
                            
                            def on_dark_toggle(e):
                                if e.value: dark.enable()
                                else: dark.disable()
                                state.dark_mode = e.value
                                log_event(f"Theme: {'dark' if e.value else 'light'}")
                            ui.switch('Dark mode', value=state.dark_mode, on_change=on_dark_toggle)
                            
                            def on_engine(e): state.default_engine = e.value; log_event(f"Default engine: {e.value}")
                            ui.select(get_engines(), value=state.default_engine, label='Default translation engine', on_change=on_engine).classes('w-full mt-2')
                            
                            def on_quality(e): state.quality_passes = int(e.value); log_event(f"Quality: {e.value}")
                            ui.number(label='Default quality passes', value=state.quality_passes, min=1, max=5, on_change=on_quality).classes('w-full mt-2')
                            
                            def on_context(e): state.context_window = int(e.value); log_event(f"Context: {e.value}")
                            ui.number(label='Context window size', value=state.context_window, min=1, max=20, on_change=on_context).classes('w-full mt-2')
                        
                        with ui.element('div').classes('card'):
                            ui.label('Processing Defaults').classes('card-title')
                            
                            def on_mask(e): state.default_masking = e.value
                            ui.switch('Enable formula/URL masking', value=state.default_masking, on_change=on_mask)
                            
                            def on_tables(e): state.translate_tables = e.value
                            ui.switch('Translate table content', value=state.translate_tables, on_change=on_tables)
                            
                            def on_figures(e): state.translate_figures = e.value
                            ui.switch('Translate figure captions', value=state.translate_figures, on_change=on_figures)
                        
                        with ui.element('div').classes('card'):
                            ui.label('About').classes('card-title')
                            ui.markdown('''
**SciTrans-LLMs** v1.0

Scientific document translation system with:
- Layout-preserving PDF translation
- Terminology management via glossaries
- Multiple translation backends
- Evaluation metrics (BLEU, glossary adherence)
                            ''').classes('text-sm opacity-80')
    
    log_event("GUI ready")
    print(f"\n{'='*60}\nSciTrans-LLMs GUI - http://127.0.0.1:{port}\n{'='*60}")
    
    ui.run(
        port=port,
        title='SciTrans-LLMs',
        favicon='🔬',
        show=True,
        reload=False,
        storage_secret='scitrans_stable_session_2024',
        reconnect_timeout=60,
    )

if __name__ == "__main__":
    launch()
