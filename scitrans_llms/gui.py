"""SciTrans-LLMs GUI - Scientific Document Translation Interface

Fixed version with:
- Proper session persistence using app.storage.user
- Layout that fits to screen without scrolling  
- Working drag & drop upload
- Reliable URL fetch
- Real-time logging
- Error recovery
"""

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
    
    def get_engines():
        e = ['free', 'dictionary', 'improved-offline', 'huggingface', 'ollama']
        if km.get_key('openai'): e.append('openai')
        if km.get_key('deepseek'): e.append('deepseek')
        if km.get_key('anthropic'): e.append('anthropic')
        return list(set(e))
    
    def get_preview(path: str, page_num: int = 0) -> str:
        """Get PDF preview as base64 image."""
        try:
            import fitz
            doc = fitz.open(path)
            if page_num < len(doc):
                pix = doc[page_num].get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                b64 = base64.b64encode(pix.tobytes("png")).decode()
                doc.close()
                return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;max-height:100%;object-fit:contain;display:block;margin:auto;"/>'
            doc.close()
        except Exception as ex:
            log_event(f"Preview error: {ex}", "ERROR")
        return '<div style="padding:40px;text-align:center;color:#888;">No preview</div>'
    
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
    
    # CSS optimized for no-scroll layout
    CSS = """<style>
:root { 
    --bg-card: rgba(30,35,45,0.95); 
    --border: rgba(100,100,120,0.3);
    --accent: #6366f1;
}

html, body { 
    margin:0; padding:0; 
    height:100vh!important; 
    overflow:hidden!important;
}

.nicegui-content, .q-page-container { 
    height:calc(100vh - 48px)!important; 
    overflow:hidden!important;
}

.q-tab-panels { 
    height:100%!important; 
    overflow:hidden!important;
}

.q-tab-panel { 
    height:100%!important; 
    padding:0!important;
    overflow:hidden!important;
}

/* Main two-column layout */
.main-row { 
    display:flex; 
    height:calc(100vh - 100px)!important;
    padding:8px; 
    gap:8px; 
    box-sizing:border-box;
    overflow:hidden!important;
}

.panel { 
    flex:1; 
    display:flex; 
    flex-direction:column; 
    gap:6px;
    overflow-y:auto;
    overflow-x:hidden;
    max-height:100%;
}

.panel::-webkit-scrollbar { width:6px; }
.panel::-webkit-scrollbar-thumb { background:#555; border-radius:3px; }

.card { 
    background:var(--bg-card); 
    border:1px solid var(--border); 
    border-radius:6px; 
    padding:10px;
    flex-shrink:0;
}

.card-title { 
    font-weight:600; 
    font-size:12px; 
    margin-bottom:6px; 
    border-bottom:1px solid var(--border); 
    padding-bottom:4px;
    color:#fff;
}

/* Upload zone */
.upload-zone {
    border:2px dashed var(--border);
    border-radius:6px;
    padding:20px;
    text-align:center;
    cursor:pointer;
    transition:all 0.2s;
    min-height:80px;
    display:flex;
    flex-direction:column;
    align-items:center;
    justify-content:center;
    background:rgba(99,102,241,0.03);
}

.upload-zone:hover {
    border-color:var(--accent);
    background:rgba(99,102,241,0.1);
}

/* Preview area - constrained height */
.preview-card {
    flex:1;
    display:flex;
    flex-direction:column;
    min-height:200px;
    max-height:calc(100vh - 300px);
}

.preview-area {
    flex:1;
    display:flex;
    align-items:center;
    justify-content:center;
    background:#1a1a2e;
    border-radius:4px;
    overflow:auto;
    min-height:150px;
    max-height:calc(100vh - 380px);
}

.preview-area img {
    max-width:100%;
    max-height:100%;
    object-fit:contain;
}

/* Buttons */
.q-btn { font-size:12px!important; }

.action-row {
    display:flex;
    gap:6px;
    margin-top:6px;
}

.action-row .q-btn { flex:1; }

/* Pagination */
.pagination-row {
    display:flex;
    align-items:center;
    justify-content:center;
    gap:6px;
    padding:6px 0;
}

/* Light mode */
body.body--light {
    --bg-card: rgba(255,255,255,0.98);
    --border: rgba(0,0,0,0.12);
}
body.body--light .preview-area { background:#e8e8e8; }
body.body--light .card-title { color:#000; }

/* Developer tab logs */
.log-area {
    font-family:monospace;
    font-size:11px;
    min-height:200px;
    max-height:300px;
    overflow-y:auto!important;
}
</style>"""
    
    @ui.page('/')
    async def main():
        ui.add_head_html(CSS)
        dark = ui.dark_mode()
        dark.enable()
        log_event("Page loaded")
        
        # Initialize storage for session persistence
        storage = app.storage.user
        
        # UI State class with storage integration
        class UIState:
            def __init__(self):
                self._dark_mode = storage.get('dark_mode', True)
                self._uploaded_path = storage.get('uploaded_path', None)
                self._uploaded_name = storage.get('uploaded_name', None)
                self._translated_path = storage.get('translated_path', None)
                self._direction = storage.get('direction', 'en-fr')
                self._engine = storage.get('engine', 'free')
                self._source_page = 0
                self._translated_page = 0
                self.custom_glossary = None
            
            @property
            def dark_mode(self):
                return self._dark_mode
            
            @dark_mode.setter
            def dark_mode(self, v):
                self._dark_mode = v
                storage['dark_mode'] = v
            
            @property
            def uploaded_path(self):
                return self._uploaded_path
            
            @uploaded_path.setter
            def uploaded_path(self, v):
                self._uploaded_path = v
                storage['uploaded_path'] = v
            
            @property
            def uploaded_name(self):
                return self._uploaded_name
            
            @uploaded_name.setter
            def uploaded_name(self, v):
                self._uploaded_name = v
                storage['uploaded_name'] = v
            
            @property
            def translated_path(self):
                return self._translated_path
            
            @translated_path.setter
            def translated_path(self, v):
                self._translated_path = v
                storage['translated_path'] = v
            
            @property
            def direction(self):
                return self._direction
            
            @direction.setter
            def direction(self, v):
                self._direction = v
                storage['direction'] = v
            
            @property
            def engine(self):
                return self._engine
            
            @engine.setter
            def engine(self, v):
                self._engine = v
                storage['engine'] = v
        
        state = UIState()
        
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
            ui.button(icon='dark_mode', on_click=toggle_dark).props('flat round dense text-color=white').tooltip('Toggle theme')
        
        # Tabs
        with ui.tabs().classes('w-full bg-gray-800 text-white') as tabs:
            t_translate = ui.tab('Translate', icon='translate')
            t_testing = ui.tab('Testing', icon='science')
            t_glossary = ui.tab('Glossary', icon='book')
            t_developer = ui.tab('Developer', icon='code')
            t_settings = ui.tab('Settings', icon='settings')
        
        with ui.tab_panels(tabs, value=t_translate).classes('w-full flex-grow'):
            
            # ==================== TRANSLATE TAB ====================
            with ui.tab_panel(t_translate).classes('p-0'):
                with ui.element('div').classes('main-row'):
                    
                    # LEFT PANEL - Source & Settings
                    with ui.element('div').classes('panel').style('max-width:45%;'):
                        
                        # Source Document Card
                        with ui.element('div').classes('card'):
                            ui.label('Source Document').classes('card-title')
                            upload_status = ui.label('No file selected').classes('text-xs opacity-70 mb-2')
                            
                            # Restore previous upload if exists
                            if state.uploaded_path and Path(state.uploaded_path).exists():
                                upload_status.text = f'✓ {state.uploaded_name}'
                            
                            # Upload tabs
                            with ui.tabs().classes('w-full mb-2').style('min-height:24px;') as upload_tabs:
                                upload_tab = ui.tab('Upload')
                                url_tab = ui.tab('URL')
                            
                            with ui.tab_panels(upload_tabs, value=upload_tab).classes('w-full'):
                                
                                # Upload panel
                                with ui.tab_panel(upload_tab).classes('p-0'):
                                    async def handle_upload(e):
                                        try:
                                            log_event("Upload started")
                                            content = None
                                            fname = 'document.pdf'
                                            
                                            if hasattr(e, 'content'):
                                                if hasattr(e.content, 'read'):
                                                    content = await e.content.read()
                                                else:
                                                    content = e.content
                                                fname = getattr(e, 'name', fname)
                                            
                                            if not content or len(content) == 0:
                                                ui.notify('Empty file', type='warning')
                                                return
                                            
                                            if not isinstance(content, bytes):
                                                ui.notify('Invalid format', type='warning')
                                                return
                                            
                                            if content[:4] != b'%PDF':
                                                ui.notify('Not a valid PDF', type='warning')
                                                return
                                            
                                            tmp = Path(tempfile.mkdtemp()) / fname
                                            tmp.write_bytes(content)
                                            
                                            state.uploaded_path = str(tmp)
                                            state.uploaded_name = fname
                                            state.translated_path = None
                                            
                                            upload_status.text = f'✓ {fname}'
                                            preview_html.set_content(get_preview(str(tmp), 0))
                                            state._source_page = 0
                                            update_source_page_label()
                                            translated_preview.set_content('<div style="padding:40px;text-align:center;color:#888;">No translation yet</div>')
                                            
                                            log_event(f"Uploaded: {fname}")
                                            ui.notify(f'Loaded: {fname}', type='positive')
                                        except Exception as ex:
                                            log_event(f"Upload error: {ex}", "ERROR")
                                            ui.notify(f'Error: {str(ex)[:50]}', type='negative')
                                    
                                    ui.upload(
                                        on_upload=handle_upload,
                                        auto_upload=True,
                                        max_files=1
                                    ).props('accept=".pdf" label="📄 Drop PDF or click to browse"').classes('w-full')
                                
                                # URL panel
                                with ui.tab_panel(url_tab).classes('p-0'):
                                    with ui.row().classes('w-full gap-2 items-center'):
                                        url_input = ui.input(placeholder='https://arxiv.org/pdf/...').classes('flex-grow').props('dense')
                                        
                                        async def fetch_url():
                                            url = url_input.value.strip()
                                            if not url:
                                                ui.notify('Enter URL', type='warning')
                                                return
                                            
                                            try:
                                                upload_status.text = 'Fetching...'
                                                import urllib.request, ssl
                                                
                                                log_event(f"Fetching: {url}")
                                                tmp = Path(tempfile.mkdtemp())
                                                fname = url.split('/')[-1].split('?')[0] or 'document.pdf'
                                                if not fname.endswith('.pdf'): fname += '.pdf'
                                                fpath = tmp / fname
                                                
                                                ctx = ssl.create_default_context()
                                                ctx.check_hostname = False
                                                ctx.verify_mode = ssl.CERT_NONE
                                                
                                                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                                                
                                                def download():
                                                    with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
                                                        return resp.read()
                                                
                                                loop = asyncio.get_event_loop()
                                                content = await loop.run_in_executor(None, download)
                                                fpath.write_bytes(content)
                                                
                                                if content[:4] != b'%PDF':
                                                    raise ValueError("Not a valid PDF")
                                                
                                                state.uploaded_path = str(fpath)
                                                state.uploaded_name = fname
                                                state.translated_path = None
                                                
                                                upload_status.text = f'✓ {fname}'
                                                preview_html.set_content(get_preview(str(fpath), 0))
                                                state._source_page = 0
                                                update_source_page_label()
                                                
                                                log_event(f"Downloaded: {fname}")
                                                ui.notify('Downloaded', type='positive')
                                            except Exception as ex:
                                                upload_status.text = 'Failed'
                                                log_event(f"URL error: {ex}", "ERROR")
                                                ui.notify(f'Error: {str(ex)[:50]}', type='negative')
                                        
                                        ui.button('Fetch', on_click=fetch_url, icon='download').props('dense size=sm')
                        
                        # Translation Settings Card
                        with ui.element('div').classes('card'):
                            ui.label('Translation Settings').classes('card-title')
                            
                            # Direction toggle
                            with ui.row().classes('w-full gap-2 mb-2'):
                                def set_dir(d):
                                    state.direction = d
                                    dir_en_fr.props('color=primary' if d == 'en-fr' else '')
                                    dir_fr_en.props('color=primary' if d == 'fr-en' else '')
                                
                                dir_en_fr = ui.button('EN → FR', on_click=lambda: set_dir('en-fr')).props('size=sm' + (' color=primary' if state.direction == 'en-fr' else ''))
                                dir_fr_en = ui.button('FR → EN', on_click=lambda: set_dir('fr-en')).props('size=sm' + (' color=primary' if state.direction == 'fr-en' else ''))
                            
                            engine_sel = ui.select(get_engines(), value=state.engine, label='Engine').classes('w-full').props('dense')
                            engine_sel.on('update:model-value', lambda e: setattr(state, 'engine', e.args))
                            
                            with ui.row().classes('w-full gap-2'):
                                pages_sel = ui.select(
                                    {'all': 'All', '1': '1', '2': '1-2', '3': '1-3', '5': '1-5', '10': '1-10'},
                                    value='all', label='Pages'
                                ).classes('flex-grow').props('dense')
                                pages_custom = ui.input(placeholder='1-5', label='Custom').classes('w-24').props('dense')
                            
                            quality_sel = ui.select(
                                {1: '1 pass', 2: '2 passes', 3: '3 passes'},
                                value=2, label='Quality'
                            ).classes('w-full').props('dense')
                            
                            candidates_sel = ui.select(
                                {1: '1 candidate', 3: '3 candidates', 5: '5 candidates'},
                                value=3, label='Candidates'
                            ).classes('w-full').props('dense')
                        
                        # Options Card
                        with ui.element('div').classes('card'):
                            with ui.expansion('Advanced', icon='tune').classes('w-full').props('dense'):
                                mask_chk = ui.checkbox('Mask formulas/URLs', value=True).props('dense')
                                struct_chk = ui.checkbox('Preserve numbering', value=True).props('dense')
                                tables_chk = ui.checkbox('Preserve tables', value=True).props('dense')
                                show_prompt_chk = ui.checkbox('Show LLM prompt', value=False).props('dense')
                        
                        # Translate Button
                        translate_btn = ui.button('Translate', icon='translate').classes('w-full').props('color=primary')
                    
                    # RIGHT PANEL - Preview & Actions
                    with ui.element('div').classes('panel').style('max-width:55%;'):
                        
                        # Preview Card
                        with ui.element('div').classes('card preview-card'):
                            ui.label('Preview').classes('card-title')
                            
                            with ui.tabs().classes('w-full').style('min-height:24px;') as preview_tabs:
                                src_tab = ui.tab('Source')
                                trans_tab = ui.tab('Translated')
                            
                            with ui.tab_panels(preview_tabs, value=src_tab).style('flex:1;min-height:0;overflow:hidden;'):
                                
                                # Source preview
                                with ui.tab_panel(src_tab).classes('p-0').style('height:100%;display:flex;flex-direction:column;'):
                                    with ui.element('div').classes('preview-area'):
                                        initial_preview = '<div style="padding:40px;text-align:center;color:#888;">Upload a document</div>'
                                        if state.uploaded_path and Path(state.uploaded_path).exists():
                                            initial_preview = get_preview(state.uploaded_path, 0)
                                        preview_html = ui.html(initial_preview, sanitize=False)
                                    
                                    with ui.row().classes('pagination-row'):
                                        def prev_src():
                                            if state._source_page > 0:
                                                state._source_page -= 1
                                                if state.uploaded_path:
                                                    preview_html.set_content(get_preview(state.uploaded_path, state._source_page))
                                                update_source_page_label()
                                        
                                        def next_src():
                                            if state.uploaded_path:
                                                import fitz
                                                doc = fitz.open(state.uploaded_path)
                                                mx = len(doc)
                                                doc.close()
                                                if state._source_page < mx - 1:
                                                    state._source_page += 1
                                                    preview_html.set_content(get_preview(state.uploaded_path, state._source_page))
                                                update_source_page_label()
                                        
                                        ui.button(icon='chevron_left', on_click=prev_src).props('flat dense')
                                        src_page_label = ui.label('Page 1/1').classes('text-sm')
                                        ui.button(icon='chevron_right', on_click=next_src).props('flat dense')
                                        
                                        def update_source_page_label():
                                            if state.uploaded_path and Path(state.uploaded_path).exists():
                                                try:
                                                    import fitz
                                                    doc = fitz.open(state.uploaded_path)
                                                    mx = len(doc)
                                                    doc.close()
                                                    src_page_label.text = f'Page {state._source_page+1}/{mx}'
                                                except:
                                                    src_page_label.text = f'Page {state._source_page+1}'
                                        
                                        update_source_page_label()
                                
                                # Translated preview
                                with ui.tab_panel(trans_tab).classes('p-0').style('height:100%;display:flex;flex-direction:column;'):
                                    with ui.element('div').classes('preview-area'):
                                        trans_initial = '<div style="padding:40px;text-align:center;color:#888;">No translation yet</div>'
                                        if state.translated_path and Path(state.translated_path).exists():
                                            trans_initial = get_preview(state.translated_path, 0)
                                        translated_preview = ui.html(trans_initial, sanitize=False)
                                    
                                    with ui.row().classes('pagination-row'):
                                        def prev_trans():
                                            if state._translated_page > 0:
                                                state._translated_page -= 1
                                                if state.translated_path:
                                                    translated_preview.set_content(get_preview(state.translated_path, state._translated_page))
                                                update_trans_page_label()
                                        
                                        def next_trans():
                                            if state.translated_path:
                                                import fitz
                                                doc = fitz.open(state.translated_path)
                                                mx = len(doc)
                                                doc.close()
                                                if state._translated_page < mx - 1:
                                                    state._translated_page += 1
                                                    translated_preview.set_content(get_preview(state.translated_path, state._translated_page))
                                                update_trans_page_label()
                                        
                                        ui.button(icon='chevron_left', on_click=prev_trans).props('flat dense')
                                        trans_page_label = ui.label('Page 1/1').classes('text-sm')
                                        ui.button(icon='chevron_right', on_click=next_trans).props('flat dense')
                                        
                                        def update_trans_page_label():
                                            if state.translated_path and Path(state.translated_path).exists():
                                                try:
                                                    import fitz
                                                    doc = fitz.open(state.translated_path)
                                                    mx = len(doc)
                                                    doc.close()
                                                    trans_page_label.text = f'Page {state._translated_page+1}/{mx}'
                                                except:
                                                    trans_page_label.text = f'Page {state._translated_page+1}'
                        
                        # Progress Card
                        with ui.element('div').classes('card'):
                            ui.label('Progress').classes('card-title')
                            prog_bar = ui.linear_progress(value=0, show_value=False).classes('w-full')
                            prog_text = ui.label('Ready').classes('text-xs opacity-70')
                            log_area = ui.textarea().props('readonly rows=3 dense').classes('w-full text-xs font-mono log-area')
                        
                        # Actions Card
                        with ui.element('div').classes('card'):
                            with ui.row().classes('action-row'):
                                download_btn = ui.button('Download', icon='download').props('color=positive size=sm disabled')
                                new_btn = ui.button('New', icon='refresh').props('outline size=sm disabled')
                                tweak_btn = ui.button('Tweak', icon='tune').props('outline size=sm disabled')
                
                # Translation logic
                async def do_translate():
                    if not state.uploaded_path:
                        ui.notify('Upload a document first', type='warning')
                        return
                    
                    translate_btn.disable()
                    download_btn.props('disabled')
                    new_btn.props('disabled')
                    tweak_btn.props('disabled')
                    prog_bar.value = 0
                    translated_preview.set_content('<div style="padding:40px;text-align:center;color:#888;">Translating...</div>')
                    
                    logs = []
                    
                    def log(m):
                        try:
                            logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {m}")
                            log_area.value = '\n'.join(logs[-5:])
                            log_event(m)
                        except:
                            pass
                    
                    try:
                        log("Starting...")
                        prog_bar.value = 0.1
                        prog_text.text = "Parsing..."
                        
                        inp = Path(state.uploaded_path)
                        out = Path(tempfile.mkdtemp()) / f'translated_{inp.name}'
                        
                        pages_val = pages_sel.value
                        if pages_custom.value.strip():
                            pages_val = pages_custom.value.strip()
                        
                        log(f"Engine: {engine_sel.value}, Direction: {state.direction}")
                        
                        def cb(m):
                            try:
                                log(m)
                                if 'pars' in m.lower(): prog_bar.value = 0.2
                                elif 'translat' in m.lower() and 'block' in m.lower():
                                    prog_bar.value = 0.5
                                    prog_text.text = "Translating..."
                                elif 'render' in m.lower(): prog_bar.value = 0.85
                            except:
                                pass
                        
                        await asyncio.sleep(0.1)
                        prog_bar.value = 0.25
                        prog_text.text = "Translating..."
                        
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(None, lambda: translate_document(
                            input_path=str(inp),
                            output_path=str(out),
                            engine=engine_sel.value,
                            direction=state.direction,
                            pages=pages_val,
                            quality_loops=int(quality_sel.value),
                            enable_rerank=True,
                            use_mineru=True,
                            num_candidates=int(candidates_sel.value),
                            progress=cb
                        ))
                        
                        prog_bar.value = 1.0
                        
                        if result.success:
                            log(f"Stats: {result.stats.get('translated_blocks', 0)} blocks translated")
                            log("Complete!")
                            prog_text.text = "Complete"
                            
                            if out.exists():
                                state.translated_path = str(out)
                                download_btn.props(remove='disabled')
                                new_btn.props(remove='disabled')
                                tweak_btn.props(remove='disabled')
                                translated_preview.set_content(get_preview(str(out), 0))
                                state._translated_page = 0
                                update_trans_page_label()
                            
                            ui.notify('Translation complete', type='positive')
                        else:
                            log(f"Errors: {result.errors[:2]}")
                            prog_text.text = "Completed with errors"
                    except Exception as ex:
                        log(f"Error: {ex}")
                        prog_text.text = "Error"
                        log_event(f"Translation error: {ex}", "ERROR")
                        ui.notify(f'Error: {str(ex)[:60]}', type='negative')
                    finally:
                        translate_btn.enable()
                
                translate_btn.on_click(do_translate)
                download_btn.on_click(lambda: ui.download(state.translated_path) if state.translated_path else None)
                
                def do_new():
                    state.translated_path = None
                    translated_preview.set_content('<div style="padding:40px;text-align:center;color:#888;">No translation yet</div>')
                    download_btn.props('disabled')
                    new_btn.props('disabled')
                    tweak_btn.props('disabled')
                    prog_bar.value = 0
                    prog_text.text = 'Ready'
                    log_area.value = ''
                    ui.notify('Ready for new translation', type='info')
                
                new_btn.on_click(do_new)
                tweak_btn.on_click(lambda: ui.notify('Adjust settings and click Translate', type='info'))
            
            # ==================== TESTING TAB ====================
            with ui.tab_panel(t_testing).classes('p-0'):
                with ui.element('div').classes('main-row'):
                    with ui.element('div').classes('panel'):
                        with ui.element('div').classes('card'):
                            ui.label('Text Translation Test').classes('card-title')
                            test_input = ui.textarea(label='Source', placeholder='Enter text...').props('rows=8').classes('w-full')
                            with ui.row().classes('w-full gap-2 mt-2'):
                                test_dir = ui.select({'en-fr': 'EN → FR', 'fr-en': 'FR → EN'}, value='en-fr', label='Direction').classes('w-1/3').props('dense')
                                test_eng = ui.select(get_engines(), value='dictionary', label='Engine').classes('w-1/3').props('dense')
                            with ui.row().classes('w-full gap-2 mt-2'):
                                test_run = ui.button('Translate', icon='translate').props('color=primary size=sm')
                                test_clear = ui.button('Clear').props('outline size=sm')
                    
                    with ui.element('div').classes('panel'):
                        with ui.element('div').classes('card'):
                            ui.label('Result').classes('card-title')
                            test_output = ui.textarea(label='Translation').props('readonly rows=8').classes('w-full')
                            test_status = ui.label('').classes('text-xs opacity-70')
                        
                        with ui.element('div').classes('card'):
                            ui.label('Metrics').classes('card-title')
                            with ui.row().classes('gap-4'):
                                with ui.column():
                                    ui.label('BLEU').classes('text-xs opacity-70')
                                    bleu_lbl = ui.label('--').classes('text-lg font-bold')
                                with ui.column():
                                    ui.label('Terms').classes('text-xs opacity-70')
                                    terms_lbl = ui.label('--').classes('text-lg font-bold')
                
                def run_test():
                    if not test_input.value:
                        ui.notify('Enter text', type='warning')
                        return
                    test_status.text = 'Translating...'
                    try:
                        s, t = test_dir.value.split('-')
                        config = PipelineConfig(
                            source_lang=s, target_lang=t,
                            translator_backend=test_eng.value,
                            enable_masking=True, enable_glossary=True
                        )
                        pipeline = TranslationPipeline(config)
                        doc = Document.from_text(test_input.value, s, t)
                        result = pipeline.translate(doc)
                        test_output.value = result.translated_text
                        terms_lbl.text = str(result.stats.get('refined_blocks', 0))
                        test_status.text = 'Done'
                    except Exception as ex:
                        test_output.value = f'Error: {ex}'
                        test_status.text = 'Error'
                
                test_run.on_click(run_test)
                test_clear.on_click(lambda: (setattr(test_input, 'value', ''), setattr(test_output, 'value', '')))
            
            # ==================== GLOSSARY TAB ====================
            with ui.tab_panel(t_glossary).classes('p-0'):
                with ui.element('div').style('padding:8px;'):
                    with ui.element('div').classes('card'):
                        ui.label('Scientific Glossary').classes('card-title')
                        search_input = ui.input(label='Search').classes('w-full mb-2').props('dense')
                        try:
                            g = get_default_glossary()
                            all_entries = [{'en': e.source, 'fr': e.target, 'domain': e.domain} for e in g.entries]
                        except:
                            all_entries = []
                        
                        gloss_table = ui.table(
                            columns=[
                                {'name': 'en', 'label': 'English', 'field': 'en', 'sortable': True},
                                {'name': 'fr', 'label': 'French', 'field': 'fr', 'sortable': True},
                                {'name': 'domain', 'label': 'Domain', 'field': 'domain'}
                            ],
                            rows=all_entries[:50],
                            row_key='en',
                            pagination={'rowsPerPage': 20}
                        ).classes('w-full').style('max-height:calc(100vh - 200px);')
                        
                        def filter_gloss():
                            s = search_input.value.lower()
                            gloss_table.rows = [e for e in all_entries if not s or s in e['en'].lower() or s in e['fr'].lower()][:50]
                        
                        search_input.on('keyup', filter_gloss)
            
            # ==================== DEVELOPER TAB ====================
            with ui.tab_panel(t_developer).classes('p-0'):
                with ui.element('div').classes('main-row'):
                    with ui.element('div').classes('panel'):
                        with ui.element('div').classes('card').style('flex:1;display:flex;flex-direction:column;'):
                            ui.label('System Logs').classes('card-title')
                            with ui.row().classes('gap-2 mb-2'):
                                def refresh_logs(): dev_logs.value = '\n'.join(system_logs[-100:])
                                def clear_logs(): system_logs.clear(); dev_logs.value = ''
                                def export_logs():
                                    f = Path(tempfile.mktemp(suffix='.log'))
                                    f.write_text('\n'.join(system_logs))
                                    ui.download(str(f), 'logs.log')
                                ui.button('Refresh', on_click=refresh_logs).props('dense outline size=sm')
                                ui.button('Clear', on_click=clear_logs).props('dense outline size=sm')
                                ui.button('Export', on_click=export_logs).props('dense outline size=sm')
                            dev_logs = ui.textarea().props('readonly').classes('w-full font-mono text-xs log-area').style('flex:1;')
                            dev_logs.value = '\n'.join(system_logs[-50:])
                            
                            # Auto-refresh timer
                            def auto_refresh():
                                dev_logs.value = '\n'.join(system_logs[-50:])
                            ui.timer(2.0, auto_refresh)
                    
                    with ui.element('div').classes('panel'):
                        with ui.element('div').classes('card'):
                            ui.label('System Info').classes('card-title')
                            import sys, platform
                            for k, v in [
                                ('Python', f'{sys.version_info.major}.{sys.version_info.minor}'),
                                ('Platform', platform.system()),
                                ('Machine', platform.machine())
                            ]:
                                with ui.row().classes('w-full justify-between py-1'):
                                    ui.label(k).classes('opacity-70')
                                    ui.label(v).classes('font-mono text-sm')
                        
                        with ui.element('div').classes('card'):
                            ui.label('Backends').classes('card-title')
                            for name, status in [
                                ('free', 'Ready'),
                                ('dictionary', 'Ready'),
                                ('ollama', 'Ready'),
                                ('openai', 'Ready' if km.get_key('openai') else 'Needs key'),
                                ('deepseek', 'Ready' if km.get_key('deepseek') else 'Needs key'),
                            ]:
                                with ui.row().classes('w-full justify-between py-1'):
                                    ui.label(name).classes('font-mono')
                                    ui.label(status).classes('text-xs ' + ('text-green-400' if 'Ready' in status else 'text-yellow-400'))
            
            # ==================== SETTINGS TAB ====================
            with ui.tab_panel(t_settings).classes('p-0'):
                with ui.element('div').style('padding:8px;max-width:700px;margin:auto;'):
                    with ui.element('div').classes('card'):
                        ui.label('API Keys').classes('card-title')
                        ui.label('Configure keys for premium services').classes('text-xs opacity-70 mb-2')
                        
                        for svc in ['openai', 'deepseek', 'anthropic']:
                            k = km.get_key(svc)
                            with ui.row().classes('w-full justify-between py-1'):
                                ui.label(svc.capitalize())
                                ui.label('✓' if k else '✗').classes('text-green-400' if k else 'text-red-400')
                        
                        ui.separator().classes('my-2')
                        key_svc = ui.select(['openai', 'deepseek', 'anthropic'], value='openai', label='Service').classes('w-full').props('dense')
                        key_val = ui.input(label='API Key', password=True).classes('w-full').props('dense')
                        key_msg = ui.label('').classes('text-xs')
                        
                        def save_key():
                            if not key_val.value.strip():
                                key_msg.text = 'Enter a key'
                                return
                            try:
                                storage_loc = km.set_key(key_svc.value, key_val.value.strip())
                                key_msg.text = f'Saved to {storage_loc}'
                                key_val.value = ''
                                log_event(f"Key saved: {key_svc.value}")
                                ui.notify('Saved', type='positive')
                            except Exception as ex:
                                key_msg.text = f'Error: {ex}'
                        
                        ui.button('Save', on_click=save_key).classes('w-full mt-2').props('color=primary')
                    
                    with ui.element('div').classes('card'):
                        ui.label('Appearance').classes('card-title')
                        def on_dark(e):
                            if e.value: dark.enable()
                            else: dark.disable()
                            state.dark_mode = e.value
                        ui.switch('Dark mode', value=state.dark_mode, on_change=on_dark)
                    
                    with ui.element('div').classes('card'):
                        ui.label('About').classes('card-title')
                        ui.markdown('''
**SciTrans-LLMs** v1.0

Scientific document translation with:
- Layout-preserving PDF translation
- Terminology management
- Multiple translation backends
- BLEU and glossary metrics
                        ''').classes('text-sm opacity-80')
    
    log_event("GUI ready")
    print(f"\n{'='*50}\nSciTrans-LLMs GUI: http://127.0.0.1:{port}\n{'='*50}")
    
    ui.run(
        port=port,
        title='SciTrans-LLMs',
        favicon='🔬',
        show=True,
        reload=False,
        storage_secret='scitrans_session_key_2024',
        reconnect_timeout=120,
    )

if __name__ == "__main__":
    launch()
