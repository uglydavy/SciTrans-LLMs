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
        # Use free backend for better quality by default
        default_engine = 'free'
        default_masking = True  # ALWAYS enable masking
        default_reranking = True  # ALWAYS enable reranking
        quality_passes = 2  # Multiple passes for better quality
        context_window = 8  # Larger context for better coherence
        translate_tables = True  # ALWAYS translate tables
        translate_figures = True  # ALWAYS translate figures
        preserve_structure = True  # ALWAYS preserve structure
        uploaded_pdf_path = None
        uploaded_pdf_name = None
        translated_pdf_path = None
        custom_glossary = None
    
    state = State()
    
    def get_engines():
        e = ['free', 'dictionary', 'improved-offline', 'huggingface', 'ollama']
        if km.get_key('openai'): e.append('openai')
        if km.get_key('deepseek'): e.append('deepseek')
        if km.get_key('anthropic'): e.append('anthropic')
        return list(set(e))
    
    def get_preview(path: str, page_num: int = 0) -> str:
        """Get PDF preview as base64 image, constrained to fit preview area."""
        try:
            import fitz
            doc = fitz.open(path)
            if page_num < len(doc):
                # Use 2x zoom for better quality, CSS will constrain size
                pix = doc[page_num].get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                b64 = base64.b64encode(pix.tobytes("png")).decode()
                doc.close()
                # Inline style ensures image fits in container
                return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;max-height:calc(100vh - 450px);width:auto;height:auto;object-fit:contain;display:block;margin:auto;"/>'
            doc.close()
        except Exception as ex:
            log_event(f"Preview error: {ex}", "ERROR")
        return '<div style="padding:20px;text-align:center;color:#888;font-size:11px;">No preview available</div>'
    
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
/* ========== ROOT VARIABLES ========== */
:root { 
  --bg-card: rgba(30,35,45,0.95); 
  --border: rgba(100,100,120,0.3); 
  --font-base: 15px;
  --font-title: 16px;
  --font-label: 14px;
  --btn-height: 44px;
  --card-padding: 16px;
  --gap: 12px;
}

/* ========== BASE LAYOUT ========== */
html, body { 
  margin:0; 
  padding:0; 
  overflow-x:hidden!important; 
  height:100vh!important; 
  font-size:var(--font-base);
}
.nicegui-content, .q-page-container { 
  height:calc(100vh - 56px)!important; 
  overflow-x:hidden!important; 
}
.q-tab-panels { 
  height:100%!important; 
  overflow:hidden!important; 
  padding:0!important; 
}
.q-tab-panel { 
  height:100%!important; 
  overflow-y:auto!important; 
  overflow-x:hidden!important;
  padding:0!important; 
}

/* ========== MAIN LAYOUT ========== */
.main-row { 
  display:flex; 
  width:100%; 
  min-height:calc(100vh - 110px); 
  padding:var(--gap); 
  gap:var(--gap); 
  box-sizing:border-box; 
}

/* ========== PANELS ========== */
.panel { 
  flex:1; 
  display:flex; 
  flex-direction:column; 
  gap:var(--gap); 
  min-width:0; 
  max-width:50%;
}
.panel.left-panel { max-width:45%; }
.panel.right-panel { max-width:55%; }
.panel::-webkit-scrollbar { width:8px; } 
.panel::-webkit-scrollbar-thumb { background:#555; border-radius:4px; }

/* ========== CARDS ========== */
.card { 
  background:var(--bg-card); 
  border:1px solid var(--border); 
  border-radius:8px; 
  padding:var(--card-padding); 
  flex-shrink:0; 
}
.card-title { 
  font-weight:600; 
  font-size:var(--font-title); 
  margin-bottom:12px; 
  border-bottom:1px solid var(--border); 
  padding-bottom:8px; 
  color:#fff;
}

/* ========== UPLOAD ZONE ========== */
.upload-zone { 
  border:3px dashed var(--border); 
  border-radius:8px; 
  padding:32px; 
  text-align:center; 
  cursor:pointer; 
  transition:all 0.3s; 
  min-height:140px; 
  display:flex; 
  flex-direction:column; 
  align-items:center; 
  justify-content:center;
  font-size:var(--font-label);
  position:relative;
  background:rgba(99,102,241,0.05);
}
.upload-zone:hover { 
  border-color:#6366f1; 
  background:rgba(99,102,241,0.15); 
  transform:scale(1.01);
}
.upload-zone .upload-icon { font-size:36px!important; opacity:0.6; margin-bottom:8px; }
.upload-zone .upload-label { font-size:14px; opacity:0.8; }

/* ========== PREVIEW ========== */
.preview-card { 
  flex:1!important; 
  display:flex; 
  flex-direction:column; 
  min-height:0; 
  overflow:hidden; 
}
.preview-area { 
  flex:1; 
  display:flex; 
  align-items:center; 
  justify-content:center; 
  overflow:auto;
  background:#1a1a2e; 
  min-height:400px;
  max-height:calc(100vh - 380px);
  border-radius:8px; 
  position:relative;
}
.preview-area img { 
  max-width:100%!important; 
  max-height:calc(100vh - 420px)!important; 
  width:auto!important;
  height:auto!important;
  object-fit:contain!important; 
  display:block;
  margin:auto;
}

/* ========== BUTTONS & INPUTS ========== */
.q-btn { 
  min-height:var(--btn-height)!important; 
  font-size:var(--font-label)!important; 
  font-weight:500!important;
  padding:0 20px!important;
}
.action-row { 
  display:flex; 
  gap:var(--gap); 
  margin-top:var(--gap); 
}
.action-row .q-btn { 
  flex:1;
}
input, select, textarea, .q-field { 
  font-size:var(--font-label)!important;
}

/* ========== PAGINATION ========== */
.pagination-btn { 
  min-width:44px!important;
  min-height:44px!important;
  font-size:16px!important;
  padding:8px!important;
}
.pagination-row { 
  display:flex; 
  align-items:center; 
  gap:8px; 
  justify-content:center; 
  padding:8px 0;
}

/* ========== LIGHT MODE ========== */
body.body--light { 
  --bg-card: rgba(255,255,255,0.98); 
  --border: rgba(0,0,0,0.12); 
}
body.body--light .q-header { background:#4f46e5!important; }
body.body--light .q-tab-panels { background:#f5f5f5!important; }
body.body--light .preview-area { background:#e8e8e8; }
body.body--light .card-title { color:#000; }

/* ========== SCROLLBAR STYLING ========== */
*::-webkit-scrollbar { width:10px; height:10px; }
*::-webkit-scrollbar-track { background:#1a1a2e; }
*::-webkit-scrollbar-thumb { background:#555; border-radius:5px; }
*::-webkit-scrollbar-thumb:hover { background:#777; }

/* ========== TEXT AREAS ========== */
textarea, .q-textarea { 
  min-height:180px!important; 
  font-family:monospace!important; 
}
textarea.log-area { 
  min-height:120px!important;
  overflow-y:auto!important;
}
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
            ui.button(icon='dark_mode', on_click=toggle_dark).props('flat round dense text-color=white size=md').tooltip('Toggle dark mode')
        
        # Tabs with icons
        with ui.tabs().classes('w-full bg-gray-800 text-white') as tabs:
            t_translate = ui.tab('Translate', icon='translate')
            t_testing = ui.tab('Testing', icon='science')
            t_glossary = ui.tab('Glossary', icon='book')
            t_developer = ui.tab('Developer', icon='code')
            t_settings = ui.tab('Settings', icon='settings')
        
        with ui.tab_panels(tabs, value=t_translate).classes('w-full flex-grow'):
            
            # ==================== TRANSLATE TAB ====================
            with ui.tab_panel(t_translate).classes('p-0').style('height:100%; overflow:hidden;'):
                with ui.element('div').classes('main-row'):
                    # LEFT PANEL
                    with ui.element('div').classes('panel left-panel'):
                        # Source Document with sliding tabs
                        with ui.element('div').classes('card'):
                            ui.label('Source Document').classes('card-title')
                            upload_status = ui.label('No file selected').classes('text-xs opacity-70 mb-2')
                            
                            # Upload handler function
                            async def handle_upload(e):
                                try:
                                    # Handle NiceGUI upload event
                                    content = None
                                    fname = 'document.pdf'
                                    
                                    # Method 1: e.content (most common)
                                    if hasattr(e, 'content'):
                                        if hasattr(e.content, 'read'):
                                            content = await e.content.read()
                                        else:
                                            content = e.content
                                        fname = getattr(e, 'name', fname)
                                    # Method 2: e.files
                                    elif hasattr(e, 'files'):
                                        files = e.files if isinstance(e.files, list) else [e.files]
                                        if files and len(files) > 0:
                                            f = files[0]
                                            if hasattr(f, 'read'):
                                                content = await f.read()
                                            else:
                                                content = f
                                            fname = getattr(f, 'name', fname)
                                    # Method 3: Direct read
                                    else:
                                        try:
                                            if hasattr(e, 'read'):
                                                content = await e.read()
                                            else:
                                                content = e
                                        except:
                                            pass
                                    
                                    if not content:
                                        ui.notify('Empty file received', type='warning')
                                        return
                                    
                                    # Ensure content is bytes
                                    if not isinstance(content, bytes):
                                        ui.notify('Invalid file format - must be PDF', type='warning')
                                        return
                                    
                                    if len(content) == 0:
                                        ui.notify('Empty file received', type='warning')
                                        return
                                    
                                    # Verify it's a valid PDF by checking magic bytes
                                    if content[:4] != b'%PDF':
                                        ui.notify('Invalid PDF file', type='warning')
                                        return
                                    
                                    tmp = Path(tempfile.mkdtemp()) / fname
                                    tmp.write_bytes(content)
                                    
                                    # Verify file was written correctly
                                    if not tmp.exists() or tmp.stat().st_size == 0:
                                        ui.notify('Failed to save file', type='warning')
                                        return
                                    
                                    state.uploaded_pdf_path = str(tmp)
                                    state.uploaded_pdf_name = fname
                                    upload_status.text = f'✓ Loaded: {fname}'
                                    
                                    # Reset translated preview
                                    state.translated_pdf_path = None
                                    translated_preview_html.set_content('<div style="padding:20px;text-align:center;color:#888;">No translation yet</div>')
                                    
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
                            
                            # Sliding tabs for Upload/URL
                            with ui.tabs().classes('w-full mb-2').style('min-height:28px;') as upload_tabs:
                                upload_tab = ui.tab('Upload', icon='upload_file')
                                url_tab = ui.tab('URL', icon='link')
                            
                            with ui.tab_panels(upload_tabs, value=upload_tab).classes('w-full'):
                                # Upload tab
                                with ui.tab_panel(upload_tab).classes('p-0'):
                                    with ui.element('div').classes('upload-zone').style('position:relative; min-height:140px; cursor:pointer;') as upload_zone:
                                        ui.icon('cloud_upload', size='lg').classes('upload-icon')
                                        ui.label('Drop PDF here or click to browse').classes('upload-label')
                                        ui.label('Supports .pdf files up to 50MB').classes('text-xs opacity-50 mt-1')
                                    upload_comp = ui.upload(
                                        on_upload=handle_upload,
                                        auto_upload=True,
                                        max_files=1
                                    ).props('accept=".pdf"').style('position:absolute; inset:0; opacity:0; cursor:pointer; z-index:10;').classes('w-full h-full')
                                    # Make entire zone clickable
                                    upload_zone.on('click', lambda e: upload_comp.run_method('pickFiles'))
                                
                                # URL tab
                                with ui.tab_panel(url_tab).classes('p-0'):
                                    with ui.row().classes('w-full gap-2 items-center'):
                                        url_input = ui.input(placeholder='https://arxiv.org/pdf/...').classes('flex-grow').props('dense')
                                        
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
                                                translated_preview_html.set_content('<div style="padding:20px;text-align:center;color:#888;">No translation yet</div>')
                                                
                                                preview_html.set_content(get_preview(str(fpath), 0))
                                                current_page_num.value = 0
                                                update_page_count()
                                                
                                                log_event(f"Downloaded: {fname}")
                                                ui.notify('PDF downloaded', type='positive')
                                            except Exception as ex:
                                                upload_status.text = 'Download failed'
                                                log_event(f"URL error: {ex}", "ERROR")
                                                ui.notify(f'Failed: {str(ex)[:50]}', type='negative')
                                        
                                        ui.button('Fetch', on_click=fetch_url, icon='download').props('dense size=sm')
                        
                        # Translation Settings
                        with ui.element('div').classes('card'):
                            ui.label('Translation Settings').classes('card-title')
                            
                            # Direction as toggle buttons
                            with ui.row().classes('w-full gap-2 mb-2'):
                                direction_en_fr = ui.button('EN → FR', on_click=lambda: set_direction('en-fr')).props('toggle size=sm')
                                direction_fr_en = ui.button('FR → EN', on_click=lambda: set_direction('fr-en')).props('toggle size=sm')
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
                            
                            engine = ui.select(get_engines(), value=state.default_engine, label='Engine').classes('w-full').props('dense')
                            
                            # Pages selection: dropdown + input
                            with ui.row().classes('w-full gap-2'):
                                pages_dropdown = ui.select(
                                    {'all': 'All pages', **{str(i): f'Pages 1-{i}' for i in range(1, 21)}},
                                    value='all',
                                    label='Pages'
                                ).classes('flex-grow').props('dense')
                                
                                pages_custom = ui.input(placeholder='e.g., 1-5, 1,3,5', label='Custom').classes('w-32').props('dense')
                            
                            quality = ui.select(
                                {1: '1 pass', 2: '2 passes', 3: '3 passes', 4: '4 passes', 5: '5 passes'},
                                value=state.quality_passes,
                                label='Quality passes'
                            ).classes('w-full').props('dense')
                            candidates = ui.select(
                                {1: '1 candidate', 3: '3 candidates', 5: '5 candidates'},
                                value=3,
                                label='Candidates per block'
                            ).classes('w-full mt-1').props('dense')
                        
                        # Advanced Settings
                        with ui.element('div').classes('card'):
                            with ui.expansion('Advanced Settings', icon='tune').classes('w-full').props('dense'):
                                ui.markdown('**Note:** Reranking is enabled by default.').classes('text-xs opacity-70 mb-1')
                                masking_chk = ui.checkbox('Mask formulas/URLs', value=state.default_masking).props('dense')
                                structure_chk = ui.checkbox('Preserve numbering', value=state.preserve_structure).props('dense')
                                tables_chk = ui.checkbox('Preserve tables', value=state.translate_tables).props('dense')
                                figures_chk = ui.checkbox('Preserve figures', value=state.translate_figures).props('dense')
                                prompt_preview_chk = ui.checkbox('Show LLM prompt for first block', value=False).props('dense')
                                ui.number(label='Context window', value=state.context_window, min=1, max=20).classes('w-full mt-1').props('dense')
                        
                        # Glossary
                        with ui.element('div').classes('card'):
                            ui.label('Custom Glossary').classes('card-title')
                            gloss_status = ui.label('Using default glossary').classes('text-xs opacity-70 mb-1')
                            
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
                            
                            ui.upload(on_upload=handle_gloss, auto_upload=True).props('accept=".csv,.txt,.json" label="Upload glossary" dense').classes('w-full')
                        
                        # Translate Button
                        translate_btn = ui.button('Translate Document', icon='translate').classes('w-full').props('color=primary size=sm')
                    
                    # RIGHT PANEL - Preview and Actions
                    with ui.element('div').classes('panel right-panel'):
                        # Preview card
                        with ui.element('div').classes('card preview-card'):
                            ui.label('Preview').classes('card-title')
                            
                            # Tabs for Source/Translated
                            with ui.tabs().classes('w-full').style('min-height:24px;') as preview_tabs:
                                source_tab = ui.tab('Source')
                                translated_tab = ui.tab('Translated')
                            
                            with ui.tab_panels(preview_tabs, value=source_tab).style('flex:1; min-height:0; overflow:hidden;'):
                                # Source preview
                                with ui.tab_panel(source_tab).classes('p-0').style('height:100%; display:flex; flex-direction:column;'):
                                    with ui.element('div').classes('preview-area'):
                                        preview_html = ui.html('<div style="padding:20px;text-align:center;color:#666;font-size:11px;">Upload a document</div>', sanitize=False)
                                    
                                    # Page navigation
                                    with ui.row().classes('pagination-row w-full').style('flex-shrink:0;'):
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
                                        
                                        ui.button(icon='chevron_left', on_click=prev_source).props('flat dense').classes('pagination-btn')
                                        current_page_num = ui.number(value=0, min=0).props('dense readonly').style('display:none;')
                                        page_label_source = ui.label('Page 0/0').classes('text-sm font-mono')
                                        ui.button(icon='chevron_right', on_click=next_source).props('flat dense').classes('pagination-btn')
                                        
                                        def update_page_count():
                                            if state.uploaded_pdf_path:
                                                try:
                                                    import fitz
                                                    doc = fitz.open(state.uploaded_pdf_path)
                                                    max_pages = len(doc)
                                                    doc.close()
                                                    page_label_source.text = f'Page {int(current_page_num.value)+1}/{max_pages}'
                                                except:
                                                    page_label_source.text = f'Page {int(current_page_num.value)+1}'
                                
                                # Translated preview
                                with ui.tab_panel(translated_tab).classes('p-0').style('height:100%; display:flex; flex-direction:column;'):
                                    with ui.element('div').classes('preview-area'):
                                        translated_preview_html = ui.html('<div style="padding:20px;text-align:center;color:#666;font-size:11px;">No translation yet</div>', sanitize=False)
                                    
                                    # Page navigation
                                    with ui.row().classes('pagination-row w-full').style('flex-shrink:0;'):
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
                                        
                                        ui.button(icon='chevron_left', on_click=prev_translated).props('flat dense').classes('pagination-btn')
                                        translated_page_num = ui.number(value=0, min=0).props('dense readonly').style('display:none;')
                                        page_label_translated = ui.label('Page 0/0').classes('text-sm font-mono')
                                        ui.button(icon='chevron_right', on_click=next_translated).props('flat dense').classes('pagination-btn')
                                        
                                        def update_translated_page_count():
                                            if state.translated_pdf_path:
                                                try:
                                                    import fitz
                                                    doc = fitz.open(state.translated_pdf_path)
                                                    max_pages = len(doc)
                                                    doc.close()
                                                    page_label_translated.text = f'Page {int(translated_page_num.value)+1}/{max_pages}'
                                                except:
                                                    page_label_translated.text = f'Page {int(translated_page_num.value)+1}'
                        
                        # Progress card
                        with ui.element('div').classes('card'):
                            ui.label('Translation Progress').classes('card-title')
                            prog_bar = ui.linear_progress(value=0, show_value=False).classes('w-full')
                            prog_text = ui.label('Ready').classes('text-xs opacity-70 mb-2')
                            log_area = ui.textarea().props('readonly rows=4 dense auto-grow').classes('w-full text-xs font-mono log-area').style('min-height:120px; max-height:180px; overflow-y:auto;')
                        
                        # Actions card
                        with ui.element('div').classes('card'):
                            with ui.row().classes('action-row'):
                                download_btn = ui.button('Download', icon='download').props('color=positive size=xs dense disabled')
                                retranslate_btn = ui.button('New', icon='refresh').props('outline size=xs dense disabled')
                                tweak_btn = ui.button('Tweak', icon='tune').props('outline size=xs dense disabled')
                
                # Translate logic
                async def do_translate():
                    if not state.uploaded_pdf_path:
                        ui.notify('Upload a document first', type='warning')
                        return
                    
                    try:
                        translate_btn.disable()
                        download_btn.props('disabled')
                        retranslate_btn.props('disabled')
                        tweak_btn.props('disabled')
                        prog_bar.value = 0
                        translated_preview_html.set_content('<div style="padding:20px;text-align:center;color:#888;">Translating...</div>')
                    except:
                        pass
                    
                    logs = []
                    state.translated_pdf_path = None
                    
                    def log(m):
                        try:
                            logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {m}")
                            log_area.value = '\n'.join(logs[-6:])
                            # Auto-scroll to bottom
                            try:
                                log_area.run_method('scrollTo', {'top': 99999})
                            except:
                                pass
                            log_event(m)
                        except:
                            pass
                    
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
                        
                        # Track progress to prevent connection timeout
                        progress_tracker = {'blocks_done': 0, 'total_blocks': 0}
                        
                        def cb(m):
                            try:
                                log(m)
                                log_event(m)
                                
                                # Parse block progress from messages like "Translating block 5/100"
                                if 'translating block' in m.lower():
                                    try:
                                        parts = m.split('block')[1].strip().split('/')
                                        if len(parts) == 2:
                                            progress_tracker['blocks_done'] = int(parts[0])
                                            progress_tracker['total_blocks'] = int(parts[1])
                                            # Update progress bar based on blocks
                                            pct = 0.25 + 0.6 * (progress_tracker['blocks_done'] / progress_tracker['total_blocks'])
                                            prog_bar.value = min(0.85, pct)
                                            prog_text.text = f"Translating {progress_tracker['blocks_done']}/{progress_tracker['total_blocks']}"
                                    except:
                                        pass
                                
                                if 'pars' in m.lower(): 
                                    prog_bar.value = 0.2
                                elif 'translat' in m.lower() and 'block' not in m.lower(): 
                                    prog_bar.value = 0.5
                                elif 'render' in m.lower(): 
                                    prog_bar.value = 0.85
                            except:
                                pass
                        
                        await asyncio.sleep(0.1)
                        try:
                            prog_bar.value = 0.25
                            prog_text.text = "Translating..."
                        except:
                            pass
                        
                        # Run translation with reranking enabled
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(None, lambda: translate_document(
                            input_path=str(inp),
                            output_path=str(out),
                            engine=engine.value,
                            direction=current_direction['value'],
                            pages=pages_val,
                            quality_loops=int(quality.value),
                            enable_rerank=True,
                            use_mineru=True,
                            num_candidates=int(candidates.value),
                            progress=cb
                        ))
                        
                        try:
                            prog_bar.value = 1.0
                        except:
                            pass
                        
                        if result.success:
                            try:
                                # Log basic quality-related stats
                                stats = result.stats or {}
                                total_blocks = stats.get('total_blocks', 0)
                                translated_blocks = stats.get('translated_blocks', 0)
                                refined_blocks = stats.get('refined_blocks', 0)
                                masks_applied = stats.get('masks_applied', 0)
                                log(f"Stats: {translated_blocks}/{total_blocks} blocks translated, {refined_blocks} refined, {masks_applied} masks applied")

                                # Optionally surface the LLM prompts for the first translated block
                                if prompt_preview_chk.value:
                                    meta = stats.get('first_translation_metadata') or {}
                                    sys_prompt = meta.get('system_prompt')
                                    user_prompt = meta.get('user_prompt')
                                    if sys_prompt:
                                        snippet = sys_prompt[:1200]
                                        if len(sys_prompt) > 1200:
                                            snippet += ' ...[truncated]'
                                        log('=== LLM system prompt (first block) ===')
                                        log(snippet)
                                    if user_prompt:
                                        snippet_u = user_prompt[:800]
                                        if len(user_prompt) > 800:
                                            snippet_u += ' ...[truncated]'
                                        log('=== LLM user prompt (first block) ===')
                                        log(snippet_u)

                                log("Translation complete")
                                prog_text.text = "Complete"
                                if out.exists():
                                    state.translated_pdf_path = str(out)
                                    download_btn.props(remove='disabled')
                                    retranslate_btn.props(remove='disabled')
                                    tweak_btn.props(remove='disabled')
                                    translated_preview_html.set_content(get_preview(str(out), 0))
                                    translated_page_num.value = 0
                                    update_translated_page_count()
                                ui.notify('Translation complete', type='positive')
                            except:
                                log_event("Translation complete", "INFO")
                        else:
                            try:
                                log(f"Errors: {result.errors[:2]}")
                                prog_text.text = "Completed with errors"
                            except:
                                pass
                    except Exception as ex:
                        try:
                            log(f"Error: {ex}")
                            prog_text.text = "Error"
                            log_event(f"Translation error: {ex}", "ERROR")
                            import traceback
                            log_event(traceback.format_exc(), "ERROR")
                            ui.notify(f'Translation failed: {str(ex)[:80]}', type='negative')
                        except:
                            log_event(f"Translation error: {ex}", "ERROR")
                    finally:
                        try:
                            translate_btn.enable()
                        except:
                            pass
                
                translate_btn.on_click(do_translate)
                download_btn.on_click(lambda: ui.download(state.translated_pdf_path) if state.translated_pdf_path else None)
                
                def do_retranslate():
                    """Reset and allow retranslation"""
                    state.translated_pdf_path = None
                    translated_preview_html.set_content('<div style="padding:40px;text-align:center;color:#888;">No translation yet</div>')
                    download_btn.props('disabled')
                    retranslate_btn.props('disabled')
                    tweak_btn.props('disabled')
                    prog_bar.value = 0
                    prog_text.text = 'Ready'
                    log_area.value = ''
                    ui.notify('Ready for new translation', type='info')
                
                retranslate_btn.on_click(do_retranslate)
                
                def do_tweak():
                    """Scroll to settings"""
                    ui.notify('Adjust settings above and click Translate', type='info')
                
                tweak_btn.on_click(do_tweak)
            
            # ==================== TESTING TAB ====================
            with ui.tab_panel(t_testing).classes('p-0').style('height:100%; overflow:hidden;'):
                with ui.element('div').classes('main-row'):
                    # LEFT - Input & Settings
                    with ui.element('div').classes('panel left-panel'):
                        with ui.element('div').classes('card'):
                            ui.label('Text Translation Test').classes('card-title')
                            test_input = ui.textarea(label='Source text', placeholder='Enter text to translate...').props('rows=10 auto-grow').classes('w-full').style('min-height:200px;')
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
                    with ui.element('div').classes('panel right-panel'):
                        with ui.element('div').classes('card'):
                            ui.label('Translation Output').classes('card-title')
                            test_output = ui.textarea(label='Result').props('rows=10 readonly auto-grow').classes('w-full').style('min-height:200px;')
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
                            test_ref = ui.textarea(placeholder='Optional: paste reference translation for BLEU score').props('rows=4 auto-grow').classes('w-full').style('min-height:100px;')
                        
                        with ui.element('div').classes('card'):
                            ui.label('Dictionary Training from Corpus').classes('card-title')
                            ui.markdown('''
Download parallel corpora and build dictionaries to improve offline translation quality.
The dictionary backend will automatically use trained dictionaries matching your language pair.
                            ''').classes('text-xs opacity-70 mb-2')
                            
                            corpus_train_status = ui.label('No corpus loaded').classes('text-xs opacity-70 mb-2')
                            
                            with ui.row().classes('w-full gap-2'):
                                corpus_name_sel = ui.select(
                                    {'opensubtitles': 'OpenSubtitles (80MB) - Fast', 'europarl': 'Europarl (150MB) - High Quality', 'wikipedia': 'Wikipedia (30MB) - Quick'},
                                    value='opensubtitles',
                                    label='Corpus'
                                ).classes('flex-grow')
                                corpus_pair_sel = ui.select(
                                    {'en-fr': 'EN → FR', 'fr-en': 'FR → EN'},
                                    value='en-fr',
                                    label='Language Pair'
                                ).classes('w-32')
                            
                            with ui.row().classes('w-full gap-2 mt-2'):
                                corpus_limit = ui.number(label='Max entries', value=10000, min=1000, max=100000, step=1000).classes('w-1/3')
                                corpus_download_btn = ui.button('Download & Build Dictionary').props('color=primary')
                                corpus_status_btn = ui.button('Show Status').props('outline')
                
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
                
                def corpus_download_and_build():
                    """Download corpus and build dictionary for dictionary backend"""
                    corpus_name = corpus_name_sel.value
                    pair = corpus_pair_sel.value
                    src_lang, tgt_lang = pair.split('-')
                    limit = int(corpus_limit.value)
                    
                    corpus_train_status.text = 'Downloading corpus...'
                    corpus_download_btn.props('loading')
                    
                    try:
                        from scitrans_llms.translate.corpus_manager import download_corpus, get_corpus_dictionary
                        import csv
                        
                        # Download
                        def update(msg: str, pct: float):
                            corpus_train_status.text = f"{msg} ({int(pct*100)}%)"
                        
                        corpus_path = download_corpus(corpus_name, src_lang, tgt_lang, update)
                        
                        corpus_train_status.text = 'Building dictionary...'
                        
                        # Build dictionary
                        dictionary = get_corpus_dictionary(corpus_name, src_lang, tgt_lang, limit)
                        
                        # Save to default location for DictionaryTranslator
                        dict_root = Path.home() / ".scitrans" / "dictionaries"
                        dict_root.mkdir(parents=True, exist_ok=True)
                        dict_path = dict_root / f"{corpus_name}_{src_lang}_{tgt_lang}.tsv"
                        
                        with open(dict_path, 'w', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f, delimiter='\t')
                            for src, tgt in dictionary.items():
                                writer.writerow([src, tgt])
                        
                        corpus_train_status.text = f'✓ Dictionary built: {len(dictionary)} entries saved to {dict_path.name}'
                        log_event(f"Corpus dictionary built: {corpus_name} ({src_lang}-{tgt_lang}), {len(dictionary)} entries")
                        ui.notify(f'Dictionary ready! {len(dictionary)} translation pairs', type='positive', position='top')
                        
                    except Exception as ex:
                        corpus_train_status.text = f'Error: {str(ex)[:80]}'
                        log_event(f"Corpus training error: {ex}", "ERROR")
                        ui.notify(f'Training failed: {str(ex)[:50]}', type='negative')
                    finally:
                        corpus_download_btn.props(remove='loading')
                
                def corpus_show_status():
                    """Show downloaded corpora status"""
                    try:
                        from scitrans_llms.translate.corpus_manager import CorpusManager
                        cm = CorpusManager()
                        downloaded = cm.list_downloaded()
                        
                        # Check for built dictionaries
                        dict_root = Path.home() / ".scitrans" / "dictionaries"
                        if dict_root.exists():
                            dicts = [f.name for f in dict_root.glob("*.tsv")]
                        else:
                            dicts = []
                        
                        msg = f"Downloaded corpora: {', '.join(downloaded) if downloaded else 'None'}\n"
                        msg += f"Built dictionaries: {', '.join(dicts) if dicts else 'None'}"
                        corpus_train_status.text = msg
                        ui.notify('Status updated', type='info')
                        log_event("Corpus status checked")
                        
                    except Exception as ex:
                        corpus_train_status.text = f'Error: {str(ex)[:80]}'
                        log_event(f"Corpus status error: {ex}", "ERROR")
                
                corpus_download_btn.on_click(corpus_download_and_build)
                corpus_status_btn.on_click(corpus_show_status)
                
                test_run_btn.on_click(run_test)
                test_clear_btn.on_click(lambda: (setattr(test_input, 'value', ''), setattr(test_output, 'value', ''), setattr(bleu_label, 'text', '--'), setattr(terms_label, 'text', '--'), setattr(blocks_label, 'text', '--')))
            
            # ==================== GLOSSARY TAB ====================
            with ui.tab_panel(t_glossary).classes('p-0').style('height:100%; overflow:auto;'):
                with ui.element('div').classes('main-row'):
                    with ui.element('div').classes('panel').style('max-width:100%;'):
                        with ui.element('div').classes('card').style('min-height:calc(100vh - 180px);'):
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
                                rows=all_entries[:50], row_key='en', pagination={'rowsPerPage': 20}
                            ).classes('w-full').style('max-height:calc(100vh - 280px);')
                            
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
            with ui.tab_panel(t_developer).classes('p-0').style('height:100%; overflow:hidden;'):
                with ui.element('div').classes('main-row'):
                    with ui.element('div').classes('panel left-panel'):
                        with ui.element('div').classes('card').style('flex:1; display:flex; flex-direction:column; min-height:calc(100vh - 180px);'):
                            ui.label('System Logs').classes('card-title')
                            with ui.row().classes('gap-2 mb-2'):
                                def refresh(): dev_logs.value = '\n'.join(system_logs[-50:])
                                def clear(): system_logs.clear(); dev_logs.value = ''
                                def export():
                                    f = Path(tempfile.mktemp(suffix='.log'))
                                    f.write_text('\n'.join(system_logs))
                                    ui.download(str(f), 'scitrans_logs.log')
                                ui.button('Refresh', on_click=refresh).props('dense outline')
                                ui.button('Clear', on_click=clear).props('dense outline')
                                ui.button('Export', on_click=export).props('dense outline')
                            dev_logs = ui.textarea().props('readonly auto-grow').classes('w-full font-mono text-xs').style('flex:1; min-height:400px; max-height:calc(100vh - 300px); overflow-y:auto;')
                            dev_logs.value = '\n'.join(system_logs[-100:])
                            
                            # Auto-refresh logs every 2 seconds
                            def update_logs():
                                if len(system_logs) > 0:
                                    dev_logs.value = '\n'.join(system_logs[-50:])
                            ui.timer(2.0, update_logs)
                    
                    with ui.element('div').classes('panel right-panel'):
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
            with ui.tab_panel(t_settings).classes('p-0').style('height:100%; overflow:auto;'):
                with ui.element('div').style('padding:var(--gap); display:flex; flex-direction:column; gap:var(--gap); max-width:900px; margin:auto;'):
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
