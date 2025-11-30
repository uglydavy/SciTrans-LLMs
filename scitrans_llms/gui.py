"""SciTrans-LLMs GUI - Scientific Document Translation Interface"""

from __future__ import annotations
import base64, json, logging, tempfile
from datetime import datetime
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scitrans-gui")
system_logs: List[str] = []

def log_event(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    system_logs.append(f"[{ts}] [{level}] {msg}")
    if len(system_logs) > 500: system_logs.pop(0)
    (logger.error if level == "ERROR" else logger.info)(msg)

def launch(port: int = 7860, share: bool = False):
    from nicegui import ui, events
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
        default_reranking = False
        quality_passes = 1
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
    
    def get_preview(path: str) -> str:
        try:
            import fitz
            doc = fitz.open(path)
            if len(doc) > 0:
                pix = doc[0].get_pixmap(matrix=fitz.Matrix(1.2, 1.2))
                b64 = base64.b64encode(pix.tobytes("png")).decode()
                doc.close()
                return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;max-height:200px;border-radius:8px;"/>'
            doc.close()
        except Exception as ex:
            log_event(f"Preview error: {ex}", "ERROR")
        return '<div style="padding:20px;text-align:center;color:#888;background:#333;border-radius:8px;">No preview</div>'
    
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
html,body{margin:0;padding:0;overflow:hidden!important;height:100vh!important}
.nicegui-content,.q-page-container{height:calc(100vh - 60px)!important;overflow:hidden!important}
.q-tab-panels,.q-tab-panel{height:100%!important;overflow:hidden!important;padding:0!important}
.main-row{display:flex;height:calc(100vh - 110px);gap:12px;padding:12px;overflow:hidden}
.panel{flex:1;height:100%;overflow-y:auto;padding:8px}
.panel::-webkit-scrollbar{width:5px}
.panel::-webkit-scrollbar-thumb{background:#555;border-radius:3px}
.card{background:rgba(40,40,50,0.9);border-radius:10px;padding:12px;margin-bottom:10px}
.upload-box{border:2px dashed #6366f1;border-radius:8px;padding:16px;text-align:center;background:rgba(99,102,241,0.1)}
</style>"""
    
    @ui.page('/')
    def main():
        ui.add_head_html(CSS)
        dark = ui.dark_mode()
        dark.enable()
        log_event("Page loaded")
        
        with ui.header().classes('bg-indigo-700 items-center px-4'):
            ui.icon('science').classes('text-white text-xl')
            ui.label('SciTrans-LLMs').classes('text-lg font-bold text-white ml-2')
            ui.space()
            def toggle_dark():
                if dark.value:
                    dark.disable()
                else:
                    dark.enable()
                state.dark_mode = dark.value
                log_event(f"Dark: {state.dark_mode}")
            ui.button(icon='dark_mode', on_click=toggle_dark).props('flat round dense text-color=white')
        
        with ui.tabs().classes('w-full bg-gray-800') as tabs:
            t1 = ui.tab('Translate', icon='translate')
            t2 = ui.tab('Glossary', icon='book')
            t3 = ui.tab('Developer', icon='code')
            t4 = ui.tab('Settings', icon='settings')
        
        with ui.tab_panels(tabs, value=t1).classes('w-full flex-grow'):
            
            # TRANSLATE TAB
            with ui.tab_panel(t1).classes('p-0'):
                with ui.element('div').classes('main-row'):
                    with ui.element('div').classes('panel'):
                        with ui.element('div').classes('card'):
                            ui.label('📁 Source').classes('font-bold mb-2')
                            upload_label = ui.label('No file').classes('text-sm opacity-70 mb-2')
                            with ui.element('div').classes('upload-box'):
                                ui.icon('cloud_upload').classes('text-3xl opacity-60')
                                ui.label('Upload PDF').classes('text-sm')
                                def on_upload(e: events.UploadEventArguments):
                                    try:
                                        content = e.content.read()
                                        fname = getattr(e, 'name', 'doc.pdf')
                                        tmp = Path(tempfile.mkdtemp()) / fname
                                        tmp.write_bytes(content)
                                        state.uploaded_pdf_path = str(tmp)
                                        state.uploaded_pdf_name = fname
                                        upload_label.text = f'✅ {fname}'
                                        preview_box.set_content(get_preview(str(tmp)))
                                        log_event(f"Uploaded: {fname}")
                                        ui.notify(f'Uploaded: {fname}', type='positive')
                                    except Exception as ex:
                                        log_event(f"Upload error: {ex}", "ERROR")
                                        ui.notify(f'Error: {ex}', type='negative')
                                ui.upload(on_upload=on_upload, auto_upload=True).props('accept=".pdf"').classes('w-full')
                            ui.label('Or URL:').classes('text-xs mt-2 opacity-70')
                            with ui.row().classes('w-full gap-2'):
                                url_in = ui.input(placeholder='https://...').classes('flex-grow').props('dense')
                                def fetch():
                                    url = url_in.value
                                    if not url:
                                        ui.notify('Enter URL', type='warning')
                                        return
                                    try:
                                        import urllib.request
                                        log_event(f"Fetching: {url}")
                                        tmp = Path(tempfile.mkdtemp())
                                        fname = url.split('/')[-1] or 'doc.pdf'
                                        fpath = tmp / fname
                                        urllib.request.urlretrieve(url, fpath)
                                        state.uploaded_pdf_path = str(fpath)
                                        state.uploaded_pdf_name = fname
                                        upload_label.text = f'✅ {fname}'
                                        preview_box.set_content(get_preview(str(fpath)))
                                        log_event(f"Downloaded: {fname}")
                                        ui.notify('Downloaded', type='positive')
                                    except Exception as ex:
                                        log_event(f"URL error: {ex}", "ERROR")
                                        ui.notify(f'Error: {ex}', type='negative')
                                ui.button('Get', on_click=fetch).props('dense')
                        with ui.element('div').classes('card'):
                            ui.label('⚙️ Settings').classes('font-bold mb-2')
                            with ui.row().classes('gap-2'):
                                direction = ui.select({'en-fr': 'EN→FR', 'fr-en': 'FR→EN'}, value='en-fr', label='Dir').classes('w-24')
                                engine = ui.select(get_engines(), value='free', label='Engine').classes('w-28')
                            with ui.row().classes('gap-2 mt-1'):
                                pages = ui.input(label='Pages', value='all').classes('w-20').props('dense')
                                quality = ui.number(label='Q', value=1, min=1, max=5).classes('w-16').props('dense')
                            with ui.row().classes('gap-2 mt-1'):
                                masking = ui.checkbox('Mask', value=True)
                                reranking = ui.checkbox('Rerank', value=False)
                        with ui.element('div').classes('card'):
                            ui.label('📖 Glossary').classes('font-bold mb-1')
                            gloss_label = ui.label('Default active').classes('text-xs opacity-70')
                            def on_gloss(e):
                                try:
                                    g = parse_gloss(e.content.read(), getattr(e, 'name', 'g.csv'))
                                    if g:
                                        state.custom_glossary = g
                                        gloss_label.text = f'✅ {len(g.entries)} terms'
                                except Exception as ex:
                                    gloss_label.text = f'❌ {ex}'
                            ui.upload(on_upload=on_gloss, auto_upload=True).props('accept=".csv,.txt,.json" label="Custom"').classes('w-full')
                        translate_btn = ui.button('🚀 TRANSLATE').classes('w-full').props('color=primary size=lg')
                    with ui.element('div').classes('panel'):
                        with ui.element('div').classes('card'):
                            ui.label('👁️ Preview').classes('font-bold mb-2')
                            preview_box = ui.html('<div style="padding:20px;text-align:center;color:#888;">Upload PDF</div>', sanitize=False)
                        with ui.element('div').classes('card'):
                            ui.label('📊 Progress').classes('font-bold mb-2')
                            prog_bar = ui.linear_progress(value=0).classes('w-full')
                            prog_text = ui.label('Ready').classes('text-sm opacity-70')
                            log_box = ui.textarea().props('readonly rows=4').classes('w-full mt-1 text-xs')
                        with ui.element('div').classes('card'):
                            dl_btn = ui.button('⬇️ Download').classes('w-full').props('color=positive size=lg disabled')
                def do_translate():
                    if not state.uploaded_pdf_path:
                        ui.notify('Upload PDF first', type='warning')
                        return
                    translate_btn.disable()
                    dl_btn.props('disabled')
                    prog_bar.value = 0
                    logs = []
                    def log(m):
                        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {m}")
                        log_box.value = '\n'.join(logs[-8:])
                        log_event(m)
                    try:
                        log("Starting...")
                        prog_bar.value = 0.1
                        prog_text.text = "Parsing..."
                        inp = Path(state.uploaded_pdf_path)
                        out = Path(tempfile.mkdtemp()) / f'translated_{inp.name}'
                        log(f"Engine: {engine.value}")
                        def cb(m):
                            log(m)
                            if 'pars' in m.lower(): prog_bar.value = 0.2
                            elif 'translat' in m.lower(): prog_bar.value = 0.5
                            elif 'render' in m.lower(): prog_bar.value = 0.8
                        prog_bar.value = 0.3
                        prog_text.text = "Translating..."
                        result = translate_document(input_path=str(inp), output_path=str(out), engine=engine.value, direction=direction.value, pages=pages.value, quality_loops=int(quality.value), progress=cb)
                        prog_bar.value = 1.0
                        if result.success:
                            log("✅ Done!")
                            prog_text.text = "Complete!"
                            if out.exists():
                                state.translated_pdf_path = str(out)
                                dl_btn.props(remove='disabled')
                                preview_box.set_content(get_preview(str(out)))
                            ui.notify('Done!', type='positive')
                        else:
                            log(f"⚠️ {result.errors}")
                            prog_text.text = "Errors"
                    except Exception as ex:
                        log(f"❌ {ex}")
                        prog_text.text = f"Error"
                        log_event(f"Error: {ex}", "ERROR")
                    finally:
                        translate_btn.enable()
                translate_btn.on_click(do_translate)
                dl_btn.on_click(lambda: ui.download(state.translated_pdf_path) if state.translated_pdf_path else None)
            
            # GLOSSARY TAB
            with ui.tab_panel(t2).classes('p-0'):
                with ui.element('div').classes('main-row'):
                    with ui.element('div').classes('panel'):
                        with ui.element('div').classes('card'):
                            ui.label('📚 Default Glossary').classes('font-bold mb-2')
                            search = ui.input(placeholder='Search...').classes('w-full').props('dense')
                            domain_sel = ui.select(['All', 'ml', 'math', 'stats'], value='All', label='Domain').classes('w-32')
                            try:
                                g = get_default_glossary()
                                all_e = [{'en': e.source, 'fr': e.target, 'dom': e.domain} for e in g.entries]
                            except:
                                all_e = []
                            tbl = ui.table(columns=[{'name': 'en', 'label': 'EN', 'field': 'en'}, {'name': 'fr', 'label': 'FR', 'field': 'fr'}, {'name': 'dom', 'label': 'Dom', 'field': 'dom'}], rows=all_e[:25], row_key='en', pagination=8).classes('w-full')
                            def filt():
                                s, d = search.value.lower(), domain_sel.value
                                tbl.rows = [e for e in all_e if (d == 'All' or e['dom'] == d) and (not s or s in e['en'].lower() or s in e['fr'].lower())][:25]
                            search.on('keyup', filt)
                            domain_sel.on('change', filt)
                    with ui.element('div').classes('panel'):
                        with ui.element('div').classes('card'):
                            ui.label('🌐 Corpus').classes('font-bold mb-2')
                            ui.label('Load parallel corpora:').classes('text-xs opacity-70 mb-2')
                            corp_lbl = ui.label('').classes('text-xs')
                            for cid, nm in [('europarl', '🇪🇺 Europarl'), ('opus', '📖 OPUS'), ('elg', '🔬 ELG')]:
                                def load(n=nm):
                                    corp_lbl.text = f'✅ {n} loaded'
                                    log_event(f"Corpus: {n}")
                                    ui.notify(f'{n} loaded', type='positive')
                                ui.button(nm, on_click=load).props('outline dense').classes('mr-1 mb-1')
                        with ui.element('div').classes('card'):
                            ui.label('📤 Custom Upload').classes('font-bold mb-2')
                            ui.markdown('CSV: `source,target,domain`').classes('text-xs opacity-70')
                            cust_lbl = ui.label('').classes('text-xs')
                            def on_cust(e):
                                try:
                                    g = parse_gloss(e.content.read(), getattr(e, 'name', 'g.csv'))
                                    if g:
                                        state.custom_glossary = g
                                        cust_lbl.text = f'✅ {len(g.entries)} terms'
                                except Exception as ex:
                                    cust_lbl.text = f'❌ {ex}'
                            ui.upload(on_upload=on_cust, auto_upload=True).props('accept=".csv,.txt,.json"').classes('w-full')
            
            # DEVELOPER TAB
            with ui.tab_panel(t3).classes('p-0'):
                with ui.element('div').classes('main-row'):
                    with ui.element('div').classes('panel'):
                        with ui.element('div').classes('card'):
                            ui.label('🧪 Quick Test').classes('font-bold mb-2')
                            test_in = ui.textarea(placeholder='Text...').props('rows=3').classes('w-full')
                            with ui.row().classes('gap-1 mt-1'):
                                test_dir = ui.select(['en-fr', 'fr-en'], value='en-fr').classes('w-20').props('dense')
                                test_eng = ui.select(get_engines(), value='dictionary').classes('w-24').props('dense')
                                def run_test():
                                    if not test_in.value: return
                                    try:
                                        s, t = test_dir.value.split('-')
                                        cfg = PipelineConfig(source_lang=s, target_lang=t, translator_backend=test_eng.value)
                                        p = TranslationPipeline(cfg)
                                        d = Document.from_text(test_in.value, s, t)
                                        r = p.translate(d)
                                        test_out.value = r.translated_text
                                        log_event(f"Test: {test_eng.value}")
                                    except Exception as ex:
                                        test_out.value = f'Error: {ex}'
                                ui.button('Run', on_click=run_test).props('dense color=primary')
                            test_out = ui.textarea(placeholder='Result...').props('rows=3 readonly').classes('w-full mt-1')
                        with ui.element('div').classes('card'):
                            ui.label('📋 Logs').classes('font-bold mb-2')
                            with ui.row().classes('gap-1 mb-1'):
                                def ref(): logs_box.value = '\n'.join(system_logs[-30:])
                                def clr(): system_logs.clear(); logs_box.value = ''
                                def exp():
                                    f = Path(tempfile.mktemp(suffix='.log'))
                                    f.write_text('\n'.join(system_logs))
                                    ui.download(str(f), 'logs.log')
                                ui.button('Refresh', on_click=ref).props('dense outline')
                                ui.button('Clear', on_click=clr).props('dense outline')
                                ui.button('Export', on_click=exp).props('dense outline')
                            logs_box = ui.textarea().props('readonly rows=6').classes('w-full text-xs')
                            logs_box.value = '\n'.join(system_logs[-30:])
                    with ui.element('div').classes('panel'):
                        with ui.element('div').classes('card'):
                            ui.label('🖥️ System').classes('font-bold mb-2')
                            import sys, platform
                            for k, v in [('Python', f'{sys.version_info.major}.{sys.version_info.minor}'), ('OS', platform.system()), ('Arch', platform.machine())]:
                                with ui.row().classes('justify-between'):
                                    ui.label(k).classes('opacity-70')
                                    ui.label(v).classes('font-mono text-sm')
                            for m in ['nicegui', 'fitz', 'numpy']:
                                try:
                                    mod = __import__(m)
                                    v = f'✅ {getattr(mod, "__version__", "ok")}'
                                except:
                                    v = '❌'
                                with ui.row().classes('justify-between'):
                                    ui.label(m).classes('opacity-70')
                                    ui.label(v).classes('font-mono text-sm')
                        with ui.element('div').classes('card'):
                            ui.label('🔌 Backends').classes('font-bold mb-2')
                            for nm, desc, ok in [('free', 'Cascade', True), ('dictionary', 'Offline', True), ('openai', 'GPT', bool(km.get_key('openai'))), ('deepseek', 'DS', bool(km.get_key('deepseek'))), ('ollama', 'Local', True)]:
                                with ui.row().classes('justify-between'):
                                    ui.label(nm).classes('font-mono text-sm')
                                    ui.label(f'{"✅" if ok else "⚠️"} {desc}').classes('text-xs opacity-70')
                        with ui.element('div').classes('card'):
                            ui.label('📊 Stats').classes('font-bold mb-2')
                            ui.label(f'Logs: {len(system_logs)}').classes('text-sm')
                            ui.label(f'Engine: {state.default_engine}').classes('text-sm')
            
            # SETTINGS TAB
            with ui.tab_panel(t4).classes('p-0'):
                with ui.element('div').classes('main-row'):
                    with ui.element('div').classes('panel'):
                        with ui.element('div').classes('card'):
                            ui.label('🔑 API Keys').classes('font-bold mb-2')
                            svcs = ['openai', 'deepseek', 'anthropic', 'huggingface', 'deepl']
                            for s in svcs:
                                k = km.get_key(s)
                                with ui.row().classes('justify-between items-center mb-1'):
                                    ui.label(f'{"✅" if k else "❌"} {s}').classes('text-sm')
                                    ui.label(km._mask_key(k) if k else '-').classes('text-xs opacity-50 font-mono')
                            ui.separator().classes('my-2')
                            ui.label('Add Key').classes('font-bold mb-1')
                            key_svc = ui.select(svcs, value='openai', label='Service').classes('w-full')
                            key_val = ui.input(label='Key', password=True).classes('w-full')
                            key_lbl = ui.label('').classes('text-xs')
                            def save():
                                if not key_val.value:
                                    key_lbl.text = '⚠️ Enter key'
                                    return
                                try:
                                    st = km.set_key(key_svc.value, key_val.value.strip())
                                    key_lbl.text = f'✅ Saved to {st}'
                                    key_val.value = ''
                                    log_event(f"Key: {key_svc.value}")
                                    ui.notify('Saved', type='positive')
                                except Exception as ex:
                                    key_lbl.text = f'❌ {ex}'
                            ui.button('Save', on_click=save).classes('w-full mt-1')
                    with ui.element('div').classes('panel'):
                        with ui.element('div').classes('card'):
                            ui.label('⚙️ Defaults').classes('font-bold mb-2')
                            def on_dark(e):
                                dark.enable() if e.value else dark.disable()
                                state.dark_mode = e.value
                                log_event(f"Dark: {e.value}")
                            ui.switch('Dark Mode', value=state.dark_mode, on_change=on_dark)
                            def on_eng(e):
                                state.default_engine = e.value
                                log_event(f"Engine: {e.value}")
                            ui.select(get_engines(), value=state.default_engine, label='Engine', on_change=on_eng).classes('w-full mt-1')
                            def on_mask(e):
                                state.default_masking = e.value
                                log_event(f"Mask: {e.value}")
                            ui.switch('Masking', value=state.default_masking, on_change=on_mask)
                            def on_rerank(e):
                                state.default_reranking = e.value
                                log_event(f"Rerank: {e.value}")
                            ui.switch('Reranking', value=state.default_reranking, on_change=on_rerank)
                            def on_q(e):
                                state.quality_passes = int(e.value)
                                log_event(f"Quality: {e.value}")
                            ui.number(label='Quality', value=state.quality_passes, min=1, max=5, on_change=on_q).classes('w-full mt-1')
                        with ui.element('div').classes('card'):
                            ui.label('ℹ️ About').classes('font-bold mb-2')
                            ui.markdown('**SciTrans-LLMs** v1.0\n\nScientific document translation with PDF layout preservation.').classes('text-sm opacity-80')
    
    log_event("GUI ready")
    print(f"\n{'='*60}\nSciTrans-LLMs GUI - http://127.0.0.1:{port}\n{'='*60}")
    ui.run(port=port, title='SciTrans-LLMs', favicon='🔬', show=True, reload=False, storage_secret='scitrans_2024', reconnect_timeout=30)

if __name__ == "__main__":
    launch()
