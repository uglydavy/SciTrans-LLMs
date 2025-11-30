"""
SciTrans-LLMs GUI - Professional Scientific Document Translation Interface

A comprehensive, feature-rich GUI for scientific document translation with:
- PDF translation with layout preservation
- Real-time progress tracking
- Glossary management with corpus integration
- Developer tools and debugging
- Comprehensive settings management
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import os
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("scitrans-gui")

# Global log storage for UI
system_logs: List[str] = []

def log_event(message: str, level: str = "INFO"):
    """Log an event with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] [{level}] {message}"
    system_logs.append(entry)
    if len(system_logs) > 1000:  # Keep last 1000 entries
        system_logs.pop(0)
    logger.info(message)


def launch(port: int = 7860, share: bool = False):
    """Launch the SciTrans-LLMs GUI."""
    try:
        from nicegui import ui, app
    except ImportError:
        raise ImportError("NiceGUI not installed. Run: pip install nicegui>=1.4.0")
    
    # Import components
    from scitrans_llms.keys import KeyManager, SERVICES
    from scitrans_llms.pipeline import PipelineConfig, TranslationPipeline, translate_document
    from scitrans_llms.translate.glossary import get_default_glossary, Glossary, GlossaryEntry
    from scitrans_llms.translate.base import create_translator
    from scitrans_llms.models import Document
    
    km = KeyManager()
    log_event("GUI initialization started")
    
    # ==========================================================================
    # STATE MANAGEMENT
    # ==========================================================================
    
    class AppState:
        """Global application state."""
        dark_mode: bool = True
        default_engine: str = 'free'
        default_masking: bool = True
        default_reranking: bool = False
        quality_passes: int = 1
        uploaded_pdf_path: Optional[str] = None
        uploaded_pdf_name: Optional[str] = None
        translated_pdf_path: Optional[str] = None
        custom_glossary: Optional[Glossary] = None
        glossary_mode: str = 'preserve'  # 'preserve', 'translate', 'context'
        translation_logs: List[str] = []
    
    state = AppState()
    
    # ==========================================================================
    # HELPER FUNCTIONS
    # ==========================================================================
    
    def get_available_engines() -> List[str]:
        """Get list of available translation engines."""
        engines = ['free', 'dictionary', 'dummy', 'improved-offline']
        
        # Check for API-based engines
        if km.get_key('openai'):
            engines.extend(['openai', 'gpt-4', 'gpt-4o'])
        if km.get_key('deepseek'):
            engines.append('deepseek')
        if km.get_key('anthropic'):
            engines.append('anthropic')
        if km.get_key('deepl'):
            engines.append('deepl')
        
        # Always available
        engines.extend(['huggingface', 'ollama', 'googlefree'])
        
        return list(set(engines))
    
    def parse_glossary_file(content: bytes, filename: str) -> Optional[Glossary]:
        """Parse glossary from uploaded file."""
        try:
            text = content.decode('utf-8')
            entries = []
            
            if filename.endswith('.json'):
                data = json.loads(text)
                for item in data:
                    if isinstance(item, dict):
                        entries.append(GlossaryEntry(
                            source=item.get('source', item.get('en', '')),
                            target=item.get('target', item.get('fr', '')),
                            domain=item.get('domain', 'custom'),
                        ))
                    elif isinstance(item, list) and len(item) >= 2:
                        entries.append(GlossaryEntry(source=item[0], target=item[1], domain='custom'))
            
            elif filename.endswith('.csv'):
                lines = text.strip().split('\n')
                for line in lines[1:] if lines[0].lower().startswith('source') else lines:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        entries.append(GlossaryEntry(
                            source=parts[0].strip().strip('"'),
                            target=parts[1].strip().strip('"'),
                            domain=parts[2].strip().strip('"') if len(parts) > 2 else 'custom',
                        ))
            
            else:  # txt or other
                lines = text.strip().split('\n')
                for line in lines:
                    if '\t' in line:
                        parts = line.split('\t')
                    elif ',' in line:
                        parts = line.split(',')
                    elif ':' in line:
                        parts = line.split(':')
                    else:
                        continue
                    
                    if len(parts) >= 2:
                        entries.append(GlossaryEntry(
                            source=parts[0].strip(),
                            target=parts[1].strip(),
                            domain='custom',
                        ))
            
            if entries:
                log_event(f"Parsed {len(entries)} glossary entries from {filename}")
                return Glossary(name='custom', entries=entries)
            
        except Exception as e:
            log_event(f"Error parsing glossary: {e}", "ERROR")
        
        return None
    
    def get_pdf_preview_html(pdf_path: str, page: int = 0) -> str:
        """Generate HTML for PDF preview."""
        try:
            import fitz
            doc = fitz.open(pdf_path)
            if page < len(doc):
                pix = doc[page].get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                img_data = pix.tobytes("png")
                img_b64 = base64.b64encode(img_data).decode()
                doc.close()
                return f'<img src="data:image/png;base64,{img_b64}" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);" />'
            doc.close()
        except Exception as e:
            log_event(f"PDF preview error: {e}", "ERROR")
        
        return '<div style="padding: 40px; text-align: center; color: #888;">PDF preview not available</div>'
    
    # ==========================================================================
    # CUSTOM CSS
    # ==========================================================================
    
    CUSTOM_CSS = """
    <style>
    /* Prevent scrolling - fit to viewport */
    html, body {
        overflow: hidden !important;
        height: 100vh !important;
    }
    
    .q-page-container {
        height: calc(100vh - 80px) !important;
        overflow: hidden !important;
    }
    
    .q-tab-panels {
        height: 100% !important;
        overflow: hidden !important;
    }
    
    .q-tab-panel {
        height: 100% !important;
        overflow: hidden !important;
        padding: 0 !important;
    }
    
    /* Centered container */
    .centered-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
        height: 100%;
        padding: 16px;
        overflow-y: auto;
        max-height: calc(100vh - 140px);
    }
    
    /* Card styling */
    .feature-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }
    
    .dark .feature-card {
        background: rgba(255, 255, 255, 0.05);
    }
    
    /* Upload area */
    .upload-area {
        border: 2px dashed rgba(99, 102, 241, 0.5);
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        transition: all 0.3s ease;
        background: rgba(99, 102, 241, 0.05);
    }
    
    .upload-area:hover {
        border-color: rgba(99, 102, 241, 0.8);
        background: rgba(99, 102, 241, 0.1);
    }
    
    /* Preview area */
    .preview-area {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 16px;
        min-height: 200px;
    }
    
    /* Progress styling */
    .progress-container {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 8px;
        padding: 12px;
    }
    
    /* Log area */
    .log-area {
        font-family: 'Monaco', 'Menlo', monospace;
        font-size: 11px;
        background: #1a1a2e;
        color: #a0a0a0;
        border-radius: 8px;
        padding: 12px;
        max-height: 300px;
        overflow-y: auto;
    }
    
    /* Compact inputs */
    .compact-input .q-field__control {
        height: 36px !important;
        min-height: 36px !important;
    }
    
    /* Hide scrollbars but allow scrolling */
    .scroll-hidden::-webkit-scrollbar {
        display: none;
    }
    
    .scroll-hidden {
        -ms-overflow-style: none;
        scrollbar-width: none;
    }
    </style>
    """
    
    # ==========================================================================
    # MAIN PAGE
    # ==========================================================================
    
    @ui.page('/')
    async def main_page():
        # Add custom CSS
        ui.add_head_html(CUSTOM_CSS)
        
        # Apply initial dark mode
        dark = ui.dark_mode()
        dark.enable()
        log_event("Main page loaded")
        
        # Header
        with ui.header().classes('bg-gradient-to-r from-indigo-600 to-purple-600 items-center'):
            ui.icon('science', size='md').classes('text-white')
            ui.label('SciTrans-LLMs').classes('text-xl font-bold text-white ml-2')
            ui.space()
            ui.label('Scientific Document Translation').classes('text-sm text-white opacity-80')
            ui.space()
            
            # Quick dark mode toggle in header
            def toggle_dark():
                if dark.value:
                    dark.disable()
                    state.dark_mode = False
                else:
                    dark.enable()
                    state.dark_mode = True
                log_event(f"Dark mode {'enabled' if state.dark_mode else 'disabled'}")
            
            ui.button(icon='dark_mode', on_click=toggle_dark).props('flat round dense').classes('text-white')
        
        # Main tabs
        with ui.tabs().classes('w-full bg-gray-800') as tabs:
            translate_tab = ui.tab('translate', label='📄 Translate', icon='translate')
            glossary_tab = ui.tab('glossary', label='📚 Glossary', icon='book')
            developer_tab = ui.tab('developer', label='🛠️ Developer', icon='code')
            settings_tab = ui.tab('settings', label='⚙️ Settings', icon='settings')
        
        with ui.tab_panels(tabs, value=translate_tab).classes('w-full'):
            
            # =================================================================
            # TRANSLATE TAB
            # =================================================================
            with ui.tab_panel(translate_tab):
                await render_translate_panel(state, km, dark)
            
            # =================================================================
            # GLOSSARY TAB
            # =================================================================
            with ui.tab_panel(glossary_tab):
                await render_glossary_panel(state)
            
            # =================================================================
            # DEVELOPER TAB
            # =================================================================
            with ui.tab_panel(developer_tab):
                await render_developer_panel(state, km)
            
            # =================================================================
            # SETTINGS TAB
            # =================================================================
            with ui.tab_panel(settings_tab):
                await render_settings_panel(state, km, dark)
    
    # ==========================================================================
    # TRANSLATE PANEL
    # ==========================================================================
    
    async def render_translate_panel(state: AppState, km: KeyManager, dark):
        """Render the main translation panel."""
        
        # References for UI elements
        refs = {
            'pdf_preview': None,
            'progress_bar': None,
            'progress_label': None,
            'log_area': None,
            'download_btn': None,
            'translate_btn': None,
            'status_label': None,
        }
        
        with ui.row().classes('w-full h-full gap-4 p-4').style('max-width: 1400px; margin: 0 auto;'):
            
            # ================== LEFT SIDE: Controls ==================
            with ui.column().classes('w-1/2 gap-3 scroll-hidden').style('max-height: calc(100vh - 140px); overflow-y: auto;'):
                
                # Source Document Card
                with ui.card().classes('w-full feature-card'):
                    ui.label('📁 Source Document').classes('text-lg font-bold mb-3')
                    
                    with ui.tabs().classes('w-full') as source_tabs:
                        upload_tab = ui.tab('upload', label='Upload File')
                        url_tab = ui.tab('url', label='From URL')
                    
                    with ui.tab_panels(source_tabs, value=upload_tab).classes('w-full'):
                        
                        # File Upload Tab
                        with ui.tab_panel(upload_tab):
                            with ui.column().classes('w-full upload-area'):
                                ui.icon('cloud_upload', size='xl').classes('opacity-60')
                                upload_label = ui.label('Drop PDF here or click to upload').classes('text-center')
                                ui.label('Supports: PDF').classes('text-xs opacity-50')
                                
                                def handle_upload(e):
                                    try:
                                        # NiceGUI upload event handling
                                        content = e.content.read()
                                        filename = e.name if hasattr(e, 'name') else 'uploaded.pdf'
                                        
                                        temp_dir = Path(tempfile.mkdtemp())
                                        file_path = temp_dir / filename
                                        file_path.write_bytes(content)
                                        
                                        state.uploaded_pdf_path = str(file_path)
                                        state.uploaded_pdf_name = filename
                                        
                                        upload_label.text = f'✅ {filename}'
                                        log_event(f"File uploaded: {filename}")
                                        
                                        # Update preview
                                        if refs['pdf_preview']:
                                            html = get_pdf_preview_html(str(file_path))
                                            refs['pdf_preview'].set_content(html)
                                        
                                        ui.notify(f'Uploaded: {filename}', type='positive')
                                    except Exception as ex:
                                        log_event(f"Upload error: {ex}", "ERROR")
                                        ui.notify(f'Upload failed: {ex}', type='negative')
                                
                                ui.upload(
                                    on_upload=handle_upload,
                                    auto_upload=True,
                                ).props('accept=".pdf"').classes('w-full mt-2')
                        
                        # URL Tab
                        with ui.tab_panel(url_tab):
                            url_input = ui.input(
                                label='PDF URL',
                                placeholder='https://example.com/paper.pdf',
                            ).classes('w-full')
                            
                            url_status = ui.label('').classes('text-xs opacity-70')
                            
                            async def fetch_url():
                                url = url_input.value
                                if not url:
                                    ui.notify('Enter a URL', type='warning')
                                    return
                                
                                try:
                                    import urllib.request
                                    url_status.text = 'Downloading...'
                                    log_event(f"Fetching URL: {url}")
                                    
                                    temp_dir = Path(tempfile.mkdtemp())
                                    filename = url.split('/')[-1] or 'document.pdf'
                                    file_path = temp_dir / filename
                                    
                                    urllib.request.urlretrieve(url, file_path)
                                    
                                    state.uploaded_pdf_path = str(file_path)
                                    state.uploaded_pdf_name = filename
                                    url_status.text = f'✅ Downloaded: {filename}'
                                    log_event(f"URL downloaded: {filename}")
                                    
                                    # Update preview
                                    if refs['pdf_preview']:
                                        html = get_pdf_preview_html(str(file_path))
                                        refs['pdf_preview'].set_content(html)
                                    
                                    ui.notify('PDF downloaded', type='positive')
                                except Exception as ex:
                                    url_status.text = f'❌ {ex}'
                                    log_event(f"URL fetch error: {ex}", "ERROR")
                            
                            ui.button('Fetch PDF', icon='download', on_click=fetch_url).classes('mt-2')
                
                # Translation Settings Card
                with ui.card().classes('w-full feature-card'):
                    ui.label('🔧 Translation Settings').classes('text-lg font-bold mb-3')
                    
                    with ui.row().classes('w-full gap-4'):
                        direction_select = ui.select(
                            options={'en-fr': '🇬🇧 English → 🇫🇷 French', 'fr-en': '🇫🇷 French → 🇬🇧 English'},
                            value='en-fr',
                            label='Direction',
                        ).classes('w-1/2')
                        
                        engine_select = ui.select(
                            options=get_available_engines(),
                            value=state.default_engine,
                            label='Engine',
                        ).classes('w-1/2')
                    
                    with ui.row().classes('w-full gap-4 mt-2'):
                        pages_input = ui.input(
                            label='Pages',
                            value='all',
                            placeholder='all, 1-5, 1,3,5',
                        ).classes('w-1/3')
                        
                        quality_slider = ui.slider(
                            min=1, max=5, value=state.quality_passes,
                        ).classes('w-2/3').props('label-always')
                        ui.label('Quality Passes').classes('text-xs opacity-70')
                    
                    # Advanced Options
                    with ui.expansion('Advanced Options', icon='tune').classes('w-full mt-2'):
                        with ui.row().classes('gap-4 flex-wrap'):
                            masking_check = ui.checkbox('Mask Formulas/URLs', value=state.default_masking)
                            reranking_check = ui.checkbox('Enable Reranking', value=state.default_reranking)
                            context_check = ui.checkbox('Context Awareness', value=True)
                            coherence_check = ui.checkbox('Cross-page Coherence', value=True)
                
                # Glossary Card
                with ui.card().classes('w-full feature-card'):
                    ui.label('📖 Custom Glossary').classes('text-lg font-bold mb-3')
                    
                    glossary_mode = ui.select(
                        options={
                            'preserve': '🔒 Preserve terms (no translation)',
                            'translate': '🔄 Translate with glossary hints',
                            'context': '📝 Add context for terms',
                        },
                        value=state.glossary_mode,
                        label='Glossary Mode',
                    ).classes('w-full')
                    
                    glossary_status = ui.label('No custom glossary loaded').classes('text-xs opacity-70 mt-2')
                    
                    def handle_glossary_upload(e):
                        try:
                            content = e.content.read()
                            filename = e.name if hasattr(e, 'name') else 'glossary.csv'
                            
                            gloss = parse_glossary_file(content, filename)
                            if gloss:
                                state.custom_glossary = gloss
                                glossary_status.text = f'✅ Loaded {len(gloss.entries)} terms from {filename}'
                                ui.notify(f'Glossary loaded: {len(gloss.entries)} terms', type='positive')
                            else:
                                glossary_status.text = '❌ Failed to parse glossary'
                                ui.notify('Failed to parse glossary file', type='negative')
                        except Exception as ex:
                            log_event(f"Glossary upload error: {ex}", "ERROR")
                            glossary_status.text = f'❌ {ex}'
                    
                    ui.upload(
                        on_upload=handle_glossary_upload,
                        auto_upload=True,
                    ).props('accept=".csv,.txt,.json"').classes('w-full mt-2')
                    
                    with ui.expansion('Format Instructions', icon='info').classes('w-full mt-2'):
                        ui.markdown("""
                        **CSV format:**
                        ```
                        source,target,domain
                        machine learning,apprentissage automatique,ml
                        neural network,réseau neuronal,ml
                        ```
                        
                        **JSON format:**
                        ```json
                        [{"source": "machine learning", "target": "apprentissage automatique"}]
                        ```
                        
                        **TXT format (tab or comma separated):**
                        ```
                        machine learning    apprentissage automatique
                        neural network    réseau neuronal
                        ```
                        """)
                
                # Translate Button
                refs['translate_btn'] = ui.button(
                    '🚀 Translate Document',
                    on_click=lambda: do_translation(),
                ).classes('w-full mt-2').props('color=primary size=lg')
            
            # ================== RIGHT SIDE: Preview & Progress ==================
            with ui.column().classes('w-1/2 gap-3 scroll-hidden').style('max-height: calc(100vh - 140px); overflow-y: auto;'):
                
                # Preview Card
                with ui.card().classes('w-full feature-card'):
                    ui.label('👁️ Document Preview').classes('text-lg font-bold mb-3')
                    
                    with ui.column().classes('w-full preview-area'):
                        refs['pdf_preview'] = ui.html(
                            '<div style="padding: 40px; text-align: center; color: #888;">Upload a PDF to see preview</div>',
                            sanitize=False,
                        ).classes('w-full')
                
                # Progress Card
                with ui.card().classes('w-full feature-card'):
                    ui.label('📊 Translation Progress').classes('text-lg font-bold mb-3')
                    
                    with ui.column().classes('w-full progress-container'):
                        refs['progress_bar'] = ui.linear_progress(value=0).classes('w-full')
                        refs['progress_label'] = ui.label('Ready to translate').classes('text-sm opacity-70 mt-2')
                        refs['status_label'] = ui.label('').classes('text-xs opacity-50')
                    
                    # Log area
                    ui.label('Translation Log').classes('text-sm font-bold mt-3')
                    refs['log_area'] = ui.textarea().classes('w-full log-area').props('readonly rows=6')
                
                # Download Card
                with ui.card().classes('w-full feature-card'):
                    refs['download_btn'] = ui.button(
                        '⬇️ Download Translated PDF',
                        on_click=lambda: download_result(),
                    ).classes('w-full').props('color=positive size=lg disabled')
                    
                    ui.label('Translation will be available here').classes('text-xs opacity-50 text-center mt-2')
        
        # Translation function
        async def do_translation():
            if not state.uploaded_pdf_path:
                ui.notify('Please upload a PDF first', type='warning')
                return
            
            refs['translate_btn'].disable()
            refs['download_btn'].props('disabled')
            refs['progress_bar'].value = 0
            refs['progress_label'].text = 'Starting translation...'
            state.translation_logs = []
            
            def log(msg: str):
                timestamp = datetime.now().strftime("%H:%M:%S")
                entry = f"[{timestamp}] {msg}"
                state.translation_logs.append(entry)
                refs['log_area'].value = '\n'.join(state.translation_logs)
                log_event(msg)
            
            try:
                log('Initializing translation pipeline...')
                refs['progress_bar'].value = 0.1
                
                # Create output path
                input_path = Path(state.uploaded_pdf_path)
                output_dir = Path(tempfile.mkdtemp())
                output_path = output_dir / f'translated_{input_path.name}'
                
                log(f'Input: {state.uploaded_pdf_name}')
                log(f'Direction: {direction_select.value}')
                log(f'Engine: {engine_select.value}')
                
                # Progress callback
                def progress_cb(msg: str):
                    log(msg)
                    # Update progress based on keywords
                    if 'parsing' in msg.lower():
                        refs['progress_bar'].value = 0.2
                    elif 'translating' in msg.lower():
                        refs['progress_bar'].value = 0.5
                    elif 'rendering' in msg.lower():
                        refs['progress_bar'].value = 0.8
                    elif 'complete' in msg.lower():
                        refs['progress_bar'].value = 1.0
                
                refs['progress_bar'].value = 0.2
                refs['progress_label'].text = 'Parsing PDF...'
                log('Parsing document layout...')
                
                # Run translation
                result = translate_document(
                    input_path=str(input_path),
                    output_path=str(output_path),
                    engine=engine_select.value,
                    direction=direction_select.value,
                    pages=pages_input.value,
                    quality_loops=int(quality_slider.value),
                    progress=progress_cb,
                )
                
                refs['progress_bar'].value = 1.0
                
                if result.success:
                    log('✅ Translation complete!')
                    log(f'   Blocks translated: {result.stats.get("translated_blocks", 0)}')
                    refs['progress_label'].text = 'Translation complete!'
                    
                    # Check output
                    if output_path.exists():
                        state.translated_pdf_path = str(output_path)
                        refs['download_btn'].props(remove='disabled')
                        
                        # Update preview with translated
                        html = get_pdf_preview_html(str(output_path))
                        refs['pdf_preview'].set_content(html)
                        
                        ui.notify('Translation complete!', type='positive')
                    else:
                        text_path = output_path.with_suffix('.txt')
                        if text_path.exists():
                            state.translated_pdf_path = str(text_path)
                            refs['download_btn'].props(remove='disabled')
                            log(f'   Saved as text: {text_path.name}')
                else:
                    log('⚠️ Translation completed with errors:')
                    for err in result.errors:
                        log(f'   - {err}')
                    refs['progress_label'].text = 'Completed with errors'
                
            except Exception as ex:
                log(f'❌ Error: {ex}')
                log_event(f"Translation error: {ex}", "ERROR")
                refs['progress_label'].text = f'Error: {ex}'
                ui.notify(f'Translation failed: {ex}', type='negative')
            finally:
                refs['translate_btn'].enable()
        
        async def download_result():
            if state.translated_pdf_path and Path(state.translated_pdf_path).exists():
                ui.download(state.translated_pdf_path)
                log_event(f"Downloaded: {state.translated_pdf_path}")
            else:
                ui.notify('No translated file available', type='warning')
    
    # ==========================================================================
    # GLOSSARY PANEL
    # ==========================================================================
    
    async def render_glossary_panel(state: AppState):
        """Render the glossary management panel."""
        
        with ui.column().classes('centered-container w-full').style('max-width: 1200px; margin: 0 auto;'):
            
            with ui.row().classes('w-full gap-4'):
                
                # Left: Default Glossary
                with ui.card().classes('w-1/2 feature-card'):
                    ui.label('📚 Default Scientific Glossary').classes('text-lg font-bold mb-3')
                    
                    # Search
                    search_input = ui.input(
                        label='Search terms',
                        placeholder='Type to search...',
                    ).classes('w-full')
                    
                    domain_filter = ui.select(
                        options=['All', 'ml', 'math', 'stats', 'physics', 'chemistry', 'biology', 'custom'],
                        value='All',
                        label='Domain',
                    ).classes('w-48 mt-2')
                    
                    # Load glossary
                    try:
                        gloss = get_default_glossary()
                        all_entries = [
                            {'source': e.source, 'target': e.target, 'domain': e.domain}
                            for e in gloss.entries
                        ]
                    except:
                        all_entries = []
                    
                    columns = [
                        {'name': 'source', 'label': 'English', 'field': 'source', 'sortable': True},
                        {'name': 'target', 'label': 'French', 'field': 'target', 'sortable': True},
                        {'name': 'domain', 'label': 'Domain', 'field': 'domain', 'sortable': True},
                    ]
                    
                    glossary_table = ui.table(
                        columns=columns,
                        rows=all_entries[:50],  # Show first 50
                        row_key='source',
                        pagination=10,
                    ).classes('w-full').style('max-height: 300px;')
                    
                    def filter_glossary():
                        search = search_input.value.lower()
                        domain = domain_filter.value
                        
                        filtered = []
                        for entry in all_entries:
                            if domain != 'All' and entry['domain'] != domain:
                                continue
                            if search and search not in entry['source'].lower() and search not in entry['target'].lower():
                                continue
                            filtered.append(entry)
                        
                        glossary_table.rows = filtered[:50]
                    
                    search_input.on('keyup', filter_glossary)
                    domain_filter.on('change', filter_glossary)
                
                # Right: Corpus & Custom
                with ui.card().classes('w-1/2 feature-card'):
                    ui.label('🌐 Corpus Integration').classes('text-lg font-bold mb-3')
                    
                    ui.markdown("""
                    Load parallel corpora to enhance translation quality with verified sources.
                    These corpora provide domain-specific terminology for scientific translation.
                    """).classes('text-sm opacity-70 mb-3')
                    
                    corpus_options = [
                        ('europarl_en_fr', '🇪🇺 Europarl EN-FR', 'European Parliament proceedings'),
                        ('opus_science', '📖 OPUS Scientific', 'Scientific publications corpus'),
                        ('elg_terminology', '🔬 ELG Terminology', 'European Language Grid terms'),
                    ]
                    
                    corpus_status = ui.label('No corpus loaded').classes('text-xs opacity-70')
                    
                    for corpus_id, name, desc in corpus_options:
                        with ui.row().classes('w-full items-center gap-2 mb-2'):
                            
                            async def load_corpus(cid=corpus_id, cname=name):
                                corpus_status.text = f'Loading {cname}...'
                                log_event(f"Loading corpus: {cname}")
                                # Simulate loading (actual implementation would fetch from APIs)
                                await asyncio.sleep(1)
                                corpus_status.text = f'✅ {cname} loaded (simulated)'
                                ui.notify(f'{cname} loaded', type='positive')
                            
                            ui.button(name, on_click=lambda cid=corpus_id, cname=name: load_corpus(cid, cname)).props('outline dense')
                            ui.label(desc).classes('text-xs opacity-50')
                    
                    ui.separator().classes('my-4')
                    
                    # Upload custom glossary
                    ui.label('📤 Upload Custom Glossary').classes('font-bold mb-2')
                    
                    custom_glossary_status = ui.label('').classes('text-xs opacity-70')
                    
                    def handle_custom_upload(e):
                        try:
                            content = e.content.read()
                            filename = e.name if hasattr(e, 'name') else 'glossary.csv'
                            gloss = parse_glossary_file(content, filename)
                            if gloss:
                                state.custom_glossary = gloss
                                custom_glossary_status.text = f'✅ {len(gloss.entries)} terms loaded'
                                # Add to table
                                new_entries = [
                                    {'source': e.source, 'target': e.target, 'domain': 'custom'}
                                    for e in gloss.entries
                                ]
                                all_entries.extend(new_entries)
                                glossary_table.rows = all_entries[:50]
                                ui.notify(f'Custom glossary: {len(gloss.entries)} terms', type='positive')
                        except Exception as ex:
                            custom_glossary_status.text = f'❌ {ex}'
                    
                    ui.upload(
                        on_upload=handle_custom_upload,
                        auto_upload=True,
                    ).props('accept=".csv,.txt,.json"').classes('w-full')
    
    # ==========================================================================
    # DEVELOPER PANEL
    # ==========================================================================
    
    async def render_developer_panel(state: AppState, km: KeyManager):
        """Render the developer tools panel."""
        
        with ui.column().classes('centered-container w-full').style('max-width: 1200px; margin: 0 auto;'):
            
            # Sub-tabs
            with ui.tabs().classes('w-full') as dev_tabs:
                testing_tab = ui.tab('testing', label='🧪 Testing')
                logs_tab = ui.tab('logs', label='📋 Logs')
                debug_tab = ui.tab('debug', label='🐛 Debug')
            
            with ui.tab_panels(dev_tabs, value=testing_tab).classes('w-full'):
                
                # Testing Tab
                with ui.tab_panel(testing_tab):
                    with ui.card().classes('w-full feature-card'):
                        ui.label('Quick Translation Test').classes('text-lg font-bold mb-3')
                        
                        with ui.row().classes('w-full gap-4'):
                            with ui.column().classes('w-1/2'):
                                test_input = ui.textarea(
                                    label='Source Text',
                                    placeholder='Enter text to translate...',
                                ).classes('w-full').props('rows=6')
                                
                                with ui.row().classes('gap-2 mt-2'):
                                    test_direction = ui.select(
                                        options=['en-fr', 'fr-en'],
                                        value='en-fr',
                                        label='Direction',
                                    ).classes('w-32')
                                    
                                    test_engine = ui.select(
                                        options=get_available_engines(),
                                        value='dictionary',
                                        label='Engine',
                                    ).classes('w-32')
                                
                                test_btn = ui.button('Translate', icon='translate').props('color=primary')
                                test_clear = ui.button('Clear', icon='clear').props('outline')
                            
                            with ui.column().classes('w-1/2'):
                                test_output = ui.textarea(
                                    label='Translation',
                                ).classes('w-full').props('rows=6 readonly')
                                
                                test_status = ui.label('').classes('text-xs opacity-70 mt-2')
                        
                        def do_test():
                            if not test_input.value:
                                return
                            
                            test_status.text = 'Translating...'
                            log_event(f"Test translation: {test_engine.value}")
                            
                            try:
                                src, tgt = test_direction.value.split('-')
                                config = PipelineConfig(
                                    source_lang=src,
                                    target_lang=tgt,
                                    translator_backend=test_engine.value,
                                    enable_glossary=True,
                                )
                                pipeline = TranslationPipeline(config)
                                doc = Document.from_text(test_input.value, src, tgt)
                                result = pipeline.translate(doc)
                                
                                test_output.value = result.translated_text
                                test_status.text = f'✅ Complete ({result.stats.get("translated_blocks", 0)} blocks)'
                            except Exception as ex:
                                test_output.value = f'Error: {ex}'
                                test_status.text = f'❌ {ex}'
                                log_event(f"Test error: {ex}", "ERROR")
                        
                        def clear_test():
                            test_input.value = ''
                            test_output.value = ''
                            test_status.text = ''
                        
                        test_btn.on_click(do_test)
                        test_clear.on_click(clear_test)
                
                # Logs Tab
                with ui.tab_panel(logs_tab):
                    with ui.card().classes('w-full feature-card'):
                        ui.label('📋 System Logs').classes('text-lg font-bold mb-3')
                        
                        with ui.row().classes('gap-2 mb-3'):
                            def refresh_logs():
                                logs_area.value = '\n'.join(system_logs[-100:])
                            
                            def clear_logs():
                                system_logs.clear()
                                logs_area.value = ''
                                log_event("Logs cleared")
                            
                            def export_logs():
                                content = '\n'.join(system_logs)
                                temp_file = Path(tempfile.mktemp(suffix='.log'))
                                temp_file.write_text(content)
                                ui.download(str(temp_file), 'scitrans_logs.log')
                            
                            ui.button('Refresh', icon='refresh', on_click=refresh_logs).props('outline dense')
                            ui.button('Clear', icon='delete', on_click=clear_logs).props('outline dense')
                            ui.button('Export', icon='download', on_click=export_logs).props('outline dense')
                        
                        logs_area = ui.textarea().classes('w-full log-area').props('readonly rows=15')
                        logs_area.value = '\n'.join(system_logs[-100:])
                
                # Debug Tab
                with ui.tab_panel(debug_tab):
                    with ui.row().classes('w-full gap-4'):
                        
                        # System Info
                        with ui.card().classes('w-1/2 feature-card'):
                            ui.label('🖥️ System Information').classes('text-lg font-bold mb-3')
                            
                            import sys
                            import platform
                            
                            info_items = [
                                ('Python Version', f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'),
                                ('Platform', platform.system()),
                                ('Architecture', platform.machine()),
                            ]
                            
                            # Check modules
                            modules = ['nicegui', 'fitz', 'numpy', 'pandas']
                            for mod in modules:
                                try:
                                    m = __import__(mod)
                                    version = getattr(m, '__version__', 'installed')
                                    info_items.append((mod, f'✅ {version}'))
                                except ImportError:
                                    info_items.append((mod, '❌ not installed'))
                            
                            for label, value in info_items:
                                with ui.row().classes('w-full justify-between'):
                                    ui.label(label).classes('opacity-70')
                                    ui.label(value)
                        
                        # Translation Backends
                        with ui.card().classes('w-1/2 feature-card'):
                            ui.label('🔌 Translation Backends').classes('text-lg font-bold mb-3')
                            
                            backends = [
                                ('free', 'Smart cascade', True),
                                ('dictionary', 'Offline glossary', True),
                                ('dummy', 'Testing', True),
                                ('openai', 'GPT-4/4o', bool(km.get_key('openai'))),
                                ('deepseek', 'DeepSeek', bool(km.get_key('deepseek'))),
                                ('anthropic', 'Claude', bool(km.get_key('anthropic'))),
                                ('huggingface', 'HF Inference', True),
                                ('ollama', 'Local LLM', True),
                            ]
                            
                            for name, desc, available in backends:
                                status = '✅' if available else '⚠️ needs key'
                                with ui.row().classes('w-full justify-between'):
                                    ui.label(f'{name}').classes('font-mono')
                                    ui.label(f'{status} {desc}').classes('text-xs opacity-70')
    
    # ==========================================================================
    # SETTINGS PANEL
    # ==========================================================================
    
    async def render_settings_panel(state: AppState, km: KeyManager, dark):
        """Render the settings panel."""
        
        with ui.column().classes('centered-container w-full').style('max-width: 1000px; margin: 0 auto;'):
            
            with ui.row().classes('w-full gap-4'):
                
                # Left: API Keys
                with ui.card().classes('w-1/2 feature-card'):
                    ui.label('🔑 API Keys').classes('text-lg font-bold mb-3')
                    
                    ui.markdown('Configure API keys for premium translation services.').classes('text-sm opacity-70 mb-3')
                    
                    services = ['openai', 'deepseek', 'anthropic', 'huggingface', 'deepl', 'google']
                    
                    for service in services:
                        key = km.get_key(service)
                        status = '✅' if key else '❌'
                        masked = km._mask_key(key) if key else 'Not set'
                        
                        with ui.row().classes('w-full items-center gap-2 mb-2'):
                            ui.label(f'{status} {service.capitalize()}').classes('w-32')
                            ui.label(masked).classes('flex-grow text-xs opacity-50 font-mono')
                    
                    ui.separator().classes('my-3')
                    
                    # Add key
                    ui.label('Add/Update Key').classes('font-bold mb-2')
                    
                    key_service = ui.select(
                        options=services,
                        value='openai',
                        label='Service',
                    ).classes('w-full')
                    
                    key_input = ui.input(
                        label='API Key',
                        password=True,
                        placeholder='Enter key...',
                    ).classes('w-full')
                    
                    key_status = ui.label('').classes('text-xs')
                    
                    def save_key():
                        if not key_input.value:
                            key_status.text = '⚠️ Enter a key'
                            return
                        try:
                            storage = km.set_key(key_service.value, key_input.value.strip())
                            key_status.text = f'✅ Saved to {storage}'
                            key_input.value = ''
                            log_event(f"API key saved: {key_service.value}")
                            ui.notify(f'Key saved for {key_service.value}', type='positive')
                        except Exception as ex:
                            key_status.text = f'❌ {ex}'
                    
                    ui.button('Save Key', icon='save', on_click=save_key).classes('w-full mt-2')
                
                # Right: Application Settings
                with ui.card().classes('w-1/2 feature-card'):
                    ui.label('⚙️ Application Settings').classes('text-lg font-bold mb-3')
                    
                    # Dark Mode
                    def toggle_dark_mode(e):
                        if e.value:
                            dark.enable()
                            state.dark_mode = True
                        else:
                            dark.disable()
                            state.dark_mode = False
                        log_event(f"Dark mode: {state.dark_mode}")
                    
                    ui.switch('Dark Mode', value=state.dark_mode, on_change=toggle_dark_mode)
                    
                    ui.separator().classes('my-3')
                    
                    # Default Engine
                    def set_default_engine(e):
                        state.default_engine = e.value
                        log_event(f"Default engine: {e.value}")
                    
                    ui.select(
                        options=get_available_engines(),
                        value=state.default_engine,
                        label='Default Translation Engine',
                        on_change=set_default_engine,
                    ).classes('w-full')
                    
                    # Default Options
                    def set_masking(e):
                        state.default_masking = e.value
                        log_event(f"Default masking: {e.value}")
                    
                    def set_reranking(e):
                        state.default_reranking = e.value
                        log_event(f"Default reranking: {e.value}")
                    
                    def set_quality(e):
                        state.quality_passes = int(e.value)
                        log_event(f"Quality passes: {e.value}")
                    
                    ui.switch('Default: Mask Formulas/URLs', value=state.default_masking, on_change=set_masking)
                    ui.switch('Default: Enable Reranking', value=state.default_reranking, on_change=set_reranking)
                    
                    ui.number(
                        label='Default Quality Passes',
                        value=state.quality_passes,
                        min=1, max=5,
                        on_change=set_quality,
                    ).classes('w-full mt-2')
                    
                    ui.separator().classes('my-3')
                    
                    ui.label('About').classes('font-bold')
                    ui.markdown("""
                    **SciTrans-LLMs** v1.0.0
                    
                    Scientific Document Translation with LLMs.
                    Features: PDF layout preservation, glossary management,
                    multiple engines, and more.
                    """).classes('text-sm opacity-70')
    
    # ==========================================================================
    # RUN APPLICATION
    # ==========================================================================
    
    log_event("GUI ready to launch")
    
    print(f"\n{'='*60}")
    print(f"SciTrans-LLMs GUI Starting on http://127.0.0.1:{port}")
    print(f"{'='*60}")
    
    ui.run(
        port=port,
        title='SciTrans-LLMs',
        favicon='🔬',
        show=True,
        reload=False,
        storage_secret='scitrans_secure_session_key_2024',
    )


if __name__ == "__main__":
    launch()
