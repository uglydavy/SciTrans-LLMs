"""
SciTrans-LLMs GUI - Clean NiceGUI Implementation

A functional GUI for scientific document translation.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scitrans-gui")


def launch(port: int = 7860, share: bool = False):
    """Launch the SciTrans-LLMs GUI."""
    try:
        from nicegui import ui, app
    except ImportError:
        raise ImportError("NiceGUI not installed. Run: pip install nicegui>=1.4.0")
    
    # Import components
    from scitrans_llms.keys import KeyManager
    from scitrans_llms.pipeline import PipelineConfig, TranslationPipeline, translate_document
    from scitrans_llms.translate.glossary import get_default_glossary
    from scitrans_llms.models import Document
    
    km = KeyManager()
    
    # State
    uploaded_file_path = [None]  # Use list for mutable reference
    translation_log = []
    
    # =========================================================================
    # MAIN PAGE
    # =========================================================================
    
    @ui.page('/')
    def main_page():
        # Apply dark mode
        ui.dark_mode().enable()
        
        # Header
        with ui.header().classes('bg-primary'):
            ui.label('🔬 SciTrans-LLMs').classes('text-2xl font-bold')
            ui.space()
            ui.label('Scientific Document Translation')
        
        # Main content
        with ui.tabs().classes('w-full') as tabs:
            text_tab = ui.tab('Text Translation', icon='edit')
            pdf_tab = ui.tab('PDF Translation', icon='picture_as_pdf')
            glossary_tab = ui.tab('Glossary', icon='book')
            settings_tab = ui.tab('Settings', icon='settings')
        
        with ui.tab_panels(tabs, value=text_tab).classes('w-full'):
            
            # =================================================================
            # TEXT TRANSLATION TAB
            # =================================================================
            with ui.tab_panel(text_tab):
                with ui.row().classes('w-full gap-4 p-4'):
                    # Left: Input
                    with ui.column().classes('w-1/2'):
                        ui.label('Source Text').classes('text-lg font-bold')
                        text_input = ui.textarea(
                            placeholder='Enter text to translate...',
                        ).classes('w-full').props('rows=12')
                        
                        with ui.row().classes('gap-4 mt-4'):
                            text_direction = ui.select(
                                options=['English → French', 'French → English'],
                                value='English → French',
                                label='Direction',
                            ).classes('w-48')
                            
                            text_engine = ui.select(
                                options=['free', 'dictionary', 'dummy', 'openai', 'deepseek'],
                                value='free',
                                label='Engine',
                            ).classes('w-48')
                        
                        with ui.row().classes('gap-4 mt-2'):
                            text_glossary = ui.checkbox('Use Glossary', value=True)
                            text_masking = ui.checkbox('Mask Formulas', value=True)
                        
                        text_translate_btn = ui.button(
                            'Translate',
                            icon='translate',
                        ).classes('mt-4').props('color=primary size=lg')
                    
                    # Right: Output
                    with ui.column().classes('w-1/2'):
                        ui.label('Translation').classes('text-lg font-bold')
                        text_output = ui.textarea(
                            placeholder='Translation will appear here...',
                        ).classes('w-full').props('rows=12 readonly')
                        
                        text_status = ui.label('').classes('text-sm opacity-70 mt-4')
                
                # Translation handler
                def translate_text():
                    source = text_input.value
                    if not source.strip():
                        text_status.text = '⚠️ Please enter some text to translate.'
                        return
                    
                    text_status.text = '🔄 Translating...'
                    text_translate_btn.disable()
                    
                    try:
                        # Parse direction
                        if text_direction.value == 'English → French':
                            src, tgt = 'en', 'fr'
                        else:
                            src, tgt = 'fr', 'en'
                        
                        config = PipelineConfig(
                            source_lang=src,
                            target_lang=tgt,
                            translator_backend=text_engine.value,
                            enable_glossary=text_glossary.value,
                            enable_masking=text_masking.value,
                            enable_refinement=True,
                        )
                        
                        pipeline = TranslationPipeline(config)
                        doc = Document.from_text(source, src, tgt)
                        result = pipeline.translate(doc)
                        
                        text_output.value = result.translated_text
                        text_status.text = f'✅ Translation complete! ({result.stats.get("translated_blocks", 0)} blocks)'
                        
                    except Exception as e:
                        logger.exception("Translation error")
                        text_output.value = f'Error: {str(e)}'
                        text_status.text = f'❌ Translation failed: {str(e)}'
                    finally:
                        text_translate_btn.enable()
                
                text_translate_btn.on_click(translate_text)
            
            # =================================================================
            # PDF TRANSLATION TAB
            # =================================================================
            with ui.tab_panel(pdf_tab):
                with ui.row().classes('w-full gap-4 p-4'):
                    # Left: Upload and Settings
                    with ui.column().classes('w-1/2'):
                        ui.label('Upload PDF').classes('text-lg font-bold')
                        
                        # File upload area
                        with ui.card().classes('w-full'):
                            ui.icon('upload_file', size='xl').classes('mx-auto')
                            upload_status = ui.label('Drop PDF here or click to upload').classes('text-center')
                            
                            def handle_upload(e):
                                try:
                                    # Save to temp file
                                    temp_dir = Path(tempfile.mkdtemp())
                                    file_path = temp_dir / e.name
                                    file_path.write_bytes(e.content.read())
                                    uploaded_file_path[0] = str(file_path)
                                    upload_status.text = f'✅ Uploaded: {e.name}'
                                    ui.notify(f'File uploaded: {e.name}', type='positive')
                                except Exception as ex:
                                    logger.exception("Upload error")
                                    upload_status.text = f'❌ Upload failed: {str(ex)}'
                                    ui.notify(f'Upload error: {str(ex)}', type='negative')
                            
                            ui.upload(
                                on_upload=handle_upload,
                                auto_upload=True,
                            ).props('accept=".pdf"').classes('w-full')
                        
                        # Settings
                        ui.label('Settings').classes('text-lg font-bold mt-4')
                        
                        with ui.row().classes('gap-4'):
                            pdf_direction = ui.select(
                                options=['English → French', 'French → English'],
                                value='English → French',
                                label='Direction',
                            ).classes('w-48')
                            
                            pdf_engine = ui.select(
                                options=['free', 'dictionary', 'dummy', 'openai', 'deepseek'],
                                value='free',
                                label='Engine',
                            ).classes('w-48')
                        
                        with ui.row().classes('gap-4 mt-2'):
                            pdf_pages = ui.input(
                                label='Pages',
                                value='all',
                                placeholder='all, 1-5, or 1,3,5',
                            ).classes('w-32')
                            
                            pdf_quality = ui.number(
                                label='Quality',
                                value=1,
                                min=1,
                                max=5,
                            ).classes('w-24')
                        
                        with ui.row().classes('gap-4 mt-2'):
                            pdf_glossary = ui.checkbox('Use Glossary', value=True)
                            pdf_masking = ui.checkbox('Preserve Formulas', value=True)
                        
                        pdf_translate_btn = ui.button(
                            'Translate PDF',
                            icon='translate',
                        ).classes('mt-4').props('color=primary size=lg')
                    
                    # Right: Progress and Download
                    with ui.column().classes('w-1/2'):
                        ui.label('Translation Progress').classes('text-lg font-bold')
                        
                        pdf_progress = ui.linear_progress(value=0).classes('w-full')
                        pdf_progress.visible = False
                        
                        pdf_log = ui.textarea(
                            placeholder='Translation log will appear here...',
                        ).classes('w-full').props('rows=10 readonly')
                        
                        pdf_download_link = ui.link('', '').classes('hidden')
                        
                        pdf_download_btn = ui.button(
                            'Download Translated PDF',
                            icon='download',
                        ).classes('mt-4').props('color=positive size=lg')
                        pdf_download_btn.visible = False
                
                # Translated file storage
                translated_file_path = [None]
                
                def translate_pdf():
                    if not uploaded_file_path[0]:
                        ui.notify('Please upload a PDF first', type='warning')
                        return
                    
                    pdf_translate_btn.disable()
                    pdf_progress.visible = True
                    pdf_progress.value = 0
                    pdf_download_btn.visible = False
                    log_lines = []
                    
                    def log(msg):
                        log_lines.append(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}')
                        pdf_log.value = '\n'.join(log_lines)
                    
                    try:
                        log('Starting translation...')
                        pdf_progress.value = 0.1
                        
                        # Parse direction
                        if pdf_direction.value == 'English → French':
                            direction = 'en-fr'
                        else:
                            direction = 'fr-en'
                        
                        # Create output path
                        input_path = Path(uploaded_file_path[0])
                        output_dir = Path(tempfile.mkdtemp())
                        output_path = output_dir / f'translated_{input_path.name}'
                        
                        log(f'Input: {input_path.name}')
                        log(f'Direction: {direction}')
                        log(f'Engine: {pdf_engine.value}')
                        
                        def progress_callback(msg):
                            log(msg)
                        
                        log('Parsing PDF...')
                        pdf_progress.value = 0.3
                        
                        result = translate_document(
                            input_path=str(input_path),
                            output_path=str(output_path),
                            engine=pdf_engine.value,
                            direction=direction,
                            pages=pdf_pages.value,
                            quality_loops=int(pdf_quality.value),
                            progress=progress_callback,
                        )
                        
                        pdf_progress.value = 0.9
                        
                        if result.success:
                            log(f'✅ Translation complete!')
                            log(f'   Blocks: {result.stats.get("translated_blocks", 0)}')
                            
                            # Check for output file
                            if output_path.exists():
                                translated_file_path[0] = str(output_path)
                                log(f'   Output: {output_path.name}')
                                pdf_download_btn.visible = True
                                ui.notify('Translation complete!', type='positive')
                            else:
                                # Check for text fallback
                                text_path = output_path.with_suffix('.txt')
                                if text_path.exists():
                                    translated_file_path[0] = str(text_path)
                                    log(f'   Output: {text_path.name} (text fallback)')
                                    pdf_download_btn.visible = True
                                else:
                                    log('   ⚠️ Output file not created')
                        else:
                            log(f'⚠️ Translation completed with errors:')
                            for err in result.errors:
                                log(f'   - {err}')
                        
                        pdf_progress.value = 1.0
                        
                    except Exception as e:
                        logger.exception("PDF translation error")
                        log(f'❌ Error: {str(e)}')
                        ui.notify(f'Translation failed: {str(e)}', type='negative')
                    finally:
                        pdf_translate_btn.enable()
                
                pdf_translate_btn.on_click(translate_pdf)
                
                async def download_pdf():
                    if translated_file_path[0] and Path(translated_file_path[0]).exists():
                        ui.download(translated_file_path[0])
                    else:
                        ui.notify('No translated file available', type='warning')
                
                pdf_download_btn.on_click(download_pdf)
            
            # =================================================================
            # GLOSSARY TAB
            # =================================================================
            with ui.tab_panel(glossary_tab):
                ui.label('Scientific Terminology Glossary').classes('text-xl font-bold p-4')
                
                with ui.row().classes('w-full gap-4 p-4'):
                    glossary_search = ui.input(
                        label='Search',
                        placeholder='Type to search...',
                    ).classes('w-64')
                    
                    glossary_domain = ui.select(
                        options=['All', 'ml', 'math', 'stats', 'physics', 'chemistry', 'biology'],
                        value='All',
                        label='Domain',
                    ).classes('w-48')
                
                # Load glossary
                try:
                    gloss = get_default_glossary()
                    glossary_data = [
                        {'source': e.source, 'target': e.target, 'domain': e.domain}
                        for e in gloss.entries
                    ]
                except Exception as e:
                    glossary_data = [{'source': 'Error', 'target': str(e), 'domain': ''}]
                
                columns = [
                    {'name': 'source', 'label': 'English', 'field': 'source', 'sortable': True},
                    {'name': 'target', 'label': 'French', 'field': 'target', 'sortable': True},
                    {'name': 'domain', 'label': 'Domain', 'field': 'domain', 'sortable': True},
                ]
                
                glossary_table = ui.table(
                    columns=columns,
                    rows=glossary_data,
                    row_key='source',
                    pagination=20,
                ).classes('w-full')
                
                def filter_glossary():
                    search = glossary_search.value.lower()
                    domain = glossary_domain.value
                    
                    filtered = []
                    for entry in glossary_data:
                        if domain != 'All' and entry['domain'] != domain:
                            continue
                        if search and search not in entry['source'].lower() and search not in entry['target'].lower():
                            continue
                        filtered.append(entry)
                    
                    glossary_table.rows = filtered
                
                glossary_search.on('keyup', filter_glossary)
                glossary_domain.on('change', filter_glossary)
            
            # =================================================================
            # SETTINGS TAB
            # =================================================================
            with ui.tab_panel(settings_tab):
                ui.label('Settings').classes('text-xl font-bold p-4')
                
                with ui.card().classes('w-full max-w-2xl mx-auto p-4'):
                    ui.label('API Keys').classes('text-lg font-bold mb-4')
                    
                    # Show status for each service
                    services = ['openai', 'deepseek', 'anthropic', 'huggingface', 'deepl']
                    
                    for service in services:
                        key = km.get_key(service)
                        status = '✅ Set' if key else '❌ Not set'
                        masked = km._mask_key(key) if key else '-'
                        
                        with ui.row().classes('w-full items-center gap-4 mb-2'):
                            ui.label(service.capitalize()).classes('w-24 font-bold')
                            ui.label(status).classes('w-20')
                            ui.label(masked).classes('flex-grow opacity-70')
                    
                    ui.separator().classes('my-4')
                    
                    # Add new key
                    ui.label('Add/Update API Key').classes('text-lg font-bold mb-4')
                    
                    with ui.row().classes('w-full gap-4'):
                        key_service = ui.select(
                            options=services,
                            label='Service',
                            value='openai',
                        ).classes('w-48')
                        
                        key_input = ui.input(
                            label='API Key',
                            password=True,
                            placeholder='Enter API key...',
                        ).classes('flex-grow')
                        
                        def save_key():
                            if not key_input.value.strip():
                                ui.notify('Key cannot be empty', type='warning')
                                return
                            try:
                                storage = km.set_key(key_service.value, key_input.value.strip())
                                ui.notify(f'Key saved to {storage}', type='positive')
                                key_input.value = ''
                            except Exception as e:
                                ui.notify(f'Error: {str(e)}', type='negative')
                        
                        ui.button('Save', icon='save', on_click=save_key).props('color=primary')
                
                with ui.card().classes('w-full max-w-2xl mx-auto p-4 mt-4'):
                    ui.label('About').classes('text-lg font-bold mb-4')
                    ui.markdown("""
                    **SciTrans-LLMs** - Scientific Document Translation
                    
                    Features:
                    - 🔬 Scientific terminology preservation
                    - 📄 PDF translation with layout preservation
                    - 📋 List structure preservation (1., 1.1, 2., etc.)
                    - 🧠 Multiple translation engines
                    - ⚡ Formula and URL masking
                    """)
    
    # Run
    print(f"\n{'='*60}")
    print(f"SciTrans-LLMs GUI Starting on http://127.0.0.1:{port}")
    print(f"{'='*60}")
    
    ui.run(
        port=port,
        title='SciTrans-LLMs',
        favicon='🔬',
        show=True,
        reload=False,
        storage_secret='scitrans_secret_key',
    )


if __name__ == "__main__":
    launch()
