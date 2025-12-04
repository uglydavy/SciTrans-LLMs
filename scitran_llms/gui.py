"""
NiceGUI-based interface for SciTrans-LLMs with dark mode and full features.
"""

import asyncio
import io
import tempfile
import os
from pathlib import Path
from typing import Optional, Dict, Any
from nicegui import ui, app, events
from nicegui.events import UploadEventArguments
import logging

from .pipeline import TranslationPipeline, PipelineConfig
from .models import Document, Block, BlockType, Glossary
from .config import BACKENDS, LANGUAGES, DEFAULT_BACKEND, MODEL_DESCRIPTIONS, GUI_PORT
from . import __version__

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')
logger = logging.getLogger('scitran-gui')

# Global state
state = {
    "dark_mode": True,
    "current_document": None,
    "translated_document": None,
    "pipeline": None,
    "glossary": Glossary(),
    "source_pdf": None,
    "backend": DEFAULT_BACKEND,
    "source_lang": "en",
    "target_lang": "fr",
    "enable_masking": True,
    "enable_glossary": True,
    "enable_reranking": False,
    "num_candidates": 1,
}


def launch(port: int = GUI_PORT, reload: bool = False):
    """Launch the GUI."""
    logger.info("GUI init")
    
    # Dark mode CSS
    dark_css = """
    <style>
    .dark {
        background-color: #1e1e1e !important;
        color: #e0e0e0 !important;
    }
    .q-page {
        min-height: 100vh;
    }
    .q-tab-panels {
        height: calc(100vh - 150px);
        overflow-y: auto;
    }
    .q-card {
        background-color: #2d2d2d;
        color: #e0e0e0;
    }
    .dark .q-card {
        background-color: #2d2d2d !important;
    }
    .dark .q-btn {
        background-color: #3d3d3d !important;
    }
    .dark .q-input, .dark .q-select {
        background-color: #2d2d2d !important;
        color: #e0e0e0 !important;
    }
    </style>
    """
    
    @ui.page('/')
    async def main_page():
        """Main page with tabs."""
        # Add dark CSS
        ui.add_head_html(dark_css)
        
        # Apply dark mode class
        if state["dark_mode"]:
            ui.query('body').classes('dark')
        
        # Header
        with ui.header().classes('bg-primary'):
            with ui.row().classes('w-full items-center'):
                ui.label('SciTrans-LLMs').classes('text-2xl font-bold')
                ui.space()
                
                # Dark mode toggle
                dark_switch = ui.switch('Dark Mode', value=state["dark_mode"])
                dark_switch.on('update:model-value', lambda e: toggle_dark_mode())
                
                ui.label(f'v{__version__}').classes('text-sm')
        
        # Main content with tabs
        with ui.tabs().classes('w-full') as tabs:
            translate_tab = ui.tab('Translate', icon='translate')
            testing_tab = ui.tab('Testing', icon='science')
            glossary_tab = ui.tab('Glossary', icon='book')
            developer_tab = ui.tab('Developer', icon='code')
            settings_tab = ui.tab('Settings', icon='settings')
        
        with ui.tab_panels(tabs, value=translate_tab).classes('w-full q-tab-panels'):
            # Translate Tab
            with ui.tab_panel(translate_tab).classes('p-4'):
                await create_translate_tab()
            
            # Testing Tab  
            with ui.tab_panel(testing_tab).classes('p-4'):
                await create_testing_tab()
            
            # Glossary Tab
            with ui.tab_panel(glossary_tab).classes('p-4'):
                await create_glossary_tab()
            
            # Developer Tab
            with ui.tab_panel(developer_tab).classes('p-4'):
                await create_developer_tab()
            
            # Settings Tab
            with ui.tab_panel(settings_tab).classes('p-4'):
                await create_settings_tab()
    
    def toggle_dark_mode():
        """Toggle dark mode."""
        state["dark_mode"] = not state["dark_mode"]
        if state["dark_mode"]:
            ui.query('body').classes('dark')
        else:
            ui.query('body').classes(remove='dark')
    
    async def create_translate_tab():
        """Create translation tab."""
        with ui.grid(columns=2).classes('w-full gap-4'):
            # Left column - Input
            with ui.column().classes('gap-2'):
                ui.label('Input').classes('text-lg font-bold')
                
                # Language and backend selection
                with ui.row().classes('w-full gap-2'):
                    source_select = ui.select(
                        options=list(LANGUAGES.keys()),
                        value=state["source_lang"],
                        label='Source Language'
                    ).classes('flex-1')
                    source_select.on('update:model-value', 
                                   lambda e: update_state('source_lang', e.value))
                    
                    ui.icon('arrow_forward')
                    
                    target_select = ui.select(
                        options=list(LANGUAGES.keys()),
                        value=state["target_lang"],
                        label='Target Language'
                    ).classes('flex-1')
                    target_select.on('update:model-value',
                                   lambda e: update_state('target_lang', e.value))
                
                # Backend selection with description
                backend_select = ui.select(
                    options=list(BACKENDS.keys()),
                    value=state["backend"],
                    label='Translation Backend'
                ).classes('w-full')
                backend_select.on('update:model-value',
                                lambda e: update_state('backend', e.value))
                
                # Show backend description
                backend_desc = ui.label(MODEL_DESCRIPTIONS.get(state["backend"], ""))
                backend_desc.classes('text-sm text-gray-600 dark:text-gray-400')
                
                # Options
                with ui.row().classes('gap-4'):
                    ui.switch('Enable Masking', value=state["enable_masking"]).on(
                        'update:model-value', lambda e: update_state('enable_masking', e.value))
                    ui.switch('Use Glossary', value=state["enable_glossary"]).on(
                        'update:model-value', lambda e: update_state('enable_glossary', e.value))
                
                # Input methods
                ui.separator()
                
                # Text input
                text_input = ui.textarea('Enter text to translate', 
                                        placeholder='Type or paste text here...')
                text_input.classes('w-full h-48')
                
                # File upload
                ui.label('Or upload a file:').classes('mt-4')
                upload = ui.upload(
                    on_upload=lambda e: handle_file_upload(e),
                    auto_upload=True,
                    label='Upload PDF/TXT'
                ).classes('w-full')
                upload.props('accept=".pdf,.txt,.docx"')
                
                # Translate button
                translate_btn = ui.button('Translate', icon='translate', 
                                        on_click=lambda: translate_content(text_input.value))
                translate_btn.classes('w-full mt-4').props('color=primary size=lg')
            
            # Right column - Output
            with ui.column().classes('gap-2'):
                ui.label('Output').classes('text-lg font-bold')
                
                # Output display
                output_area = ui.textarea('Translation will appear here', 
                                         placeholder='Waiting for translation...')
                output_area.classes('w-full h-96')
                output_area.props('readonly')
                
                # Stats
                stats_label = ui.label('Ready').classes('text-sm')
                
                # Export options
                with ui.row().classes('gap-2 mt-4'):
                    ui.button('Copy', icon='content_copy',
                             on_click=lambda: copy_output(output_area.value))
                    ui.button('Download TXT', icon='download',
                             on_click=lambda: download_txt(output_area.value))
                    ui.button('Download PDF', icon='picture_as_pdf',
                             on_click=lambda: download_pdf())
        
        # Progress indicator
        progress = ui.linear_progress(value=0, show_value=False).classes('mt-4')
        progress.visible = False
        
        # Store references
        state['output_area'] = output_area
        state['stats_label'] = stats_label
        state['progress'] = progress
        state['backend_desc'] = backend_desc
        state['text_input'] = text_input
    
    async def create_testing_tab():
        """Create testing tab with advanced features."""
        ui.label('Advanced Testing & Training').classes('text-xl font-bold mb-4')
        
        with ui.tabs().classes('w-full') as test_tabs:
            corpus_tab = ui.tab('Corpus Management')
            prompt_tab = ui.tab('Prompt Training')
            eval_tab = ui.tab('Evaluation')
            benchmark_tab = ui.tab('Benchmarks')
        
        with ui.tab_panels(test_tabs, value=corpus_tab).classes('w-full'):
            # Corpus Management
            with ui.tab_panel(corpus_tab).classes('p-4'):
                ui.label('Parallel Corpus Management').classes('text-lg font-bold')
                
                with ui.row().classes('gap-4 mb-4'):
                    ui.button('Download WMT Corpus', icon='download')
                    ui.button('Download UN Corpus', icon='download')
                    ui.button('Upload Custom Corpus', icon='upload')
                
                ui.separator()
                
                ui.label('Dictionary Training').classes('text-lg font-bold mt-4')
                with ui.row().classes('gap-2'):
                    ui.button('Train from Corpus', icon='model_training')
                    ui.button('Export Dictionary', icon='save')
                    ui.button('Import Dictionary', icon='folder_open')
            
            # Prompt Training
            with ui.tab_panel(prompt_tab).classes('p-4'):
                ui.label('Prompt Optimization').classes('text-lg font-bold')
                
                ui.textarea('Custom Prompt Template', 
                          value='Translate the following {source_lang} text to {target_lang}:\\n{text}',
                          placeholder='Enter prompt template...')
                          
                ui.label('Prompt Training Algorithm').classes('mt-4')
                with ui.row().classes('gap-2'):
                    ui.select(['Random Search', 'Bayesian Optimization', 'Genetic Algorithm'],
                            value='Bayesian Optimization', label='Algorithm')
                    ui.number('Iterations', value=100, min=10, max=1000)
                
                ui.button('Start Training', icon='play_arrow').classes('mt-4')
            
            # Evaluation
            with ui.tab_panel(eval_tab).classes('p-4'):
                ui.label('Translation Evaluation').classes('text-lg font-bold')
                
                with ui.row().classes('gap-4'):
                    ui.select(['BLEU', 'chrF', 'METEOR', 'BERTScore', 'COMET'],
                            value='BLEU', label='Metric')
                    ui.button('Run Evaluation', icon='analytics')
                
                # Results display
                ui.label('Evaluation Results').classes('text-lg mt-4')
                with ui.card().classes('w-full'):
                    ui.label('No evaluation results yet').classes('text-gray-500')
            
            # Benchmarks
            with ui.tab_panel(benchmark_tab).classes('p-4'):
                ui.label('Benchmark Testing').classes('text-lg font-bold')
                
                ui.button('Download Test PDFs', icon='download').classes('mb-4')
                
                with ui.row().classes('gap-2'):
                    ui.button('Run Full Benchmark', icon='speed')
                    ui.button('Generate Report', icon='assessment')
                    ui.button('Export Graphs', icon='show_chart')
    
    async def create_glossary_tab():
        """Create glossary management tab."""
        ui.label('Glossary Management').classes('text-xl font-bold mb-4')
        
        # Glossary table
        columns = [
            {'name': 'source', 'label': 'Source Term', 'field': 'source'},
            {'name': 'target', 'label': 'Target Term', 'field': 'target'},
            {'name': 'domain', 'label': 'Domain', 'field': 'domain'},
        ]
        
        rows = [entry.to_dict() for entry in state['glossary'].entries]
        
        table = ui.table(columns=columns, rows=rows, row_key='source')
        table.classes('w-full')
        
        # Add entry form
        with ui.row().classes('gap-2 mt-4'):
            source_input = ui.input('Source term')
            target_input = ui.input('Target term')
            domain_input = ui.input('Domain', value='general')
            ui.button('Add', icon='add', on_click=lambda: add_glossary_entry(
                source_input.value, target_input.value, domain_input.value, table))
        
        # Import/Export
        with ui.row().classes('gap-2 mt-4'):
            ui.button('Import CSV', icon='upload')
            ui.button('Export CSV', icon='download')
            ui.button('Clear All', icon='delete')
    
    async def create_developer_tab():
        """Create developer tools tab."""
        ui.label('Developer Tools').classes('text-xl font-bold mb-4')
        
        # System info
        with ui.card().classes('w-full mb-4'):
            ui.label('System Information').classes('text-lg font-bold')
            ui.label(f'Version: {__version__}')
            ui.label(f'Python: {os.sys.version.split()[0]}')
            ui.label(f'Platform: {os.sys.platform}')
        
        # API testing
        with ui.card().classes('w-full mb-4'):
            ui.label('API Testing').classes('text-lg font-bold')
            
            test_text = ui.input('Test text', value='Hello world')
            test_btn = ui.button('Test All Backends', icon='bug_report')
            
            results_area = ui.textarea('Results will appear here')
            results_area.classes('w-full h-48 mt-2')
            results_area.props('readonly')
        
        # Logs
        with ui.card().classes('w-full'):
            ui.label('System Logs').classes('text-lg font-bold')
            log_area = ui.log(max_lines=20).classes('w-full h-48')
    
    async def create_settings_tab():
        """Create settings tab."""
        ui.label('Settings').classes('text-xl font-bold mb-4')
        
        # API Keys
        with ui.card().classes('w-full mb-4'):
            ui.label('API Keys').classes('text-lg font-bold mb-2')
            
            with ui.column().classes('gap-2'):
                openai_key = ui.input('OpenAI API Key', 
                                     value=os.getenv('OPENAI_API_KEY', ''),
                                     password=True)
                deepseek_key = ui.input('DeepSeek API Key',
                                       value=os.getenv('DEEPSEEK_API_KEY', ''),
                                       password=True)
                anthropic_key = ui.input('Anthropic API Key',
                                        value=os.getenv('ANTHROPIC_API_KEY', ''),
                                        password=True)
                
                ui.button('Save Keys', icon='save').classes('mt-2')
        
        # Advanced Settings
        with ui.card().classes('w-full'):
            ui.label('Advanced Settings').classes('text-lg font-bold mb-2')
            
            with ui.column().classes('gap-2'):
                ui.number('Context Window', value=3, min=0, max=10)
                ui.number('Chunk Size', value=5000, min=100, max=10000)
                ui.number('Max Retries', value=3, min=1, max=10)
                ui.switch('Enable Caching', value=True)
                ui.switch('Debug Mode', value=False)
    
    def update_state(key: str, value: Any):
        """Update state variable."""
        state[key] = value
        if key == 'backend' and 'backend_desc' in state:
            state['backend_desc'].text = MODEL_DESCRIPTIONS.get(value, "")
    
    async def handle_file_upload(e: UploadEventArguments):
        """Handle file upload."""
        content = e.content.read()
        filename = e.name
        
        # Save temporarily
        temp_path = Path(tempfile.gettempdir()) / filename
        temp_path.write_bytes(content)
        
        state['source_file'] = temp_path
        ui.notify(f'File uploaded: {filename}', type='positive')
        
        # If PDF, parse and display
        if filename.lower().endswith('.pdf'):
            from .ingest import parse_pdf
            document = parse_pdf(temp_path)
            state['current_document'] = document
            
            # Display first page text in input
            if document.blocks and 'text_input' in state:
                preview_text = '\\n\\n'.join(b.text[:200] for b in document.blocks[:3])
                state['text_input'].value = preview_text + '\\n\\n[... PDF continues ...]'
    
    async def translate_content(text: str):
        """Translate content."""
        if 'progress' in state:
            state['progress'].visible = True
            state['progress'].value = 0
        
        if 'stats_label' in state:
            state['stats_label'].text = 'Translating...'
        
        try:
            # Create pipeline
            config = PipelineConfig(
                backend=state['backend'],
                source_lang=state['source_lang'],
                target_lang=state['target_lang'],
                enable_masking=state['enable_masking'],
                enable_glossary=state['enable_glossary'],
                enable_reranking=state['enable_reranking'],
                num_candidates=state['num_candidates'],
            )
            
            pipeline = TranslationPipeline(config)
            
            # Set glossary if enabled
            if state['enable_glossary'] and state['glossary']:
                pipeline.set_glossary(state['glossary'])
            
            # Translate
            if state.get('current_document'):
                # Translate document
                result = pipeline.translate_document(state['current_document'])
                state['translated_document'] = result.document
                
                # Display translation
                translation = '\\n\\n'.join(
                    b.translation or b.text for b in result.document.blocks
                )
                
                if 'stats_label' in state:
                    state['stats_label'].text = (
                        f"Success rate: {result.success_rate:.1%} | "
                        f"Time: {result.time_taken:.2f}s"
                    )
            else:
                # Translate text
                translation = pipeline.translate_text(text)
                
                if 'stats_label' in state:
                    state['stats_label'].text = 'Translation complete'
            
            # Update output
            if 'output_area' in state:
                state['output_area'].value = translation
            
            ui.notify('Translation complete!', type='positive')
            
        except Exception as e:
            ui.notify(f'Translation error: {str(e)}', type='negative')
            if 'stats_label' in state:
                state['stats_label'].text = f'Error: {str(e)}'
        
        finally:
            if 'progress' in state:
                state['progress'].visible = False
    
    def copy_output(text: str):
        """Copy output to clipboard."""
        ui.run_javascript(f'navigator.clipboard.writeText(`{text}`)')
        ui.notify('Copied to clipboard!', type='positive')
    
    def download_txt(text: str):
        """Download text file."""
        # Create download link
        ui.download(text.encode('utf-8'), 'translation.txt')
    
    def download_pdf():
        """Download PDF file."""
        if state.get('translated_document'):
            # Render to PDF
            from .render import render_pdf
            temp_path = Path(tempfile.gettempdir()) / 'translation.pdf'
            render_pdf(state['translated_document'], temp_path)
            
            # Download
            ui.download(temp_path.read_bytes(), 'translation.pdf')
        else:
            ui.notify('No translated document available', type='warning')
    
    def add_glossary_entry(source: str, target: str, domain: str, table):
        """Add glossary entry."""
        if source and target:
            state['glossary'].add_entry(source, target, domain)
            # Update table
            table.rows = [entry.to_dict() for entry in state['glossary'].entries]
            table.update()
            ui.notify(f'Added: {source} → {target}', type='positive')
    
    logger.info("GUI ready")
    print("\\n" + "="*60)
    print(f"SciTrans-LLMs GUI - http://127.0.0.1:{port}")
    print("="*60)
    
    ui.run(
        port=port,
        title='SciTrans-LLMs',
        favicon='🌍',
        dark=state['dark_mode'],
        reload=reload,
        show=False,
    )
