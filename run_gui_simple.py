#!/usr/bin/env python3
"""Simple GUI launcher without dependency issues"""

import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    try:
        from scitran_llms.gui import launch
        print("\n" + "="*50)
        print("Launching SciTrans-LLMs GUI...")
        print("="*50)
        print("\nGUI will open at: http://localhost:7860")
        print("Press Ctrl+C to stop\n")
        launch()
    except ImportError as e:
        print(f"Error importing GUI: {e}")
        print("\nTrying alternative method...")
        
        # Alternative: Direct GUI creation
        try:
            import nicegui
            from nicegui import ui
            
            @ui.page('/')
            def main_page():
                ui.dark_mode().enable()
                ui.label("SciTrans-LLMs").classes('text-h3')
                ui.label("PDF Translation System").classes('text-subtitle1')
                
                with ui.tabs().classes('w-full') as tabs:
                    translate_tab = ui.tab('Translate')
                    testing_tab = ui.tab('Testing')
                    
                with ui.tab_panels(tabs, value=translate_tab).classes('w-full'):
                    with ui.tab_panel(translate_tab):
                        ui.label("Upload a PDF to translate")
                        ui.upload(label="Upload PDF", accept='.pdf')
                        
                        with ui.row():
                            ui.select(['en', 'fr'], label='Source Language', value='en')
                            ui.select(['en', 'fr'], label='Target Language', value='fr')
                        
                        ui.button('Translate', color='primary')
                        
                    with ui.tab_panel(testing_tab):
                        ui.label("Testing Panel")
                        ui.textarea(label="Test Text", value="Enter text to translate")
                        ui.button('Test Translation')
            
            ui.run(host='0.0.0.0', port=7860, title='SciTrans-LLMs')
            
        except Exception as e2:
            print(f"Alternative method failed: {e2}")
            print("\nPlease run: python3 setup.sh")

if __name__ == "__main__":
    main()
