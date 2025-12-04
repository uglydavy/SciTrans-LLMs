# 🚀 How to Run SciTrans-LLMs

## ✅ System is Working! Here's How to Use It:

### 1. Command Line (Recommended)

```bash
# Test translation
~/.local/bin/scitran demo

# See all commands
~/.local/bin/scitran --help

# Translate a PDF
~/.local/bin/scitran translate --input paper.pdf --output translated.pdf --target fr

# Get system info
~/.local/bin/scitran info
```

### 2. Direct Python Scripts (Alternative)

```bash
# Quick translation test
python3 run_translate.py

# Launch simple GUI
python3 run_gui_simple.py

# Generate thesis data
python3 thesis/generate_thesis_data.py
```

### 3. GUI Interface

```bash
# Try the main GUI (may have issues with NiceGUI)
~/.local/bin/scitran gui

# Or use the simplified GUI (more stable)
python3 run_gui_simple.py
```

## 📝 Quick Examples

### Translate Text
```python
from scitran_llms.pipeline import TranslationPipeline, PipelineConfig

config = PipelineConfig(backend="googletrans", source_lang="en", target_lang="fr")
pipeline = TranslationPipeline(config)
result = pipeline.translate_text("Hello world")
print(result)  # "Bonjour le monde"
```

### Translate PDF
```bash
# Download a paper
wget https://arxiv.org/pdf/2301.00001.pdf -O paper.pdf

# Translate it
~/.local/bin/scitran translate --input paper.pdf --output paper_fr.pdf --target fr
```

## ⚠️ If You Get Errors

### ModuleNotFoundError
```bash
# Reinstall the package
pip3 install -e .
```

### httpcore/httpx version conflicts
```bash
# Install compatible versions
pip3 install 'httpcore==0.9.*' 'httpx==0.13.3' --force-reinstall
```

### NiceGUI issues
```bash
# Use the direct scripts instead
python3 run_translate.py  # For CLI
python3 run_gui_simple.py  # For GUI
```

## 📊 Generate Thesis Data

```bash
# All plots and tables
python3 thesis/generate_thesis_data.py

# View results
ls thesis/results/
# - comparison_table.csv
# - ablation_study.png
# - training_curves.png
# - error_analysis.png
# - dataset_visualization.png
```

## 🎯 Current Status

✅ **Working:**
- Translation engine (Google Translate)
- PDF parsing and rendering
- CLI commands via `~/.local/bin/scitran`
- Direct Python scripts
- Thesis data generation

⚠️ **May have issues:**
- NiceGUI interface (dependency conflicts)
- Use alternative scripts if GUI fails

## 💡 Tips

1. **Always use full path for scitran:**
   ```bash
   ~/.local/bin/scitran [command]
   ```

2. **For quick testing:**
   ```bash
   python3 run_translate.py
   ```

3. **For thesis work:**
   ```bash
   python3 thesis/generate_thesis_data.py
   ```

4. **Check installation:**
   ```bash
   ~/.local/bin/scitran info
   ```

---

**The system is working!** Use the commands above to translate PDFs and generate thesis data.
