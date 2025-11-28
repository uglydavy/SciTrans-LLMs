# 🌐 SciTrans-LLMs

**Adaptive Document Translation Enhanced by Technology based on LLMs**

Research-grade layout-preserving PDF translator with multiple backends from free to premium.

## Quick Start (30 seconds)

```bash
# 1. Install
pip install -e .

# 2. Translate for FREE (no API key needed!)
scitrans translate --text "Machine learning is amazing" --backend free

# 3. Or translate a PDF
scitrans translate --input paper.pdf --output paper_fr.pdf --backend free
```

📖 **[Read the complete USER_GUIDE.md →](USER_GUIDE.md)**

## Features

✅ **Free Translation** - Multiple free services with smart fallbacks  
✅ **Latest AI Models** - GPT-4o, o1, Claude 3.5, DeepSeek  
✅ **Layout Preservation** - Maintains PDF formatting  
✅ **Smart Caching** - Offline capability for cached translations  
✅ **Custom Glossaries** - Domain-specific terminology  
✅ **Multiple Backends** - From free to premium quality  

## Translation Backends

| Backend | Cost | Quality | Best For |
|---------|------|---------|----------|
| `free` ⭐ | FREE | ⭐⭐⭐⭐ | Testing, personal use |
| `deepseek` 💰 | $0.01/100 pages | ⭐⭐⭐⭐⭐ | Best value |
| `gpt4o` | $0.19/100 pages | ⭐⭐⭐⭐⭐⭐ | Premium quality |
| `o1-mini` | $0.23/100 pages | ⭐⭐⭐⭐⭐⭐ | Most accurate |

[See all backends →](USER_GUIDE.md#available-translation-backends)

## Installation

See [INSTALL.md](INSTALL.md) for detailed instructions.

```bash
# Basic installation
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# With all features
pip install -e ".[full]"
```

## Usage Examples

### Free Translation (No API Key)

```bash
# The new cascading free translator
scitrans translate --input paper.pdf --output paper_fr.pdf --backend free
```

### Paid Translation (Best Quality)

```bash
# DeepSeek - Best value ($0.01/100 pages)
export DEEPSEEK_API_KEY="sk-..."
scitrans translate --input paper.pdf --output paper_fr.pdf --backend deepseek

# GPT-4o - Premium quality
export OPENAI_API_KEY="sk-..."
scitrans translate --input paper.pdf --output paper_fr.pdf --backend gpt4o
```

### With Custom Glossary

```bash
scitrans translate \
  --input paper.pdf \
  --output paper_fr.pdf \
  --glossary my_terms.csv \
  --backend deepseek
```

### GUI Interface

```bash
pip install "gradio>=4.0.0"
scitrans gui
```

## 📚 Documentation

- **[USER_GUIDE.md](USER_GUIDE.md)** ⭐ - Complete usage guide (START HERE!)
- **[INSTALL.md](INSTALL.md)** - Installation instructions  
- **[EXPERIMENTS.md](EXPERIMENTS.md)** - Running experiments
- **[THESIS_GUIDE.md](THESIS_GUIDE.md)** - Research documentation
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contributing guidelines

## API Keys Setup

### DeepSeek (Recommended - $0.01/100 pages)

1. Sign up at https://platform.deepseek.com (get $5 free!)
2. Get your API key
3. Set environment variable:

```bash
export DEEPSEEK_API_KEY="sk-..."
```

### Other Providers

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic (Claude)
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Commands

```bash
# Translate
scitrans translate --input file.pdf --output file_fr.pdf --backend free

# Check API keys
scitrans keys list

# Show system info
scitrans info

# Run demo
scitrans demo

# Launch GUI
scitrans gui

# Get help
scitrans --help
```

## Architecture

```
Document Input
    ↓
PDF Parser (Layout Detection)
    ↓
Block Segmentation
    ↓
Masking (LaTeX, URLs, etc.)
    ↓
Translation (Multiple Backends)
    ↓
Refinement & Glossary
    ↓
Unmasking & Layout Reconstruction
    ↓
Translated Document
```

## Backend Options

### Free Backends
- **`free`** - Cascading translator (Lingva → LibreTranslate → MyMemory)
- **`dictionary`** - Offline with 1000+ words
- **`dummy`** - Testing only

### Paid Backends
- **`deepseek`** - Best value ($0.001/1K tokens)
- **`gpt4o`** - OpenAI's best general model
- **`gpt4mini`** - Cheaper OpenAI model
- **`o1-mini`** - Reasoning model
- **`claude-3-5-sonnet`** - Anthropic's best

[Full comparison →](USER_GUIDE.md#backend-selection-guide)

## Research

This tool was developed as part of research on LLM-based document translation:

- Layout-preserving translation
- Glossary-aware translation
- Context-aware translation
- Multi-backend evaluation

See [THESIS_GUIDE.md](THESIS_GUIDE.md) for research details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{scitrans_llms,
  title={SciTrans-LLMs: Adaptive Document Translation Enhanced by Technology based on LLMs},
  author={Tchienkoua Franck-Davy},
  year={2024},
  url={https://github.com/uglydavy/SciTrans-LLMs}
}
```

## Support

- 📖 Read the [USER_GUIDE.md](USER_GUIDE.md)
- 🐛 Report issues on GitHub
- 💬 Use `--help` for any command

---

**Get started in 30 seconds:**

```bash
scitrans translate --text "Hello, world!" --backend free
```
