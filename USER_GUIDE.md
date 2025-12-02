# SciTrans-LLMs User Guide

Complete guide to using SciTrans-LLMs for document translation.

## Quick Start (30 seconds)

```bash
# 1. Activate virtual environment
source .venv/bin/activate  # or: .venv/bin/scitrans for direct use

# 2. Translate with FREE backend (no API key needed!)
scitrans translate --text "Machine learning is amazing" --backend free

# 3. See translation instantly!
```

## Available Translation Backends

### 🆓 FREE Options (No API Key Required)

#### `free` - Smart Cascading Translator ⭐ RECOMMENDED
**Best free option!** Tries multiple services automatically with caching.

- **How**: Lingva → LibreTranslate → MyMemory → Dictionary
- **Quality**: ⭐⭐⭐⭐ (Excellent)
- **Speed**: Fast (cached translations instant)
- **Offline**: Works for cached content

```bash
scitrans translate --input paper.pdf --backend free
```

#### `dictionary` - Offline with 1000+ Words
**Good for**: Offline work, academic terms

- **Quality**: ⭐⭐⭐ (Good)
- **Offline**: Yes (after first download)
- **Words**: 1000+ automatically loaded; can be further improved from corpora via `scitrans corpus build-dict`

```bash
scitrans translate --input paper.pdf --backend dictionary
```

### 💰 Paid Options (Best Quality)

#### `deepseek` - Best Value ⭐
**Cost**: $0.01 per 100 pages | **Quality**: ⭐⭐⭐⭐⭐

```bash
export DEEPSEEK_API_KEY="sk-..."
scitrans translate --input paper.pdf --backend deepseek
```

#### OpenAI Models

| Backend | Cost/100 pages | Quality | Best For |
|---------|---------------|---------|----------|
| `gpt4mini` | $0.01 | ⭐⭐⭐⭐⭐ | Cheap & excellent |
| `gpt4o` | $0.19 | ⭐⭐⭐⭐⭐⭐ | Premium quality |
| `o1-mini` | $0.23 | ⭐⭐⭐⭐⭐⭐ | Best accuracy |
| `o1-preview` | $1.13 | ⭐⭐⭐⭐⭐⭐ | Publication-grade |

```bash
export OPENAI_API_KEY="sk-..."
scitrans translate --input paper.pdf --backend gpt4o
```

#### Anthropic Claude Models

| Backend | Cost/100 pages | Quality |
|---------|---------------|---------|
| `claude-3-5-haiku` | $0.08 | ⭐⭐⭐⭐⭐ |
| `claude-3-5-sonnet` | $0.23 | ⭐⭐⭐⭐⭐⭐ |

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
scitrans translate --input paper.pdf --backend claude-3-5-sonnet
```

## Common Commands

### Translate Text

```bash
# Quick translation
scitrans translate --text "Hello world" --backend free

# With specific languages
scitrans translate --text "Hello" --source en --target fr --backend free
```

### Translate Files

```bash
# PDF to PDF
scitrans translate --input paper.pdf --output paper_fr.pdf --backend free

# Text to text
scitrans translate --input doc.txt --output doc_fr.txt --backend free
```

### Use Custom Glossary

```bash
# Create glossary.csv:
# machine learning,apprentissage automatique
# neural network,réseau de neurones

scitrans translate --input paper.pdf --glossary glossary.csv --backend free
```

### Check System Status

```bash
# Check version
scitrans --version

# List API keys
scitrans keys list

# Show available backends
scitrans info
```

### Launch GUI

```bash
# Install Gradio first (if needed)
pip install "gradio>=4.0.0"

# Launch
scitrans gui
```

## Backend Selection Guide

### Choose `free` when:
- ✅ Budget is $0
- ✅ Testing the system
- ✅ Personal projects
- ✅ Learning/experimenting

### Choose `deepseek` when:
- ✅ Need good quality at low cost ($0.01/100 pages)
- ✅ Production work
- ✅ Large volumes

### Choose premium (`gpt4o`, `claude`, `o1`) when:
- ✅ Publication-quality needed
- ✅ Critical documents
- ✅ Best accuracy required

## Setup API Keys

### DeepSeek (Recommended - Almost Free)

1. Sign up at https://platform.deepseek.com
2. Get $5 free credits
3. Set your key:

```bash
export DEEPSEEK_API_KEY="sk-..."
# Or add to ~/.bashrc or ~/.zshrc
```

### OpenAI

```bash
export OPENAI_API_KEY="sk-..."
```

### Anthropic (Claude)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Examples

### Example 1: Free Translation

```bash
# No setup required!
scitrans translate \
  --input my_paper.pdf \
  --output my_paper_fr.pdf \
  --backend free
```

### Example 2: High-Quality Translation

```bash
# Sign up for DeepSeek ($5 free credits)
export DEEPSEEK_API_KEY="sk-..."

scitrans translate \
  --input important_paper.pdf \
  --output important_paper_fr.pdf \
  --backend deepseek
```

### Example 3: Batch Processing

```bash
# Process multiple files
for file in papers/*.pdf; do
  output="translated/${file%.pdf}_fr.pdf"
  scitrans translate --input "$file" --output "$output" --backend free
done
```

### Example 4: With Custom Terms

```bash
# Create my_terms.csv:
# deep learning,apprentissage profond
# neural network,réseau de neurones
# attention mechanism,mécanisme d'attention

scitrans translate \
  --input paper.pdf \
  --output paper_fr.pdf \
  --glossary my_terms.csv \
  --backend deepseek
```

## Cost Comparison (100-Page Document)

| Backend | Cost | Quality | Speed | Recommendation |
|---------|------|---------|-------|----------------|
| `free` | FREE | ⭐⭐⭐⭐ | Fast | ✅ Start here |
| `deepseek` | $0.01 | ⭐⭐⭐⭐⭐ | Fast | ⭐ Best value |
| `gpt4mini` | $0.01 | ⭐⭐⭐⭐⭐ | Fast | Good alternative |
| `gpt4o` | $0.19 | ⭐⭐⭐⭐⭐⭐ | Fast | Premium |
| `claude-3-5-sonnet` | $0.23 | ⭐⭐⭐⭐⭐⭐ | Medium | Best prose |
| `o1-mini` | $0.23 | ⭐⭐⭐⭐⭐⭐ | Slow | Most accurate |

## FAQ

### Q: Which backend should I use?
**A:** Start with `free` (zero cost). Upgrade to `deepseek` ($0.01) for better quality.

### Q: What's the best free option?
**A:** The `free` backend - it cascades through multiple services and caches results.

### Q: Is GPT-5 available?
**A:** No, GPT-5 doesn't exist yet. Latest is GPT-4o and o1-series.

### Q: How do I work offline?
**A:** Use `dictionary` backend. First use downloads 1000+ words, then works offline.

### Q: What if I hit rate limits with free backends?
**A:** The `free` backend uses caching and multiple services, so limits are rare. For unlimited use, try `deepseek` at $0.01/100 pages.

### Q: Can I use this commercially?
**A:** Free backends have usage limits. For commercial use, we recommend paid backends like `deepseek`.


## Troubleshooting

### "scitrans: command not found"

```bash
# Option 1: Use full path
.venv/bin/scitrans --version

# Option 2: Activate virtual environment
source .venv/bin/activate
scitrans --version
```

### "Translation failed" with free backend

- Check internet connection
- Try again (service might be temporarily down)
- Or upgrade to `deepseek`: `--backend deepseek`

### API key not recognized

```bash
# Check if set
echo $OPENAI_API_KEY

# Set it
export OPENAI_API_KEY="sk-..."

# Verify
scitrans keys list
```

## Advanced Usage

### Pipeline Options

```bash
# Disable refinement (faster)
scitrans translate --input paper.pdf --no-refinement --backend free

# Disable glossary
scitrans translate --input paper.pdf --no-glossary --backend free

# Disable masking
scitrans translate --input paper.pdf --no-masking --backend free
```

### Evaluation

```bash
# Evaluate translation quality
scitrans evaluate \
  --hyp translated.txt \
  --ref reference.txt
```

### Run Experiments

```bash
# Full pipeline
python3 scripts/full_pipeline.py --backend deepseek
```

## Getting Help

```bash
# General help
scitrans --help

# Command-specific help
scitrans translate --help
scitrans keys --help
scitrans evaluate --help
```

## Summary

1. **Start free**: Use `free` backend (no setup)
2. **Upgrade cheap**: Use `deepseek` ($0.01/100 pages)
3. **Go premium**: Use `gpt4o` or `claude` when quality matters

**The system is ready to use right now!**

---

For installation instructions, see `INSTALL.md`  
For development/research, see `EXPERIMENTS.md` and `THESIS_GUIDE.md`



