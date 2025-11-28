# Translation System Improvements

This document describes the improvements made to address translation completeness and provide free testing options.

## Problems Solved

### 1. Incomplete Translations with Dummy/Dictionary Backends

**Problem**: When using `dummy` or `dictionary` backends, only parts of text were translated, leaving much in the original language.

**Solution**: Created `improved-offline` translator that:
- Uses an expanded glossary (500+ common words)
- Applies learned translation patterns from examples
- Uses rule-based translation for common structures
- Provides much better coverage than simple dictionary lookup

**Usage**:
```bash
scitrans translate --text "Machine learning is revolutionizing AI" --backend improved-offline
```

### 2. Learning System for Offline Translation

**New Feature**: A learning system that can build translation models from parallel examples.

**Usage**:
```python
from scitrans_llms.translate.offline import learn_from_examples

# Learn from parallel texts
source_texts = ["Machine learning", "Deep learning", "Neural networks"]
target_texts = ["Apprentissage automatique", "Apprentissage profond", "Réseaux de neurones"]

model = learn_from_examples(source_texts, target_texts, output_path="learned_model.json")

# Use learned model
from scitrans_llms.translate.offline import ImprovedOfflineTranslator, LearnedModel
learned_model = LearnedModel.load("learned_model.json")
translator = ImprovedOfflineTranslator(learned_model=learned_model)
```

### 3. Free API Options for Testing

**Problem**: You want to test with real API models but have no balance/credits.

**Solutions**: Added three free options:

#### A. Hugging Face Inference API (Free Tier)
- **Free tier**: 1000 requests/month
- **No credit card required**
- **Setup**: Get free API key from https://huggingface.co/settings/tokens (optional, works without key too)

```bash
# Optional: Set API key for higher rate limits
export HUGGINGFACE_API_KEY="your_key_here"

# Use Hugging Face
scitrans translate --text "Hello world" --backend huggingface
```

#### B. Ollama (Local, Completely Free)
- **Completely free** - runs locally
- **Setup**: 
  1. Install from https://ollama.ai
  2. Pull a model: `ollama pull llama3`
  3. Use the translator

```bash
scitrans translate --text "Hello world" --backend ollama
```

#### C. Google Translate Free (Unofficial)
- **Free** but unofficial API
- **Setup**: `pip install googletrans==4.0.0rc1`

```bash
scitrans translate --text "Hello world" --backend googlefree
```

### 4. GPT-5.1 and Better Model Support

**New**: Added support for GPT-5.1 and other newer models.

**Usage**:
```bash
# Use GPT-5.1 (when available)
scitrans translate --text "Hello world" --backend gpt-5.1

# Or specify model explicitly
scitrans translate --text "Hello world" --backend openai --model gpt-5.1
```

**Supported Models**:
- `gpt-4o` (default)
- `gpt-5` 
- `gpt-5.1`
- `gpt-4`
- `gpt-3.5-turbo`

### 5. Expanded Glossary

**Improvement**: Expanded default glossary from ~60 to 500+ terms covering:
- Common scientific terms
- ML/NLP terminology
- General academic vocabulary
- Common words and phrases

## Quick Start Guide

### Testing Without API Balance

1. **Use Improved Offline Translator** (best for testing):
```bash
scitrans translate --text "Your text here" --backend improved-offline
```

2. **Use Hugging Face** (free API):
```bash
scitrans translate --text "Your text here" --backend huggingface
```

3. **Use Ollama** (local, completely free):
```bash
# First install Ollama and pull a model
ollama pull llama3

# Then use it
scitrans translate --text "Your text here" --backend ollama
```

4. **Use Google Free** (unofficial):
```bash
pip install googletrans==4.0.0rc1
scitrans translate --text "Your text here" --backend googlefree
```

### Learning from Examples

Create a learning script:

```python
from scitrans_llms.translate.offline import learn_from_examples
from pathlib import Path

# Load your parallel corpus
source_texts = []
target_texts = []

# Read from files or corpus
with open("source.txt", "r") as f:
    source_texts = [line.strip() for line in f]

with open("target.txt", "r") as f:
    target_texts = [line.strip() for line in f]

# Learn model
model = learn_from_examples(
    source_texts,
    target_texts,
    output_path=Path("learned_model.json")
)

print(f"Learned {len(model.word_translations)} word translations")
print(f"Learned {len(model.phrase_translations)} phrase translations")
```

Then use it:

```python
from scitrans_llms.translate.offline import ImprovedOfflineTranslator, LearnedModel
from scitrans_llms.translate.glossary import get_default_glossary

glossary = get_default_glossary()
learned_model = LearnedModel.load("learned_model.json")

translator = ImprovedOfflineTranslator(
    glossary=glossary,
    learned_model=learned_model
)

result = translator.translate("Your text here")
print(result.text)
```

## Available Backends Summary

| Backend | Cost | Quality | Setup |
|---------|------|---------|-------|
| `improved-offline` | Free | Good | None |
| `huggingface` | Free (1000/mo) | Good | Optional API key |
| `ollama` | Free | Good | Install Ollama |
| `googlefree` | Free | Good | `pip install googletrans` |
| `openai` | Paid | Excellent | API key required |
| `gpt-5.1` | Paid | Excellent | API key required |
| `deepseek` | Paid (cheap) | Good | API key required |
| `anthropic` | Paid | Excellent | API key required |
| `dictionary` | Free | Basic | None |
| `dummy` | Free | Testing only | None |

## Recommendations

1. **For testing without balance**: Use `improved-offline` or `huggingface`
2. **For best quality**: Use `gpt-5.1` or `openai` with `gpt-4o`
3. **For cost-effective**: Use `deepseek` or `huggingface`
4. **For completely offline**: Use `improved-offline` with learned model
5. **For local processing**: Use `ollama` with a good model

## Next Steps

1. Try the improved offline translator:
   ```bash
   scitrans translate --text "Machine learning is revolutionizing AI" --backend improved-offline
   ```

2. Test with free APIs:
   ```bash
   scitrans translate --text "Your text" --backend huggingface
   ```

3. Build a learned model from your corpus (see learning example above)

4. Check available backends:
   ```bash
   scitrans info
   ```

