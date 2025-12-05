# GUI Fixed - December 2025

## What Was Broken

The GUI was defaulting to an **experimental Gradio interface** that had a critical bug in Gradio 4.44's API introspection system. This caused the GUI to crash immediately on launch with:

```
TypeError: argument of type 'bool' is not iterable
```

This was NOT a bug in our code, but a third-party bug in Gradio's `get_api_info()` method.

## What Was Fixed

### 1. CLI Default Changed to NiceGUI
- **Before**: `scitrans gui` launched broken Gradio interface
- **After**: `scitrans gui` launches **working NiceGUI interface** (stable, tested)
- Gradio moved to `--experimental` flag (not recommended)

### 2. Corpus Training Added to GUI
The Testing tab now includes a **Dictionary Training from Corpus** section that allows you to:
- Download parallel corpora (Europarl, OPUS, Tatoeba)
- Build custom dictionaries for offline translation
- Improve translation quality with domain-specific terminology

### 3. Documentation Updated
- All user-facing documentation now correctly describes NiceGUI as default
- Removed confusing Gradio references
- Added corpus training instructions

## How to Use the GUI (Quickstart)

### Launch the GUI

```bash
cd /Users/kv.kn/Desktop/Research/SciTrans-LLMs
source .venv/bin/activate
scitrans gui
```

This will:
- Start the **NiceGUI interface** on port 7860
- Open http://127.0.0.1:7860 in your browser
- Show 5 tabs: Translate, Testing, Glossary, Developer, Settings

### Tab 1: Translate (PDF Translation)

**Upload PDF:**
1. Click the upload zone or drag & drop a PDF
2. Preview appears in left panel
3. Select translation settings:
   - Engine: `dictionary` (offline, fast) or `free` (online, better quality)
   - Enable masking (protects equations, code)
   - Enable reranking (quality improvement)
   - Page range (optional)
4. Click **Translate**
5. Download translated PDF from right panel

**URL Translation:**
1. Paste PDF URL in the URL input box
2. Click **Fetch**
3. Translation settings appear
4. Click **Translate**

### Tab 2: Testing (Text Translation + Corpus Training)

**Quick Text Test:**
1. Enter source text in the text area
2. Select direction (EN→FR or FR→EN)
3. Select engine (try `dictionary` for offline)
4. Click **Run Translation**
5. See results, BLEU score, and metrics

**Train Dictionary from Corpus (NEW!):**
1. Select a corpus:
   - **opus-euconst** (5MB) - Recommended for testing
   - **europarl** (200MB) - Large, high quality
   - **tatoeba** (50MB) - Community-sourced
2. Select language pair (EN→FR or FR→EN)
3. Set max entries (default: 10,000)
4. Click **Download & Build Dictionary**
5. Wait for download and processing
6. Dictionary saves to `~/.scitrans/dictionaries/`
7. The `dictionary` backend will automatically use it!

**Check Dictionary Status:**
- Click **Show Status** to see all downloaded corpora and built dictionaries

### Tab 3: Glossary

**View Default Scientific Glossary:**
- 181 scientific terms (ML, math, stats, physics)
- Search by term or domain
- Used automatically in all translations

**Upload Custom Glossary:**
- Supported formats: CSV, TXT (tab-separated), JSON
- CSV format: `source,target,domain`
- Example:
  ```csv
  neural network,réseau neuronal,ml
  gradient descent,descente de gradient,ml
  ```
- Upload via the file picker
- Custom glossary merges with default

### Tab 4: Developer

**View System Logs:**
- Real-time logs of all GUI operations
- Translation events, errors, corpus downloads
- Useful for debugging

**System Diagnostics:**
- Check installed dependencies
- View backend availability
- API key status

### Tab 5: Settings

**API Keys:**
- Configure OpenAI, DeepSeek, Anthropic keys
- Keys are stored securely in `~/.scitrans/keys.json`
- Not required for `free` and `dictionary` backends

**Performance Options:**
- Context window size
- Number of refinement passes
- Reranking settings

## Recommended Workflow for Thesis

### Step 1: Build a Domain Dictionary
```bash
scitrans gui
# → Go to Testing tab
# → Select "opus-euconst" corpus
# → Click "Download & Build Dictionary"
# → Wait 1-2 minutes
```

This creates `~/.scitrans/dictionaries/opus-euconst_en_fr.tsv` with 10K translation pairs.

### Step 2: Test Text Translation
```bash
# In Testing tab
# → Enter: "Neural networks are trained using backpropagation"
# → Engine: dictionary
# → Click "Run Translation"
```

Compare results with and without the corpus dictionary!

### Step 3: Translate Research PDFs
```bash
# In Translate tab
# → Upload your thesis PDF
# → Engine: dictionary (fast) or free (better quality)
# → Enable masking (protects equations)
# → Click Translate
# → Download result
```

### Step 4: Evaluate Quality
```bash
# CLI evaluation
scitrans evaluate hypothesis.txt reference.txt --metrics bleu chrf
```

## Troubleshooting

### GUI won't start
```bash
# Check dependencies
pip install 'nicegui>=1.4.0'

# Try launching directly
python -m scitrans_llms.cli gui
```

### Corpus download fails
```bash
# Try smaller corpus first
# opus-euconst (5MB) instead of europarl (200MB)

# Check internet connection
curl https://opus.nlpl.eu/download.php

# CLI alternative
scitrans corpus download opus-euconst en fr
scitrans corpus build-dict opus-euconst en fr --limit 10000
```

### Dictionary not being used
```bash
# Check it exists
ls ~/.scitrans/dictionaries/

# View dictionary entries
head ~/.scitrans/dictionaries/opus-euconst_en_fr.tsv

# Force dictionary backend
scitrans translate --input paper.pdf --backend dictionary
```

### Translation quality is poor
1. **Build a larger dictionary**: Increase limit to 50,000 or 100,000
2. **Use a better corpus**: Try `europarl` instead of `opus-euconst`
3. **Enable masking**: Protects equations and code
4. **Try free backend**: Better than dictionary for complex text
5. **Add custom glossary**: Domain-specific terms

## File Locations

- **GUI code**: `scitrans_llms/gui.py` (NiceGUI, stable)
- **Experimental GUI**: `scitrans_llms/gui_gradio.py` (Gradio, broken)
- **CLI entrypoint**: `scitrans_llms/cli.py`
- **Corpus manager**: `scitrans_llms/translate/corpus_manager.py`
- **Dictionaries**: `~/.scitrans/dictionaries/`
- **Corpora cache**: `~/.scitrans/corpora/`
- **API keys**: `~/.scitrans/keys.json`

## What's Next

### High-Priority Improvements (Recommended)
1. **Translation Caching** - Avoid retranslating identical content (50-90% speedup)
2. **Font-Based Masking** - Better math detection using font patterns
3. **Async Translation** - Non-blocking UI with progress streaming
4. **More Corpora** - Add scientific paper corpora (arXiv, PubMed)

### Low-Priority (Can Wait)
- Fix Gradio interface (wait for Gradio 5.0)
- Add PDF annotations/highlights
- Export to other formats (DOCX, HTML)

## Testing Checklist

Before starting thesis experiments, verify:

- [ ] GUI launches without errors: `scitrans gui`
- [ ] Can upload PDF and see preview
- [ ] Can translate with `dictionary` backend
- [ ] Can translate with `free` backend
- [ ] Can download corpus (opus-euconst)
- [ ] Dictionary improves translation (compare before/after)
- [ ] Can upload custom glossary
- [ ] Text translation test works
- [ ] CLI translation works: `scitrans translate --input test.pdf`

## Support

If you encounter issues:

1. **Check logs**: Developer tab in GUI
2. **Run CLI**: `scitrans info` to check system status
3. **Test pipeline**: `scitrans demo` for quick sanity check
4. **Check dependencies**: `pip list | grep -E "nicegui|pymupdf|sacrebleu"`

## Summary

✅ **GUI is now working** - NiceGUI is default, stable, tested  
✅ **Corpus training added** - Build dictionaries from parallel corpora  
✅ **Documentation updated** - Clear instructions and troubleshooting  
✅ **Ready for thesis work** - All core features functional  

**Just run `scitrans gui` and start translating!** 🚀
