# SciTrans-LM (EN↔FR) – Modern GUI + CLI

A layout-preserving scientific PDF translator for **English ↔ French** with:
- Mandatory YOLO layout analysis (DocLayout-style) for figures/tables/formulas
- Placeholder masking (math/tables) and glossary enforcement
- Translation memory + adaptive prompts for coherence across sections
- Reranking and self-evaluation to reject weak translations
- Multiple engines (OpenAI, DeepL, Google, DeepSeek, Perplexity placeholders, **free Google Translate via deep-translator**) + **offline glossary/dictionary fallback**
- Modern web GUI (Gradio) with drag-and-drop, **Pipeline Lab** for testing masking/rerank/BLEU, and page-range selection
- CLI parity for automation + document inspector for layout debugging, now with a visible progress bar
- Secure API key handling with OS keychain (`keyring`)
- Evaluation (SacreBLEU) + refine (spacing fixes, glossary post-processing)
- Iterative prompt-guided refinement (up to 4 passes) for more reliable offline/online translations
- Cross-platform: macOS, Linux, Windows

> **Note:** Model files (YOLO weights) and large bilingual corpora are not bundled in this ZIP to keep it light. The first-run
> *setup** will download/train as needed and create default glossaries.

---

## What’s new vs. PDFMathTranslate?

SciTrans-LM started as a learning project inspired by PDFMathTranslate, but it now adds research-oriented capabilities:

- **Adaptive glossary enforcement**: built-in + user-uploaded glossaries are merged, post-enforced, and shared with online/offline engines.
- **Translation memory prompts**: recent segments are summarized in the prompt so layout blocks stay terminologically coherent.
- **Hybrid dictionary**: offline lexicon backed by an on-demand online lookup cache for rare terms; case-aware replacements avoid blank outputs.
- **Quality control loop**: iterative prompting with reranking that scores glossary hits, fluency signals, and change-ratios.
- **Content inspector**: `python -m scitrans_lm inspect` surfaces block/heading/caption detection to verify layout extraction.
- **Model-specific prompting**: OpenAI, DeepSeek, Perplexity, DeepL, and Google get tailored system prompts while sharing the same guardrails (masking + placeholder preservation).
- **GUI ergonomics**: sliders for refinement depth, rerank toggle, and glossary upload panel clearly signal what each feature does.
- **Licensing clarity**: the entire codebase is MIT-licensed; external dictionaries remain optional downloads.

---

## Quick Start

### 1) Create a clean virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate  # Windows PowerShell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you see a `No such file or directory` error, delete any partially created `.venv` folder first (`rm -rf .venv`) and rerun the commands above.

> **NumPy note:** Some dependencies (pandas/matplotlib/bottleneck) still ship wheels built against NumPy 1.x. The pinned `numpy<2` in `requirements.txt` avoids `_ARRAY_API` crashes; if your existing environment forces NumPy 2.x, reinstall with `python -m pip install "numpy<2"` before launching the GUI/CLI.

> **Torch & YOLO:** If PyTorch isn't auto-installed by `ultralytics`, install a wheel appropriate for your system:
> https://pytorch.org/get-started/locally/

### 2) Already have an environment? Install/refresh dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3) One-time setup (ensure model + glossary)

You can run all setup steps now **or** later from GUI.

```bash
python3 -m scitrans_lm setup --all
# or run only specific steps:
# python3 -m scitrans_lm setup --yolo
# python3 -m scitrans_lm setup --glossary
```

This step ensures `data/layout/layout_model.pt` exists and builds a **50+ term default glossary** at `data/glossary/default_en_fr.csv`.
If you prefer to train DocLayout-YOLO on your data, see `scitrans_lm/yolo/train.py`.
If networked downloads fail, the app will still generate the built-in glossary so offline dictionary translation keeps working.

### 4) Store API keys securely (optional, only for online engines)

```bash
python3 -m scitrans_lm set-key openai
python3 -m scitrans_lm set-key deepl
python3 -m scitrans_lm set-key google
python3 -m scitrans_lm set-key deepseek
python3 -m scitrans_lm set-key perplexity
```

This uses your OS keychain via `keyring` so you won’t be prompted each run.
For offline-only usage, skip this and choose the `dictionary` engine (aliases: `offline`, `local`, `lexicon`) in GUI/CLI. If an online engine fails at runtime, the pipeline will **automatically fall back** to the glossary/dictionary translator so the job still finishes. You can also pick `google-free` for a keyless (community) Google Translate backend powered by `deep-translator`.

### 5) Run a quick health check (deps/models/keys)

```bash
python3 -m scitrans_lm doctor
# or JSON output for CI/logs
python3 -m scitrans_lm doctor --json
```

Use the **System Check** tab in the GUI to run the same diagnostics without leaving the browser.

### 5) Run a quick health check (deps/models/keys)

```bash
python3 -m scitrans_lm doctor
# or JSON output for CI/logs
python3 -m scitrans_lm doctor --json
```

Use the **System Check** tab in the GUI to run the same diagnostics without leaving the browser.

### Free/offline usage cheat sheet

- No keys? Pick `google-free` (keyless) or `dictionary`/`offline` (pure glossary + adaptive web dictionary for rare terms).
- The offline dictionary merges the built-in glossary with your uploads and a cached MyMemory lookup so you still get reasonable translations without paid APIs.
- If any engine errors mid-run, the block falls back to the offline dictionary so the translation completes instead of crashing.

### 6) Launch GUI (modern web UI)

```bash
python3 -m scitrans_lm gui
```

- **Left:** Upload (drag & drop) the source PDF with centered controls
- **Top bar:** Engine selection, EN↔FR direction, page range (auto-filled), preserve figures/formulas toggle
- **Quality controls:** refinement loop slider + reranking toggle
- **Right:** Live preview of translated text before download + glossary upload/status panel
- **Pipeline timeline:** a single log shows the major steps (layout → translation → rerank → rendering) instead of scattered progress snippets
- **Tabs:**
  - **Translate:** main workflow with upload/URL fetch and the pipeline timeline
  - **Debug / QA:** run the analyzer only to check segmentation before translating
  - **Pipeline Lab:** test masking, glossary-aware reranking, BLEU, and a quick first-page layout snapshot without committing to a full translation

### 7) CLI usage

```bash
python3 -m scitrans_lm translate --input path/to/input.pdf --output path/to/output.pdf --engine openai --direction en-fr --pages 1-5 --preserve-figures --quality-loops 4
# Quick preview without opening the PDF
python3 -m scitrans_lm translate -i input.pdf -o output.pdf --engine google-free --preview
# List engines + key requirements
python3 -m scitrans_lm engines
# Check which keys are already stored (masked)
python3 -m scitrans_lm keys
# Run environment/model/key diagnostics from the terminal
python3 -m scitrans_lm doctor
# View a concise architecture map to locate modules quickly
python3 -m scitrans_lm map
```

### 8) Inspect layout extraction

```bash
python3 -m scitrans_lm inspect -i input.pdf --pages 1-3 --json report.json
```

### 9) Evaluate translations (BLEU)

```bash
python3 -m scitrans_lm evaluate --ref data/refs.txt --hyp outputs/my_run.txt
# or compare folders of .txt files with matching names
python3 -m scitrans_lm evaluate --ref refs_dir --hyp hyps_dir
```

### 10) Find the right file quickly (architecture map)

- CLI: `python3 -m scitrans_lm map`
- GUI: open **System Check → Show code map**

Both views summarize the main modules (ingestion, masking, translation backends, refinement, rendering, diagnostics) with their filenames so contributors know where to jump for changes.

## Features & Notes

- **YOLO mandatory:** The pipeline requires a YOLO layout model at `data/layout/layout_model.pt`. First-run setup will create a placeholder if download/training is unavailable.
- **Placeholder masking:** Mathematical expressions and tables are temporarily replaced with unique tokens before translation and restored after.
- **Glossary:** A populated EN↔FR glossary (50+ core research terms) is created on install. You can **upload your own** `.csv`, `.txt`, or `.docx` glossary from the GUI, or place files under `data/glossary/`.
- **Engines:** `openai`, `deepl`, `google`, `google-free` (keyless), `deepseek`, `perplexity` (pluggable). If a services SDK isn’t installed, you’ll get a friendly message.
- **Offline fallback:** If an online engine fails (missing key, API credit, etc.), translation automatically switches to the dictionary/glossary engine instead of aborting. The keyless `google-free` backend offers a free option when paid APIs are unreachable.
- **Dependency-friendly keyless Google:** The `google-free` backend now uses `deep-translator` (no strict `httpx` pin), so it installs cleanly alongside Gradio and the rest of the stack.
- **Progress visibility:** CLI logs each stage (layout parse, detection, per-block translation, rerank, overlay) with a Rich spinner/bar. The GUI shows concise status while Pipeline Lab and rerank logs keep details handy without scrolling.
- **Translation memory & rerank:** Prompts include recent segments; reranking scores glossary hits and fluency before returning output.
- **Inspection:** `inspect` reveals headings/captions vs paragraphs to validate layout parsing.
- **No dummy/identity in GUI:** Test backends exist for developers but are **hidden** from end users in the GUI.

## Requirements

See `requirements.txt`. Key items:
- Python 3.10–3.12
- PyMuPDF (fitz), ultralytics (YOLOv8), gradio (GUI)
- keyring (secure key storage), sacrebleu (evaluation)
- requests (adaptive dictionary fetch)

Optional:
- pytesseract for OCR of images inside figures (if you choose to translate figure text)
- deepl / google-cloud-translate / openai SDKs (only if using those engines)

## Security & Keys

- Keys are kept in the OS keychain (`keyring`) by default.
- If keyring is unavailable, the app falls back to `~/.scitranslm/config.json` (excluded by `.gitignore`).

## License

The code is under the **MIT License** (see `LICENSE`). External dictionaries/glossaries downloaded at runtime may carry their own terms; verify before redistribution.

---

## Troubleshooting

- **Torch not found / CUDA issues:** Install PyTorch matching your OS/GPU. Then reinstall `ultralytics` if needed.
- **Model missing:** Run `python3 -m scitrans_lm setup --yolo` to download/train a layout model. If only the placeholder exists, layout detection is skipped gracefully.
- **Blank PDF outputs:** Fixed by overlay-render. If you still see blank pages, ensure PyMuPDF ≥ 1.23 and the `wrap_contents` step is invoked.
- **Figure translation looks odd:** Try enabling *Preserve figures/formulas* (default) so images and equations are not OCR-translated.
