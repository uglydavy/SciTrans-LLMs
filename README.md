
# SciTrans-LM (EN↔FR) – Modern GUI + CLI

A layout-preserving scientific PDF translator for **English ↔ French** with:
- Mandatory YOLO layout analysis (DocLayout-style) for figures/tables/formulas
- Placeholder masking (math/tables) and glossary enforcement
- Multiple engines (OpenAI, DeepL, Google, DeepSeek, Perplexity placeholders) + **offline glossary/dictionary fallback**
- Modern web GUI (Gradio) with drag-and-drop and page-range selection
- CLI parity for automation
- Secure API key handling with OS keychain (`keyring`)
- Evaluation (SacreBLEU) + refine (spacing fixes, glossary post-processing)
- Cross-platform: macOS, Linux, Windows

> **Note:** Model files (YOLO weights) and large bilingual corpora are not bundled in this ZIP to keep it light. The first-run **setup** will download/train as needed and create default glossaries.

---

## Quick Start

### 1) Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate  # Windows PowerShell
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

> **Torch & YOLO:** If PyTorch isn't auto-installed by `ultralytics`, install a wheel appropriate for your system:
> https://pytorch.org/get-started/locally/

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
For offline-only usage, skip this and choose the `dictionary` engine in GUI/CLI. If an online engine fails at runtime, the pipeline will **automatically fall back** to the glossary/dictionary translator so the job still finishes.

### 5) Launch GUI (modern web UI)

```bash
python3 -m scitrans_lm gui
```

- **Left:** Upload (drag & drop) the source PDF and preview.
- **Top bar:** Engine selection ("dictionary" for offline, "google"/"google-free" for keyless online, other engines require keys), EN↔FR direction, page range (auto-filled), preserve figures/formulas toggle.
- **Right:** Live preview of translated result (sample pages), Download button for full PDF.

If your browser doesn't open automatically, navigate to the printed Gradio URL (usually http://127.0.0.1:7860). The GUI accepts file paths or uploaded files—no need to pass `.name` attributes anymore.

### 6) CLI usage

```bash
python3 -m scitrans_lm translate   -i path/to/input.pdf   -o path/to/output.pdf   --engine openai   --direction en-fr   --pages 1-5   --preserve-figures
```

### 7) Evaluate translations (BLEU)

```bash
python3 -m scitrans_lm evaluate --ref data/refs.txt --hyp outputs/my_run.txt
# or compare folders of .txt files with matching names
python3 -m scitrans_lm evaluate --ref refs_dir --hyp hyps_dir
```

## Features & Notes

- **YOLO mandatory:** The pipeline requires a YOLO layout model at `data/layout/layout_model.pt`. First-run setup will create a placeholder if download/training is unavailable.
- **Placeholder masking:** Mathematical expressions and tables are temporarily replaced with unique tokens before translation and restored after.
- **Glossary:** A populated EN↔FR glossary (70+ core research terms) is created on install. You can **upload your own** `.csv`, `.tsv`, or `.txt` glossary from the GUI, or place files under `data/glossary/`.
- **Engines:** `openai`, `deepl`, `google`/`google-free` (keyless), `deepseek`, `perplexity` (pluggable). If a service’s SDK isn’t installed, you’ll get a friendly message.
- **Offline fallback:** If an online engine fails (missing key, API credit, etc.), translation automatically switches to the dictionary/glossary engine instead of aborting.
- **Evaluation:** Use `python3 -m scitrans_lm evaluate --ref ref.txt --hyp hyp.txt` for BLEU (SacreBLEU). For document sets, point to folders.
- **No dummy/identity in GUI:** Test backends exist for developers but are **hidden** from end users in the GUI.

## Requirements

See `requirements.txt`. Key items:
- Python 3.10–3.12
- PyMuPDF (fitz), ultralytics (YOLOv8), gradio (GUI)
- keyring (secure key storage), sacrebleu (evaluation)

Optional:
- pytesseract for OCR of images inside figures (if you choose to translate figure text)
- deepl / google-cloud-translate / openai SDKs (only if using those engines)
- googletrans (bundled) enables free Google Translate calls without credentials.

### Glossary and dictionary sources

- A 70+ term EN↔FR scientific glossary ships with the repo for offline use.
- You can place additional `.csv`, `.tsv`, or `.txt` glossaries under `data/glossary/`; they will be merged automatically.
- To pull a larger bilingual archive (FreeDict wordlist) or your own CSV/TSV URL:

```bash
python3 -m scitrans_lm setup --glossary --refresh-remote
# or
python3 -m scitrans_lm setup --glossary --refresh-remote --glossary-url https://example.com/my_en_fr.csv
```

The downloaded file is saved as `data/glossary/remote_en_fr.csv` and is used by both CLI and GUI. If the download fails (network/offline), the built-in glossary is still used so translation continues.

## Security & Keys

- Keys are kept in the OS keychain (`keyring`) by default.
- If keyring is unavailable, the app falls back to `~/.scitranslm/config.json` (excluded by `.gitignore`).

## License

The code is under the **MIT License** (see `LICENSE`). Default dictionary/glossary downloads use open datasets (e.g., FreeDict). Check their licenses before redistribution.

---

## Troubleshooting

- **Torch not found / CUDA issues:** Install PyTorch matching your OS/GPU. Then reinstall `ultralytics` if needed.
- **Model missing:** Run `python3 -m scitrans_lm setup --yolo` to download/train a layout model. If only the placeholder exists, layout detection is skipped gracefully.
- **Blank PDF outputs:** Fixed by overlay-render. If you still see blank pages, ensure PyMuPDF ≥ 1.23 and the `wrap_contents` step is invoked.
- **Figure translation looks odd:** Try enabling *Preserve figures/formulas* (default) so images and equations are not OCR-translated.
- **API credit errors:** If OpenAI/DeepL/other paid services reject requests, switch the engine to `google`/`google-free` or `dictionary`. The pipeline will automatically fall back to the offline glossary when any online engine fails.
