# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project overview

SciTrans-LLMs is a research-grade, layout-preserving scientific document translation system. It exposes:
- A Typer-based CLI entrypoint (`scitrans`) for translation, evaluation, corpus management, and diagnostics.
- A NiceGUI-based web UI for interactive PDF translation and inspection.
- A configurable, test-covered translation pipeline that supports free and paid LLM backends, masking, glossaries, and reranking.

Key documentation (start here before deep changes):
- `README.md` – high-level features and quick CLI examples.
- `INSTALL.md` – environment and dependency setup.
- `USER_GUIDE.md` – day-to-day usage of `scitrans` (backends, glossaries, common commands).
- `EXPERIMENTS.md` and `THESIS_GUIDE.md` – end-to-end experiment and thesis workflows.
- `corpus/README.md` – structure of the evaluation corpus.
- `thesis/README.md` – how generated LaTeX tables integrate into a thesis.

## Environment, install, and core commands

Assume a Unix-like shell (macOS/Linux) unless otherwise specified.

### Virtual environment and installation

```bash
# From repo root
python3 -m venv .venv
source .venv/bin/activate

# Full install with LLM backends, layout, GUI, and dev tools
pip install --upgrade pip
pip install -e ".[dev,full]"

# Minimal runtime install (no dev tooling or extras)
# pip install -e .
```

If `scitrans` is not on PATH, prefer the venv binary directly:

```bash
.venv/bin/scitrans --version
# or
python3 -m scitrans_llms.cli --help
```

### Core CLI usage

Common `scitrans` commands (see `scitrans --help` and `USER_GUIDE.md` for full details):

```bash
# Quick free translation (no API key)
scitrans translate --text "Machine learning is amazing" --backend free

# PDF → PDF with free backend
scitrans translate --input paper.pdf --output paper_fr.pdf --backend free

# Use a paid backend after configuring API keys (see INSTALL/USER_GUIDE)
scitrans translate --input paper.pdf --output paper_fr.pdf --backend deepseek

# With custom glossary CSV
scitrans translate \
  --input paper.pdf \
  --output paper_fr.pdf \
  --glossary my_terms.csv \
  --backend free

# System information and backends
scitrans info

# API keys management
scitrans keys list
scitrans keys set openai

# Corpus tools (download/build dictionaries, etc.)
scitrans corpus --help

# Quick pipeline demo
scitrans demo
```

### GUI

The GUI is built with NiceGUI and wraps the same pipeline used by the CLI.

```bash
# Ensure GUI deps are installed (via ".[full]" or manually)
scitrans gui
```

This launches a browser-based interface for PDF upload, engine selection, glossary upload, and inspection of system logs.

### Experiments and thesis pipeline

For thesis/experiment workflows, the supported entrypoints are Python scripts under `scripts/` (see `EXPERIMENTS.md` and `THESIS_GUIDE.md` for all options). Common ones:

```bash
# Sanity check that the environment and core pipeline work
python3 scripts/quick_test.py

# Run a lightweight experiment sweep (dictionary backend)
python3 scripts/run_experiments.py --quick

# Full experiments with a specific backend
python3 scripts/run_experiments.py --backend deepseek

# End-to-end thesis pipeline (ablation + main experiments)
python3 scripts/full_pipeline.py --backend openai
```

Experimental results and LaTeX tables are written into `results/` and `results/thesis/` as described in `EXPERIMENTS.md` and `thesis/README.md`.

### Tests, linting, and type checking

Pytest, Ruff, and MyPy are configured in `pyproject.toml` and expected to be installed via the `dev` extra.

```bash
# Run the full test suite
pytest -v
# or, as configured
pytest tests/ -v

# Run a single test
pytest tests/test_core.py::TestPipeline::test_simple_pipeline -v

# Lint (Ruff)
ruff check scitrans_llms/ tests/

# Type checking
mypy scitrans_llms/
```

`tests/test_core.py` is the main regression suite for the core document model, masking, glossary mechanisms, translator stubs, and pipeline orchestration.

## High-level architecture

At a high level, both CLI and GUI drive the same `TranslationPipeline` defined in `scitrans_llms/pipeline.py`. The pipeline operates over a structured `Document` model and composes ingestion, masking, translation backends, refinement, and rendering.

### Core data model (`scitrans_llms.models`)

- `Document`, `Segment`, `Block`, and `BlockType` represent the parsed document, its logical segments (typically pages), and block-level units (paragraphs, headings, equations, figures, etc.).
- Each `Block` carries:
  - `source_text` (original content) and `translated_text` (final output).
  - Optional `masked_text` used during placeholder protection.
  - Layout metadata via a `BoundingBox` and font/structural hints for preservation.
- The model exposes helpers like `Document.from_text(...)`, `Document.from_paragraphs(...)`, and JSON (de)serialization used in tests and experiments.

### Translation pipeline (`scitrans_llms/pipeline.py`)

- `PipelineConfig` is the central configuration object; it toggles:
  - Source/target languages and the selected translator backend.
  - Masking (on/off + `MaskConfig`).
  - Glossary usage and whether terms appear in prompts / post-processing.
  - Document-level context window size.
  - Refinement mode (e.g., glossary-only vs. LLM-based) and candidate reranking.
- `TranslationPipeline.translate(Document)` orchestrates the full flow:
  1. **Masking** – uses `mask_document` to replace formulas, code, URLs, etc. with stable placeholders, recording them in a `MaskRegistry`.
  2. **Context construction** – builds a `DocumentContext` that tracks previous source/target segments for document-level coherence.
  3. **Block-wise translation** – iterates over `document.all_blocks` where `block.is_translatable` is true, calling the configured translator with a `TranslationContext` that includes local context and (optionally) glossary hints.
  4. **Reranking (optional)** – when `num_candidates > 1`, integrates with `refine.rerank` to score and choose the best candidate per block.
  5. **Refinement** – passes the translated document to a `Refiner` (from `refine.base`) which can enforce glossary adherence and surface-level quality fixes.
  6. **Unmasking** – restores masked content using `unmask_document`, ensuring formulas, URLs, and code blocks are faithfully preserved.
- `PipelineResult` returns the translated `Document`, accumulated stats (block counts, refined blocks, masks applied, etc.), and any errors.
- Convenience entrypoints:
  - `translate_text(...)` for quick one-off string translation.
  - `translate_document(...)` for PDF-to-PDF, used by the GUI, which also handles page ranges and calls the render layer.

### PDF ingestion and layout (`scitrans_llms/ingest`)

- `ingest/pdf.py` contains the main layout-aware PDF parser:
  - Uses PyMuPDF to extract low-level text spans, positions, fonts, and images into `PageContent` + `TextSpan` structures.
  - Applies a `LayoutDetector` to assign `BlockType` labels (paragraph, heading, equation, list item, caption, reference, header/footer, etc.).
  - Two detection strategies exist:
    - `HeuristicLayoutDetector` – rule-based; always available.
    - `YOLOLayoutDetector` – DocLayout-YOLO-backed; used when weights and `ultralytics` are available, otherwise gracefully falls back to heuristics.
  - Spans are grouped into logical blocks based on position and font continuity, then converted into `Segment` + `Block` structures with layout metadata.
- The top-level parser class `PDFParser` exposes `parse(...)` which returns a `Document` ready for the pipeline. CLI and GUI both delegate PDF parsing to this component (either directly or via helpers like `scitrans_llms.ingest.parse_pdf`).

### Masking (`scitrans_llms.mask` and `scitrans_llms.masking`)

- Masking is responsible for protecting content that must not be altered by translation:
  - Inline and display LaTeX math.
  - URLs.
  - Inline code and code blocks.
- The main pieces are:
  - `MaskConfig` – toggles which constructs to protect and how strictly to preserve structure (section numbers, bullets, indentation) for layout-aware rendering.
  - `mask_text` / `unmask_text` for string-level operations.
  - `mask_document` / `unmask_document` for document-level operations.
  - `validate_placeholders` to ensure translated text hasn’t dropped any placeholders (used in tests and quality checks).
- Tests in `tests/test_core.py::TestMasking` codify expected behavior and are a good reference when changing or extending masking rules.

### Translation backends and glossary (`scitrans_llms/translate`)

- Backends are split into two layers:
  - Low-level, multi-segment batch translators in `translate/backends.py` (`BaseTranslator` implementations like `DictionaryTranslator`, `OpenAITranslator`, `DeepLTranslator`, `GoogleTranslator`, `GoogleFreeTranslator`, `DeepSeekTranslator`, `PerplexityTranslator`).
  - Higher-level translator abstractions in `translate.base` (e.g., `Translator`, `DummyTranslator`, `create_translator`, `TranslationContext`, and translation memory support).
- `DictionaryTranslator` is the key offline fallback:
  - Bootstraps a base lexicon of scientific terms.
  - Optionally augments that lexicon from online dictionary sources with caching.
  - Enforces glossary and translation memory hits aggressively, supporting purely offline runs.
- LLM-compatible translators inherit from `_OpenAICompatTranslator`, which unifies prompt construction and client setup for OpenAI-compatible APIs (OpenAI, DeepSeek, Perplexity, etc.) using `get_key(...)` from `scitrans_llms.keys`.
- Glossary handling lives in `translate/glossary.py`:
  - `Glossary`, `GlossaryEntry`, and helpers such as `get_default_glossary`, `enforce_glossary`, `check_glossary_adherence`.
  - Default glossaries are used by both CLI (`scitrans glossary`, `scitrans translate`) and pipeline refinement to keep terminology consistent.

### Refinement, reranking, and quality (`scitrans_llms/refine`)

- `refine/base.py` defines the `Refiner` abstraction and factory helpers used by the pipeline.
- `refine/prompting.py` builds translation prompts that include glossary terms and structural hints; the CLI can expose them via `--preview-prompt`.
- `refine/rerank.py` provides reranking over multiple candidate translations, optionally using scoring heuristics or LLM scoring; wired into `TranslationPipeline` when `num_candidates > 1`.
- `refine/scoring.py` and `eval/metrics.py` leverage metrics such as BLEU and chrF via SacreBLEU for quick quality evaluation.
- `refine/postprocess.py` handles structural fixes such as list and numbering preservation.

### Rendering (`scitrans_llms/render`)

- `render/pdf.py` takes a translated `Document` and the original PDF to produce a layout-preserving translated PDF:
  - Uses bounding boxes and block types to place translated text back into the original page geometry.
  - Can fall back to plain-text output when rendering fails, which is surfaced in `PipelineResult.errors` and via CLI messaging.
- `TranslationPipeline.translate_document(...)` and the CLI’s `translate` command both eventually delegate PDF writing to this module.

### CLI, GUI, and service integration

- `scitrans_llms/cli.py` (entrypoint `scitrans` from `pyproject.toml`):
  - `translate` – wraps the pipeline for text and PDF inputs, exposes masking/glossary/context flags, multi-pass refinement, reranking, and interactive review.
  - `glossary` – inspects and queries the default glossary.
  - `evaluate` – calls `eval.runner.EvaluationRunner` to compute metrics over hypothesis/reference files.
  - `ablation` – orchestrates ablation studies using `eval.ablation.AblationStudy`.
  - `corpus` – interfaces with `translate.corpus_manager` for downloading corpora and building dictionaries.
  - `info` – surfaces available backends, dependency status (PyMuPDF, SacreBLEU), and API key health.
  - `keys` – front-end to `keys.KeyManager` for secure key storage and export.
  - `gui` – thin wrapper around `gui.launch` with dependency checks and error reporting.
- `scitrans_llms/gui.py` (NiceGUI app):
  - Provides tabs for translation, testing, glossary management, and developer/system diagnostics.
  - Uses `translate_document(...)` under the hood for PDF translation and `KeyManager` to adapt the available engine list.
  - Maintains lightweight in-memory logs (`system_logs`) for UI debugging.

### Experiments and evaluation (`scitrans_llms/experiments` and `scitrans_llms/eval`)

- `experiments/` encapsulates research workflows:
  - `experiments/corpus.py` – utilities for loading the structured corpus under `corpus/`.
  - `experiments/runner.py` – high-level `ExperimentRunner` and `ExperimentConfig` abstractions for running sweeps over backends and pipeline settings.
  - `experiments/thesis.py` – exports thesis-ready LaTeX tables and figures (used by `thesis/` templates and scripts in `EXPERIMENTS.md` / `THESIS_GUIDE.md`).
- `eval/` provides reusable evaluation building blocks:
  - `eval/metrics.py` – BLEU/chrF and related metrics via SacreBLEU.
  - `eval/baselines.py` – baselines such as naive LLM, Google Translate, and PDFMathTranslate.
  - `eval/runner.py` – `EvaluationRunner`, used by `scitrans evaluate` and custom experiments.
  - `eval/ablation.py` – ablation configuration and result aggregation.

### Keys, configuration, diagnostics, and bootstrap

- `keys.py` implements `KeyManager` and service-specific key handling:
  - Aggregates keys from environment variables, local config files (`~/.scitrans/`), and (optionally) OS keychains.
  - Exposes methods used by both CLI (`scitrans keys ...`) and translators (`get_key(...)`).
- `config.py` and `bootstrap.py` coordinate configuration of default paths, YOLO weights, and data/glossary assets used at runtime.
- `diagnostics.py` centralizes environment-health checks (dependencies, assets, API keys); surfaced through CLI and the GUI’s system-check views.

## Working effectively in this codebase

- When adding new backends, prefer wiring them through the existing translator abstractions in `translate.base` and `translate/backends.py` so that CLI, GUI, and experiments all see them consistently.
- When changing layout or masking behavior, keep the contract between `ingest.pdf.PDFParser`, `masking`, and `render.pdf` intact (blocks must remain correctly typed and have valid bounding boxes), and extend tests in `tests/test_core.py` accordingly.
- For research-oriented changes, consider whether they belong in the generic pipeline (`pipeline.py`) or in experiment-specific layers (`experiments/` and `eval/`); the CLI/GUI pipeline should remain a stable, user-facing core while experiments can evolve faster.
