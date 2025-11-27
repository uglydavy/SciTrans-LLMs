# 🌐 SciTrans-LLMs

**Adaptive Document Translation Enhanced by Technology based on LLMs**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 👥 Project Information

**Author:** TCHIENKOUA FRANCK-DAVY  
**Email:** aknk.v@pm.me  
**Institution:** Wenzhou University  
**Program:** Master's in Computer Science & Artificial Intelligence  
**Supervisor:** Dr. Chen Ang  
**Year:** 2025

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Research Contributions](#-research-contributions)
- [Installation](#-installation)
- [API Key Setup](#-api-key-setup)
- [Quick Start](#-quick-start)
- [GUI Usage](#-gui-usage)
- [Command Line Interface](#-command-line-interface)
- [Python API](#-python-api)
- [System Architecture](#-system-architecture)
- [Module Documentation](#-module-documentation)
- [Configuration](#-configuration)
- [Experiments & Evaluation](#-experiments--evaluation)
- [Development](#-development)
- [Troubleshooting](#-troubleshooting)
- [Citation](#-citation)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

We present **SciTrans-LLMs**, a research-grade translation system specifically designed for scientific PDF documents. Our system implements an end-to-end pipeline that preserves document layout while providing high-quality translations between English and French.

This project is part of our master's thesis research on "Adaptive Document Translation Enhanced by Technology based on LLMs" and represents a significant improvement over existing document translation systems.

### ✨ Key Features

- 📄 **Layout-Preserving PDF Translation** - Maintains document structure, fonts, and formatting
- 🔢 **Advanced Masking** - Protects mathematical formulas, code blocks, URLs, and DOIs
- 📚 **User-Owned Glossaries** - Enforce domain-specific terminology
- 🧠 **Context-Aware Prompting** - Document-level coherence across pages
- 🔄 **Translation Reranking** - Multiple candidate generation with quality scoring
- 📊 **Research-Grade Evaluation** - BLEU, chrF++, COMET metrics
- 🧪 **Ablation Study Framework** - Systematic component evaluation
- 🖥️ **Interactive GUI** - User-friendly web interface powered by Gradio

---

## 🔬 Research Contributions

Our system addresses key challenges in scientific document translation through three main contributions:

### 1️⃣ Terminology-Constrained, Layout-Preserving Translation

We implement a structured document representation that:
- Preserves PDF layout using YOLO-based detection + heuristic analysis
- Protects non-translatable content (formulas like `$x^2$`, code, URLs, DOIs)
- Enforces bilingual glossaries through both prompting and post-processing
- Reconstructs translated PDFs with faithful formatting

### 2️⃣ Document-Level LLM Context and Refinement

Our system maintains coherence through:
- Sliding context windows with translation history
- Multi-turn LLM translation for better coherence
- Document-level refinement passes (terminology, style, coherence)
- Candidate reranking with quality scoring

### 3️⃣ Research-Grade Evaluation Framework

We provide comprehensive evaluation tools:
- Automatic metrics: BLEU, chrF++, COMET
- Glossary adherence measurement
- Numeric consistency checks
- Placeholder preservation validation
- Ablation study framework for systematic experiments
- Baseline comparison tools (PDFMathTranslate, Google Translate, etc.)

---

## 📦 Installation

### Prerequisites

- **Python 3.9+** (we use Python 3.9)
- Virtual environment (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/uglydavy/SciTrans-LLMs
cd SciTrans-LLMs
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install Dependencies

**Option A: Full Installation** (includes all features - GUI, YOLO, all LLM backends)

```bash
pip install -e ".[full]"
```

**Option B: Minimal Installation** (core features only)

```bash
pip install -e .
```

**Option C: Custom Installation**

```bash
# Core only
pip install -e .

# Add OpenAI support
pip install -e ".[openai]"

# Add all LLM backends
pip install -e ".[all-llm]"

# Add layout detection (YOLO)
pip install -e ".[layout]"

# Development tools
pip install -e ".[dev]"
```

### Step 4: Verify Installation

```bash
scitrans info
```

This command displays:
- ✅ Available translation backends
- ✅ Installed dependencies
- ✅ API key status
- ✅ System configuration

---

## 🔑 API Key Setup

Our system supports multiple LLM backends, each requiring API keys. We provide **three secure ways** to configure your API keys:

### Method 1: Environment Variables (Recommended)

Add to your shell profile (`~/.bashrc`, `~/.zshrc`, or `~/.bash_profile`):

```bash
export OPENAI_API_KEY="sk-..."
export DEEPSEEK_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export DEEPL_API_KEY="..."
export GOOGLE_API_KEY="..."
```

Then reload your shell:

```bash
source ~/.bashrc  # or ~/.zshrc
```

### Method 2: Using Our Key Management CLI

We provide a secure key management system that stores keys in your OS keychain:

```bash
# Set a key
scitrans keys set openai

# You'll be prompted to enter the key securely

# List all configured keys
scitrans keys list

# Check specific key status
scitrans keys status openai

# Delete a key
scitrans keys delete openai
```

The keys are stored securely using:
1. **macOS**: Keychain
2. **Linux**: Secret Service (GNOME Keyring, KWallet)
3. **Windows**: Windows Credential Locker
4. **Fallback**: Encrypted local file (`~/.scitrans/keys.json`)

### Method 3: Using Setup Script

For interactive setup:

```bash
python3 scripts/setup_keys.py
```

### Supported Services

| Service | Environment Variable | Usage |
|---------|---------------------|-------|
| OpenAI | `OPENAI_API_KEY` | GPT-4, GPT-4o models |
| DeepSeek | `DEEPSEEK_API_KEY` | DeepSeek models |
| Anthropic | `ANTHROPIC_API_KEY` | Claude 3+ models |
| DeepL | `DEEPL_API_KEY` | DeepL baseline |
| Google | `GOOGLE_API_KEY` | Google Translate baseline |
| COMET | `COMET_API_KEY` | COMET evaluation |

### Verifying Your Keys

```bash
# Check all keys status
scitrans keys list

# Test translation with your key
scitrans translate --text "Hello world" --backend openai
```

---

## 🚀 Quick Start

### Basic Text Translation

```bash
# Translate simple text
scitrans translate --text "Machine learning is revolutionizing AI" --backend dummy

# With OpenAI
scitrans translate --text "Machine learning is revolutionizing AI" --backend openai
```

### PDF Translation

```bash
# Translate a PDF document
scitrans translate -i paper.pdf -o paper_fr.pdf --backend openai

# Specify language pair
scitrans translate -i paper.pdf -o paper_de.pdf --source en --target de --backend openai
```

### Using Custom Glossary

```bash
# With custom glossary file
scitrans translate -i paper.pdf -o paper_fr.pdf --glossary my_terms.csv --backend openai
```

### Running the Demo

```bash
# Quick demo with sample text
scitrans demo
```

---

## 🖥️ GUI Usage

We provide an interactive web-based GUI built with Gradio for easy document translation.

### Starting the GUI

**Method 1: Using CLI**

```bash
scitrans gui
```

**Method 2: Using Python**

```bash
python3 -m scitrans_llms.gui
```

**Method 3: Using Script**

```bash
python3 scripts/full_pipeline.py
```

### GUI Features

The GUI interface provides:

1. **📤 PDF Upload**
   - Drag and drop your PDF file
   - Or click to browse and select

2. **⚙️ Configuration Panel**
   - **Backend Selection**: Choose from OpenAI, DeepSeek, Anthropic, or dummy
   - **Language Direction**: EN→FR or FR→EN
   - **Page Range**: Translate specific pages (e.g., "1-5,10,15-20")
   - **Quality Options**: Enable/disable reranking for better quality
   - **Layout Preservation**: Toggle figure preservation

3. **🎯 Translation Process**
   - Real-time progress updates
   - Stage-by-stage status (parsing → translating → rendering)
   - Live preview of translated text

4. **📥 Output**
   - Download translated PDF
   - View translation statistics
   - See processing timeline
   - Preview translated text

### GUI Workflow

```
1. Start GUI → 2. Upload PDF → 3. Configure settings → 4. Click "Translate" → 5. Download result
```

### Advanced GUI Options

- **Preserve Figures**: Keep images and figures in original positions
- **Quality Loops**: Number of refinement iterations (1-3 recommended)
- **Enable Reranking**: Generate multiple candidates and select best (improves quality but slower)
- **Custom Glossary**: Upload your own terminology CSV file

### Accessing the GUI

After starting, the GUI will be available at:
- **Local**: http://127.0.0.1:7860
- **Network**: The URL will be displayed in the terminal

The interface automatically opens in your default browser.

---

## 💻 Command Line Interface

Our CLI provides comprehensive functionality for translation, evaluation, and system management.

### Core Commands

#### `scitrans translate`

Translate text or documents.

**Basic Usage:**

```bash
# Translate text
scitrans translate --text "Hello world" --backend openai

# Translate PDF
scitrans translate -i input.pdf -o output.pdf --backend openai

# Translate text file
scitrans translate -i document.txt -o translated.txt --backend openai
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--text` | `-t` | Text to translate (for quick tests) |
| `--input` | `-i` | Input file (PDF or TXT) |
| `--output` | `-o` | Output file path |
| `--source` | `-s` | Source language code (default: en) |
| `--target` | `-l` | Target language code (default: fr) |
| `--backend` | `-b` | Backend: dummy, dictionary, openai, deepseek, anthropic |
| `--model` | `-m` | Model name for LLM backends |
| `--glossary` | `-g` | Custom glossary CSV file |
| `--no-glossary` | | Disable glossary (for ablation) |
| `--no-refinement` | | Disable refinement pass |
| `--no-masking` | | Disable formula/code masking |
| `--no-context` | | Disable document context |

**Examples:**

```bash
# Translate with specific model
scitrans translate -i paper.pdf -o paper_fr.pdf --backend openai --model gpt-4o

# Ablation: without glossary
scitrans translate -i paper.pdf -o output.pdf --backend openai --no-glossary

# Ablation: without context
scitrans translate -i paper.pdf -o output.pdf --backend openai --no-context
```

#### `scitrans glossary`

Manage and view glossaries.

```bash
# List all terms in default glossary
scitrans glossary --list

# Search for a term
scitrans glossary --search "neural network"

# Filter by domain
scitrans glossary --domain ml
```

#### `scitrans evaluate`

Evaluate translation quality.

```bash
# Basic evaluation
scitrans evaluate --hyp output.txt --ref reference.txt --source source.txt

# With custom glossary
scitrans evaluate --hyp output.txt --ref reference.txt --glossary terms.csv

# Save results
scitrans evaluate --hyp output.txt --ref reference.txt -o results.json
```

**Output formats:** JSON, CSV, LaTeX

#### `scitrans ablation`

Run systematic ablation studies.

```bash
# Run ablation on a corpus
scitrans ablation --input docs/ --refs references/ --output results.json

# Test specific backends
scitrans ablation --input docs/ --refs refs/ --backends openai,deepseek
```

#### `scitrans keys`

Manage API keys securely.

```bash
# Set a key
scitrans keys set openai

# List all keys
scitrans keys list

# Check specific key
scitrans keys status openai

# Delete a key
scitrans keys delete openai

# Export to environment
scitrans keys export
```

#### `scitrans info`

Display system information.

```bash
scitrans info
```

Shows:
- Version information
- Available backends and status
- API key configuration
- Installed dependencies
- System capabilities

#### `scitrans demo`

Run a quick demonstration.

```bash
scitrans demo
```

Demonstrates:
- Text parsing
- Masking of formulas and URLs
- Glossary enforcement
- Translation pipeline
- Statistics reporting

### Additional Commands

#### `scitrans gui`

Launch the web GUI.

```bash
scitrans gui

# With custom port
scitrans gui --port 8080

# Share publicly
scitrans gui --share
```

#### `scitrans version`

Show version information.

```bash
scitrans --version
```

---

## 🐍 Python API

Our system provides a comprehensive Python API for programmatic use.

### Basic Translation

```python
from scitrans_llms import Document, TranslationPipeline
from scitrans_llms.pipeline import PipelineConfig

# Create document from text
doc = Document.from_text("""
Machine learning has revolutionized natural language processing.
The formula $E=mc^2$ shows mass-energy equivalence.
See https://arxiv.org/abs/1234.5678 for details.
""")

# Configure pipeline
config = PipelineConfig(
    source_lang="en",
    target_lang="fr",
    translator_backend="openai",
    enable_glossary=True,
    enable_refinement=True,
    enable_masking=True,
)

# Translate
pipeline = TranslationPipeline(config)
result = pipeline.translate(doc)

# Access results
print(result.translated_text)
print(result.stats)
print(f"Success: {result.success}")
```

### PDF Translation

```python
from scitrans_llms.ingest import parse_pdf
from scitrans_llms.render import render_pdf
from scitrans_llms.pipeline import TranslationPipeline, PipelineConfig

# Parse PDF with layout detection
doc = parse_pdf("paper.pdf", source_lang="en", target_lang="fr")
print(f"Parsed {len(doc.all_blocks)} blocks from {len(doc.pages)} pages")

# Configure and translate
config = PipelineConfig(
    translator_backend="openai",
    enable_glossary=True,
    enable_refinement=True,
)
pipeline = TranslationPipeline(config)
result = pipeline.translate(doc)

# Render translated PDF
render_pdf(doc, "paper.pdf", "paper_fr.pdf")
print(f"Saved to paper_fr.pdf")
```

### LLM Translation with Context

```python
from scitrans_llms.translate.llm import OpenAITranslator, LLMConfig, MultiTurnTranslator

# Configure LLM
config = LLMConfig(
    model="gpt-4o",
    temperature=0.3,
    max_tokens=4096,
)

# Create translator with context memory
base_translator = OpenAITranslator(config=config)
translator = MultiTurnTranslator(base_translator, max_history=10)

# Translate with context
result1 = translator.translate("The model uses attention mechanisms.")
result2 = translator.translate("It achieves state-of-the-art results.")
# "It" correctly references "The model" from previous context
```

### Custom Glossary

```python
from scitrans_llms.translate.glossary import Glossary, GlossaryEntry, load_glossary_csv

# Create custom glossary
glossary = Glossary(
    name="ML Terms",
    source_lang="en",
    target_lang="fr",
    entries=[
        GlossaryEntry(source="machine learning", target="apprentissage automatique", domain="ml"),
        GlossaryEntry(source="neural network", target="réseau de neurones", domain="ml"),
        GlossaryEntry(source="deep learning", target="apprentissage profond", domain="ml"),
    ]
)

# Or load from CSV
glossary = load_glossary_csv("my_terms.csv")

# Use in pipeline
config = PipelineConfig(
    glossary=glossary,
    enable_glossary=True,
)
```

### Evaluation

```python
from scitrans_llms.eval import run_evaluation, EvaluationRunner
from scitrans_llms.translate import get_default_glossary

# Quick evaluation
report = run_evaluation(
    hypotheses=["Le modèle utilise l'attention."],
    references=["Le modèle utilise le mécanisme d'attention."],
    sources=["The model uses attention."],
    glossary=get_default_glossary(),
)

print(report.summary())
print(f"BLEU: {report.bleu:.2f}")
print(f"chrF++: {report.chrf:.2f}")

# Save results
report.save("evaluation_results.json")
```

### Ablation Studies

```python
from scitrans_llms.eval.ablation import AblationStudy, AblationConfig
from scitrans_llms.models import Document

# Configure ablation
config = AblationConfig(
    name="thesis_ablation",
    test_glossary=True,
    test_context=True,
    test_refinement=True,
    test_masking=True,
    backends=["openai", "dummy"],
)

# Prepare test data
documents = [
    Document.from_text("Machine learning is powerful."),
    Document.from_text("Neural networks learn patterns."),
]

references = [
    ["L'apprentissage automatique est puissant."],
    ["Les réseaux de neurones apprennent des motifs."],
]

# Run study
study = AblationStudy(config=config)
study.run(documents, references)

# Results
print(study.summary())
print(study.to_latex_table())
study.save("ablation_results.json")
```

### Advanced: Custom Translator Backend

```python
from scitrans_llms.translate.base import Translator, TranslationResult

class MyCustomTranslator(Translator):
    """Custom translator implementation."""
    
    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: list[str] = None,
        glossary=None,
    ) -> TranslationResult:
        # Your custom translation logic here
        translated = my_translation_function(text)
        
        return TranslationResult(
            translated_text=translated,
            source_text=text,
            backend="custom",
            model="my-model",
        )

# Use in pipeline
config = PipelineConfig(translator_backend="custom")
# Register your translator...
```

---

## 🏗️ System Architecture

Our system is organized into modular components:

```
scitrans_llms/
├── 📦 Core Modules
│   ├── __init__.py          # Package exports and version
│   ├── models.py            # Document, Block, Segment data structures
│   ├── pipeline.py          # Main translation orchestration
│   ├── config.py            # Configuration management
│   ├── utils.py             # Utility functions
│   └── bootstrap.py         # Initialization and model downloads
│
├── 🎨 User Interfaces
│   ├── cli.py               # Command-line interface (Typer)
│   └── gui.py               # Web GUI (Gradio)
│
├── 🔒 Masking & Protection
│   ├── mask.py              # Placeholder masking engine
│   └── masking.py           # Masking utilities and patterns
│
├── 🔑 API Key Management
│   └── keys.py              # Secure key storage and retrieval
│
├── 📄 PDF Processing
│   ├── ingest/
│   │   ├── pdf.py           # PyMuPDF parser
│   │   └── analyzer.py      # Layout analysis
│   └── render/
│       └── pdf.py           # PDF reconstruction
│
├── 🌐 Translation Backends
│   └── translate/
│       ├── base.py          # Translator interface
│       ├── backends.py      # Backend implementations
│       ├── llm.py           # OpenAI, DeepSeek, Anthropic
│       ├── glossary.py      # Glossary management
│       ├── context.py       # Document-level context
│       ├── memory.py        # Translation memory
│       └── online_dictionary.py  # Dictionary lookups
│
├── ✨ Refinement & Reranking
│   └── refine/
│       ├── base.py          # Refiner interface
│       ├── llm.py           # LLM-based refinement
│       ├── rerank.py        # Candidate reranking
│       ├── scoring.py       # Quality scoring
│       ├── postprocess.py   # Post-processing
│       └── prompting.py     # Prompt templates
│
├── 📊 Evaluation Framework
│   └── eval/
│       ├── metrics.py       # BLEU, chrF++, COMET
│       ├── runner.py        # Batch evaluation
│       ├── ablation.py      # Ablation studies
│       └── baselines.py     # Baseline system wrappers
│
├── 🎯 Layout Detection
│   └── yolo/
│       ├── predictor.py     # YOLO inference
│       └── train.py         # YOLO training utilities
│
├── 🗂️ Data
│   └── data/
│       ├── glossary/        # Default terminology glossaries
│       └── layout/          # Pre-trained YOLO models
│
└── 🧪 Experiments
    └── experiments/
        ├── corpus.py        # Corpus management
        ├── runner.py        # Experiment orchestration
        └── thesis.py        # Thesis-specific experiments
```

---

## 📚 Module Documentation

### Core Modules

#### `models.py` - Data Structures

Defines the core data structures for document representation:

- **`Document`**: Top-level document container
  - `pages`: List of Page objects
  - `metadata`: Document metadata (title, author, etc.)
  - `all_blocks`: Flat list of all blocks across pages
  - `from_text()`: Create document from plain text
  - `from_pdf()`: Create document from PDF file

- **`Page`**: Represents a single page
  - `blocks`: List of Block objects
  - `page_number`: Page index
  - `dimensions`: Page width and height

- **`Block`**: Translatable text block
  - `source_text`: Original text
  - `translated_text`: Translated text
  - `block_type`: Type (paragraph, heading, list, etc.)
  - `is_translatable`: Whether block should be translated
  - `segments`: List of Segment objects

- **`Segment`**: Sub-block text segment
  - `text`: Segment content
  - `is_masked`: Whether segment is protected
  - `mask_type`: Type of mask (formula, code, url, etc.)

#### `pipeline.py` - Translation Orchestration

Main translation pipeline that coordinates all components:

- **`TranslationPipeline`**: Orchestrates translation process
  - `translate(doc)`: Translate a document
  - `translate_block(block)`: Translate a single block
  - Handles: masking → translation → refinement → unmasking

- **`PipelineConfig`**: Configuration for pipeline
  - Language settings
  - Backend selection
  - Feature toggles (masking, glossary, context, refinement)
  - Model parameters

- **`TranslationResult`**: Translation output
  - `translated_text`: Final translated text
  - `success`: Whether translation succeeded
  - `errors`: List of errors
  - `stats`: Translation statistics

#### `masking.py` / `mask.py` - Content Protection

Protects non-translatable content during translation:

- **Formula masking**: `$x^2$`, `$$\int f(x)dx$$`
- **Code blocks**: ` ```python ... ``` `
- **URLs**: `https://example.com`
- **DOIs**: `10.1234/example`
- **Emails**: `user@example.com`

Functions:
- `mask_protected_segments(text)`: Apply masks to text
- `unmask(text, mapping)`: Restore original content
- `detect_formulas(text)`: Find mathematical formulas
- `detect_code_blocks(text)`: Find code snippets

#### `keys.py` - API Key Management

Secure API key storage and retrieval:

- **`KeyManager`**: Manages API keys
  - `get_key(service)`: Retrieve key for service
  - `set_key(service, key)`: Store key securely
  - `delete_key(service)`: Remove stored key
  - `list_keys()`: Show all configured keys

- **Storage hierarchy**:
  1. Environment variables (highest priority)
  2. OS keychain (macOS Keychain, Windows Credential Locker, Linux Secret Service)
  3. Local encrypted file (`~/.scitrans/keys.json`)

### Translation Modules

#### `translate/base.py` - Translator Interface

Base classes for all translators:

- **`Translator`**: Abstract base class
  - `translate(text, source_lang, target_lang)`: Core translation method
  - `batch_translate(texts)`: Batch translation

- **`DummyTranslator`**: For testing
  - Returns uppercase text with markers

- **`DictionaryTranslator`**: Glossary-only translation
  - Word-by-word dictionary lookup

#### `translate/llm.py` - LLM Backends

LLM-based translation implementations:

- **`OpenAITranslator`**: GPT models
  - Supports GPT-4, GPT-4o, GPT-3.5-turbo
  - Streaming support
  - Token counting

- **`DeepSeekTranslator`**: DeepSeek models
  - Uses OpenAI-compatible API
  - Cost-effective alternative

- **`AnthropicTranslator`**: Claude models
  - Claude 3 Opus, Sonnet, Haiku
  - Long context support

- **`MultiTurnTranslator`**: Context-aware wrapper
  - Maintains translation history
  - Provides document-level coherence
  - Configurable history window

#### `translate/glossary.py` - Terminology Management

Bilingual glossary support:

- **`Glossary`**: Terminology database
  - `entries`: List of GlossaryEntry objects
  - `get_target(source)`: Lookup translation
  - `filter_by_domain(domain)`: Filter by domain
  - `enforce_in_text(text)`: Apply glossary to text

- **`GlossaryEntry`**: Single term
  - `source`: Source term
  - `target`: Target translation
  - `domain`: Subject domain (ml, math, physics, etc.)
  - `case_sensitive`: Whether matching is case-sensitive

- Functions:
  - `load_glossary_csv(path)`: Load from CSV file
  - `get_default_glossary()`: Get built-in glossary
  - `merge_glossaries(glossaries)`: Combine multiple glossaries

#### `translate/context.py` - Context Management

Document-level context tracking:

- **`ContextWindow`**: Manages translation history
  - `add_translation(source, target)`: Add to history
  - `get_context(max_length)`: Get recent context
  - `clear()`: Reset context

- **`ContextBuilder`**: Builds prompts with context
  - `build_prompt(text, context)`: Create contextualized prompt
  - `format_history(entries)`: Format history for prompt

### Refinement Modules

#### `refine/base.py` - Refiner Interface

Post-translation refinement:

- **`Refiner`**: Base class for refiners
  - `refine(text, source, glossary)`: Refine translation

- **`GlossaryRefiner`**: Enforce glossary terms
  - Post-processing glossary enforcement
  - Handles word boundaries and inflections

- **`PlaceholderRefiner`**: Fix placeholder issues
  - Ensures masked content is preserved
  - Fixes spacing around placeholders

#### `refine/llm.py` - LLM Refinement

LLM-based post-editing:

- **`LLMRefiner`**: Uses LLM for refinement
  - Improves coherence
  - Fixes terminology inconsistencies
  - Polishes style
  - Provides multiple refinement passes

#### `refine/rerank.py` - Candidate Reranking

Multiple candidate generation and selection:

- **`CandidateGenerator`**: Generate multiple candidates
  - Temperature sampling
  - Beam search
  - Multiple model outputs

- **`Reranker`**: Select best candidate
  - Quality scoring
  - Reference-free ranking
  - Ensemble methods

- **Scoring functions**:
  - `score_by_bleu(candidate, reference)`
  - `score_by_glossary_adherence(candidate, glossary)`
  - `score_by_fluency(candidate)`

### Evaluation Modules

#### `eval/metrics.py` - Quality Metrics

Automatic evaluation metrics:

- **BLEU**: Precision-based metric
  - `compute_bleu(hypothesis, reference)`
  - n-gram overlap

- **chrF++**: Character-based metric
  - `compute_chrf(hypothesis, reference)`
  - Better for morphologically rich languages

- **COMET**: Neural metric
  - `compute_comet(hypothesis, reference, source)`
  - Quality estimation

- **Glossary adherence**:
  - `compute_glossary_adherence(translation, glossary)`
  - Measures terminology compliance

- **Numeric consistency**:
  - `check_numeric_consistency(source, translation)`
  - Validates number preservation

#### `eval/runner.py` - Batch Evaluation

Evaluation orchestration:

- **`EvaluationRunner`**: Run evaluations
  - `evaluate_files(hyp, ref, src)`: Evaluate from files
  - `evaluate_lists(hyps, refs, srcs)`: Evaluate lists
  - `save_report(report, path)`: Save results

- **`EvaluationReport`**: Results container
  - Metrics (BLEU, chrF++, COMET)
  - Statistics
  - Error analysis
  - `summary()`: Human-readable summary
  - `to_latex()`: LaTeX table export

#### `eval/ablation.py` - Ablation Studies

Systematic component evaluation:

- **`AblationStudy`**: Ablation framework
  - `run(documents, references)`: Run study
  - `summary()`: Results summary
  - `to_latex_table()`: LaTeX export

- **`AblationConfig`**: Study configuration
  - `test_glossary`: Test with/without glossary
  - `test_context`: Test with/without context
  - `test_refinement`: Test with/without refinement
  - `test_masking`: Test with/without masking
  - `backends`: Backends to test

### PDF Processing

#### `ingest/pdf.py` - PDF Parsing

Extract text and layout from PDFs:

- **`parse_pdf(path, source_lang, target_lang)`**: Main parser
  - Extracts text blocks
  - Detects layout (columns, figures, tables)
  - Preserves formatting metadata
  - Returns Document object

- **`extract_text_with_layout(pdf_path)`**: Layout-aware extraction
  - Uses PyMuPDF
  - Maintains block positions
  - Preserves font information

#### `ingest/analyzer.py` - Layout Analysis

Analyze document structure:

- **`analyze_document(doc)`**: Document analysis
  - Detect columns
  - Identify headers/footers
  - Find figures and tables
  - Classify block types

- **YOLO integration**: Deep learning-based layout detection
  - Trained on scientific papers
  - Detects: text, figures, tables, equations, captions

#### `render/pdf.py` - PDF Reconstruction

Create translated PDFs:

- **`render_pdf(doc, original_path, output_path)`**: Render translated PDF
  - Preserves original layout
  - Maintains formatting
  - Handles fonts and styles
  - Supports RTL languages

### Experiments

#### `experiments/corpus.py` - Corpus Management

Manage test corpora:

- **`Corpus`**: Corpus container
  - `load_documents()`: Load source documents
  - `load_references()`: Load references
  - `split_train_test()`: Create splits

#### `experiments/runner.py` - Experiment Orchestration

Run large-scale experiments:

- **`ExperimentRunner`**: Run experiments
  - `run_experiment(config)`: Execute experiment
  - `collect_results()`: Aggregate results
  - `generate_report()`: Create reports

---

## ⚙️ Configuration

### Pipeline Configuration

```python
from scitrans_llms.pipeline import PipelineConfig

config = PipelineConfig(
    # Language settings
    source_lang="en",
    target_lang="fr",

    # Backend selection
    translator_backend="openai",  # dummy, dictionary, openai, deepseek, anthropic
    translator_kwargs={
        "model": "gpt-4o",
        "temperature": 0.3,
        "max_tokens": 4096,
    },
    
    # Feature toggles (useful for ablations)
    enable_masking=True,      # Protect formulas, code, URLs
    enable_glossary=True,     # Use terminology glossary
    enable_context=True,      # Document-level context
    enable_refinement=True,   # Post-translation refinement
    
    # Glossary settings
    glossary=my_glossary,         # Custom glossary (or None for default)
    glossary_in_prompt=True,      # Include in LLM prompt
    glossary_post_process=True,   # Enforce after translation
    
    # Refinement settings
    refiner_mode="default",  # none, glossary, default, llm
    num_candidates=1,        # >1 enables reranking

    # Context settings
    max_context_blocks=10,   # Number of previous blocks in context
)
```

### LLM Configuration

```python
from scitrans_llms.translate.llm import LLMConfig

config = LLMConfig(
    model="gpt-4o",           # Model name
    temperature=0.3,          # Sampling temperature (0.0-1.0)
    max_tokens=4096,          # Maximum output tokens
    top_p=1.0,               # Nucleus sampling
    frequency_penalty=0.0,   # Repetition penalty
    presence_penalty=0.0,    # Topic diversity penalty
    api_key=None,            # API key (None = use env var)
    timeout=60,              # Request timeout (seconds)
)
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required for OpenAI backend |
| `DEEPSEEK_API_KEY` | DeepSeek API key | Required for DeepSeek backend |
| `ANTHROPIC_API_KEY` | Anthropic API key | Required for Anthropic backend |
| `DEEPL_API_KEY` | DeepL API key | For baseline comparisons |
| `GOOGLE_API_KEY` | Google Translate API key | For baseline comparisons |
| `SCITRANS_DATA_DIR` | Data directory | `~/.scitrans/data` |
| `SCITRANS_CACHE_DIR` | Cache directory | `~/.scitrans/cache` |

---

## 🧪 Experiments & Evaluation

### Running Evaluations

```bash
# Evaluate translation quality
scitrans evaluate \
  --hyp system_output.txt \
  --ref human_reference.txt \
  --source original.txt \
  --output results.json
```

### Ablation Studies

```bash
# Run full ablation study
scitrans ablation \
  --input corpus/sources/ \
  --refs corpus/references/ \
  --output ablation_results.json \
  --backends openai,dummy
```

### Using Python API

```python
from scitrans_llms.eval.ablation import AblationStudy, AblationConfig
from scitrans_llms.models import Document

# Configure ablation
config = AblationConfig(
    name="my_ablation",
    test_glossary=True,
    test_context=True,
    test_refinement=True,
    test_masking=True,
    backends=["openai", "dummy"],
)

# Prepare data
docs = [Document.from_text(text) for text in sources]
refs = [ref.split("\n") for ref in references]

# Run
study = AblationStudy(config=config)
study.run(docs, refs)

# Results
print(study.summary())
study.save("results.json")
```

### Baseline Comparisons

We support comparison with:

- **PDFMathTranslate**: Layout-preserving PDF translation
- **Google Translate**: General-purpose MT
- **DeepL**: High-quality commercial MT
- **Opus-MT**: Open-source neural MT

---

## 🛠️ Development

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/uglydavy/SciTrans-LLMs
cd SciTrans-LLMs

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with development dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_core.py -v

# Run specific test
pytest tests/test_core.py::TestPipeline::test_translation -v

# With coverage
pytest tests/ --cov=scitrans_llms --cov-report=html
```

### Code Quality

```bash
# Linting
ruff check scitrans_llms/

# Type checking
mypy scitrans_llms/

# Formatting
ruff format scitrans_llms/
```

### Project Structure for Contributors

```
scitrans_llms/        # Main package
tests/                # Unit and integration tests
scripts/              # Utility scripts
thesis/               # Thesis-related materials
docs/                 # Documentation (if any)
```

### Adding New Translators

To add a new translation backend:

1. Create class inheriting from `Translator` in `translate/backends.py`
2. Implement `translate()` method
3. Register backend in `get_translator()` factory
4. Add tests in `tests/test_translators.py`

### Adding New Refiners

To add a new refiner:

1. Create class inheriting from `Refiner` in `refine/base.py`
2. Implement `refine()` method
3. Register in refiner factory
4. Add tests

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'scitrans_llms'`

**Solution**:
```bash
# Reinstall in editable mode
pip install -e .

# Or with full dependencies
pip install -e ".[full]"
```

#### 2. API Key Not Found

**Problem**: `ValueError: API key for 'openai' not found`

**Solutions**:

```bash
# Option 1: Set environment variable
export OPENAI_API_KEY="sk-..."

# Option 2: Use key manager
scitrans keys set openai

# Option 3: Verify key is set
scitrans keys list
```

#### 3. GUI Not Starting

**Problem**: GUI doesn't launch or shows errors

**Solutions**:

```bash
# Install GUI dependencies
pip install ".[full]"

# Or just Gradio
pip install gradio>=4.0.0

# Try alternative launch method
python3 -m scitrans_llms.gui
```

#### 4. PDF Processing Errors

**Problem**: PDF parsing fails

**Solutions**:

```bash
# Install PDF dependencies
pip install PyMuPDF>=1.23.0

# Try with different PDF
scitrans translate -i different.pdf -o output.pdf --backend dummy
```

#### 5. Commands Not Found

**Problem**: `scitrans: command not found`

**Solutions**:

```bash
# Reinstall package
pip install -e .

# Or use module form
python3 -m scitrans_llms.cli translate --help

# Check PATH
which scitrans

# Try with full path
~/.venv/bin/scitrans translate --help
```

#### 6. NumPy Version Conflicts

**Problem**: `_ARRAY_API` import errors

**Solution**:
```bash
# Install NumPy 1.x (already in requirements.txt)
pip install "numpy>=1.25.0,<2.0.0"
```

#### 7. Slow Translation

**Solutions**:
- Use faster models (GPT-3.5-turbo instead of GPT-4)
- Disable refinement: `--no-refinement`
- Disable reranking in GUI
- Translate specific pages: `--pages 1-5`

### Getting Help

- **Check logs**: Most commands show detailed progress
- **Run demo**: `scitrans demo` to verify setup
- **System info**: `scitrans info` to check configuration
- **Verbose mode**: Add `-v` flag to commands for detailed output

### Reporting Issues

When reporting issues, include:

1. Python version: `python3 --version`
2. System info: `scitrans info`
3. Full error message
4. Command or code that caused the error
5. Expected vs. actual behavior

---

## 📖 Citation

If you use this system in your research, please cite:

```bibtex
@mastersthesis{tchienkoua2025scitrans,
  title = {Adaptive Document Translation Enhanced by Technology based on LLMs},
  author = {Tchienkoua Franck-Davy},
  year = {2025},
  school = {Wenzhou University},
  address = {Wenzhou, China},
  type = {Master's thesis},
  supervisor = {Chen Ang},
  note = {Software available at \url{https://github.com/uglydavy/SciTrans-LLMs}}
}
```

---

## 🙏 Acknowledgments

We would like to thank:

- **Dr. Chen Ang** - Our thesis supervisor for guidance and support
- **Wenzhou University** - For providing resources and research environment
- **PDFMathTranslate** - For inspiration on layout-preserving PDF translation
- **DocuTranslate** - For ideas on LLM-based document translation
- **WMT Community** - For research on MT quality and reranking
- **Open Source Community** - For excellent tools: PyMuPDF, Gradio, Typer, and more

### Built With

- 🐍 **Python 3.9+**
- 🤖 **OpenAI GPT, DeepSeek, Anthropic Claude** - LLM backends
- 📄 **PyMuPDF** - PDF processing
- 🖼️ **Ultralytics YOLO** - Layout detection
- 🎨 **Gradio** - Web GUI
- 💻 **Typer** - CLI framework
- 📊 **SacreBLEU** - Evaluation metrics
- 🔐 **Keyring** - Secure key storage

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔗 Links

- **Repository**: https://github.com/uglydavy/SciTrans-LLMs
- **Documentation**: [README.md](README.md)
- **Thesis Guide**: [THESIS_GUIDE.md](THESIS_GUIDE.md)
- **Experiments**: [EXPERIMENTS.md](EXPERIMENTS.md)

---

## 📧 Contact

**TCHIENKOUA FRANCK-DAVY**  
📧 Email: aknk.v@pm.me  
🎓 Wenzhou University  
🏫 Master's in Computer Science & AI

---

<div align="center">

**Made with ❤️ for the NLP Research Community**

⭐ Star us on GitHub if you find this project useful!

</div>
