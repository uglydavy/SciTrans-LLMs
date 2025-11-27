# SciTrans-Next

**Adaptive Document Translation Enhanced by LLMs**

A research-grade, layout-preserving translation system for scientific PDFs (EN↔FR).

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

SciTrans-Next implements a complete scientific document translation pipeline with three core research contributions:

### 1. Terminology-Constrained, Layout-Preserving Translation
- Structured document representation (blocks, segments, pages)
- Placeholder masking for formulas (`$x^2$`), code, URLs, DOIs
- Bilingual glossary enforcement via prompting and post-processing
- PDF layout detection (heuristic + YOLO) and faithful reconstruction

### 2. Document-Level LLM Context and Refinement
- Context window with sliding history of previous translations
- Multi-turn LLM translation for coherence
- Document-level refinement pass (coherence, terminology, style)
- Candidate reranking with quality scoring

### 3. Research-Grade Evaluation Framework
- BLEU, chrF++, and COMET metrics
- Glossary adherence measurement
- Numeric consistency and placeholder preservation checks
- Ablation study framework for systematic experiments
- Baseline comparison tools (PDFMathTranslate, Google Translate, etc.)

## Quick Start

```bash
# Clone and setup
git clone https://github.com/scitrans-next/scitrans-next
cd scitrans-next

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with all features
pip install -e ".[full]"

# Or minimal install
pip install -e .

# Run the demo
python -m scitrans_next.cli demo

# Check available backends
python -m scitrans_next.cli info
```

## Usage

### Python API

```python
from scitrans_next import Document, TranslationPipeline
from scitrans_next.pipeline import PipelineConfig

# Create document from text
doc = Document.from_text("""
Machine learning has revolutionized natural language processing.
The formula $E=mc^2$ shows mass-energy equivalence.
""")

# Configure pipeline
config = PipelineConfig(
    source_lang="en",
    target_lang="fr",
    translator_backend="openai",  # or "dummy", "dictionary", "deepseek"
    enable_glossary=True,
    enable_refinement=True,
    enable_masking=True,
)

# Translate
pipeline = TranslationPipeline(config)
result = pipeline.translate(doc)

print(result.translated_text)
print(result.stats)
```

### PDF Translation

```python
from scitrans_next.ingest import parse_pdf
from scitrans_next.render import render_pdf
from scitrans_next.pipeline import TranslationPipeline, PipelineConfig

# Parse PDF with layout detection
doc = parse_pdf("paper.pdf", source_lang="en", target_lang="fr")

# Translate
pipeline = TranslationPipeline(PipelineConfig(translator_backend="openai"))
result = pipeline.translate(doc)

# Render translated PDF
render_pdf(doc, "paper.pdf", "paper_fr.pdf")
```

### LLM Translation with Context

```python
from scitrans_next.translate.llm import OpenAITranslator, LLMConfig, MultiTurnTranslator

# Configure LLM
config = LLMConfig(model="gpt-4o", temperature=0.3)
base_translator = OpenAITranslator(config=config)

# Wrap with multi-turn for document-level coherence
translator = MultiTurnTranslator(base_translator, max_history=10)

# Each translation builds on previous context
result1 = translator.translate("The model uses attention.")
result2 = translator.translate("It achieves state of the art results.")  # "It" references previous
```

### Evaluation

```python
from scitrans_next.eval import run_evaluation, EvaluationRunner
from scitrans_next.translate import get_default_glossary

# Quick evaluation
report = run_evaluation(
    hypotheses=["Le modèle utilise l'attention."],
    references=["Le modèle utilise le mécanisme d'attention."],
    sources=["The model uses attention."],
    glossary=get_default_glossary(),
)
print(report.summary())

# Full evaluation with file I/O
runner = EvaluationRunner(glossary=get_default_glossary())
report = runner.evaluate_files(
    hypothesis_file="output.txt",
    reference_file="reference.txt",
    source_file="source.txt",
)
runner.save_report(report, "results.json")
```

### Ablation Studies

```python
from scitrans_next.eval.ablation import AblationStudy, AblationConfig
from scitrans_next.models import Document

# Configure ablation
config = AblationConfig(
    name="thesis_ablation",
    test_glossary=True,      # With vs without glossary
    test_context=True,       # With vs without document context
    test_refinement=True,    # With vs without refinement
    test_masking=True,       # With vs without masking
    backends=["dummy"],      # Backends to test
)

# Run study
study = AblationStudy(config=config)
study.run(
    documents=[doc1, doc2],
    references=[[ref1_blocks], [ref2_blocks]],
    sources=[[src1_blocks], [src2_blocks]],
)

# Results
print(study.summary())
print(study.to_latex_table())
study.save("ablation_results.json")
```

### Command Line

```bash
# Translate text
scitrans translate --text "Hello world" --backend openai

# Translate PDF
scitrans translate -i paper.pdf -o paper_fr.pdf --backend openai

# With ablation flags
scitrans translate -i doc.txt --no-glossary --no-refinement

# View glossary
scitrans glossary --list
scitrans glossary --search "neural network"
scitrans glossary --domain ml

# Evaluate
scitrans evaluate --hyp output.txt --ref reference.txt --source source.txt

# Run ablation study
scitrans ablation --input docs/ --refs refs/ --output results.json

# System info
scitrans info
```

## Architecture

```
scitrans_next/
├── __init__.py              # Package exports
├── models.py                # Document, Block, Segment data structures
├── masking.py               # Placeholder protection (formulas, code, URLs)
├── pipeline.py              # Main translation orchestration
├── cli.py                   # Command-line interface
│
├── translate/               # Translation backends
│   ├── base.py              # Translator interface, dummy/dictionary backends
│   ├── glossary.py          # Glossary loading and enforcement
│   ├── context.py           # Document-level context management
│   └── llm.py               # OpenAI, DeepSeek, Anthropic backends
│
├── refine/                  # Post-translation refinement
│   ├── base.py              # Refiner interface, glossary/placeholder refiners
│   ├── llm.py               # LLM-based coherence refinement
│   └── rerank.py            # Candidate reranking
│
├── ingest/                  # PDF parsing
│   └── pdf.py               # PyMuPDF parser, layout detection
│
├── render/                  # PDF output
│   └── pdf.py               # Layout-preserving PDF rendering
│
└── eval/                    # Evaluation framework
    ├── metrics.py           # BLEU, chrF++, glossary adherence
    ├── runner.py            # Batch evaluation runner
    ├── ablation.py          # Ablation study framework
    └── baselines.py         # Baseline system wrappers
```

## Configuration

### Pipeline Configuration

```python
from scitrans_next.pipeline import PipelineConfig

config = PipelineConfig(
    # Languages
    source_lang="en",
    target_lang="fr",
    
    # Translation backend
    translator_backend="openai",  # dummy, dictionary, openai, deepseek, anthropic
    translator_kwargs={"model": "gpt-4o"},
    
    # Feature toggles (for ablations)
    enable_masking=True,          # Protect formulas, code, URLs
    enable_glossary=True,         # Use terminology glossary
    enable_context=True,          # Document-level context
    enable_refinement=True,       # Post-translation refinement
    
    # Glossary
    glossary=my_glossary,         # Custom glossary (or use default)
    glossary_in_prompt=True,      # Include in LLM prompt
    glossary_post_process=True,   # Enforce after translation
    
    # Refinement
    refiner_mode="default",       # none, glossary, default, llm
    
    # Candidate generation
    num_candidates=1,             # >1 enables reranking
)
```

### LLM Configuration

```python
from scitrans_next.translate.llm import LLMConfig

config = LLMConfig(
    model="gpt-4o",
    temperature=0.3,
    max_tokens=4096,
    api_key=None,  # Uses OPENAI_API_KEY env var if None
)
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for GPT models |
| `DEEPSEEK_API_KEY` | DeepSeek API key |
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude |
| `DEEPL_API_KEY` | DeepL API key (for baselines) |

## Thesis Chapter Mapping

| Component | Thesis Section | Research Question |
|-----------|---------------|-------------------|
| `masking.py` | 3.1 Layout Preservation | How to protect non-translatable content? |
| `glossary.py` | 3.2 Terminology Control | How to enforce domain terminology? |
| `context.py` | 3.3 Document-Level Context | How does context improve coherence? |
| `refine/llm.py` | 3.4 LLM Refinement | Does post-translation refinement help? |
| `rerank.py` | 3.5 Candidate Selection | Can reranking improve quality? |
| `eval/ablation.py` | 4.0 Experiments | What is each component's contribution? |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run specific test
pytest tests/test_core.py::TestPipeline -v

# Type checking
mypy scitrans_next/

# Linting
ruff check scitrans_next/
```

## Dependencies

### Required
- Python 3.10+
- pydantic, typer, rich (CLI/config)
- PyMuPDF (PDF processing)
- sacrebleu (evaluation)

### Optional
- `openai` - OpenAI GPT backends
- `anthropic` - Claude backends
- `ultralytics` - YOLO layout detection
- `unbabel-comet` - COMET evaluation
- `deep-translator` - Google Translate baseline

## License

MIT License - See [LICENSE](LICENSE)

## Citation

```bibtex
@mastersthesis{scitrans_next_2024,
  title = {Adaptive Document Translation Enhanced by Technology based on LLMs},
  author = {Your Name},
  year = {2024},
  school = {Your University},
  note = {Software available at https://github.com/scitrans-next}
}
```

## Acknowledgments

This project builds on ideas from:
- [PDFMathTranslate](https://github.com/Byaidu/PDFMathTranslate) - Layout-preserving PDF translation
- [DocuTranslate](https://github.com/Docutranslate) - LLM-based document translation
- WMT24 submissions using LLM reranking for MT quality
