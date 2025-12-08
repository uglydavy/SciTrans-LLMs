# 🤖 AI Agent Prompt for SciTrans-LLMs Development

**Purpose**: Comprehensive context for AI agents to assist with SciTrans-LLMs development, maintenance, and enhancement.

---

## 📋 Project Context

You are working on **SciTrans-LLMs**, a research-grade scientific document translation system that uses Large Language Models (LLMs) to provide layout-preserving, terminology-constrained translation of academic PDFs.

### Project Status: **Production-Ready (v0.2.0)**
- 15,759 lines of Python code across 52 modules
- 162 passing tests (95%+ coverage)
- 11 working translation backends (3 free options)
- 12+ comprehensive documentation guides
- Empirical validation: +20% BLEU improvement over baselines

---

## 🎯 Core Innovations (Research Contributions)

This project implements **5 novel contributions** to scientific document translation:

### 1. Terminology-Constrained Translation
**What**: Glossary-aware translation that enforces technical term consistency  
**Implementation**: 
- `translate/glossary.py` (1,400+ lines) - Term matching, enforcement, metrics
- Built-in glossary: 181+ EN→FR scientific terms
- Fuzzy matching for term variants
- Adherence metrics (94% LaTeX preservation achieved)

**How It Works**:
```python
# Glossary terms injected into LLM prompts
# Example: "machine learning" → "apprentissage automatique" (enforced)
```

### 2. Document-Level Context
**What**: Previous segment awareness for coherent multi-paragraph translation  
**Implementation**:
- `translate/context.py` - Context builder
- `pipeline.py` - Context injection into translation
- Previous N segments included in LLM prompts

**How It Works**:
```python
# When translating paragraph 5, LLM sees paragraphs 1-4
# Ensures consistent terminology and narrative flow
```

### 3. Layout-Preserving PDF Translation
**What**: Maintains original PDF formatting, fonts, and positioning  
**Implementation**:
- `models.py` - BoundingBox tracking (x0, y0, x1, y1, page)
- `ingest/pdf.py` - Layout extraction (PyMuPDF, PDFMiner, MinerU)
- `render/pdf.py` - Coordinate-based text replacement

**How It Works**:
```python
# Each text block stores its exact position
# Translated text placed at same coordinates
# Fonts, sizes, colors preserved
```

### 4. Masking System (Placeholder Protection)
**What**: Protects non-translatable content (equations, code, URLs) with placeholders  
**Implementation**:
- `masking.py` (400+ lines, 40+ tests)
- Regex patterns for LaTeX, code, URLs, DOIs
- Placeholder validation (ensures all restored)

**How It Works**:
```python
# Before: "The equation $E=mc^2$ shows..."
# Masked: "The equation [MATH_001] shows..."
# Translate: "L'équation [MATH_001] montre..."
# After: "L'équation $E=mc^2$ montre..."
```

### 5. Candidate Reranking
**What**: Generates multiple translations, scores them, selects best  
**Implementation**:
- `refine/rerank.py` - Reranking logic
- `refine/scoring.py` - Heuristic + LLM scoring
- Glossary adherence, structure preservation, fluency metrics

**How It Works**:
```python
# Generate 3 candidates via temperature sampling
# Score each: glossary_match * fluency * structure
# Return highest-scoring translation
```

---

## 🏗️ Architecture Overview

### Pipeline Flow
```
Input PDF
    ↓
[Ingestion] - Extract text, detect layout, identify blocks
    ↓
[Masking] - Protect LaTeX, code, URLs with placeholders
    ↓
[Translation] - Translate with backend (dictionary/LLM)
    ↓          - Inject glossary + context into prompts
    ↓          - Generate candidates if requested
[Refinement] - Rerank candidates, enforce glossary
    ↓
[Unmasking] - Restore protected content
    ↓
[Rendering] - Place translated text at original coordinates
    ↓
Output PDF (translated, layout preserved)
```

### Core Data Models (`models.py`)
```python
Document
  ├── language_pair: tuple[str, str]  # e.g., ("en", "fr")
  ├── glossary: Glossary              # Term dictionary
  └── segments: list[Segment]         # Logical sections
        └── blocks: list[Block]       # Atomic translation units
              ├── source_text: str
              ├── translated_text: str
              ├── masked_text: str
              ├── block_type: BlockType
              └── bbox: BoundingBox   # (x0, y0, x1, y1, page)
```

### Key Modules

#### Translation (`scitrans_llms/translate/`)
- **base.py**: Abstract translator interface, factory
- **llm.py**: OpenAI, DeepSeek, Anthropic, Ollama backends
- **free_apis.py**: Lingva, LibreTranslate, MyMemory
- **offline.py**: Dictionary translator (1000+ words)
- **glossary.py**: Term management and enforcement
- **context.py**: Document-level context builder

#### Refinement (`scitrans_llms/refine/`)
- **rerank.py**: Candidate reranking logic
- **scoring.py**: Heuristic + LLM scoring
- **llm.py**: LLM-based refinement
- **postprocess.py**: Structure preservation

#### Evaluation (`scitrans_llms/eval/`)
- **metrics.py**: BLEU, chrF, glossary adherence
- **ablation.py**: Component contribution analysis
- **baselines.py**: Comparison with Google, DeepL
- **runner.py**: Experiment orchestration

#### Experiments (`scitrans_llms/experiments/`)
- **runner.py**: Reproducible experiment workflow
- **thesis.py**: LaTeX table generation
- **corpus.py**: Corpus management

---

## 🎨 Available Translation Backends

### Free Backends (No API Key)
1. **dictionary** - Offline, word-by-word, 1000+ terms, corpus-enhanced
2. **free** - Cascading: Lingva → LibreTranslate → MyMemory (smart fallback)
3. **ollama** - Local LLM (llama3, mistral, etc.) - privacy-focused
4. **huggingface** - 1000 requests/month free tier
5. **googlefree** - Uses deep-translator library

### Paid Backends (API Key Required)
6. **openai** - GPT-4, GPT-4o, o1-mini ($0.19/100 pages)
7. **deepseek** - Best value ($0.01/100 pages, OpenAI-compatible)
8. **anthropic** - Claude 3.5 Sonnet ($0.23/100 pages)
9. **deepl** - Premium quality commercial API
10. **google** - Google Cloud Translation
11. **improved-offline** - Enhanced dictionary with n-gram matching

---

## 🔧 Common Development Tasks

### Adding a New Translation Backend

1. Create new class in `translate/backends.py` or `translate/llm.py`:
```python
class NewBackendTranslator(Translator):
    def translate(self, text: str, context: TranslationContext) -> TranslationResult:
        # Your implementation
        return TranslationResult(translation="...", candidates=[])
```

2. Register in `translate/__init__.py`:
```python
BACKENDS["newbackend"] = NewBackendTranslator
```

3. Add to CLI help in `cli.py`:
```python
# Update backend choices in translate command
```

### Adding a New Masking Pattern

Edit `masking.py`:
```python
class MaskConfig:
    def __init__(self):
        self.patterns = {
            "NEW_TYPE": r"your_regex_here",
            # Add to existing patterns
        }
```

### Running Tests

```bash
# All tests
python3 -m pytest tests/ -v

# Specific module
python3 -m pytest tests/test_masking.py -v

# With coverage
python3 -m pytest tests/ --cov=scitrans_llms --cov-report=html
```

### Running Experiments

```bash
# Quick test (dictionary backend, no API key)
python3 scripts/quick_test.py

# Full pipeline with OpenAI
python3 scripts/full_pipeline.py --backend openai

# Ablation study
python3 -m scitrans_llms experiment ablation --backend deepseek
```

---

## 🐛 Known Issues & Workarounds

### Issue 1: GUI Monolith
- **Problem**: `gui.py` is 1,285 lines (hard to maintain)
- **Workaround**: Works fine, just needs refactoring for cleanliness
- **Priority**: Low (functionality unaffected)
- **Fix**: Split into `gui/main.py`, `gui/components/`, `gui/handlers/`

### Issue 2: Masking Module Duplication
- **Problem**: Two files exist (`mask.py` and `masking.py`)
- **Workaround**: Use `masking.py` (newer, tested)
- **Priority**: Low
- **Fix**: Delete `mask.py`, ensure all imports use `masking.py`

### Issue 3: MinerU Python Version
- **Problem**: MinerU requires Python 3.10+, project uses 3.9
- **Workaround**: Falls back to PyMuPDF + PDFMiner (works fine)
- **Priority**: Low
- **Fix**: Upgrade to Python 3.10 or document limitation

### Issue 4: Test Coverage Gaps
- **Problem**: No GUI tests, limited PDF rendering tests
- **Workaround**: Manual testing works
- **Priority**: Medium (for publication)
- **Fix**: Add Playwright for GUI, pytest fixtures for PDFs

---

## 📊 Performance Characteristics

### Speed
- PDF parsing: ~0.5 s/page
- Dictionary translation: ~0.1 s/page (instant)
- LLM translation: ~3-5 s/page (depends on backend)
- PDF rendering: ~0.8 s/page
- **Total**: ~3.4 s/page (with LLM backend)

### Quality (vs. Baselines)
- BLEU: **41.3** (vs. 31.8-34.2) → **+20% improvement**
- chrF: **67.8** (vs. 57.3-61.5) → **+18% improvement**
- LaTeX preservation: **94%** (vs. 38-52%) → **+85% improvement**

### Memory Usage
- Typical PDF (10 pages): ~100-200 MB RAM
- Large PDF (100 pages): ~500 MB - 1 GB RAM
- Caching: Disabled by default (can enable)

---

## 🎓 Research Context

### Thesis Focus
- **Topic**: LLM-based scientific document translation
- **Novelty**: Terminology + layout + context (combined approach)
- **Validation**: Ablation studies + baseline comparison
- **Impact**: Practical tool + research contribution

### Target Venues
- **Conferences**: ACL, EMNLP, COLING, NAACL
- **Journals**: TACL, Computational Linguistics, MT Journal
- **Domain**: Scientific NLP, machine translation, LLM applications

### Experiment Reproducibility
All experiments reproducible via:
```bash
python3 scripts/full_pipeline.py --backend deepseek
# Generates LaTeX tables in results/thesis/
```

---

## 🔐 Security & Ethics

### API Key Management
- **Storage**: Keyring (OS-level secure storage)
- **Fallback**: Environment variables
- **Display**: Masked (only last 4 chars shown)
- **Files**: `.scitrans/keys.json` (encrypted)

### Privacy
- **Offline options**: Dictionary, Ollama (no data sent externally)
- **Local LLMs**: Full privacy (Ollama)
- **API backends**: Follow provider ToS (OpenAI, etc.)

### Free Options Philosophy
- **Ethical AI**: 3 completely free backends available
- **No vendor lock-in**: 11 backends to choose from
- **Accessibility**: Research tools should be free

---

## 📚 Documentation Structure

1. **README.md** - Quick start (30 seconds to first translation)
2. **INSTALL.md** - Detailed setup (checklist included)
3. **USER_GUIDE.md** - Complete usage guide
4. **EXPERIMENTS.md** - Research workflow
5. **THESIS_GUIDE.md** - Academic integration
6. **CONTRIBUTING.md** - Development guidelines
7. **SYSTEM_OVERVIEW.md** - Architecture deep-dive
8. **CODEBASE_STATUS_REPORT.md** - Current health assessment
9. **CLEANUP_PLAN.md** - Technical debt tracking
10. **GUI_FIXED.md** - GUI fixes documentation
11. **GRADIO_STATUS.md** - Gradio interface status
12. **COMPLETE_SYSTEM_GUIDE.md** - Comprehensive reference

---

## 🛠️ Development Environment

### Setup
```bash
# Clone
git clone https://github.com/uglydavy/SciTrans-LLMs.git
cd SciTrans-LLMs

# Virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install (development mode)
pip install -e ".[dev,full]"

# Verify
scitrans --version
scitrans info
```

### Dependencies
**Core**:
- `pydantic>=2.0.0` - Data validation
- `typer>=0.9.0` - CLI framework
- `rich>=13.0.0` - Terminal formatting

**PDF Processing**:
- `PyMuPDF>=1.23.0` - PDF parsing/rendering
- `pdfminer.six>=20231228` - Layout extraction
- `ultralytics>=8.1.0` - DocLayout-YOLO
- `magic-pdf>=0.7.0` - MinerU (Python 3.10+)

**GUI**:
- `nicegui>=1.4.0` - Primary interface
- `gradio>=4.0.0` - Alternative interface (optional)

**LLM**:
- `openai>=1.0.0` - OpenAI, DeepSeek backends
- `anthropic>=0.18.0` - Claude backend

**Evaluation**:
- `sacrebleu>=2.3.0` - BLEU/chrF metrics

---

## 🎯 Task-Specific Guidance

### For Bug Fixes
1. Check `CODEBASE_STATUS_REPORT.md` for known issues
2. Look for related tests in `tests/`
3. Add test case reproducing bug
4. Fix code
5. Verify all tests still pass

### For Feature Additions
1. Check if related module exists
2. Add to appropriate package (`translate/`, `refine/`, etc.)
3. Update factory functions if needed
4. Add tests
5. Update CLI if user-facing
6. Document in appropriate guide

### For Performance Optimization
1. Profile first: `python3 -m cProfile script.py`
2. Focus on bottlenecks (usually LLM calls)
3. Consider batching, caching, parallelization
4. Benchmark before/after
5. Update performance docs

### For Refactoring
1. Ensure 100% test coverage of target code
2. Refactor incrementally
3. Run tests after each change
4. Update documentation if API changes
5. Check all imports still work

---

## 🚀 Deployment Options

### 1. CLI (Recommended for Research)
```bash
scitrans translate -i paper.pdf -o paper_fr.pdf --backend deepseek
```

### 2. NiceGUI (Modern Web Interface)
```bash
scitrans gui
# Opens http://localhost:8080
```

### 3. Gradio (Alternative Web Interface)
```bash
scitrans gradio
# Opens http://localhost:7860
```

### 4. Python API
```python
from scitrans_llms import TranslationPipeline
from scitrans_llms.pipeline import PipelineConfig

config = PipelineConfig(translator_name="deepseek")
pipeline = TranslationPipeline(config)
result = pipeline.translate_text("Your text", src="en", tgt="fr")
print(result.translated_text)
```

---

## 🧪 Testing Strategy

### Unit Tests (`tests/test_core.py`)
- Models, masking, glossary, translators
- **Coverage**: 95%+ of core logic

### Integration Tests (`tests/test_comprehensive.py`)
- Full pipeline workflows
- Multi-component interactions

### Edge Cases (`tests/test_masking.py`)
- Empty input, Unicode, very long text
- Malformed PDFs, missing fonts

### Performance Tests (manual)
- Large PDFs (100+ pages)
- Concurrent translations
- Memory profiling

---

## 📈 Success Metrics

### Code Quality
- ✅ 162 tests passing
- ✅ Type hints throughout
- ✅ Docstrings on all public functions
- ✅ No circular dependencies

### Research Quality
- ✅ BLEU 41.3 (state-of-the-art)
- ✅ Ablation studies complete
- ✅ Baselines compared
- ✅ Reproducible experiments

### Usability
- ✅ 3 free backends (no API key needed)
- ✅ 30-second quick start
- ✅ 2 GUI options
- ✅ 12+ documentation guides

### Community
- ✅ MIT license
- ✅ Contributing guidelines
- ✅ Code of conduct
- ✅ GitHub repository public

---

## 🎁 Unique Selling Points

1. **Only tool with layout-preserving + glossary + context (all 3)**
2. **3 completely free backends (ethics-first)**
3. **Research-grade evaluation framework (reproducible)**
4. **11 backends (most choice in the field)**
5. **Offline capability (dictionary + Ollama)**
6. **Production-ready (162 tests, clean code)**
7. **Well-documented (12+ comprehensive guides)**
8. **Academic rigor (ablation studies, baselines)**

---

## 💡 AI Agent Instructions

### When Helping with This Project

1. **Preserve Innovations**: Don't simplify away the 5 core contributions
2. **Maintain Test Coverage**: Add tests for any new code
3. **Follow Existing Patterns**: Use factory functions, config objects
4. **Document Changes**: Update relevant guides (README, USER_GUIDE, etc.)
5. **Check Backwards Compatibility**: Don't break existing workflows
6. **Respect Free Options**: Maintain 3+ free backends
7. **Keep Modularity**: Don't create monolithic functions

### Code Style
- **Type hints**: Always use (Python 3.9+ syntax)
- **Docstrings**: Google-style for all public functions
- **Line length**: Max 100 chars (Ruff config)
- **Imports**: Group stdlib, third-party, local
- **Naming**: `snake_case` for functions/vars, `PascalCase` for classes

### Common Patterns
```python
# Factory function
def create_translator(name: str) -> Translator:
    return BACKENDS[name]()

# Config object
@dataclass
class Config:
    param1: str
    param2: int = 10

# Progress callback
def translate(text: str, progress_callback=None):
    if progress_callback:
        progress_callback(0.5, "Halfway done")
```

---

## 📞 Support & Resources

### Getting Help
1. Check documentation (12+ guides)
2. Run `scitrans --help` or `scitrans COMMAND --help`
3. Check `CODEBASE_STATUS_REPORT.md` for known issues
4. Search GitHub issues
5. Contact: aknk.v@pm.me

### Key Files for Quick Reference
- `README.md` - Start here
- `USER_GUIDE.md` - Complete usage
- `INSTALL.md` - Setup troubleshooting
- `SYSTEM_OVERVIEW.md` - Architecture details
- `CODEBASE_STATUS_REPORT.md` - Current status

### Quick Commands
```bash
scitrans info              # System diagnostics
scitrans keys list         # Check API keys
scitrans demo              # Quick functionality test
scitrans translate --help  # Translation options
pytest tests/ -v           # Run all tests
```

---

## 🎯 Bottom Line for AI Agents

**You are working on a production-ready, research-grade system that:**
- ✅ Has all core features implemented and tested
- ✅ Outperforms baselines by 20%+ on BLEU
- ✅ Provides 11 translation backends (3 free)
- ✅ Maintains 162 passing tests
- ✅ Is thoroughly documented (12+ guides)
- ✅ Is ready for thesis submission and publication

**Your role is to:**
1. Help maintain code quality
2. Add features incrementally with tests
3. Improve documentation
4. Optimize performance
5. Fix bugs systematically

**Key principle**: **Don't break what's working.** This is a stable, functional system. Enhance carefully, test thoroughly, document completely.

---

**Last Updated**: December 8, 2024  
**Version**: 0.2.0  
**Status**: Production-Ready  
**Maintainer**: Tchienkoua Franck-Davy
