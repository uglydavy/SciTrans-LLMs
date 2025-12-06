# SciTrans-LLMs: Complete System Overview & Analysis

**Date**: 2025-12-02  
**Purpose**: Comprehensive analysis of system architecture, module responsibilities, quality assessment, and improvement roadmap for research publication

---

## Table of Contents
1. [System Statistics](#system-statistics)
2. [Architecture Overview](#architecture-overview)
3. [Module-by-Module Analysis](#module-by-module-analysis)
4. [Quality Assessment](#quality-assessment)
5. [Research Alignment](#research-alignment)
6. [Identified Issues](#identified-issues)
7. [Improvement Priorities](#improvement-priorities)

---

## System Statistics

- **Total Python modules**: 58
- **Core packages**: 8 (models, pipeline, translate, refine, ingest, render, eval, experiments)
- **Lines of code**: ~15,000 (estimated)
- **Test coverage**: Minimal (1 main test file)
- **Git commits**: 82+ (many with AI-style generic messages)
- **Documentation files**: 12 markdown files

---

## Architecture Overview

### Core Design Philosophy
SciTrans-LLMs implements a **modular pipeline architecture** for scientific document translation:

```
Input PDF → Ingestion → Masking → Translation → Refinement → Unmasking → Rendering → Output PDF
              ↓           ↓          ↓             ↓             ↓            ↓
         Layout     Placeholder  LLM/Dict    Glossary     Restore      Layout
        Detection   Protection    Backend   Enforcement  Protected  Preservation
```

### Key Innovations (Research Contributions)
1. **Terminology-Constrained Translation**: Glossary + masking for technical accuracy
2. **Document-Level Context**: LLM translation with previous segment awareness
3. **Layout Preservation**: Bounding box tracking + coordinate-based rendering
4. **Multi-Backend Support**: Dictionary, free APIs, commercial LLMs
5. **Candidate Reranking**: Quality scoring for best translation selection

---

## Module-by-Module Analysis

### 1. Core Data Models (`scitrans_llms/models.py`)
**Purpose**: Define document structure (Document → Segment → Block)

**Classes**:
- `BoundingBox`: PDF coordinates (x0, y0, x1, y1, page)
- `BlockType`: Enum for content types (PARAGRAPH, HEADING, EQUATION, CODE, etc.)
- `Block`: Atomic translation unit with source_text, translated_text, masked_text, bbox
- `Segment`: Logical document section (groups related blocks)
- `Document`: Top-level container with segments, language pair, glossary

**Assessment**:
- ✅ **Strength**: Clean abstraction, JSON serialization, factory methods
- ⚠️ **Issue**: `is_translatable` property excludes REFERENCE, HEADER, FOOTER - should these be configurable?
- 📝 **Research**: Central to Contribution #1 (layout preservation)

---

### 2. Translation Pipeline (`scitrans_llms/pipeline.py`)
**Purpose**: Orchestrate end-to-end translation workflow

**Key Components**:
- `PipelineConfig`: All translation settings (backend, masking, glossary, context, candidates)
- `TranslationPipeline`: Main orchestrator
  - `translate(doc)`: Full pipeline execution
  - `translate_text(text)`: Convenience for plain text
- `translate_document()`: PDF-specific wrapper for GUI

**Workflow**:
1. Mask protected content (formulas, code, URLs)
2. Build DocumentContext (previous translations for coherence)
3. Translate blocks with context + glossary
4. Rerank candidates if num_candidates > 1
5. Refine translations (glossary enforcement, structure preservation)
6. Unmask placeholders
7. Return PipelineResult with stats

**Assessment**:
- ✅ **Strength**: Well-structured, progress callbacks, comprehensive stats
- ✅ **Strength**: Captures first translation metadata for prompt inspection
- ⚠️ **Issue**: Reranking uses heuristics by default (LLM scoring requires OpenAI key)
- ⚠️ **Issue**: Multi-pass support exists but not well-exposed in API
- 📝 **Research**: Core of Contribution #2 (document-level translation)

---

### 3. Masking System (`scitrans_llms/masking.py`)
**Purpose**: Protect non-translatable content with placeholders

**Implementation**:
- `masking.py`: 
  - `MaskRegistry`: Stores placeholder → original mappings
  - `MaskConfig`: Configurable pattern detection
  - Patterns: LaTeX math, code blocks, URLs, emails, DOIs, section numbers, bullets
  - Block-level: `mask_block()`, `unmask_block()`
  - Document-level: `mask_document()`, `unmask_document()`

**Assessment**:
- ✅ **Strength**: Comprehensive regex patterns, structured config
- ✅ **Strength**: Preserves section numbers, bullets, indentation
- ⚠️ **Issue**: No post-translation validation that all placeholders were preserved
- ⚠️ **Issue**: Inline LaTeX commands without delimiters (\alpha, \frac{}) may be missed
- 📝 **Research**: Critical for Contribution #1 (technical content preservation)

---

### 4. Translation Backends (`scitrans_llms/translate/`)

#### 4.1 Base Abstractions (`base.py`)
- `Translator`: Abstract interface
- `TranslationContext`: Previous translations + glossary for document-level coherence
- `TranslationResult`: Contains translation + candidates + metadata
- `DummyTranslator`: Testing only (removed from user-facing UX)
- `DictionaryTranslator`: Offline word-by-word + corpus dictionary
- `create_translator()`: Factory function

**Assessment**:
- ✅ **Strength**: Clean interface, supports candidates
- ✅ **Strength**: DictionaryTranslator loads corpus dictionaries from `~/.scitrans/dictionaries/`
- ⚠️ **Issue**: `DummyTranslator` still in code (only for tests now)

#### 4.2 LLM Backends (`llm.py`)
- `LLMConfig`: Model, temperature, max_tokens
- `BaseLLMTranslator`: Shared prompt building
- `OpenAITranslator`: GPT-4, GPT-4o, GPT-5.1
- `DeepSeekTranslator`: OpenAI-compatible
- `AnthropicTranslator`: Claude
- `PerplexityTranslator`: Llama via Perplexity
- `MultiTurnTranslator`: Conversation-based for better coherence

**Prompts Include**:
- System: Domain rules, glossary terms, previous translations
- User: Text to translate with structure preservation instructions

**Assessment**:
- ✅ **Strength**: Unified prompt building, context injection, glossary in prompts
- ✅ **Strength**: Captures `system_prompt` + `user_prompt` in metadata for inspection
- ✅ **Strength**: Support for multiple candidates via temperature
- ⚠️ **Issue**: MultiTurnTranslator not well-tested or integrated
- 📝 **Research**: Core of Contribution #2 (LLM-based document translation)

#### 4.3 Free Backends (`free_translator.py`, `free_apis.py`)
- `FreeTranslator`: Cascading fallback (Lingva → LibreTranslate → MyMemory → Dictionary)
- `TranslationCache`: Local caching for offline capability
- `HuggingFaceTranslator`: Free tier API
- `OllamaTranslator`: Local LLM
- `GoogleFreeTranslator`: Via deep-translator library

**Assessment**:
- ✅ **Strength**: Smart cascading with caching
- ✅ **Strength**: Robust dictionary fallback using `DictionaryTranslator`
- ⚠️ **Issue**: External API reliability varies
- ⚠️ **Issue**: No candidate support (always returns single translation)

#### 4.4 Glossary (`glossary.py`)
- `GlossaryEntry`: source → target + domain
- `Glossary`: Collection with filtering, lookup, enforcement
- `get_default_glossary()`: 181 scientific terms
- `check_glossary_adherence()`: Post-translation validation

**Assessment**:
- ✅ **Strength**: Domain-specific terms (ML, math, physics)
- ✅ **Strength**: Enforcement and adherence checking
- ⚠️ **Issue**: Default glossary is small (181 terms) - needs expansion

#### 4.5 Corpus Management (`corpus_manager.py`)
- Download parallel corpora (Europarl, JW300, etc.)
- Build translation dictionaries from aligned text
- Integration with online dictionaries

**Assessment**:
- ✅ **Strength**: Supports offline dictionary training
- ⚠️ **Issue**: Not well-tested with diverse corpora
- ⚠️ **Issue**: Online dictionary sources may be unreliable

---

### 5. Refinement System (`scitrans_llms/refine/`)

#### 5.1 Base (`base.py`)
- `RefineResult`: Changed status + reason
- `Refiner`: Abstract interface
- `GlossaryRefiner`: Enforce glossary post-translation
- `LLMRefiner`: Use LLM for quality improvement
- `create_refiner()`: Factory

**Assessment**:
- ✅ **Strength**: Modular, multiple strategies
- ⚠️ **Issue**: LLMRefiner requires API key, not well-exposed

#### 5.2 Reranking (`rerank.py`)
- `CandidateScore`: Fluency, adequacy, terminology, placeholder preservation
- `CandidateReranker`: Score and select best candidate
- `DEFAULT_LLM_WEIGHTS`: adequacy 35%, placeholder 30%, terminology 25%, fluency 10%
- `DEFAULT_HEURISTIC_WEIGHTS`: placeholder 35%, terminology 30%, adequacy 25%, fluency 10%
- LLM scoring via GPT-4o-mini for cost efficiency

**Assessment**:
- ✅ **Strength**: Multi-dimensional scoring, configurable weights
- ✅ **Strength**: Falls back to heuristics when LLM unavailable
- ⚠️ **Issue**: LLM scoring requires OpenAI key (not all users have it)
- 📝 **Research**: Contribution #2 (candidate selection)

#### 5.3 Prompting (`prompting.py`)
- `build_prompt()`: Construct translation prompts
- Language-specific guidelines (French: space before punctuation, number formatting)
- Domain-specific rules (scientific, legal, medical)
- Few-shot examples

**Assessment**:
- ✅ **Strength**: Domain-aware, language-specific
- ⚠️ **Issue**: Only French target language well-supported

#### 5.4 Post-Processing (`postprocess.py`)
- `preserve_list_structure()`: Maintain numbering/bullets

**Assessment**:
- ✅ **Strength**: Targeted structure preservation
- ⚠️ **Issue**: Limited scope (only lists)

---

### 6. PDF Ingestion (`scitrans_llms/ingest/`)

#### 6.1 Main Parser (`pdf.py`)
- `PDFParser`: Extract text + layout from PDFs
- `TextSpan`: Text + bbox + font metadata
- `PageContent`: Spans + images per page
- `LayoutDetector`: Abstract interface
- `HeuristicLayoutDetector`: Rule-based classification
- `YOLOLayoutDetector`: ML-based (DocLayout-YOLO)

**Extraction Flow**:
1. PyMuPDF extracts text spans with coordinates
2. Layout detector classifies spans (paragraph, heading, equation, etc.)
3. Spans grouped into logical blocks
4. Blocks sorted by reading order
5. Convert to Document with Segments and Blocks

**Assessment**:
- ✅ **Strength**: Detailed layout extraction, multiple detectors
- ✅ **Strength**: DocLayout-YOLO enforced for block classification (no heuristic fallback)
- ⚠️ **Issue**: Requires bundled YOLO weights (`data/layout/layout_model.pt`) and ultralytics
- ⚠️ **Issue**: MinerU remains fallback; needs validation coverage
- ⚠️ **Issue**: Two-column layouts, tables, rotated text not validated
- 📝 **Research**: Critical for Contribution #3 (layout preservation)

#### 6.2 YOLO Integration (`yolo/`, `yolo_detection.py`)
- `LayoutPredictor`: Wrapper for DocLayout-YOLO
- Training script for fine-tuning on scientific PDFs

**Assessment**:
- ⚠️ **Issue**: Not fully wired into main parser
- ⚠️ **Issue**: Weights not included in repo

---

### 7. PDF Rendering (`scitrans_llms/render/pdf.py`)
**Purpose**: Reconstruct translated PDF preserving layout

- `render_pdf()`: Main entry point
- Modes: replace, overlay, side-by-side
- Uses bounding boxes to position translated text
- Preserves fonts (attempts to match original)

**Assessment**:
- ✅ **Strength**: Coordinate-based placement
- ⚠️ **Issue**: Font matching is best-effort
- ⚠️ **Issue**: Complex layouts (tables, multi-column) may fail
- ⚠️ **Issue**: Rendering failures fallback to text output silently in some paths
- 📝 **Research**: Part of Contribution #3 (layout preservation)

---

### 8. Evaluation System (`scitrans_llms/eval/`)

#### 8.1 Metrics (`metrics.py`)
- BLEU, chrF via SacreBLEU
- Glossary adherence rate
- Placeholder integrity

**Assessment**:
- ✅ **Strength**: Standard MT metrics
- ⚠️ **Issue**: Not integrated into GUI feedback

#### 8.2 Runner (`runner.py`)
- `EvaluationRunner`: Compute metrics on hypothesis vs reference
- JSON/CSV/LaTeX output

**Assessment**:
- ✅ **Strength**: Multiple output formats
- ⚠️ **Issue**: Requires reference translations (not always available)

#### 8.3 Ablation (`ablation.py`)
- `AblationStudy`: Compare configs (masking on/off, glossary on/off, etc.)
- `AblationConfig`: Define experiment matrix

**Assessment**:
- ✅ **Strength**: Systematic component evaluation
- 📝 **Research**: Critical for thesis experiments

---

### 9. Experiments (`scitrans_llms/experiments/`)

#### 9.1 Runner (`runner.py`)
- `ExperimentRunner`: Run translation experiments with config sweeps
- `ExperimentConfig`: Backend, masking, glossary, context settings

**Assessment**:
- ✅ **Strength**: Reproducible experiments
- ⚠️ **Issue**: Limited documentation on how to use

#### 9.2 Thesis (`thesis.py`)
- Export results as LaTeX tables for thesis
- Integration with `thesis/` directory

**Assessment**:
- ✅ **Strength**: Direct thesis integration
- ⚠️ **Issue**: Assumes specific thesis structure

---

### 10. GUI (`scitrans_llms/gui.py`)
**Purpose**: Web interface via NiceGUI

**Tabs**:
1. **Translate**: PDF upload, settings, translation, download
2. **Testing**: Quick text translation
3. **Glossary**: Browse default glossary
4. **Developer**: System diagnostics
5. **Settings**: API keys, preferences

**Key Features**:
- PDF preview (source and translated)
- Page navigation
- Engine selection (dictionary, free, openai, deepseek, etc.)
- Candidate control (1, 3, 5)
- Quality passes (1-5)
- Advanced settings: masking, reranking, context window, prompt preview
- Progress logging

**Assessment**:
- ✅ **Strength**: Comprehensive feature set
- ⚠️ **Critical Issue**: Panels overflow vertically, require scrolling
- ⚠️ **Issue**: Preview images don't fit in preview area (need `object-fit: contain`)
- ⚠️ **Issue**: No session persistence (page reload loses state)
- ⚠️ **Issue**: No connection status indicator
- ⚠️ **Issue**: Limited error recovery UI

---

### 11. CLI (`scitrans_llms/cli.py`)
**Purpose**: Command-line interface via Typer

**Commands**:
- `translate`: Main translation command
- `glossary`: View/search glossary
- `evaluate`: Compute BLEU/chrF
- `ablation`: Run ablation studies
- `demo`: Quick pipeline demo
- `info`: Show available backends
- `keys`: Manage API keys
- `corpus`: Download/build dictionaries

**Assessment**:
- ✅ **Strength**: Comprehensive, well-documented
- ✅ **Strength**: Supports all pipeline features
- ⚠️ **Issue**: Some commands (corpus, ablation) need better examples

---

### 12. Key Management (`scitrans_llms/keys.py`)
- `KeyManager`: Store/retrieve API keys
- Sources: environment variables, keyring, config files
- Supports: OpenAI, DeepSeek, Anthropic, HuggingFace, DeepL, Google

**Assessment**:
- ✅ **Strength**: Secure keyring integration
- ✅ **Strength**: Multiple fallback sources
- ⚠️ **Issue**: Export to env vars could be clearer

---

### 13. Configuration (`scitrans_llms/config.py`, `bootstrap.py`)
- Define paths: `DATA_DIR`, `CACHE_DIR`, `MODELS_DIR`
- Bootstrap: Download YOLO weights, glossaries

**Assessment**:
- ✅ **Strength**: Centralized configuration
- ⚠️ **Issue**: Bootstrap not always run automatically

---

### 14. Utilities (`scitrans_llms/utils.py`, `diagnostics.py`)
- `utils.py`: Minimal helpers
- `diagnostics.py`: Check dependencies, API keys, data files

**Assessment**:
- ✅ **Strength**: Useful for debugging
- ⚠️ **Issue**: Could be expanded

---

### 15. Scripts (`scripts/`)
- `quick_test.py`: Smoke test
- `run_experiments.py`: Run experiment sweeps
- `full_pipeline.py`: End-to-end thesis experiments
- `collect_corpus.py`: Download parallel corpora
- `setup_keys.py`: Interactive key setup

**Assessment**:
- ✅ **Strength**: Cover major workflows
- ⚠️ **Issue**: Insufficient inline documentation
- ⚠️ **Issue**: Not documented in README

---

### 16. Tests (`tests/test_core.py`)
- Basic pipeline tests
- Masking tests
- Glossary tests

**Assessment**:
- ⚠️ **Critical Issue**: Only 1 test file
- ⚠️ **Issue**: No GUI tests
- ⚠️ **Issue**: No PDF rendering tests
- ⚠️ **Issue**: No integration tests with real PDFs

---

### 17. Documentation

#### Core Docs
- `README.md`: Overview, quick start, features
- `INSTALL.md`: Installation instructions
- `USER_GUIDE.md`: Detailed usage
- `EXPERIMENTS.md`: How to run experiments
- `THESIS_GUIDE.md`: Thesis experiment reproduction
- `MODULE_REFERENCE.md`: Auto-generated module docs
- `WARP.md`: AI assistant context (for future sessions)

#### Specialized
- `corpus/README.md`: Corpus structure
- `thesis/README.md`: Thesis integration
- `CONTRIBUTING.md`: Contribution guidelines

#### Improvement Plans (should be removed)
- `DOCUMENTATION.txt`: Generic
- `EXTRACTION_IMPROVEMENTS.md`: Unfinished
- `QUICK_START_IMPROVEMENTS.md`: Unfinished

**Assessment**:
- ✅ **Strength**: Comprehensive core docs
- ⚠️ **Issue**: Improvement plans should be converted or removed
- ⚠️ **Issue**: MODULE_REFERENCE.md seems auto-generated
- ⚠️ **Issue**: Scripts not documented in README

---

## Quality Assessment

### Strengths
1. **Modular architecture**: Clean separation of concerns
2. **Research-focused**: Clear thesis contributions
3. **Multi-backend support**: Flexible translation options
4. **Layout preservation**: Bounding box tracking
5. **Comprehensive CLI**: Well-structured commands
6. **Glossary system**: Domain-specific terminology
7. **Reranking**: Multi-dimensional quality scoring

### Critical Issues
1. **GUI layout overflow**: Doesn't fit on screen
2. **Minimal testing**: Only 1 test file
3. **AI-generated artifacts**: 82+ generic commits, improvement docs
4. **No session persistence**: State lost on reload
5. **Incomplete YOLO integration**: Falls back to heuristics
6. **No PDF validation suite**: Untested on diverse PDFs
7. **Two masking modules**: Confusing duplication

### Moderate Issues
1. **Dictionary translation quality**: Word-by-word, not sentence-level
2. **Free backend reliability**: Depends on external services
3. **Rendering failures**: Silent fallback to text
4. **Limited error recovery**: Hard to retry failed blocks
5. **Glossary size**: Only 181 terms
6. **No placeholder validation**: After translation
7. **MultiTurnTranslator**: Not integrated

### Minor Issues
1. **DummyTranslator**: Still in codebase (tests only)
2. **Two preview implementations**: In GUI
3. **Corpus manager**: Not well-tested
4. **LLMRefiner**: Not exposed
5. **French-centric**: Other languages less supported

---

## Research Alignment

### Thesis Contributions
1. ✅ **Terminology-Constrained Translation**: Glossary + masking system works
2. ✅ **Document-Level Context**: LLM translation with previous segments
3. ⚠️ **Layout Preservation**: Works but needs validation on diverse PDFs

### Missing for Research Quality
1. **Quantitative evaluation**: Need BLEU/chrF on test corpus
2. **Ablation study results**: Need to run and document
3. **Baseline comparisons**: PDFMathTranslate, DocuTranslate, Google Translate
4. **Error analysis**: What types of errors occur?
5. **User study**: (if applicable)

---

## Identified Issues Summary

### Must Fix (Blocking Research Publication)
1. GUI layout overflow
2. No test suite for PDF processing
3. AI-generated commit messages
4. Incomplete YOLO integration
5. No quantitative evaluation on test corpus
6. Duplicate masking modules

### Should Fix (Quality/Usability)
1. Session persistence
2. Preview image fitting
3. Dictionary translation quality
4. Placeholder validation
5. Error recovery UI
6. Scripts documentation

### Nice to Have (Polish)
1. MultiTurnTranslator integration
2. Expand glossary
3. Support more target languages
4. Performance optimization
5. Accessibility features

---

## Improvement Priorities

### Phase 1: Critical (Research Blocking)
1. **GUI layout fix**: Ensure everything fits on 1920x1080
2. **PDF test suite**: 20-50 diverse PDFs with validation
3. **Consolidate masking**: Remove duplicate, add validation
4. **Run ablation studies**: Document results
5. **Baseline comparisons**: Implement and run

### Phase 2: Quality (Pre-Publication)
1. **Session persistence**: Browser localStorage
2. **Preview fitting**: CSS fixes
3. **Error recovery**: Retry UI
4. **Expand tests**: Unit + integration
5. **Document scripts**: README updates

### Phase 3: Polish (Post-Core)
1. **Dictionary improvement**: Phrase-level
2. **MultiTurnTranslator**: Full integration
3. **Performance**: Optimize bottlenecks
4. **Accessibility**: Keyboard nav, ARIA labels
5. **More languages**: Beyond English-French

---

## Next Steps

See `CLEANUP_PLAN.md` for:
- Files to remove
- Commits to squash/rewrite
- Module consolidation steps
- Git history cleanup commands
- Documentation improvements
