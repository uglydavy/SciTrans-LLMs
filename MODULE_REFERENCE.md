# SciTrans-LLMs: Complete Module Reference

> **Purpose**: This document provides a detailed explanation of every module, class, and function in the SciTrans-LLMs system. It is intended for thesis defense preparation and deep understanding of the system architecture.
>
> **Delete this file after learning** - it contains implementation details meant for your understanding, not for repository documentation.

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Core Data Models (`models.py`)](#2-core-data-models-modelspy)
3. [Configuration (`config.py`)](#3-configuration-configpy)
4. [Translation Pipeline (`pipeline.py`)](#4-translation-pipeline-pipelinepy)
5. [Masking System (`masking.py` and `mask.py`)](#5-masking-system-maskingpy-and-maskpy)
6. [Translation Backends (`translate/`)](#6-translation-backends-translate)
7. [PDF Ingestion (`ingest/`)](#7-pdf-ingestion-ingest)
8. [Refinement System (`refine/`)](#8-refinement-system-refine)
9. [Evaluation Framework (`eval/`)](#9-evaluation-framework-eval)
10. [CLI Interface (`cli.py`)](#10-cli-interface-clipy)
11. [Web GUI (`gui.py`)](#11-web-gui-guipy)
12. [Key Management (`keys.py`)](#12-key-management-keyspy)
13. [Thesis Contributions Mapping](#13-thesis-contributions-mapping)

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SciTrans-LLMs Architecture                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │    CLI      │    │    GUI      │    │   Python    │    │   Scripts   │  │
│  │  (cli.py)   │    │  (gui.py)   │    │     API     │    │   (tests)   │  │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘  │
│         │                  │                  │                  │          │
│         └──────────────────┴────────┬─────────┴──────────────────┘          │
│                                     │                                        │
│                           ┌─────────▼─────────┐                             │
│                           │   TranslationPipeline                            │
│                           │    (pipeline.py)   │                             │
│                           └─────────┬─────────┘                             │
│                                     │                                        │
│         ┌───────────────────────────┼───────────────────────────┐           │
│         │                           │                           │           │
│  ┌──────▼──────┐           ┌───────▼───────┐           ┌───────▼───────┐  │
│  │   Masking    │           │   Translator   │           │   Refiner     │  │
│  │ (masking.py) │           │  (translate/)  │           │  (refine/)    │  │
│  └──────────────┘           └───────────────┘           └───────────────┘  │
│                                     │                                        │
│         ┌───────────────────────────┼───────────────────────────┐           │
│         │                           │                           │           │
│  ┌──────▼──────┐           ┌───────▼───────┐           ┌───────▼───────┐  │
│  │  Glossary    │           │    Context     │           │  Evaluation   │  │
│  │(glossary.py) │           │  (context.py)  │           │   (eval/)     │  │
│  └──────────────┘           └───────────────┘           └───────────────┘  │
│                                                                              │
│                           ┌─────────────────┐                               │
│                           │  Data Models     │                               │
│                           │   (models.py)    │                               │
│                           └─────────────────┘                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Input**: PDF or plain text → `ingest/` module
2. **Document Creation**: Text converted to `Document` → `Segment` → `Block` hierarchy
3. **Masking**: Protected content (formulas, code, URLs) replaced with placeholders
4. **Translation**: Each block translated with document-level context
5. **Refinement**: Glossary enforcement, coherence checks
6. **Unmasking**: Placeholders restored with original content
7. **Output**: Translated document (text or PDF with layout preservation)

---

## 2. Core Data Models (`models.py`)

### Purpose
Defines the hierarchical data structures representing documents for translation.

### Key Classes

#### `BlockType` (Enum)
```python
class BlockType(Enum):
    PARAGRAPH = auto()    # Regular text - translate normally
    HEADING = auto()      # Section headings - translate, preserve structure
    LIST_ITEM = auto()    # Bullet/numbered items - translate
    TABLE = auto()        # Tables - protect or translate cells
    FIGURE = auto()       # Figures/images - protect entirely
    EQUATION = auto()     # Math formulas - protect (placeholder)
    CODE = auto()         # Code blocks - protect
    CAPTION = auto()      # Figure/table captions - translate
    FOOTNOTE = auto()     # Footnotes - translate
    REFERENCE = auto()    # Bibliography entries - partial protection
    HEADER = auto()       # Page headers - often skip
    FOOTER = auto()       # Page footers - often skip
    UNKNOWN = auto()      # Fallback
```

**Thesis Relevance**: Block types determine whether content is protected (not translated) or translated. This is fundamental to **Thesis Contribution #1** (terminology-constrained translation).

#### `BoundingBox`
```python
@dataclass
class BoundingBox:
    x0: float      # Left coordinate
    y0: float      # Top coordinate
    x1: float      # Right coordinate
    y1: float      # Bottom coordinate
    page: int = 0  # Page number
```

**Purpose**: Stores layout coordinates in PDF points (1/72 inch). Essential for layout-preserving PDF translation.

**Key Properties**:
- `width`, `height`: Computed dimensions
- `center`: Center point tuple

#### `Block`
```python
@dataclass
class Block:
    source_text: str                    # Original text
    block_type: BlockType               # Semantic type
    block_id: str                       # Unique identifier
    bbox: Optional[BoundingBox]         # Layout coordinates
    translated_text: Optional[str]      # Filled after translation
    masked_text: Optional[str]          # Text with placeholders
    metadata: dict                      # Font, style info
```

**Key Properties**:
- `is_protected`: True for EQUATION, CODE, FIGURE blocks
- `is_translatable`: True for PARAGRAPH, HEADING, LIST_ITEM, CAPTION, FOOTNOTE

**Thesis Relevance**: The atomic unit of translation. Block-level granularity enables fine-grained control over what gets translated.

#### `Segment`
```python
@dataclass
class Segment:
    blocks: list[Block]        # Content blocks
    segment_id: str            # Unique identifier
    title: Optional[str]       # Section heading
    metadata: dict             # Additional info
```

**Purpose**: Groups related blocks for document-level context. Corresponds to sections/subsections.

**Thesis Relevance**: Segments provide the context window for **Thesis Contribution #2** (document-level LLM translation).

#### `Document`
```python
@dataclass
class Document:
    segments: list[Segment]            # Document sections
    doc_id: str                        # Unique identifier
    source_lang: str = "en"            # Source language
    target_lang: str = "fr"            # Target language
    title: Optional[str]               # Document title
    metadata: dict                     # Authors, source file, etc.
    glossary_terms: dict[str, str]     # Extracted terminology
```

**Key Methods**:
- `from_text(text, source_lang, target_lang)`: Create from plain text
- `from_paragraphs(paragraphs, ...)`: Create from paragraph list
- `to_json()` / `from_json()`: Serialization
- `summary()`: Human-readable document info

---

## 3. Configuration (`config.py`)

### Purpose
Defines project-wide paths and ensures directory structure exists.

### Key Constants

```python
APP_NAME = "SciTrans-LM"               # Application name
DATA_DIR = Path(__file__) / "data"     # Main data directory
LAYOUT_DIR = DATA_DIR / "layout"       # YOLO models
GLOSSARY_DIR = DATA_DIR / "glossary"   # Terminology files
CACHE_DIR = DATA_DIR / "cache"         # Temporary files
LAYOUT_MODEL = LAYOUT_DIR / "layout_model.pt"      # YOLO weights
DEFAULT_GLOSSARY = GLOSSARY_DIR / "default_en_fr.csv"  # Default glossary
```

**Why This Matters**: Centralized configuration makes the system portable and ensures all components use consistent paths.

---

## 4. Translation Pipeline (`pipeline.py`)

### Purpose
Orchestrates the complete translation workflow from input to output.

### Key Classes

#### `PipelineConfig`
```python
@dataclass
class PipelineConfig:
    # Translation settings
    source_lang: str = "en"
    target_lang: str = "fr"
    translator_backend: str = "dummy"
    translator_kwargs: dict = {}
    
    # Feature toggles (for ablation studies)
    enable_masking: bool = True
    mask_config: MaskConfig = MaskConfig()
    enable_glossary: bool = True
    glossary: Optional[Glossary] = None
    glossary_in_prompt: bool = True
    glossary_post_process: bool = True
    enable_context: bool = True
    context_window_size: int = 5
    enable_refinement: bool = True
    refiner_mode: str = "default"
    num_candidates: int = 1
```

**Thesis Relevance**: Feature toggles enable **ablation studies** - you can disable components individually to measure their contribution.

#### `PipelineResult`
```python
@dataclass
class PipelineResult:
    document: Document                  # Translated document
    config: PipelineConfig              # Configuration used
    mask_registry: Optional[MaskRegistry]
    stats: dict                         # Translation statistics
    errors: list[str]                   # Any errors encountered
```

#### `TranslationPipeline`
```python
class TranslationPipeline:
    def __init__(self, config, progress_callback):
        self.config = config
        self.translator = create_translator(config.translator_backend)
        self.glossary = config.glossary or get_default_glossary()
        self.refiner = create_refiner(config.refiner_mode)
    
    def translate(self, document: Document) -> PipelineResult:
        # Step 1: Masking
        mask_registry = mask_document(document, self.config.mask_config)
        
        # Step 2: Translation with context
        for block in document.translatable_blocks:
            context = TranslationContext(
                previous_source=...,
                previous_target=...,
                glossary=self.glossary
            )
            result = self.translator.translate_block(block, context)
            block.translated_text = result.text
        
        # Step 3: Refinement
        if self.refiner:
            self.refiner.refine_document(document, self.glossary)
        
        # Step 4: Unmasking
        unmask_document(document, mask_registry)
        
        return PipelineResult(document, config, stats)
```

**Key Methods**:
- `translate(document)`: Full pipeline execution
- `translate_text(text)`: Convenience for plain text

### Convenience Functions

```python
def translate_text(text, source_lang, target_lang, backend, ...):
    """Quick translation of plain text."""
    
def translate_document(input_path, output_path, engine, direction, ...):
    """Translate PDF with layout preservation (used by GUI)."""
```

---

## 5. Masking System (`masking.py` and `mask.py`)

### Purpose
Protects non-translatable content (formulas, code, URLs) by replacing them with placeholders before translation, then restoring them after.

### Why Masking is Critical

1. **Formulas**: LaTeX like `$E = mc^2$` would be corrupted by translation
2. **Code**: Variable names and syntax must be preserved
3. **URLs/DOIs**: References must remain intact
4. **Numbers with units**: `42 MHz` should stay as-is

### Key Components

#### `MaskRegistry`
```python
@dataclass
class MaskRegistry:
    mappings: dict[str, str]  # placeholder → original
    counters: dict[str, int]  # prefix → count
    
    def register(self, prefix: str, original: str) -> str:
        """Create placeholder like <<MATH_001>> and store original."""
        
    def restore(self, text: str) -> str:
        """Replace all placeholders with original content."""
```

**Placeholder Format**: `<<TYPE_NNN>>` where TYPE indicates content type:
- `MATH`, `MATHDISP`, `MATHENV`: LaTeX formulas
- `CODE`, `CODEBLK`: Code snippets
- `URL`, `DOI`, `EMAIL`: References

#### `MaskConfig`
```python
@dataclass
class MaskConfig:
    mask_latex_inline: bool = True     # $...$
    mask_latex_display: bool = True    # $$...$$ or \[...\]
    mask_latex_env: bool = True        # \begin{equation}...
    mask_code_blocks: bool = True      # ```...```
    mask_inline_code: bool = True      # `...`
    mask_urls: bool = True
    mask_emails: bool = True
    mask_dois: bool = True
    mask_numbers_units: bool = False   # Often want context
```

### Pattern Definitions

```python
# LaTeX inline: $...$
LATEX_INLINE_PATTERN = re.compile(r'\$(?!\$)(.+?)(?<!\$)\$')

# LaTeX display: $$...$$ or \[...\]
LATEX_DISPLAY_PATTERN = re.compile(r'(\$\$.+?\$\$|\\\[.+?\\\])')

# Code blocks
CODE_BLOCK_PATTERN = re.compile(r'```[\s\S]*?```')

# URLs
URL_PATTERN = re.compile(r'https?://[^\s<>\[\]()"\']+'...)
```

### Key Functions

```python
def mask_text(text, registry, config) -> str:
    """Apply all configured masks to text."""
    
def unmask_text(text, registry) -> str:
    """Restore all placeholders."""
    
def mask_document(doc, config) -> MaskRegistry:
    """Mask all blocks in a document."""
    
def unmask_document(doc, registry) -> None:
    """Restore all placeholders in translated blocks."""
    
def validate_placeholders(source_masked, translated) -> list[str]:
    """Check that all placeholders survived translation."""
```

### Thesis Relevance

**Contribution #1**: Masking is the foundation of terminology-constrained translation. Without it, formulas like `\frac{\partial f}{\partial x}` would be corrupted.

---

## 6. Translation Backends (`translate/`)

### Module Structure

```
translate/
├── __init__.py
├── base.py           # Abstract interface, DummyTranslator, DictionaryTranslator
├── llm.py            # OpenAI, DeepSeek, Anthropic translators
├── free_apis.py      # HuggingFace, Ollama, GoogleFree translators
├── free_translator.py # Cascading free translator (Lingva→LibreTranslate→MyMemory)
├── offline.py        # Enhanced offline translation
├── glossary.py       # Glossary management
├── context.py        # Document-level context window
└── memory.py         # Translation memory
```

### Abstract Interface (`base.py`)

#### `TranslationResult`
```python
@dataclass
class TranslationResult:
    text: str                           # Translated text
    source_text: str                    # Original
    candidates: list[str]               # Alternative translations
    metadata: dict                      # Model, tokens, etc.
    glossary_terms_used: list[str]      # Which terms were applied
```

#### `TranslationContext`
```python
@dataclass
class TranslationContext:
    previous_source: list[str]          # Previous source segments
    previous_target: list[str]          # Previous translations
    document_summary: Optional[str]     # Optional doc summary
    glossary: Optional[Glossary]        # Terminology to enforce
    source_lang: str = "en"
    target_lang: str = "fr"
```

**Thesis Relevance**: Context enables **document-level translation** - the translator knows what came before for coherence.

#### `Translator` (Abstract Base Class)
```python
class Translator(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Translator identifier."""
    
    @abstractmethod
    def translate(self, text, context, num_candidates) -> TranslationResult:
        """Translate single text."""
    
    def translate_batch(self, texts, context) -> list[TranslationResult]:
        """Override for efficient batching."""
    
    def translate_block(self, block, context) -> TranslationResult:
        """Use masked_text if available."""
```

### Built-in Translators

#### `DummyTranslator`
```python
class DummyTranslator(Translator):
    """For testing. Applies glossary + adds [TRANSLATED] prefix."""
```
- **Use case**: Pipeline testing without API calls
- **Modes**: `prefix`, `echo`, `upper`, `reverse`

#### `DictionaryTranslator`
```python
class DictionaryTranslator(Translator):
    """Offline translation using glossary + built-in dictionary."""
    
    BASIC_DICT = {
        'the': 'le', 'is': 'est', 'machine': 'machine',
        'learning': 'apprentissage', 'neural': 'neuronal',
        'network': 'réseau', 'algorithm': 'algorithme', ...
    }
```
- **Use case**: Completely offline translation
- **Quality**: Basic but functional for testing

### LLM Translators (`llm.py`)

#### `OpenAITranslator`
```python
class OpenAITranslator(Translator):
    """GPT-4, GPT-4o, GPT-5.1 translation."""
    
    def __init__(self, config, api_key):
        self.client = OpenAI(api_key=api_key)
```

#### `DeepSeekTranslator`
```python
class DeepSeekTranslator(Translator):
    """DeepSeek translation (OpenAI-compatible API)."""
```

#### `AnthropicTranslator`
```python
class AnthropicTranslator(Translator):
    """Claude 3 translation."""
```

### Free Translators (`free_apis.py` and `free_translator.py`)

#### `HuggingFaceTranslator`
```python
class HuggingFaceTranslator(Translator):
    """Free tier: 1000 requests/month, no credit card."""
    
    def __init__(self, model="facebook/mbart-large-50-many-to-many-mmt"):
        self.api_url = f"https://api-inference.huggingface.co/models/{model}"
```

#### `OllamaTranslator`
```python
class OllamaTranslator(Translator):
    """Local LLM via Ollama - completely free and offline."""
    
    def __init__(self, model="llama3", base_url="http://localhost:11434"):
        ...
```

#### `FreeTranslator` (Cascading)
```python
class FreeTranslator(Translator):
    """Tries multiple free services with caching."""
    
    # Translation cascade:
    # 1. Local cache (instant, offline)
    # 2. Lingva Translate (privacy-focused)
    # 3. LibreTranslate (open source)
    # 4. MyMemory (reliable)
    # 5. Basic dictionary (fallback)
```

### Factory Function

```python
def create_translator(backend: str, **kwargs) -> Translator:
    """Create translator by name."""
    
    # Supported backends:
    # - dummy, echo, test
    # - dictionary, offline, glossary
    # - openai, gpt, gpt4, gpt-5.1
    # - deepseek, ds
    # - anthropic, claude
    # - huggingface, hf
    # - ollama, local-llm
    # - googlefree, googletrans
    # - free, free-cascade (recommended free option)
    # - improved-offline
```

### Glossary Management (`glossary.py`)

```python
@dataclass
class GlossaryEntry:
    source: str          # English term
    target: str          # French term
    domain: str = ""     # e.g., "ml", "math"
    notes: str = ""      # Usage notes

class Glossary:
    entries: list[GlossaryEntry]
    name: str
    source_lang: str
    target_lang: str
    
    def get_target(self, source: str) -> Optional[str]:
        """Look up translation for a term."""
    
    def filter_by_domain(self, domain: str) -> Glossary:
        """Get domain-specific subset."""
    
    @classmethod
    def from_csv(cls, path: Path) -> Glossary:
        """Load from CSV file."""

def get_default_glossary() -> Glossary:
    """Load built-in EN-FR glossary (181 terms)."""
```

---

## 7. PDF Ingestion (`ingest/`)

### Module Structure

```
ingest/
├── __init__.py
├── pdf.py        # PDF parsing with layout detection
└── analyzer.py   # Document structure analysis
```

### PDF Parsing (`pdf.py`)

#### `TextSpan`
```python
@dataclass
class TextSpan:
    text: str
    bbox: BoundingBox
    font_name: str
    font_size: float
    is_bold: bool
    is_italic: bool
```

#### `LayoutDetector` (Abstract)
```python
class LayoutDetector(ABC):
    @abstractmethod
    def detect(self, page: PageContent) -> list[tuple[BoundingBox, BlockType]]:
        """Detect layout regions on a page."""
```

#### `HeuristicLayoutDetector`
```python
class HeuristicLayoutDetector(LayoutDetector):
    """Rule-based detection without ML models."""
    
    # Uses heuristics:
    # - Font size > 14 → likely HEADING
    # - Contains \$, \frac → likely EQUATION
    # - Starts with "Figure 1:" → likely CAPTION
    # - Near page edges → likely HEADER/FOOTER
```

#### `YOLOLayoutDetector`
```python
class YOLOLayoutDetector(LayoutDetector):
    """DocLayout-YOLO for accurate detection."""
    
    LABEL_MAP = {
        "text": BlockType.PARAGRAPH,
        "title": BlockType.HEADING,
        "figure": BlockType.FIGURE,
        "table": BlockType.TABLE,
        "formula": BlockType.EQUATION,
        ...
    }
```

#### `PDFParser`
```python
class PDFParser:
    def parse(self, pdf_path, pages, source_lang, target_lang) -> Document:
        """Parse PDF into Document with blocks and layout info."""
        
        # For each page:
        # 1. Extract text spans with coordinates
        # 2. Run layout detection
        # 3. Group spans into blocks
        # 4. Classify block types
        # 5. Create Document structure
```

### Convenience Function

```python
def parse_pdf(pdf_path, pages=None, source_lang="en", target_lang="fr", use_yolo=False):
    """Parse PDF with optional YOLO detection."""
```

---

## 8. Refinement System (`refine/`)

### Module Structure

```
refine/
├── __init__.py
├── base.py           # Abstract Refiner interface
├── llm.py            # LLM-based refinement
├── postprocess.py    # Post-processing rules
├── prompting.py      # Refinement prompts
├── rerank.py         # Candidate reranking
└── scoring.py        # Translation quality scoring
```

### Purpose

Refinement improves translations after the initial pass by:
1. **Glossary enforcement**: Ensure all terms are translated correctly
2. **Coherence**: Fix pronouns, maintain consistent style
3. **Reranking**: Select best from multiple candidates

### Key Components

#### `Refiner` (Abstract)
```python
class Refiner(ABC):
    @abstractmethod
    def refine(self, block, document, glossary) -> RefineResult:
        """Refine a single block."""
    
    def refine_document(self, doc, glossary) -> list[RefineResult]:
        """Refine all blocks in document."""
```

#### `GlossaryRefiner`
```python
class GlossaryRefiner(Refiner):
    """Enforces glossary terms in translated text."""
    
    def refine(self, block, document, glossary):
        # Find glossary terms in source
        # Check they're correctly translated in target
        # Fix any incorrect translations
```

#### Candidate Reranking (`rerank.py`)

```python
def rerank_candidates(source, candidates, glossary) -> RankedResult:
    """Select best translation from candidates."""
    
    # Scoring criteria:
    # - Glossary term usage
    # - Length similarity to source
    # - Fluency indicators
```

#### Scoring (`scoring.py`)

```python
def bleu(hypothesis, reference) -> float:
    """Compute BLEU score."""

def glossary_adherence(text, glossary) -> float:
    """Percentage of glossary terms correctly used."""

def length_ratio(source, target) -> float:
    """Translation length relative to source."""
```

---

## 9. Evaluation Framework (`eval/`)

### Module Structure

```
eval/
├── __init__.py
├── metrics.py      # BLEU, COMET, TER computation
├── runner.py       # Evaluation orchestration
├── ablation.py     # Ablation study framework
└── baselines.py    # Baseline comparisons
```

### Metrics (`metrics.py`)

```python
def compute_bleu(hypothesis, reference) -> float:
    """SacreBLEU score."""

def compute_comet(source, hypothesis, reference) -> float:
    """COMET neural metric (requires model)."""

def compute_ter(hypothesis, reference) -> float:
    """Translation Edit Rate."""

def glossary_adherence(source, translation, glossary) -> float:
    """Percentage of glossary terms correctly translated."""

def numeric_preservation(source, translation) -> float:
    """Check numbers are preserved."""
```

### Ablation Studies (`ablation.py`)

```python
@dataclass
class AblationConfig:
    name: str
    backends: list[str]
    enable_masking: list[bool] = [True, False]
    enable_glossary: list[bool] = [True, False]
    enable_context: list[bool] = [True, False]
    enable_refinement: list[bool] = [True, False]

class AblationStudy:
    def run(self, docs, references, sources, glossary):
        """Run all configuration combinations and measure quality."""
        
        # For each combination:
        # 1. Configure pipeline
        # 2. Translate documents
        # 3. Compute metrics
        # 4. Record results
```

**Thesis Relevance**: Ablation studies demonstrate the contribution of each component (masking, glossary, context, refinement) to overall translation quality.

---

## 10. CLI Interface (`cli.py`)

### Commands

```bash
# Translation
scitrans translate --text "Hello" --backend free
scitrans translate --input paper.pdf --output paper_fr.pdf --backend openai

# Glossary management
scitrans glossary --list
scitrans glossary --search "algorithm"
scitrans glossary --domain ml

# Evaluation
scitrans evaluate --hyp output.txt --ref reference.txt

# Ablation studies
scitrans ablation --input docs/ --refs refs/ --backends "dummy,free,openai"

# System info
scitrans info        # Show available backends
scitrans demo        # Run demo translation

# Key management
scitrans keys list
scitrans keys set openai
scitrans keys status openai

# GUI
scitrans gui --port 7860 --share
```

### Key Options

```python
@app.command()
def translate(
    input_text: str = Option(None, "--text", "-t"),
    input_file: Path = Option(None, "--input", "-i"),
    output_file: Path = Option(None, "--output", "-o"),
    source_lang: str = Option("en", "--source", "-s"),
    target_lang: str = Option("fr", "--target", "-l"),
    backend: str = Option("dummy", "--backend", "-b"),
    model: str = Option(None, "--model", "-m"),
    glossary_file: Path = Option(None, "--glossary", "-g"),
    no_glossary: bool = Option(False, "--no-glossary"),
    no_refinement: bool = Option(False, "--no-refinement"),
    no_masking: bool = Option(False, "--no-masking"),
    no_context: bool = Option(False, "--no-context"),
):
```

---

## 11. Web GUI (`gui.py`)

### Features

1. **Translation Tab**: Upload PDF, select engine, translate
2. **Debug Tab**: Analyze PDF layout before translation
3. **Pipeline Lab**: Test masking, reranking, BLEU scoring
4. **System Check**: Verify dependencies and API keys

### Architecture

```python
def launch():
    # Check dependencies
    _check_dependencies()  # Gradio, PyMuPDF, etc.
    
    with gr.Blocks() as demo:
        # Translation tab
        with gr.Tab("Translate"):
            pdf = gr.File(label="Upload PDF")
            engine = gr.Dropdown(choices=["dictionary", "free", "openai", ...])
            go = gr.Button("Translate")
            
        # Debug tab
        with gr.Tab("Debug"):
            # Layout analysis tools
            
        # Pipeline lab
        with gr.Tab("Pipeline Lab"):
            # Masking sandbox
            # Reranking sandbox
            # BLEU calculator
            
        # System check
        with gr.Tab("System Check"):
            # Dependency status
    
    demo.launch()
```

---

## 12. Key Management (`keys.py`)

### Priority Order

1. **Environment variables** (best for CI/production)
2. **OS keychain** (secure local storage via `keyring`)
3. **Config file** (`~/.scitrans/keys.json`)

### Supported Services

```python
SERVICES = {
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepl": "DEEPL_API_KEY",
    "google": "GOOGLE_API_KEY",
    "comet": "COMET_API_KEY",
    "huggingface": "HUGGINGFACE_API_KEY",
    "ollama": "OLLAMA_HOST",
}
```

### Usage

```python
from scitrans_llms.keys import KeyManager, get_key, require_key

# Get key (returns None if not found)
key = get_key("openai")

# Get key or raise error
key = require_key("openai")

# Full management
km = KeyManager()
km.set_key("openai", "sk-...")
km.get_key_info("openai")  # Returns KeyInfo with status, source, masked value
km.export_to_env()  # Dict of env vars to export
```

### Environment Variable Persistence

```bash
# Add to ~/.bashrc or ~/.zshrc for persistence:
export OPENAI_API_KEY='sk-...'
export DEEPSEEK_API_KEY='sk-...'

# Or use the export command:
scitrans keys export  # Shows export commands for all stored keys
```

---

## 13. Thesis Contributions Mapping

### Contribution #1: Terminology-Constrained Translation

**Components**:
- `masking.py`: Protects formulas, code, URLs
- `translate/glossary.py`: Glossary management
- `refine/`: Glossary enforcement

**Evidence**:
- Ablation: Compare with/without masking
- Metric: `glossary_adherence`

### Contribution #2: Document-Level Context

**Components**:
- `translate/context.py`: Context window management
- `pipeline.py`: Context passed to translator
- `models.py`: Segment structure

**Evidence**:
- Ablation: Compare with/without context
- Metric: Coherence evaluation

### Contribution #3: Pluggable Architecture

**Components**:
- `pipeline.py`: PipelineConfig with feature toggles
- `translate/base.py`: Abstract Translator interface
- `eval/ablation.py`: Ablation study framework

**Evidence**:
- Easy addition of new backends
- Systematic ablation studies

---

## Quick Reference: File-to-Function Map

| File | Key Functions |
|------|---------------|
| `cli.py` | `translate()`, `info()`, `keys()`, `gui()`, `demo()` |
| `pipeline.py` | `TranslationPipeline.translate()`, `translate_text()` |
| `models.py` | `Document.from_text()`, `Block.is_translatable` |
| `masking.py` | `mask_document()`, `unmask_document()` |
| `translate/base.py` | `create_translator()`, `Translator.translate()` |
| `translate/glossary.py` | `get_default_glossary()`, `Glossary.get_target()` |
| `ingest/pdf.py` | `parse_pdf()`, `PDFParser.parse()` |
| `refine/base.py` | `create_refiner()`, `Refiner.refine_document()` |
| `eval/metrics.py` | `compute_bleu()`, `glossary_adherence()` |
| `keys.py` | `get_key()`, `require_key()`, `KeyManager` |
| `gui.py` | `launch()` |

---

## Debugging Tips

### Check Available Backends
```bash
scitrans info
```

### Test Translation Pipeline
```bash
scitrans translate --text "Machine learning is amazing" --backend dummy
scitrans translate --text "Machine learning is amazing" --backend dictionary
scitrans translate --text "Machine learning is amazing" --backend free
```

### Verify Masking
```python
from scitrans_llms.masking import mask_text, MaskRegistry, MaskConfig

text = "The formula $E = mc^2$ is famous."
registry = MaskRegistry()
masked = mask_text(text, registry, MaskConfig())
print(masked)  # "The formula <<MATH_000>> is famous."
print(registry.mappings)  # {'<<MATH_000>>': '$E = mc^2$'}
```

### Test Glossary Application
```python
from scitrans_llms.translate.glossary import get_default_glossary

glossary = get_default_glossary()
print(glossary.get_target("machine learning"))  # "apprentissage automatique"
```

---

**Remember**: Delete this file after mastering the system. It's for your understanding, not for the repository.

