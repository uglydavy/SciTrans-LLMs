# Running Experiments for Your Thesis

This guide walks you through all the steps needed to run experiments and generate thesis-ready results.

## Quick Start

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Quick test to verify everything works
python3 scripts/quick_test.py

# 3. Run experiments (with dummy translator for now)
python3 scripts/run_experiments.py --quick
```

## Step-by-Step Guide

### Step 1: Configure API Keys

For real LLM translation, you need API keys:

```bash
# Interactive setup
python3 scripts/setup_keys.py

# Or set individual keys
python3 scripts/setup_keys.py --service openai
python3 scripts/setup_keys.py --service deepseek

# Check configured keys
python3 scripts/setup_keys.py --list
```

**Environment Variables** (alternative):
```bash
export OPENAI_API_KEY="sk-..."
export DEEPSEEK_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-..."
```

### Step 2: Prepare Your Corpus

The `corpus/` directory should contain parallel data:

```
corpus/
├── source/
│   └── abstracts/
│       ├── paper_001.txt    # English source
│       ├── paper_002.txt
│       └── ...
├── reference/
│   └── abstracts/
│       ├── paper_001.txt    # French reference translation
│       ├── paper_002.txt
│       └── ...
└── glossary/
    └── domain.csv           # Optional: custom glossary
```

**Collecting Data:**

1. **Academic Papers**: Find papers published in both EN and FR
2. **WMT Data**: Download from [statmt.org/wmt24](https://www.statmt.org/wmt24/)
3. **OPUS Corpora**: Scientific domain from [opus.nlpl.eu](https://opus.nlpl.eu/)

**Minimum Recommended:**
- 20-50 documents for development
- 100+ documents for final evaluation

### Step 3: Run Experiments

#### A. Quick Test (Dummy Translator)
```bash
python3 scripts/run_experiments.py --quick
```

#### B. Full Ablation Study
```bash
python3 scripts/run_experiments.py --backend dummy
```

#### C. With Real LLM Backend
```bash
# OpenAI GPT-4
python3 scripts/run_experiments.py --backend openai

# DeepSeek
python3 scripts/run_experiments.py --backend deepseek

# Compare multiple backends
python3 scripts/run_experiments.py --compare-backends dummy dictionary openai
```

#### D. Custom Corpus
```bash
python3 scripts/run_experiments.py --corpus /path/to/your/corpus
```

### Step 4: Review Results

Results are saved to `results/`:

```
results/
├── ablation/
│   ├── ablation_YYYYMMDD_HHMMSS.json    # Raw results
│   ├── ablation_YYYYMMDD_HHMMSS.tex     # LaTeX table
│   └── ablation_YYYYMMDD_HHMMSS.txt     # Summary
├── backends/
│   └── ...
└── thesis/
    ├── results_table.tex       # Main results table
    ├── ablation_table.tex      # Ablation study table
    └── contribution.tex        # Component contribution analysis
```

### Step 5: Use in Thesis

Copy the LaTeX tables directly into your thesis:

```latex
\input{results/thesis/results_table.tex}
\input{results/thesis/ablation_table.tex}
\input{results/thesis/contribution.tex}
```

## Experiment Types

### 1. Ablation Study

Tests contribution of each component:

| Configuration | Description |
|---------------|-------------|
| `full` | All features enabled |
| `no_glossary` | Without glossary enforcement |
| `no_context` | Without document-level context |
| `no_refinement` | Without refinement pass |
| `no_masking` | Without placeholder masking |
| `minimal` | No features (baseline) |

### 2. Backend Comparison

Compare different translation engines:

```bash
python3 scripts/run_experiments.py \
    --compare-backends dummy dictionary openai deepseek
```

### 3. Baseline Comparison

Compare with external systems:

```python
from scitrans_llms.eval.baselines import (
    NaiveLLMBaseline,
    GoogleTranslateBaseline,
    PDFMathTranslateBaseline,
    BaselineComparison,
)

comparison = BaselineComparison()
comparison.add_baseline(NaiveLLMBaseline())
comparison.add_baseline(GoogleTranslateBaseline())
results = comparison.run_comparison(docs, refs, sources, our_outputs)
print(comparison.to_latex_table())
```

## Python API

### Run Custom Experiments

```python
from scitrans_llms.experiments import (
    load_corpus,
    ExperimentRunner,
    ExperimentConfig,
)
from scitrans_llms.experiments.thesis import ThesisExporter

# Load corpus
corpus = load_corpus("corpus/")

# Configure experiment
config = ExperimentConfig(
    name="custom_experiment",
    backend="openai",
    enable_glossary=True,
    enable_context=True,
    enable_refinement=True,
)

# Run
runner = ExperimentRunner(corpus, output_dir="results/custom")
result = runner.run_experiment(config)

# Export
exporter = ThesisExporter([result])
exporter.export_all("thesis_figures/")
```

### Evaluate Specific Outputs

```python
from scitrans_llms.eval import run_evaluation

report = run_evaluation(
    hypotheses=["Ma traduction..."],
    references=["La référence..."],
    sources=["My translation..."],
)
print(report.summary())
```

## Tips for Your Thesis

### 1. Statistical Significance

Run multiple times and report variance:

```python
config = AblationConfig(
    name="significance_test",
    num_runs=5,  # Run each configuration 5 times
)
```

### 2. Error Analysis

Look at segment-level results:

```python
for seg in report.segment_results:
    if seg.glossary_adherence < 0.8:
        print(f"Low glossary: {seg.source[:50]}...")
```

### 3. Qualitative Examples

Find interesting examples for discussion:

```python
# Find where glossary helped most
for i, (src, hyp) in enumerate(zip(sources, hypotheses)):
    if "machine learning" in src and "apprentissage automatique" in hyp:
        print(f"Good example at {i}")
```

## Troubleshooting

### API Rate Limits

```python
# Slow down for rate limits
config = PipelineConfig(
    translator_kwargs={"timeout": 120.0},
)
```

### Memory Issues

Process in batches:

```python
for batch in corpus.documents[::10]:
    results = runner.run_experiment(config, docs=[batch])
```

### Missing Dependencies

```bash
pip install "scitrans_llms[full]"
```

## Estimated Time

| Corpus Size | Dummy | OpenAI | DeepSeek |
|-------------|-------|--------|----------|
| 10 docs | <1 min | ~5 min | ~5 min |
| 50 docs | ~2 min | ~20 min | ~20 min |
| 200 docs | ~5 min | ~1 hour | ~1 hour |

---

Good luck with your thesis! 🎓

