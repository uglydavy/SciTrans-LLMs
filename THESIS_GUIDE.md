# Complete Thesis Experiment Guide

This guide walks you through running all experiments needed for your thesis.

## Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    THESIS PIPELINE                          │
├─────────────────────────────────────────────────────────────┤
│  1. Collect Corpus  ─→  2. Setup API Keys  ─→  3. Run      │
│                                                 Experiments │
│                                                     ↓       │
│  4. Review Results  ←─  5. Export Thesis  ←─  6. Include   │
│                                               in LaTeX      │
└─────────────────────────────────────────────────────────────┘
```

## Step 1: Collect Your Corpus

### Option A: Use Sample Data (Quick Testing)
```bash
python3 scripts/collect_corpus.py --source sample --target 50
```

### Option B: Download Real Data (Recommended)
```bash
# Interactive mode - choose sources
python3 scripts/collect_corpus.py --target 100

# OPUS Scientific articles
python3 scripts/collect_corpus.py --source opus_scielo --target 100

# Filter for scientific content only
python3 scripts/collect_corpus.py --source opus_emea --scientific-only --target 100
```

### Option C: Use Your Own Data

Place files in this structure:
```
corpus/
├── source/abstracts/
│   ├── doc_001.txt    # English text
│   ├── doc_002.txt
│   └── ...
└── reference/abstracts/
    ├── doc_001.txt    # French translation (same filename!)
    ├── doc_002.txt
    └── ...
```

**Recommended corpus size:**
- Development/testing: 20-50 documents
- Thesis experiments: 100-200 documents
- Publication quality: 500+ documents

## Step 2: Configure API Keys

### Interactive Setup
```bash
python3 scripts/setup_keys.py
```

### Manual Setup
```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Or DeepSeek (cheaper alternative)
export DEEPSEEK_API_KEY="sk-..."

# Check configuration
python3 scripts/setup_keys.py --list
```

### API Key Sources
| Service | Get Key From | Approx Cost |
|---------|--------------|-------------|
| OpenAI | platform.openai.com | $0.01-0.03/1K tokens |
| DeepSeek | platform.deepseek.com | $0.001-0.002/1K tokens |
| Anthropic | console.anthropic.com | $0.01-0.03/1K tokens |

## Step 3: Run Experiments

### Quick Test (No API Needed)
```bash
python3 scripts/quick_test.py
```

### Full Thesis Pipeline
```bash
# With sample data + dictionary translator (free, offline-friendly)
python3 scripts/full_pipeline.py

# With OpenAI (recommended for thesis)
python3 scripts/full_pipeline.py --backend openai

# With DeepSeek (cheaper alternative)
python3 scripts/full_pipeline.py --backend deepseek
```

### Custom Experiments
```bash
# Only ablation study
python3 scripts/full_pipeline.py --skip-baselines

# Custom corpus path
python3 scripts/full_pipeline.py --corpus /path/to/your/corpus

# Compare multiple backends
python3 scripts/run_experiments.py --compare-backends dictionary openai deepseek
```

## Step 4: Review Results

Results are saved to `results/`:

```
results/
├── main/
│   ├── main_*.json     # Raw experiment data
│   └── main_*.tex      # LaTeX tables
├── ablation/
│   ├── ablation_*.json
│   └── ablation_*.tex
└── thesis/
    ├── results_table.tex      # Main results
    ├── ablation_table.tex     # Ablation study
    ├── contribution.tex       # Component analysis
    └── stats.json             # Summary statistics
```

### View Summary
```bash
cat results/ablation/ablation_*.txt
```

## Step 5: Include in Your Thesis

### Copy Tables
```bash
# Copy to your thesis directory
cp results/thesis/*.tex /path/to/thesis/tables/
```

### Include in LaTeX
```latex
\documentclass{report}
\usepackage{booktabs}

\begin{document}

\chapter{Experiments}

Table~\ref{tab:main-results} presents the main results.

\input{tables/results_table}

Table~\ref{tab:ablation} shows the ablation study.

\input{tables/ablation_table}

Table~\ref{tab:contribution} quantifies each component's contribution.

\input{tables/contribution}

\end{document}
```

### Use the Chapter Template
```latex
\include{chapter_experiments}  % From thesis/chapter_experiments.tex
```

## Complete Commands Reference

```bash
# ═══════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════

# Install dependencies
source .venv/bin/activate
pip install -e .

# Verify installation
python3 scripts/quick_test.py

# ═══════════════════════════════════════════════════════════
# CORPUS
# ═══════════════════════════════════════════════════════════

# Create sample corpus
python3 scripts/collect_corpus.py --source sample --target 50

# Download from OPUS
python3 scripts/collect_corpus.py --source opus_scielo --target 100

# ═══════════════════════════════════════════════════════════
# API KEYS
# ═══════════════════════════════════════════════════════════

# Interactive setup
python3 scripts/setup_keys.py

# Set specific key
python3 scripts/setup_keys.py --service openai

# List configured keys
python3 scripts/setup_keys.py --list

# ═══════════════════════════════════════════════════════════
# EXPERIMENTS
# ═══════════════════════════════════════════════════════════

# Full pipeline (dummy backend, for testing)
python3 scripts/full_pipeline.py

# Full pipeline with OpenAI
python3 scripts/full_pipeline.py --backend openai

# Only main experiments (skip ablation)
python3 scripts/full_pipeline.py --skip-ablation

# Custom experiment
python3 scripts/run_experiments.py --backend openai --corpus my_corpus/

# ═══════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════

# View ablation results
cat results/ablation/ablation_*.txt

# View thesis tables
cat results/thesis/*.tex
```

## Troubleshooting

### "API key not found"
```bash
python3 scripts/setup_keys.py --service openai
# Or set environment variable:
export OPENAI_API_KEY="sk-..."
```

### "Corpus not found"
```bash
python3 scripts/collect_corpus.py --source sample --target 30
```

### "Low BLEU scores"
- BLEU with dummy translator is expected to be low (~5-10)
- Use real LLM backend (openai/deepseek) for meaningful scores
- Make sure reference translations are high quality

### "Rate limit exceeded"
- OpenAI has rate limits; wait or use DeepSeek
- Reduce corpus size for testing

### "Missing dependency"
```bash
pip install -e .
# Or specific package:
pip install openai sacrebleu
```

## Expected Timeline

| Task | Time (approx) |
|------|---------------|
| Setup & quick test | 10 minutes |
| Corpus collection (sample) | 1 minute |
| Corpus collection (real) | 10-30 minutes |
| Experiments (dummy, 50 docs) | 2 minutes |
| Experiments (OpenAI, 50 docs) | 15-30 minutes |
| Experiments (OpenAI, 200 docs) | 1-2 hours |

## Cost Estimation

For 100 documents with OpenAI GPT-4o:
- ~50,000 tokens input
- ~50,000 tokens output
- **Cost: ~$2-5 USD**

With DeepSeek:
- Same token count
- **Cost: ~$0.20-0.50 USD**

## Good Luck with Your Thesis! 🎓

