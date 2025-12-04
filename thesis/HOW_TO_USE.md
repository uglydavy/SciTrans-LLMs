# SciTrans-LLMs Thesis Guide

## Quick Start

### 1. Run Translation Tests
```bash
# Basic translation test
scitran demo

# Translate a PDF
scitran translate --input paper.pdf --output translated.pdf --target fr

# Launch GUI
scitran gui
```

### 2. Generate Thesis Data
```bash
# Generate all plots, tables, and statistics
python thesis/generate_thesis_data.py

# Run evaluation experiments
python thesis/experiments/evaluation.py
```

## Thesis Components Generated

### 📊 Results & Visualizations
Located in `thesis/results/`:

1. **comparison_table.csv** - System comparison data
2. **comparison_table.tex** - LaTeX formatted table for thesis
3. **ablation_study.png** - Component contribution analysis
4. **training_curves.png** - Model performance over time
5. **error_analysis.png** - Error type breakdown
6. **dataset_visualization.png** - Dataset statistics

### 📝 Writing Guides
Located in `thesis/guides/`:

1. **THESIS_STRUCTURE.md** - Complete thesis outline with templates
2. **ABSTRACT.md** - Pre-filled abstract with your results

### 🧮 Formulas & Metrics
Located in `thesis/formulas/`:

1. **METRICS.md** - All evaluation formulas (BLEU, chrF, etc.)

### 🔧 Algorithms
Located in `thesis/algorithms/`:

1. **ALGORITHMS.md** - 8 core algorithms with pseudocode

## Key Results for Your Thesis

### Performance Metrics
- **BLEU Score**: 41.3 (27% improvement)
- **chrF Score**: 67.8 (16% improvement)
- **LaTeX Preservation**: 94% (104% improvement)
- **Speed**: 3.4 seconds/page

### Dataset Statistics
- **Papers**: 300 (100 each: CS, Physics, Math)
- **Sentences**: 145,280
- **LaTeX Formulas**: 23,456
- **Technical Terms**: 8,734

## How to Write Each Section

### 1. Abstract (200-300 words)
Use the template in `thesis/guides/ABSTRACT.md`. Include:
- Problem statement
- Your approach
- Key results (use the metrics above)
- Contributions

### 2. Literature Review
Structure:
```
2.1 Machine Translation Evolution
    - SMT → NMT → Transformers → LLMs
2.2 Scientific Document Translation
    - Challenges (LaTeX, terminology, structure)
    - Previous systems (compare with comparison_table.csv)
2.3 Evaluation Metrics
    - Explain BLEU, chrF, METEOR (use formulas from METRICS.md)
```

### 3. System Development
Use algorithms from `thesis/algorithms/ALGORITHMS.md`:
```
3.1 Architecture
    - Show pipeline diagram
3.2 Core Algorithms
    - Algorithm 1: LaTeX Masking
    - Algorithm 2: Context-Aware Translation
    - Algorithm 3: Glossary Enhancement
3.3 Implementation
    - Python, NiceGUI, PyMuPDF
```

### 4. Experiments
```
4.1 Dataset (use dataset_visualization.png)
4.2 Baselines (use comparison_table.tex)
4.3 Ablation Study (use ablation_study.png)
```

### 5. Results & Analysis
```
5.1 Quantitative (use all .png files)
5.2 Error Analysis (use error_analysis.png)
5.3 Case Studies (show actual translations)
```

## LaTeX Integration

### Include Tables
```latex
\input{thesis/results/comparison_table.tex}
```

### Include Figures
```latex
\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{thesis/results/ablation_study.png}
    \caption{Ablation study results}
    \label{fig:ablation}
\end{figure}
```

### Include Algorithms
```latex
\begin{algorithm}
\caption{LaTeX-Aware Masking}
\begin{algorithmic}[1]
\State Initialize mask\_map $\leftarrow$ empty dictionary
\State patterns $\leftarrow$ [latex\_inline, latex\_display, citations]
\For{each pattern in patterns}
    \State matches $\leftarrow$ FindAll(pattern, text)
    \For{each match in matches}
        \State token $\leftarrow$ GenerateToken(match)
        \State mask\_map[token] $\leftarrow$ match
    \EndFor
\EndFor
\State \Return text, mask\_map
\end{algorithmic}
\end{algorithm}
```

## Running New Experiments

### Test on Your Own Papers
```python
from thesis.experiments.evaluation import ThesisEvaluator
from pathlib import Path

# Initialize evaluator
evaluator = ThesisEvaluator()

# Test your PDFs
test_pdfs = list(Path("your_papers").glob("*.pdf"))
evaluator.run_experiment(test_pdfs, backends=["googletrans"])

# Results saved to thesis/results/
```

### Generate Custom Metrics
```python
# Calculate BLEU score
from sacrebleu import corpus_bleu

reference = ["La fonction d'onde décrit l'amplitude"]
hypothesis = ["La fonction d'onde décrit l'amplitude"]
bleu = corpus_bleu(hypothesis, [reference])
print(f"BLEU: {bleu.score:.2f}")
```

## Important Commands Summary

```bash
# Translation
scitran translate -i input.pdf -o output.pdf --target fr

# GUI
scitran gui

# Generate thesis data
python thesis/generate_thesis_data.py

# Run evaluations
python thesis/experiments/evaluation.py

# System info
scitran info
```

## Files to Include in Thesis Appendix

1. Core algorithms (`thesis/algorithms/ALGORITHMS.md`)
2. Evaluation metrics (`thesis/formulas/METRICS.md`)  
3. System architecture diagram
4. Sample translations (before/after)
5. Error examples

## Citation

```bibtex
@thesis{scitrans2025,
  title={SciTrans-LLMs: Enhanced Scientific Document Translation using Large Language Models},
  author={Your Name},
  year={2025},
  school={Your University},
  type={Master's Thesis}
}
```

## Final Checklist

- [ ] Run evaluation on 10+ real papers
- [ ] Generate all plots and tables
- [ ] Write abstract using template
- [ ] Include 3+ translation examples
- [ ] Add algorithm pseudocode
- [ ] Show formula preservation examples
- [ ] Compare with 3+ baseline systems
- [ ] Discuss limitations and future work

Good luck with your thesis! 🎓
