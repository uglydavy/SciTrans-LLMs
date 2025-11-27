# Thesis Materials for SciTrans-LLMs

This directory contains LaTeX templates and generated outputs for your thesis.

## Files

| File | Description |
|------|-------------|
| `chapter_experiments.tex` | Complete experiments chapter template |
| `chapter_method.tex` | Method chapter template (coming soon) |

## Generated Outputs

After running experiments, thesis-ready outputs will be in `../results/thesis/`:

```
results/thesis/
├── results_table.tex       # Main results table
├── ablation_table.tex      # Ablation study table
├── contribution_table.tex  # Component contribution analysis
└── stats.json              # Summary statistics
```

## Usage

### 1. Run Experiments

```bash
python scripts/full_pipeline.py --backend openai
```

### 2. Copy to Your Thesis

```bash
# Copy generated tables
cp results/thesis/*.tex /path/to/your/thesis/tables/

# Or include directly
\input{../scitrans_llms/results/thesis/results_table}
```

### 3. Include Chapter

In your main thesis file:

```latex
\documentclass{report}  % or your thesis class

% Required packages
\usepackage{booktabs}
\usepackage{graphicx}

\begin{document}

% ... other chapters ...

\include{chapter_experiments}

\end{document}
```

## Customization

### Modify Tables

The generated tables use standard LaTeX with `booktabs`. To customize:

1. Edit the generated `.tex` files directly, or
2. Modify `scitrans_llms/experiments/thesis.py` to change formatting

### Add Figures

To include generated figures:

```latex
\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{results/thesis/results_bar.pdf}
    \caption{Translation quality by configuration.}
    \label{fig:results-bar}
\end{figure}
```

### Statistical Significance

For significance testing, add to your preamble:

```latex
\usepackage{siunitx}  % For proper number formatting
```

Then format p-values:

```latex
The difference is significant ($p < \num{0.01}$).
```

## Citation

Remember to cite the relevant papers:

```latex
% BLEU
@inproceedings{papineni2002bleu,
  title={BLEU: a method for automatic evaluation of machine translation},
  author={Papineni, Kishore and others},
  booktitle={ACL},
  year={2002}
}

% chrF++
@inproceedings{popovic2017chrf,
  title={chrF++: words helping character n-grams},
  author={Popovi{\'c}, Maja},
  booktitle={WMT},
  year={2017}
}
```

## Thesis Tips

1. **Be specific about numbers**: "improves by 2.3 BLEU" not "improves significantly"

2. **Explain the metrics**: Don't assume readers know what BLEU measures

3. **Include examples**: Show actual translation examples, not just numbers

4. **Discuss failures**: Honest error analysis strengthens your thesis

5. **Connect to contributions**: Always link results back to your thesis claims

