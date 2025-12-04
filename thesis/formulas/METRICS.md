# Translation Quality Metrics - Formulas

## 1. BLEU Score (Bilingual Evaluation Understudy)

### Formula:
```
BLEU = BP · exp(∑(n=1 to N) wn · log pn)
```

Where:
- **BP** = Brevity Penalty = min(1, e^(1-r/c))
  - r = reference length
  - c = candidate length
- **pn** = n-gram precision
- **wn** = weight (typically 1/N)
- **N** = maximum n-gram order (usually 4)

### N-gram Precision:
```
pn = (∑_C∈Candidates ∑_ngram∈C Count_clip(ngram)) / (∑_C∈Candidates ∑_ngram∈C Count(ngram))
```

### Python Implementation:
```python
import numpy as np
from collections import Counter

def calculate_bleu(reference, hypothesis, max_n=4):
    # Calculate n-gram precisions
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = get_ngrams(reference, n)
        hyp_ngrams = get_ngrams(hypothesis, n)
        
        ref_counts = Counter(ref_ngrams)
        hyp_counts = Counter(hyp_ngrams)
        
        clipped_counts = sum(min(hyp_counts[ng], ref_counts[ng]) 
                            for ng in hyp_counts)
        total_counts = sum(hyp_counts.values())
        
        precision = clipped_counts / total_counts if total_counts > 0 else 0
        precisions.append(precision)
    
    # Calculate brevity penalty
    r_len = len(reference.split())
    c_len = len(hypothesis.split())
    bp = min(1, np.exp(1 - r_len / c_len)) if c_len > 0 else 0
    
    # Calculate BLEU
    weights = [1/max_n] * max_n
    bleu = bp * np.exp(sum(w * np.log(p + 1e-10) 
                          for w, p in zip(weights, precisions)))
    return bleu * 100
```

## 2. chrF Score (Character F-score)

### Formula:
```
chrF = (1 + β²) · (chrP · chrR) / (β² · chrP + chrR)
```

Where:
- **chrP** = Character-level precision
- **chrR** = Character-level recall  
- **β** = Weight factor (typically 2 for chrF2)

### Precision and Recall:
```
chrP = |matched_chars| / |hypothesis_chars|
chrR = |matched_chars| / |reference_chars|
```

## 3. METEOR Score

### Formula:
```
METEOR = Fmean · (1 - Penalty)
```

Where:
```
Fmean = (10 · P · R) / (R + 9 · P)
Penalty = 0.5 · (chunks / matches)³
```

## 4. Success Rate

### Formula:
```
Success_Rate = (Successfully_Translated_Blocks / Total_Blocks) × 100%
```

## 5. LaTeX Preservation Rate

### Formula:
```
LaTeX_Preservation = (Preserved_LaTeX_Elements / Total_LaTeX_Elements) × 100%
```

Where LaTeX elements include:
- Inline math: `$...$`
- Display math: `$$...$$`
- Equations: `\begin{equation}...\end{equation}`
- Commands: `\cite{}`, `\ref{}`, etc.

## 6. Translation Speed

### Formula:
```
Speed = Total_Words / Translation_Time (words/second)
Efficiency = (Output_Quality × Speed) / Resource_Usage
```

## 7. Perplexity (for LLM-based translation)

### Formula:
```
PPL = exp(-1/N · ∑(i=1 to N) log P(wi|w1...wi-1))
```

Where:
- N = number of tokens
- P(wi|context) = probability of token wi given context

## 8. Word Error Rate (WER)

### Formula:
```
WER = (S + D + I) / N × 100%
```

Where:
- S = Substitutions
- D = Deletions  
- I = Insertions
- N = Number of words in reference

## 9. Semantic Similarity Score

### Formula (using cosine similarity):
```
Similarity = (A · B) / (||A|| × ||B||)
```

Where A and B are sentence embeddings from models like BERT or Sentence-BERT.

## 10. Domain-Specific Term Accuracy

### Formula:
```
Term_Accuracy = (Correctly_Translated_Terms / Total_Technical_Terms) × 100%
```

## Usage in Thesis

For comprehensive evaluation, combine multiple metrics:

```python
def comprehensive_score(bleu, chrf, meteor, latex_pres, weights=None):
    """
    Combined score for thesis evaluation
    """
    if weights is None:
        weights = [0.3, 0.3, 0.2, 0.2]  # Default weights
    
    scores = [bleu/100, chrf/100, meteor/100, latex_pres/100]
    combined = sum(w * s for w, s in zip(weights, scores))
    return combined * 100
```

## Statistical Significance Testing

### Paired Bootstrap Resampling:
```python
def bootstrap_significance(scores_system1, scores_system2, n_samples=1000):
    """
    Test if system1 is significantly better than system2
    """
    differences = []
    n = len(scores_system1)
    
    for _ in range(n_samples):
        indices = np.random.choice(n, n, replace=True)
        sample_diff = np.mean([scores_system1[i] - scores_system2[i] 
                               for i in indices])
        differences.append(sample_diff)
    
    p_value = sum(d <= 0 for d in differences) / n_samples
    return p_value < 0.05  # Significant if True
```
