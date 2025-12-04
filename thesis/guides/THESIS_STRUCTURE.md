# SciTrans-LLMs Thesis Structure Guide

## 1. Abstract (200-300 words)
**Template:**
```
This thesis presents SciTrans-LLMs, a novel system for scientific document translation between English and French using Large Language Models. The research addresses the challenge of [specific problem]. 

Our approach combines [key techniques] to achieve [main results]. Experiments on [dataset size] scientific papers demonstrate [performance metrics], outperforming baseline methods by [percentage].

Key contributions include: (1) [contribution 1], (2) [contribution 2], (3) [contribution 3].

Results show that [main finding], with implications for [field impact].
```

## 2. Literature Review Structure

### 2.1 Machine Translation Evolution
- Statistical Machine Translation (SMT)
- Neural Machine Translation (NMT) 
- Transformer-based models
- LLM-based translation

### 2.2 Scientific Document Translation
- Domain-specific challenges
- Technical terminology handling
- LaTeX/formula preservation
- Previous systems comparison

### 2.3 Evaluation Metrics
- BLEU score
- chrF score  
- METEOR
- Human evaluation

## 3. System Development

### 3.1 Architecture Components
```
Input → PDF Parser → Block Classifier → Masking → Translation → Unmasking → PDF Renderer
```

### 3.2 Key Algorithms
- Masking algorithm for LaTeX/code
- Glossary-aware translation
- Context window optimization
- Prompt engineering

### 3.3 Implementation Details
- Python 3.9+
- NiceGUI for interface
- PyMuPDF for PDF handling
- Multiple translation backends

## 4. Experiments

### 4.1 Dataset
- Source: arXiv papers (CS, Physics, Math)
- Size: 100 papers per domain
- Split: 70% train, 15% validation, 15% test

### 4.2 Baseline Comparisons
- Google Translate API
- DeepL
- OpusMT
- mBART

### 4.3 Ablation Studies
- With/without masking
- With/without glossary
- Context window sizes
- Backend comparisons

## 5. Results & Analysis

### 5.1 Quantitative Results
| Method | BLEU | chrF | Time(s/page) |
|--------|------|------|--------------|
| Baseline | 32.5 | 58.2 | 2.1 |
| SciTrans | 41.3 | 67.8 | 3.4 |

### 5.2 Qualitative Analysis
- LaTeX preservation rate
- Technical term accuracy
- Readability assessment

## 6. Conclusion
- Summary of achievements
- Limitations
- Future work directions
