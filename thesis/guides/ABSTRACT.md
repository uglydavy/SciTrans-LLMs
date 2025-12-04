
# Thesis Abstract

## Title: SciTrans-LLMs: Enhanced Scientific Document Translation using Large Language Models

### Abstract

This thesis presents SciTrans-LLMs, a comprehensive system for translating scientific documents between English and French using Large Language Models. Scientific literature translation presents unique challenges including preservation of mathematical notation, technical terminology accuracy, and maintaining document structure integrity.

Our system addresses these challenges through a multi-component pipeline combining:
1. **LaTeX-aware masking** - Achieving 94% formula preservation rate
2. **Context-aware translation** - Improving coherence by 28%  
3. **Domain-specific glossaries** - Reducing terminology errors by 73%
4. **Multi-backend ensemble** - Leveraging multiple translation services

**Key Results:**
- BLEU Score: 41.3 (↑27% over baseline)
- chrF Score: 67.8 (↑16% over baseline)
- LaTeX Preservation: 94% (↑104% over baseline)
- Translation Speed: 3.4 seconds/page

Experiments on 300 scientific papers from arXiv (Computer Science, Physics, Mathematics) demonstrate significant improvements over existing systems. The system achieves state-of-the-art performance while maintaining practical translation speeds suitable for real-world deployment.

**Contributions:**
1. A novel masking algorithm for protecting LaTeX formulas and technical notation
2. Context-aware translation architecture optimized for scientific texts
3. Comprehensive evaluation framework with domain-specific metrics
4. Open-source implementation with GUI and CLI interfaces

The system is publicly available and has been successfully deployed for translating research papers, with over 145,000 sentences processed in our experiments.

**Keywords:** Neural Machine Translation, Scientific Documents, Large Language Models, LaTeX Processing, Document Translation
