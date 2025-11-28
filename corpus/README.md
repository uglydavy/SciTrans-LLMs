# Test Corpus for SciTrans-LLMs

This directory contains the test corpus for evaluating the translation system.

## Directory Structure

```
corpus/
├── source/           # Source documents (English)
│   ├── abstracts/    # Paper abstracts
│   ├── paragraphs/   # Individual paragraphs
│   └── full/         # Full papers (PDF)
├── reference/        # Reference translations (French)
│   ├── abstracts/
│   ├── paragraphs/
│   └── full/
├── glossary/         # Domain-specific glossaries
│   ├── ml.csv        # Machine learning terms
│   ├── physics.csv   # Physics terms
│   └── general.csv   # General scientific terms
└── metadata/         # Document metadata
    └── corpus_info.json
```

## Data Format

### Text Files (abstracts/, paragraphs/)
- One segment per file
- UTF-8 encoding
- Matching filenames between source/ and reference/

Example:
```
source/abstracts/paper001.txt
reference/abstracts/paper001.txt
```

### Glossary Format (CSV)
```csv
source,target,domain,notes
machine learning,apprentissage automatique,ml,
neural network,réseau de neurones,ml,
```

## Creating the Corpus

### Option 1: From Parallel Corpora
Download from publicly available sources:
- [OPUS](https://opus.nlpl.eu/) - Parallel corpora
- [WMT News](https://www.statmt.org/wmt24/) - News translation data
- [SciTail](https://allenai.org/data/scitail) - Scientific text

### Option 2: Manual Collection
1. Find bilingual papers (author-provided translations)
2. Extract abstracts from EN/FR versions
3. Align at paragraph level

### Option 3: Synthetic (for Development)
Use Google Translate or DeepL as pseudo-references for initial testing.

## Usage

```python
from scitrans_llms.experiments.corpus import load_corpus

corpus = load_corpus("corpus/")
for doc in corpus.documents:
    print(f"{doc.source_file}: {len(doc.segments)} segments")
```

