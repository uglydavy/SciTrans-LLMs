# SciTrans-LLMs: Cleanup & Publication Preparation Plan

**Date**: 2025-12-02  
**Purpose**: Remove AI-generated artifacts, consolidate modules, improve documentation, and prepare for research publication

---

## Table of Contents
1. [Files to Remove](#files-to-remove)
2. [Modules to Consolidate](#modules-to-consolidate)
3. [Git History Cleanup](#git-history-cleanup)
4. [Documentation Updates](#documentation-updates)
5. [GUI Fixes](#gui-fixes)
6. [Testing Requirements](#testing-requirements)
7. [Execution Checklist](#execution-checklist)

---

## 1. Files to Remove

### Unfinished/Generic Documentation
```bash
# Remove generic improvement docs (convert useful parts to issues first)
rm DOCUMENTATION.txt
rm EXTRACTION_IMPROVEMENTS.md
rm QUICK_START_IMPROVEMENTS.md

# Remove test shell script (replace with proper test suite)
rm test_all_improvements.sh

# Remove NiceGUI storage (user-specific, shouldn't be in repo)
rm -rf .nicegui/
```

### Auto-Generated Files
```bash
# MODULE_REFERENCE.md appears auto-generated - review and either:
# Option A: Remove if redundant with WARP.md
rm MODULE_REFERENCE.md

# Option B: Keep but regenerate with better formatting
# (Decision needed after review)
```

### Legacy Code
```bash
# Remove legacy masking module after consolidation
# (See Section 2 for consolidation steps first)
# rm scitrans_llms/mask.py
```

---

## 2. Modules to Consolidate

### 2.1 Masking System Consolidation

**Problem**: Two masking modules (`masking.py` and `mask.py`) create confusion

**Solution**: Keep `masking.py`, remove `mask.py`

**Steps**:
```bash
# 1. Search for imports of mask.py
grep -r "from scitrans_llms.mask import" scitrans_llms/
grep -r "from scitrans_llms import mask" scitrans_llms/
grep -r "import scitrans_llms.mask" scitrans_llms/

# 2. Replace all imports with masking.py equivalents
# Example:
# OLD: from scitrans_llms.mask import mask_protected_segments
# NEW: from scitrans_llms.masking import mask_text, MaskRegistry

# 3. After confirming no references, remove
rm scitrans_llms/mask.py

# 4. Run tests to ensure nothing breaks
pytest tests/ -v
```

### 2.2 YOLO Integration Cleanup

**Problem**: YOLO integration is incomplete and falls back to heuristics

**Options**:
A. **Complete integration**: Wire YOLOLayoutDetector properly
B. **Document limitations**: Clearly state YOLO is experimental
C. **Remove YOLO code**: Use only heuristics

**Recommended**: Option B (document) for now, complete in future work

**Steps**:
```bash
# Add clear comments to yolo/ modules
# Update INSTALL.md to explain YOLO weights are optional
# Update pdf.py to log when falling back to heuristics
```

---

## 3. Git History Cleanup

### 3.1 Identify AI-Generated Commits

**Problem**: 82+ commits with generic messages like "Enhance...", "Improve...", "Refactor..."

**Current History (last 20)**:
```
7b1100b Improve reranking, GUI translate tab, and dictionary backend; remove dummy backend from UX
78a00c1 Enhance PDF parsing capabilities...
9030765 Refactor PDF rendering and upload handling...
3a78f92 Enhance SciTrans-LLMs GUI with improved PDF preview...
[... 78 more similar commits]
```

**Solution**: Squash related commits into meaningful milestones

### 3.2 Commit Squashing Strategy

**DO NOT** rewrite history that's already pushed to remote unless coordinating with team!

**If working alone**:
```bash
# WARNING: This rewrites history. Coordinate with collaborators!

# 1. Backup current branch
git branch backup-before-squash

# 2. Interactive rebase to first AI commit (find via git log)
git log --oneline --all | grep -n "first meaningful commit"
# Say first meaningful commit is at commit abc123

git rebase -i abc123

# 3. In editor, mark commits to squash:
# pick abc123 Initial commit
# squash def456 Enhance...
# squash ghi789 Improve...
# squash jkl012 Refactor...
# pick mno345 Add evaluation system
# ...

# 4. Rewrite commit messages to be descriptive:
# "Add GUI with PDF translation, preview, and settings tabs
#  
#  - Implement NiceGUI-based interface
#  - Support multiple translation backends
#  - Add progress tracking and logging
#  - Enable candidate reranking controls"

# 5. Force push (ONLY if coordinating with team)
# git push --force-with-lease origin main
```

**If working with others OR uncomfortable with rebase**:
Create a clean branch:
```bash
# 1. Create new clean branch from first meaningful commit
git checkout -b clean-main abc123  # abc123 = first good commit

# 2. Cherry-pick important feature commits
git cherry-pick -n def456  # -n = no commit yet
git cherry-pick -n ghi789
git commit -m "Meaningful commit message"

# 3. Repeat for logical groups

# 4. When satisfied, rename branches
git branch -m main old-main
git branch -m clean-main main
```

### 3.3 Recommended Commit Structure

Organize commits by feature/contribution:

```
1. Initial project structure and core models
2. Add translation pipeline with masking and glossary
3. Implement multi-backend support (Dictionary, LLM, Free)
4. Add PDF ingestion with layout detection
5. Add PDF rendering with coordinate preservation
6. Implement candidate reranking system
7. Add CLI with translate/evaluate/ablation commands
8. Add GUI with multi-tab interface
9. Add evaluation and experiment frameworks
10. Add documentation and user guides
```

---

## 4. Documentation Updates

### 4.1 README Improvements

**Add Missing Sections**:
```markdown
## Scripts

The `scripts/` directory contains tools for common workflows:

- `quick_test.py`: Smoke test to verify installation
  ```bash
  python scripts/quick_test.py
  ```

- `run_experiments.py`: Run translation experiments with multiple configs
  ```bash
  python scripts/run_experiments.py --backend dictionary --corpus corpus/
  ```

- `full_pipeline.py`: Run complete thesis experiment pipeline
  ```bash
  python scripts/full_pipeline.py --output results/
  ```

- `collect_corpus.py`: Download parallel corpora for training
  ```bash
  python scripts/collect_corpus.py --corpus europarl --langs en-fr
  ```

- `setup_keys.py`: Interactive API key configuration
  ```bash
  python scripts/setup_keys.py
  ```

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

For PDF-specific tests:
```bash
pytest tests/test_pdf.py -v  # (to be created)
```
```

### 4.2 CONTRIBUTING.md Updates

Add section on commit messages:
```markdown
## Commit Message Guidelines

Write clear, descriptive commit messages:

**Good**:
- "Fix placeholder restoration in mathematical equations"
- "Add multi-candidate reranking with heuristic fallback"
- "Improve PDF extraction for two-column layouts"

**Avoid**:
- "Improve system"
- "Enhance functionality"
- "Refactor code"
- "Update files"

Use imperative mood ("Add", "Fix", "Improve") and explain *what* and *why*.
```

### 4.3 Remove/Update Module-Specific Docs

```bash
# If MODULE_REFERENCE.md is truly auto-generated and redundant:
# Keep WARP.md (more comprehensive) and remove MODULE_REFERENCE.md

# Or regenerate MODULE_REFERENCE.md properly:
python -c "import scitrans_llms; help(scitrans_llms)" > MODULE_REFERENCE.md
```

---

## 5. GUI Fixes

### 5.1 Preview Image Fitting

**Problem**: PDF previews overflow their containers

**Fix** (in `gui.py`):
```python
# In get_preview() function, ensure object-fit:contain in CSS:
return f'''<img src="data:image/png;base64,{b64}" 
    style="max-width:100%;
           max-height:100%;
           object-fit:contain;
           display:block;
           margin:auto;
           width:auto;
           height:auto;"/>'''
```

### 5.2 Panel Height Constraints

**Problem**: Panels overflow vertically

**Fix** (in CSS section):
```css
.main-row { 
  height: calc(100vh - 110px) !important;  /* Fixed height */
  overflow: hidden;  /* Prevent parent scroll */
}

.panel {
  flex: 1;
  max-width: 50%;
  height: 100%;
  overflow-y: auto;  /* Scroll inside panel only */
  display: flex;
  flex-direction: column;
}

.preview-area {
  flex: 1;
  min-height: 400px;
  max-height: calc(100vh - 400px);  /* Leave room for controls */
  overflow: auto;
  background: #1a1a2e;
  border-radius: 4px;
}
```

### 5.3 Session Persistence

**Add** (in `gui.py` after `State` class):
```python
def save_session(state):
    """Save session to browser localStorage"""
    session_data = {
        'uploaded_pdf_path': state.uploaded_pdf_path,
        'uploaded_pdf_name': state.uploaded_pdf_name,
        'default_engine': state.default_engine,
        'quality_passes': state.quality_passes,
    }
    # Use ui.run_javascript to save to localStorage
    ui.run_javascript(f'''
        localStorage.setItem('scitrans_session', '{json.dumps(session_data)}');
    ''')

def restore_session(state):
    """Restore session from browser localStorage"""
    # Use ui.run_javascript to read from localStorage
    # (Implementation depends on NiceGUI async JS execution)
    pass

# Call restore_session() on page load
# Call save_session() after upload/setting changes
```

---

## 6. Testing Requirements

### 6.1 Create PDF Test Suite

```bash
# Create test data directory
mkdir -p tests/data/pdfs
mkdir -p tests/data/references

# Download diverse test PDFs
python << 'EOF'
import urllib.request
from pathlib import Path

# ArXiv papers covering different domains
test_papers = {
    'attention.pdf': 'https://arxiv.org/pdf/1706.03762.pdf',  # Transformers (ML)
    'gan.pdf': 'https://arxiv.org/pdf/1406.2661.pdf',         # GANs (ML)
    'bert.pdf': 'https://arxiv.org/pdf/1810.04805.pdf',       # BERT (NLP)
    # Add more...
}

for name, url in test_papers.items():
    dest = Path('tests/data/pdfs') / name
    if not dest.exists():
        print(f'Downloading {name}...')
        urllib.request.urlretrieve(url, dest)
EOF
```

### 6.2 Create Test Files

**`tests/test_pdf_extraction.py`**:
```python
import pytest
from pathlib import Path
from scitrans_llms.ingest import parse_pdf
from scitrans_llms.models import BlockType

TEST_PDF_DIR = Path(__file__).parent / 'data' / 'pdfs'

@pytest.mark.parametrize('pdf_name', [
    'attention.pdf',
    'gan.pdf',
    'bert.pdf',
])
def test_pdf_extraction(pdf_name):
    """Test that PDFs extract with reasonable block counts"""
    pdf_path = TEST_PDF_DIR / pdf_name
    if not pdf_path.exists():
        pytest.skip(f'{pdf_name} not found')
    
    doc = parse_pdf(pdf_path, pages=[0, 1, 2])  # First 3 pages
    
    assert len(doc.segments) > 0, "No segments extracted"
    assert len(doc.all_blocks) > 10, "Too few blocks extracted"
    
    # Check block type distribution
    types = [b.block_type for b in doc.all_blocks]
    assert BlockType.PARAGRAPH in types, "No paragraphs found"
    assert BlockType.HEADING in types, "No headings found"
```

**`tests/test_masking.py`**:
```python
from scitrans_llms.masking import mask_text, unmask_text, MaskRegistry, MaskConfig

def test_latex_masking():
    """Test LaTeX formula masking"""
    text = "The equation $E = mc^2$ is famous."
    registry = MaskRegistry()
    
    masked = mask_text(text, registry, MaskConfig())
    assert '<<MATH_' in masked
    assert '$E = mc^2$' not in masked
    
    unmasked = unmask_text(masked, registry)
    assert unmasked == text

def test_placeholder_preservation():
    """Test that all placeholders are tracked"""
    text = "See $x^2$ and $y^3$ and https://example.com"
    registry = MaskRegistry()
    
    masked = mask_text(text, registry, MaskConfig())
    
    # Count placeholders
    import re
    placeholders = re.findall(r'<<[A-Z]+_\d{3}>>', masked)
    assert len(placeholders) == 3  # 2 math + 1 URL
```

**`tests/test_translation.py`**:
```python
from scitrans_llms.pipeline import TranslationPipeline, PipelineConfig
from scitrans_llms.models import Document

def test_dictionary_translation():
    """Test dictionary backend end-to-end"""
    text = "Machine learning is a method of data analysis."
    doc = Document.from_text(text, "en", "fr")
    
    config = PipelineConfig(
        translator_backend='dictionary',
        enable_glossary=True,
    )
    pipeline = TranslationPipeline(config)
    result = pipeline.translate(doc)
    
    assert result.success
    assert result.translated_text != text
    assert 'machine' in result.translated_text.lower() or 'apprentissage' in result.translated_text.lower()
```

### 6.3 Run Tests

```bash
# Add to pyproject.toml or pytest.ini:
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_functions = "test_*"

# Run all tests
pytest tests/ -v --tb=short

# Run specific test categories
pytest tests/test_pdf_extraction.py -v
pytest tests/test_masking.py -v
pytest tests/test_translation.py -v
```

---

## 7. Execution Checklist

### Phase 1: Immediate Cleanup (Do First)
- [ ] Remove generic docs: `DOCUMENTATION.txt`, `EXTRACTION_IMPROVEMENTS.md`, `QUICK_START_IMPROVEMENTS.md`
- [ ] Remove test script: `test_all_improvements.sh`
- [ ] Remove NiceGUI storage: `.nicegui/` and add to `.gitignore`
- [ ] Add `.nicegui/` to `.gitignore`
- [ ] Review and handle `MODULE_REFERENCE.md` (remove or regenerate)

### Phase 2: Module Consolidation
- [ ] Search for all imports of `mask.py`
- [ ] Replace with `masking.py` equivalents
- [ ] Remove `scitrans_llms/mask.py`
- [ ] Run tests to verify: `pytest tests/ -v`

### Phase 3: Documentation Updates
- [ ] Add Scripts section to `README.md`
- [ ] Add Testing section to `README.md`
- [ ] Update `CONTRIBUTING.md` with commit guidelines
- [ ] Document YOLO limitations in `INSTALL.md`
- [ ] Add inline docstrings to all scripts

### Phase 4: GUI Fixes
- [ ] Fix preview image CSS (object-fit: contain)
- [ ] Fix panel height constraints
- [ ] Implement session persistence (localStorage)
- [ ] Add connection status indicator
- [ ] Test GUI on 1920x1080 and 1440x900 screens

### Phase 5: Testing
- [ ] Create `tests/data/pdfs/` directory
- [ ] Download test PDFs (5-10 diverse papers)
- [ ] Create `test_pdf_extraction.py`
- [ ] Create `test_masking.py` (expand existing)
- [ ] Create `test_translation.py`
- [ ] Run full test suite: `pytest tests/ -v`
- [ ] Aim for >80% coverage on core modules

### Phase 6: Git History (Optional, Coordinate!)
- [ ] Decide on strategy: squash vs clean branch
- [ ] Create backup: `git branch backup-before-cleanup`
- [ ] Execute chosen strategy
- [ ] Verify all features still work
- [ ] Update remote (if appropriate)

### Phase 7: Final Validation
- [ ] Run translation on 5 test PDFs with different backends
- [ ] Verify stats: blocks detected, masks applied, placeholders preserved
- [ ] Test GUI end-to-end: upload → translate → download
- [ ] Check that all docs are up-to-date
- [ ] Run `ruff check` and fix major issues
- [ ] Run `mypy scitrans_llms/` and address critical issues

---

## 8. Commands Summary

**Quick cleanup**:
```bash
cd /Users/kv.kn/dev/SciTrans-LLMs

# Remove unnecessary files
rm DOCUMENTATION.txt EXTRACTION_IMPROVEMENTS.md QUICK_START_IMPROVEMENTS.md test_all_improvements.sh
rm -rf .nicegui/

# Add to .gitignore
echo ".nicegui/" >> .gitignore

# Stage changes
git add -A
git commit -m "Remove AI-generated artifacts and temp files

- Remove generic improvement docs (converted to issues)
- Remove NiceGUI user storage from repo
- Add .nicegui/ to .gitignore"
```

**Consolidate masking**:
```bash
# Check for mask.py usage
grep -r "from scitrans_llms.mask import" scitrans_llms/ || echo "No imports found"

# If no imports or after fixing them:
rm scitrans_llms/mask.py
git add scitrans_llms/mask.py
git commit -m "Remove legacy mask.py module (consolidated into masking.py)"
```

**Run validation**:
```bash
# Syntax check
python3 -m py_compile scitrans_llms/**/*.py

# Run tests
pytest tests/ -v

# Quick translation test
python3 -m scitrans_llms.cli translate --text "Hello world" --backend dictionary
```

---

## 9. Post-Cleanup Validation

After executing cleanup, verify:

1. **System still works**:
   ```bash
   scitrans demo
   scitrans info
   scitrans translate --text "Test" --backend dictionary
   ```

2. **GUI launches**:
   ```bash
   scitrans gui
   # Verify in browser: http://localhost:7860
   ```

3. **Tests pass**:
   ```bash
   pytest tests/ -v
   ```

4. **Docs are current**:
   - README explains all features
   - INSTALL is accurate
   - USER_GUIDE has examples
   - Scripts are documented

5. **No obvious AI artifacts**:
   - Commit messages are descriptive
   - Docs are human-written (not template-y)
   - Code has meaningful comments
   - No "TODO: improve this" comments

---

## 10. Future Work (Post-Cleanup)

After cleanup is complete, tackle research-critical items:

1. **Quantitative Evaluation**:
   - Run BLEU/chrF on test corpus
   - Compare vs baselines (Google Translate, PDFMathTranslate)
   - Document in EXPERIMENTS.md

2. **Ablation Studies**:
   - Run with/without masking, glossary, context, reranking
   - Generate LaTeX tables for thesis
   - Document in thesis/

3. **Error Analysis**:
   - Categorize translation errors
   - Identify patterns (math errors, terminology errors, layout errors)
   - Propose improvements

4. **User Study** (if applicable):
   - Recruit participants
   - Collect feedback
   - Analyze results

---

## Notes

- **Coordinate git history rewrites** with anyone else working on the repo
- **Test thoroughly** after each phase
- **Keep backups** before major changes (`git branch backup-YYYYMMDD`)
- **Document decisions** (why you kept/removed files)
- **Commit incrementally** (don't do everything in one massive commit)

---

## Resources

- [Git Interactive Rebase](https://git-scm.com/docs/git-rebase)
- [Python Testing Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)
- [Writing Good Commit Messages](https://chris.beams.io/posts/git-commit/)
- [Research Software Engineering Guide](https://the-turing-way.netlify.app/reproducible-research/reproducible-research.html)
