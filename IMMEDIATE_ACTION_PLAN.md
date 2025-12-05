# SciTrans-LLMs: Immediate Action Plan

Based on comprehensive codebase analysis, here's what to do RIGHT NOW to prepare for thesis submission and publication.

## 🚨 TODAY (Critical - Do First)

### 1. Consolidate Masking Modules (30 min)
```bash
# Check which files use the old mask.py
grep -r "from scitran_llms.mask import" . --include="*.py"
grep -r "import scitran_llms.mask" . --include="*.py"

# If any found, update to use masking.py instead
# Then remove the duplicate:
git rm scitrans_llms/mask.py
git commit -m "refactor: Consolidate masking modules, remove duplicate mask.py"
```

### 2. Remove Redundant Files (10 min)
```bash
# Remove generic/incomplete documentation
git rm DOCUMENTATION.txt
git rm EXTRACTION_IMPROVEMENTS.md
git rm QUICK_START_IMPROVEMENTS.md
git rm test_all_improvements.sh
git rm scitrans_llms/cli.py.bak

# Remove user-specific directories
git rm -r .nicegui/ --cached

# Commit
git commit -m "cleanup: Remove redundant and incomplete documentation files"
```

### 3. Quick Test to Ensure Nothing Broke (10 min)
```bash
# Run existing tests
python -m pytest tests/ -v

# Quick smoke test
python scripts/quick_test.py
```

## 📅 THIS WEEK (High Priority)

### Day 1-2: Fix GUI Layout
- [ ] Open `scitran_llms/gui.py`
- [ ] Add responsive CSS with max-heights
- [ ] Ensure panels fit in 1920x1080 without scrolling
- [ ] Add `object-fit: contain` to PDF previews
- [ ] Test on different screen sizes

### Day 3: Add Missing Tests
- [ ] Create `tests/test_pdf_rendering.py`
- [ ] Create `tests/test_integration.py`
- [ ] Test with 5 diverse PDFs (single-column, two-column, with tables)
- [ ] Ensure 80%+ code coverage

### Day 4: Expand Glossary
- [ ] Add 300+ scientific terms to `translate/glossary.py`
- [ ] Categorize by domain (CS, Physics, Math)
- [ ] Add glossary validation test

### Day 5: Run Full Experiments
```bash
# Setup corpus if not done
python scripts/collect_corpus.py --source sample --target 100

# Run ablation study
python scripts/full_pipeline.py --backend openai

# Generate thesis tables
python thesis/generate_thesis_data.py
```

## 📊 METRICS TO TRACK

Before improvements:
- BLEU: 41.3
- LaTeX Preservation: 94%
- Speed: 3.4 s/page
- Test Coverage: ~20%

Target after improvements:
- BLEU: 43.0+
- LaTeX Preservation: 96%+
- Speed: 2.5 s/page
- Test Coverage: 80%+

## ✅ VALIDATION CHECKLIST

After each change, ensure:
- [ ] `python scripts/quick_test.py` passes
- [ ] `python -m pytest tests/` passes
- [ ] GUI still launches: `python run_gui.py`
- [ ] Translation works: `scitrans translate --input attention.pdf`

## 🎯 SUCCESS CRITERIA FOR THESIS

Your system is thesis-ready when:
1. ✅ BLEU score > 40 (ALREADY MET: 41.3)
2. ✅ LaTeX preservation > 90% (ALREADY MET: 94%)
3. ✅ Beats all baselines (ALREADY MET)
4. ⚠️ GUI works without scrolling (NEEDS FIX)
5. ⚠️ Test coverage > 70% (NEEDS WORK)
6. ⚠️ Clean git history (NEEDS CLEANUP)

## 💡 QUICK WINS

These can be done in < 5 minutes each:
1. Add `.gitignore` entry for `.nicegui/`
2. Update README.md to mention the three innovations clearly
3. Add badges to README (Python version, license, etc.)
4. Create `CITATION.cff` file for proper citations

## 🚀 COMMAND TO START

```bash
# Start with the easiest win - remove redundant files
cd /Users/kv.kn/Desktop/Research/SciTrans-LLMs
git rm DOCUMENTATION.txt EXTRACTION_IMPROVEMENTS.md QUICK_START_IMPROVEMENTS.md
git rm scitran_llms/cli.py.bak
git commit -m "cleanup: Remove redundant files and documentation"

# Then consolidate masking
# ... (follow steps above)

# You're already on the path to success!
```

## 📝 REMEMBER

Your system WORKS and BEATS BASELINES. The innovations are implemented. You just need to:
1. Clean up technical debt
2. Improve test coverage
3. Fix GUI layout
4. Document properly

You're 85% done - just need the final polish! 🎓
