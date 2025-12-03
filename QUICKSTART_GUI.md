# Quick Start: New Gradio GUI

## ✅ Fixed and Ready to Test!

### Issue Resolved
The initial `HfFolder` import error was caused by incompatible versions:
- **Problem**: Gradio 4.44 incompatible with huggingface_hub 1.x (removed `HfFolder`)
- **Solution**: Downgraded huggingface_hub to 0.22.2
- **Adaptation**: Used `gr.File()` instead of `gr.PDF()` (not available in Gradio 4.44)

### Current Setup
- ✅ Gradio 4.44.1
- ✅ huggingface_hub 0.22.2  
- ✅ All imports working
- ✅ GUI builds successfully

## Launch Instructions

```bash
cd /Users/kv.kn/Desktop/Research/SciTrans-LLMs

# Activate your virtual environment
source .venv/bin/activate

# Launch GUI
scitrans gui

# OR directly with Python
python -m scitrans_llms.gui_gradio
```

The GUI will open automatically in your browser at **http://localhost:7860**

## What to Test

### 1. File Upload (Drag & Drop)
- [ ] Drag a PDF onto the upload area
- [ ] Click to browse and select a PDF
- [ ] Verify file info appears (name, size)
- [ ] Check that uploaded file shows in Source column

### 2. URL Fetching
- [ ] Switch to "URL" tab
- [ ] Enter an arXiv URL (e.g., https://arxiv.org/pdf/1706.03762.pdf)
- [ ] Click "Fetch PDF"
- [ ] Watch progress bar
- [ ] Verify PDF downloads and appears

### 3. Translation
- [ ] Select direction (EN→FR or FR→EN)
- [ ] Choose engine (try "free" first)
- [ ] Keep default settings or adjust
- [ ] Click "Translate Document"
- [ ] Watch progress bar and log messages
- [ ] Check translated PDF appears in right column
- [ ] Click translated PDF to download

### 4. Advanced Features
- [ ] Open "Advanced Options" accordion
- [ ] Enable/disable masking
- [ ] Adjust quality passes
- [ ] Upload a custom glossary (CSV)
- [ ] Test with different engines

## Expected Behavior

### Good Signs ✅
- GUI loads in <5 seconds
- Upload works immediately (drag-drop or click)
- Progress bars show during translation
- Status logs appear at bottom
- Translated PDF downloads correctly

### Known Limitations (Gradio 4.44)
- ⚠️ PDFs show as downloadable files, not inline preview
  - This is temporary - Gradio 5.0 will have `gr.PDF()` component
  - For now, click the file to download and view
- ⚠️ No built-in PDF viewer in the GUI
  - Your browser will handle PDF viewing when you click the file

## Troubleshooting

### "Gradio not installed" Error
```bash
pip install 'gradio>=4.44.0' 'huggingface_hub<0.23.0'
```

### "Cannot import HfFolder" Error
```bash
pip install 'huggingface_hub==0.22.2'
```

### GUI Won't Launch
1. Check you're in the right directory
2. Activate virtual environment: `source .venv/bin/activate`
3. Try: `python -m scitrans_llms.gui_gradio`
4. Check port 7860 is not already in use

### Translation Fails
- Verify the backend is available (`scitrans info`)
- Check API keys if using paid backends (`scitrans keys list`)
- Try "free" backend first (no setup required)

## Comparison to Old NiceGUI

| Feature | Old (NiceGUI) | New (Gradio) |
|---------|---------------|--------------|
| PDF Preview | Base64 PNG (slow) | File download (fast) |
| Upload | Custom handler (buggy) | Native drag-drop ✅ |
| URL Fetch | Sync (blocks UI) | Async with progress ✅ |
| Progress | Manual updates | Built-in streaming ✅ |
| Layout | Fixed, CSS fragile | Responsive, native ✅ |
| Deployment | Complex | Simple (HF Spaces ready) ✅ |

## What's Next

After confirming the GUI works:
1. **Week 1 Completion**: Test all features, fix bugs
2. **Week 2**: Add translation caching (50-90% speedup!)
3. **Week 3**: Improve masking (font-based detection)
4. **Week 4**: Final polish and thesis prep

## Need Help?

If you encounter issues:
1. Check the terminal for error messages
2. Look at the status box in the GUI (bottom panel)
3. Try the legacy GUI: `scitrans gui --legacy`
4. Check logs: The terminal shows detailed progress

---

**Status**: Ready to test! 🚀  
**Commit**: c40deec  
**Pushed**: Yes ✅
