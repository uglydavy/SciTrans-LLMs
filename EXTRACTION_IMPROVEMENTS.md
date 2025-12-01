# Document Extraction & Masking Improvements

## Overview

This document describes the improvements made to the document extraction and masking system to handle inconsistent formatting, varied numbering patterns, and better preserve document structure during translation.

## Key Improvements

### 1. Enhanced Numbering Pattern Detection

The system now robustly detects various numbering formats commonly found in academic documents:

**Supported Patterns:**
- Roman numerals (uppercase): `I. Introduction`, `II. Background`
- Roman numerals (lowercase): `i. first`, `ii. second`
- Hierarchical numbering: `1.1 Section`, `3.5 Results`, `1.2.3 Subsection`
- Simple numbered: `1. Chapter`, `2) Item`
- Lettered lists: `a) item`, `B. section`
- Parenthesized: `(1) item`, `(a) item`
- Bracketed: `[1] reference`, `[a] item`

**Implementation:**
- Enhanced regex patterns in `scitrans_llms/ingest/pdf.py` (`_looks_like_list_item`, `_looks_like_numbered_section`)
- Structural marker extraction in `_extract_structural_markers`

### 2. Structure Preservation in Masking

The masking system now preserves structural elements during translation:

**Preserved Elements:**
- Section numbers (e.g., `1.1` → `<<SECNUM_001>>`)
- Bullet characters (e.g., `•` → `<<BULLET_000>>`)
- Indentation levels
- Leading whitespace

**Benefits:**
- Maintains document structure after translation
- Preserves numbering schemes
- Retains visual hierarchy

**Configuration:**
```python
config = MaskConfig(
    preserve_section_numbers=True,
    preserve_bullets=True,
    preserve_indentation=True,
)
```

### 3. MinerU Integration

Added support for MinerU (magic-pdf) as an alternative extraction backend:

**Advantages:**
- Better handling of complex layouts (multi-column, tables)
- Improved mathematical formula extraction
- Better reading order detection
- Automatic structure preservation

**Usage:**
```python
from scitrans_llms.ingest.pdf import parse_pdf

# Use MinerU if available
doc = parse_pdf("document.pdf", use_mineru=True)
```

**Installation:**
```bash
pip install magic-pdf
```

Falls back to PyMuPDF if MinerU is not available.

### 4. Enhanced YOLO Layout Detection

Implemented full YOLO-based layout detection with page rendering:

**Features:**
- Renders PDF pages to images at high resolution
- Runs YOLO detection to identify layout regions
- Maps detection labels to block types
- Extracts text from detected regions in reading order

**Location:** `scitrans_llms/ingest/yolo_detection.py`

**Usage:**
```python
from scitrans_llms.ingest.yolo_detection import detect_layout_yolo

detections = detect_layout_yolo("document.pdf", page_num=0)
```

### 5. Metadata for Structure Preservation

Each block now includes metadata for structure preservation:

**Metadata Fields:**
- `section_number`: The extracted section number (e.g., "1.1", "I")
- `numbering_style`: Type of numbering (roman_upper, numeric, etc.)
- `has_numbering`: Boolean flag
- `bullet_char`: The bullet character used
- `has_bullet`: Boolean flag
- `indent_level`: Indentation depth
- `indent_chars`: Raw indentation string

**Example:**
```python
{
    'section_number': '1.1',
    'numbering_style': 'numeric_hierarchical',
    'has_numbering': True,
    'indent_level': 0
}
```

## Files Modified

1. **`scitrans_llms/ingest/pdf.py`**
   - Enhanced `_looks_like_list_item()` with more patterns
   - Added `_looks_like_numbered_section()` method
   - Added `_extract_structural_markers()` method
   - Integrated MinerU with `MinerUPDFParser` class
   - Updated `parse_pdf()` to support MinerU

2. **`scitrans_llms/masking.py`**
   - Added `SECTION_NUMBER_PATTERN` regex
   - Added `BULLET_PATTERN` regex
   - Updated `MaskConfig` with structure preservation options
   - Enhanced `mask_block()` to preserve indentation
   - Enhanced `unmask_block()` to restore indentation

3. **`scitrans_llms/ingest/yolo_detection.py`** (NEW)
   - Full YOLO layout detection implementation
   - Page-to-image rendering
   - Detection to block type mapping
   - YOLOLayoutParser class

4. **`requirements.txt`**
   - Added `magic-pdf>=0.6.0` for MinerU support

## Testing

Run the test suite to verify improvements:

```bash
python3 test_improved_extraction.py
```

**Test Coverage:**
- Numbering pattern detection
- Structure preservation in masking
- MinerU availability check
- End-to-end document processing

## Examples

### Before
```
Input:  "1.1 Introduction with $x^2$ formula"
Masked: "1.1 Introduction with <<MATH_000>> formula"
        ↑ Section number not preserved
```

### After
```
Input:  "1.1 Introduction with $x^2$ formula"
Masked: "<<SECNUM_000>>Introduction with <<MATH_000>> formula"
        ↑ Section number preserved as placeholder
```

### Translation Flow
```
1. Source:     "1.1 Background"
2. Masked:     "<<SECNUM_000>>Background"
3. Translate:  "<<SECNUM_000>>Contexte"
4. Unmask:     "1.1 Contexte"
```

## Configuration Examples

### Preserve Everything
```python
config = MaskConfig(
    preserve_section_numbers=True,
    preserve_bullets=True,
    preserve_indentation=True,
    mask_equations=True,
    mask_code=True,
)
```

### Minimal Masking (Only Formulas)
```python
config = MaskConfig(
    preserve_section_numbers=False,
    preserve_bullets=False,
    mask_equations=True,
    mask_code=False,
)
```

## Known Limitations

1. **MinerU Dependency**: Requires `magic-pdf` package which has additional dependencies
2. **YOLO Model**: Requires trained weights for layout detection
3. **Reading Order**: Complex multi-column layouts may still have reading order issues
4. **Language-Specific Numbers**: Non-Arabic numerals (e.g., Chinese, Arabic) not yet supported

## Future Improvements

1. Support for more exotic numbering systems
2. Better handling of nested lists with mixed styles
3. Preservation of text alignment (left/center/right)
4. Support for RTL (right-to-left) languages
5. Integration with DocLayout-YOLO pre-trained models

## Migration Guide

Existing code will continue to work without changes. To use new features:

1. **Enable structure preservation:**
   ```python
   config = MaskConfig(preserve_section_numbers=True)
   ```

2. **Use MinerU:**
   ```python
   doc = parse_pdf("file.pdf", use_mineru=True)
   ```

3. **Access structural metadata:**
   ```python
   for block in doc.all_blocks:
       if 'section_number' in block.metadata:
           print(f"Section: {block.metadata['section_number']}")
   ```

## References

- [MinerU/magic-pdf](https://github.com/opendatalab/MinerU)
- [DocLayout-YOLO](https://github.com/ultralytics/ultralytics)
- [PyMuPDF](https://pymupdf.readthedocs.io/)

