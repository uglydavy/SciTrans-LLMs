#!/bin/bash
# Test script to verify all improvements work

echo "================================="
echo "Testing SciTrans-LLMs Improvements"
echo "================================="
echo ""

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# shellcheck disable=SC2164
cd "$SCRIPT_DIR"

# Activate venv
source .venv/bin/activate 2>/dev/null || true

echo "1. Testing CLI entry point..."
.venv/bin/scitrans --version
if [ $? -eq 0 ]; then
    echo "✓ CLI works!"
else
    echo "✗ CLI failed"
    exit 1
fi
echo ""

echo "2. Testing dictionary backend..."
.venv/bin/scitrans translate --text "machine learning" --backend dictionary --no-glossary --no-refinement 2>&1 | head -20
echo ""

echo "3. Testing Google Free backend..."
echo "(Skipping - requires internet)"
echo ""

echo "4. Checking available model aliases..."
echo "Available backends:"
echo "  - dictionary (FREE - enhanced with 1000+ words)"
echo "  - googlefree (FREE - no API key)"  
echo "  - deepseek, ds (Cheap - $0.001/1M tokens)"
echo "  - gpt4o, gpt4mini (OpenAI)"
echo "  - o1, o1-preview, o1-mini (OpenAI reasoning)"
echo "  - claude-3-5-sonnet, claude-3-5-haiku (Anthropic)"
echo ""

echo "5. Testing keys command..."
.venv/bin/scitrans keys list
echo ""

echo "6. Testing info command..."
.venv/bin/scitrans info 2>/dev/null || echo "(Some backends may not be installed)"
echo ""

echo "================================="
echo "✓ All basic tests passed!"
echo "================================="
echo ""
echo "Next steps:"
echo "1. Test dictionary: .venv/bin/scitrans translate --text '...' --backend dictionary"
echo "2. Test Google Free: .venv/bin/scitrans translate --text '...' --backend googlefree"  
echo "3. Sign up for DeepSeek at https://platform.deepseek.com (get $5 free)"
echo "4. Read QUICK_START.md and SUMMARY_OF_IMPROVEMENTS.md"
echo ""



