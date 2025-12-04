#!/bin/bash
# Setup script for SciTrans-LLMs

echo "====================================="
echo "Setting up SciTrans-LLMs Environment"
echo "====================================="

# Install core dependencies
echo "Installing dependencies..."
pip3 install --upgrade pip
pip3 install --upgrade "pydantic<2.0"
pip3 install --upgrade "fastapi<0.100"
pip3 install --upgrade "nicegui>=1.4.0"
pip3 install --upgrade "typer[all]"
pip3 install --upgrade "rich"
pip3 install --upgrade "PyMuPDF"
pip3 install --upgrade "googletrans==4.0.0rc1"
pip3 install --upgrade "requests"
pip3 install --upgrade "pandas"
pip3 install --upgrade "matplotlib"
pip3 install --upgrade "numpy<2"

# Install package in editable mode
echo "Installing SciTrans-LLMs..."
pip3 install -e .

# Make scitran command available
echo "Setting up scitran command..."
chmod +x ~/.local/bin/scitran 2>/dev/null || true

echo ""
echo "✅ Setup complete!"
echo ""
echo "You can now run:"
echo "  scitran --help        # Show commands"
echo "  scitran demo          # Run demo"
echo "  scitran translate     # Translate PDF"
echo "  scitran gui           # Launch GUI"
echo ""
