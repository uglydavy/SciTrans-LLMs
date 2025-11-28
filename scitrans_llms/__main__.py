"""
Entry point for running SciTrans-LLMs as a module.

Usage:
    python -m scitrans_llms --help
    python -m scitrans_llms translate --text "Hello" --backend free
    python -m scitrans_llms info
    python -m scitrans_llms gui
"""
from .cli import app


if __name__ == "__main__":
    app()
