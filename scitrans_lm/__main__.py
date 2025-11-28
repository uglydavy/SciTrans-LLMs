"""
Entry point for running SciTrans-LLMs as a module with short name.

Usage:
    python -m scitrans_lm --help
    python -m scitrans_lm translate --text "Hello" --backend free
    python -m scitrans_lm info
    python -m scitrans_lm gui

This is an alias for `python -m scitrans_llms`.
"""
from scitrans_llms.cli import app


if __name__ == "__main__":
    app()

