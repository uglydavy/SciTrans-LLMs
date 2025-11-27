"""
Project-wide configuration and directory structure.

This module defines all the paths and directories used throughout the SciTrans-LLMs system.
It ensures that necessary directories exist at import time and provides consistent
access to data files, models, and caches.

Module Contents:
    APP_NAME: Application name for display purposes
    DATA_DIR: Main data directory containing all project data
    LAYOUT_DIR: Directory for YOLO layout detection models
    GLOSSARY_DIR: Directory for terminology glossaries
    CACHE_DIR: Directory for temporary files and caches
    LAYOUT_MODEL: Path to pre-trained YOLO model
    DEFAULT_GLOSSARY: Path to default EN-FR glossary

All directories are created automatically when the module is imported.

Example:
    >>> from scitrans_llms.config import DEFAULT_GLOSSARY, LAYOUT_MODEL
    >>> print(f"Glossary at: {DEFAULT_GLOSSARY}")
    >>> print(f"Model at: {LAYOUT_MODEL}")
"""

from pathlib import Path

# Application name for display and identification
APP_NAME = "SciTrans-LM"

# Main data directory (contains models, glossaries, cache)
DATA_DIR = Path(__file__).resolve().parent / "data"

# Layout detection models directory
LAYOUT_DIR = DATA_DIR / "layout"

# Terminology glossaries directory
GLOSSARY_DIR = DATA_DIR / "glossary"

# Cache directory for temporary files
CACHE_DIR = DATA_DIR / "cache"

# Pre-trained YOLO model for layout detection
LAYOUT_MODEL = LAYOUT_DIR / "layout_model.pt"

# Default English-French glossary
DEFAULT_GLOSSARY = GLOSSARY_DIR / "default_en_fr.csv"

# Ensure all directories exist at import time
for d in (DATA_DIR, LAYOUT_DIR, GLOSSARY_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)
