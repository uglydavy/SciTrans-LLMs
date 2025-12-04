"""
Configuration for SciTrans-LLMs.
"""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / ".cache"
MODELS_DIR = DATA_DIR / "models"

# GUI Settings
GUI_HOST = "0.0.0.0"
GUI_PORT = 7860
GUI_TITLE = "SciTrans-LLMs"
GUI_RELOAD = bool(os.getenv("DEBUG", False))

# Translation Backends (simplified)
BACKENDS = {
    "googletrans": "Google Translate (free, unofficial)",
    "dictionary": "Dictionary-based translation (offline)",
    "mymemory": "MyMemory API (free, limited requests/day)",
}

# Default backend - best free option
DEFAULT_BACKEND = "googletrans"

# Languages - Only English and French needed
LANGUAGES = {
    "en": "English",
    "fr": "French",
}

# API Settings
API_TIMEOUT = 30
MAX_RETRIES = 3
CHUNK_SIZE = 5000

# Masking patterns
MASK_PATTERNS = {
    "latex": r"\$\$?[^$]+\$\$?",
    "code": r"```[^`]*```",
    "url": r"https?://[^\s]+",
    "cite": r"\\cite\{[^}]+\}",
    "ref": r"\\ref\{[^}]+\}",
}

# Model explanations
MODEL_DESCRIPTIONS = {
    "dictionary": "Fast offline translation using dictionaries. Good for technical terms but limited fluency.",
    "mymemory": "Free online translation API. Good balance of quality and speed. No API key needed.",
    "googletrans": "Unofficial Google Translate API. High quality, free, but may have rate limits.",
    "openai": "GPT-4 based translation. Excellent quality, context-aware. Requires API key ($).",
    "deepseek": "DeepSeek AI model. Good quality, competitive pricing. Requires API key ($).",
    "anthropic": "Claude 3 based translation. Very high quality, safety-focused. Requires API key ($).",
    "local": "Local LLM via Ollama or HuggingFace. Free but requires local GPU/compute.",
}
