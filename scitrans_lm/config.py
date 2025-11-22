from pathlib import Path

APP_NAME = "SciTrans-LM"
DATA_DIR = Path(__file__).resolve().parent / "data"
LAYOUT_DIR = DATA_DIR / "layout"
GLOSSARY_DIR = DATA_DIR / "glossary"
CACHE_DIR = DATA_DIR / "cache"

LAYOUT_MODEL = LAYOUT_DIR / "layout_model.pt"
DEFAULT_GLOSSARY = GLOSSARY_DIR / "default_en_fr.csv"

# Ensure directories exist at import time
for d in (DATA_DIR, LAYOUT_DIR, GLOSSARY_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)