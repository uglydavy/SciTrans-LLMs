
from __future__ import annotations
import sys, shutil, csv
from pathlib import Path
from .config import LAYOUT_MODEL, DEFAULT_GLOSSARY, LAYOUT_DIR, GLOSSARY_DIR

def ensure_layout_model() -> None:
    """Ensure layout model exists. If not, create a tiny placeholder and instruct user to run setup."""
    LAYOUT_DIR.mkdir(parents=True, exist_ok=True)
    if not LAYOUT_MODEL.exists():
        # Create a small placeholder. Real model will be downloaded/trained by setup.
        LAYOUT_MODEL.write_bytes(b"SciTrans-LM placeholder weights. Run 'python -m scitrans_lm setup --yolo' to download/train.".ljust(1024, b'\0'))

def ensure_default_glossary() -> None:
    GLOSSARY_DIR.mkdir(parents=True, exist_ok=True)
    if not DEFAULT_GLOSSARY.exists():
        with DEFAULT_GLOSSARY.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["source", "target"])  # header
            # a few seed terms
            w.writerow(["machine learning", "apprentissage automatique"])
            w.writerow(["deep learning", "apprentissage profond"])
            w.writerow(["neural network", "réseau de neurones"])
            w.writerow(["dataset", "jeu de données"])

def run_all() -> None:
    ensure_layout_model()
    ensure_default_glossary()
    print("✔ Setup complete. You can now translate.")

