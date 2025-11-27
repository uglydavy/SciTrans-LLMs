"""
Bootstrap and initialization utilities for SciTrans-LLMs.

This module handles the automatic setup and initialization of required
resources for the translation system, including:

1. YOLO layout detection models
2. Default terminology glossaries
3. Data directory structure

Functions are designed to gracefully handle missing dependencies and
network issues, allowing the system to run with reduced functionality
when resources are unavailable.

Functions:
    ensure_layout_model: Download or verify YOLO model
    ensure_default_glossary: Create or verify glossary
    download_layout_model: Download YOLO weights from URL
    run_all: Run all setup steps

Example:
    >>> from scitrans_llms.bootstrap import run_all
    >>> run_all()  # Ensures all resources are available
    ✔ Setup complete. You can now translate.
"""

from __future__ import annotations
import csv
import os
import warnings
from pathlib import Path
from typing import Iterable, Tuple
from .config import LAYOUT_MODEL, DEFAULT_GLOSSARY, LAYOUT_DIR, GLOSSARY_DIR

# A moderately sized built-in glossary to keep offline translation usable
_BUILTIN_GLOSSARY: Tuple[Tuple[str, str], ...] = (
    ("machine learning", "apprentissage automatique"),
    ("deep learning", "apprentissage profond"),
    ("neural network", "réseau de neurones"),
    ("dataset", "jeu de données"),
    ("training set", "jeu d'apprentissage"),
    ("test set", "jeu de test"),
    ("validation set", "jeu de validation"),
    ("inference", "inférence"),
    ("probability", "probabilité"),
    ("statistical model", "modèle statistique"),
    ("loss function", "fonction de perte"),
    ("gradient", "gradient"),
    ("optimizer", "optimiseur"),
    ("learning rate", "taux d'apprentissage"),
    ("epoch", "itération complète"),
    ("batch size", "taille de lot"),
    ("regularization", "régularisation"),
    ("dropout", "abandon aléatoire"),
    ("activation function", "fonction d'activation"),
    ("convolution", "convolution"),
    ("embedding", "encodage vectoriel"),
    ("sequence", "séquence"),
    ("token", "jeton"),
    ("transformer", "transformeur"),
    ("attention", "mécanisme d'attention"),
    ("cross entropy", "entropie croisée"),
    ("precision", "précision"),
    ("recall", "rappel"),
    ("f1 score", "score f1"),
    ("accuracy", "exactitude"),
    ("baseline", "référence"),
    ("hyperparameter", "hyperparamètre"),
    ("normalization", "normalisation"),
    ("standard deviation", "écart type"),
    ("variance", "variance"),
    ("statistical significance", "signification statistique"),
    ("p value", "valeur p"),
    ("confidence interval", "intervalle de confiance"),
    ("random variable", "variable aléatoire"),
    ("probability distribution", "distribution de probabilité"),
    ("gaussian", "gaussien"),
    ("bayesian", "bayésien"),
    ("posterior", "postérieure"),
    ("likelihood", "vraisemblance"),
    ("prior", "a priori"),
    ("markov chain", "chaîne de markov"),
    ("monte carlo", "monte-carlo"),
    ("stochastic", "stochastique"),
    ("deterministic", "déterministe"),
    ("experimental setup", "dispositif expérimental"),
    ("reproducibility", "reproductibilité"),
    ("open source", "code ouvert"),
    ("throughput", "débit"),
    ("latency", "latence"),
    ("performance", "performance"),
)

DEFAULT_LAYOUT_URL = "https://huggingface.co/airaria/DocLayout-YOLO/resolve/main/weights/doclaynet_yolov8_base.pt"

_PLACEHOLDER_BYTES = b"SciTrans-LM placeholder weights. Run 'python3 -m scitrans_lm setup --yolo' to download/train.".ljust(
    1024, b"\0"
)


def _write_glossary(rows: Iterable[Tuple[str, str]]) -> None:
    with DEFAULT_GLOSSARY.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target"])
        for src, tgt in rows:
            writer.writerow([src, tgt])
def _is_placeholder(path: Path) -> bool:
    try:
        return not path.exists() or path.stat().st_size <= len(_PLACEHOLDER_BYTES)
    except OSError:
        return True


def download_layout_model(url: str | None = None, timeout: int = 120) -> bool:
    """Attempt to download a YOLO layout model.

    The URL can be overridden with the ``SCITRANSLM_LAYOUT_URL`` environment
    variable or the ``url`` argument. The function returns ``True`` if a file
    was written successfully, and ``False`` on any error without raising so the
    caller can fall back to a placeholder.
    """

    target = LAYOUT_MODEL
    target.parent.mkdir(parents=True, exist_ok=True)
    url = os.environ.get("SCITRANSLM_LAYOUT_URL", url or DEFAULT_LAYOUT_URL)
    if not url:
        return False
    try:
        import requests
    except Exception:
        warnings.warn("Requests not available; cannot download YOLO weights automatically.")
        return False

    try:
        with requests.get(url, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            with target.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as exc:
        warnings.warn(
            f"Failed to download layout model from {url}: {exc}. "
            "Download manually and place the file at data/layout/layout_model.pt."
        )
        return False


def ensure_layout_model(prefer_download: bool = True, model_url: str | None = None) -> None:
    """Ensure a usable layout model exists.

    Attempts to download a DocLayout-YOLO checkpoint when missing or when a
    placeholder is detected. If download fails, a placeholder is kept so the
    rest of the pipeline can still run without detection.
    """

    LAYOUT_DIR.mkdir(parents=True, exist_ok=True)
    if not _is_placeholder(LAYOUT_MODEL):
        return
    if prefer_download and download_layout_model(model_url):
        return
    if not LAYOUT_MODEL.exists():
        LAYOUT_MODEL.write_bytes(_PLACEHOLDER_BYTES)
    warnings.warn(
        "Using placeholder layout weights. Run 'python3 -m scitrans_lm setup --yolo' "
        "with internet access or place a DocLayout-YOLO checkpoint at data/layout/layout_model.pt.",
        RuntimeWarning,
    )


def ensure_default_glossary(min_terms: int = 24, refresh_remote: bool = False) -> None:
    """Ensure the default EN↔FR glossary exists and is reasonably populated.

    ``refresh_remote`` is accepted for forward compatibility with callers that
    wish to re-download or rebuild the glossary. In this lightweight fork we
    simply regenerate the built-in glossary when refresh is requested or when
    the existing file is missing/too small.
    """

    GLOSSARY_DIR.mkdir(parents=True, exist_ok=True)
    needs_refresh = refresh_remote
    if DEFAULT_GLOSSARY.exists() and not refresh_remote:
        try:
            with DEFAULT_GLOSSARY.open("r", encoding="utf-8") as f:
                term_count = sum(1 for _ in csv.DictReader(f))
            needs_refresh = term_count < min_terms
        except Exception:
            needs_refresh = True
    if needs_refresh:
        _write_glossary(_BUILTIN_GLOSSARY)


def run_all() -> None:
    ensure_layout_model()
    ensure_default_glossary()
    print("✔ Setup complete. You can now translate.")

