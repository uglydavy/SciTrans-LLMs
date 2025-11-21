
from __future__ import annotations
import csv
import warnings
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


def _write_glossary(rows: Iterable[Tuple[str, str]]) -> None:
    with DEFAULT_GLOSSARY.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target"])
        for src, tgt in rows:
            writer.writerow([src, tgt])


def ensure_layout_model() -> None:
    """Ensure layout model exists. If not, create a tiny placeholder and instruct the user."""

    LAYOUT_DIR.mkdir(parents=True, exist_ok=True)
    if LAYOUT_MODEL.exists():
        return
    # Create a small placeholder. Real model will be downloaded/trained by setup.
    LAYOUT_MODEL.write_bytes(
        b"SciTrans-LM placeholder weights. Run 'python3 -m scitrans_lm setup --yolo' to download/train.".ljust(
            1024, b"\0"
        )
    )
    warnings.warn(
        "No layout model detected; created a placeholder. Run 'python3 -m scitrans_lm setup --yolo' to download or train a DocLayout model.",
        RuntimeWarning,
    )


def ensure_default_glossary(min_terms: int = 24) -> None:
    """Ensure the default EN↔FR glossary exists and is reasonably populated."""

    GLOSSARY_DIR.mkdir(parents=True, exist_ok=True)
    needs_refresh = True
    if DEFAULT_GLOSSARY.exists():
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

