"""
Experiments module for running systematic evaluations.

Provides:
- Corpus loading and management
- Experiment configuration
- Full pipeline experiments
- Results collection and export
"""

from scitrans_next.experiments.corpus import (
    Corpus,
    CorpusDocument,
    load_corpus,
)
from scitrans_next.experiments.runner import (
    ExperimentRunner,
    ExperimentConfig,
    ExperimentResult,
    run_experiment,
)

__all__ = [
    "Corpus",
    "CorpusDocument",
    "load_corpus",
    "ExperimentRunner",
    "ExperimentConfig",
    "ExperimentResult",
    "run_experiment",
]

