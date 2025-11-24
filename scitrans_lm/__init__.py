"""SciTrans-LM package."""

__version__ = "0.2.0"

# Re-export key management helpers and translator factory for convenience
from .keys import get_key, set_key  # noqa: F401
from .translate.backends import get_translator  # noqa: F401

__all__ = ["__version__", "get_key", "set_key", "get_translator"]
