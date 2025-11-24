"""API key storage and retrieval helpers.

Keys are stored under ``~/.config/scitrans_llm/keys.json``. Environment
variables such as ``OPENAI_API_KEY`` are also honored. This module avoids
storing secrets in the repository and keeps a single location for key
resolution so new providers (DeepSeek, Perplexity, etc.) can reuse it.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, Optional

CONFIG_DIR = Path(os.path.expanduser("~/.config/scitrans_llm"))
CONFIG_FILE = CONFIG_DIR / "keys.json"


def get_key(service: str) -> Optional[str]:
    """Retrieve the API key for a given translation service.

    Lookup proceeds in the following order:
    1. Environment variables matching common service names (e.g. ``OPENAI_API_KEY``).
    2. The JSON config file ``keys.json`` under ``~/.config/scitrans_llm``.

    Args:
        service: Name of the service (e.g. ``openai``, ``deepl``).

    Returns:
        The API key if found, otherwise ``None``.
    """

    service = (service or "").lower()
    env_map = {
        "openai": "OPENAI_API_KEY",
        "deepl": "DEEPL_API_KEY",
        "google": "GOOGLE_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "perplexity": "PERPLEXITY_API_KEY",
    }
    env_var = env_map.get(service)
    if env_var and os.getenv(env_var):
        return os.getenv(env_var)

    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get(service)
        except Exception:
            return None
    return None


def set_key(service: str, key: str) -> None:
    """Persist the API key for a translation service."""

    service = (service or "").lower()
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    data = {}
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
        except Exception:
            data = {}
    data[service] = key
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def list_keys() -> Dict[str, bool]:
    """Return a mapping of stored services -> presence (masked, not values)."""

    discovered: Dict[str, bool] = {}
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
            for name, value in data.items():
                discovered[name] = bool(value)
        except Exception:
            pass
    for svc, env_var in {
        "openai": "OPENAI_API_KEY",
        "deepl": "DEEPL_API_KEY",
        "google": "GOOGLE_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "perplexity": "PERPLEXITY_API_KEY",
    }.items():
        if env_var in os.environ:
            discovered[svc] = True
    return discovered
