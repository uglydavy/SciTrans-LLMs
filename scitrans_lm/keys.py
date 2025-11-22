"""
Utility functions for storing and retrieving API keys for translation backends.

This module provides a simple way to persist API keys used by the various
translation backends in SciTrans‑LM. Keys are stored in a JSON file in the
user's home directory under ``~/.config/scitrans_llm/keys.json``. If a key is
not found in the config file, the environment variables corresponding to
common services are checked.

Users can also set keys at runtime using the ``set_key`` function. This will
create the config directory and file if necessary and persist the key.

Note: the original upstream project includes more advanced key management via
the CLI. Here we implement a minimal subset that satisfies the needs of the
translation backends.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Optional

# Path to the keys configuration file. Use a dedicated directory under ~/.config
# to avoid cluttering the user's home directory.
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
    # Mapping of service names to environment variable names
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
    # Read from config file if it exists
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Keys may be stored by service name directly
            return data.get(service)
        except Exception:
            # If the file is corrupt or unreadable, ignore and return None
            return None
    return None


def set_key(service: str, key: str) -> None:
    """Persist the API key for a translation service.

    This function writes the provided key to the JSON config file. If the
    directory does not exist, it is created. Existing keys for other services
    are preserved.

    Args:
        service: Name of the service (e.g. ``openai``).
        key: The API key string to store.
    """
    service = (service or "").lower()
    # Ensure the config directory exists
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