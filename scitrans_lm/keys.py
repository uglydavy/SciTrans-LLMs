
from __future__ import annotations
import os, json
from pathlib import Path
from typing import Optional

try:
    import keyring
except Exception:
    keyring = None

CONFIG_DIR = Path.home() / ".scitranslm"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_FILE = CONFIG_DIR / "config.json"

def _load_cfg() -> dict:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_cfg(cfg: dict) -> None:
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

def set_key(service: str, secret: str) -> None:
    if keyring is not None:
        keyring.set_password("scitrans-lm", service, secret)
    else:
        cfg = _load_cfg()
        cfg.setdefault("keys", {})[service] = secret
        _save_cfg(cfg)

def get_key(service: str) -> Optional[str]:
    # try keyring
    if keyring is not None:
        try:
            val = keyring.get_password("scitrans-lm", service)
            if val:
                return val
        except Exception:
            pass
    # fallback env
    env_key = os.getenv(f"{service.upper()}_API_KEY") or os.getenv(f"{service.upper()}_KEY")
    if env_key:
        return env_key
    # fallback config file
    cfg = _load_cfg()
    return (cfg.get("keys") or {}).get(service)

