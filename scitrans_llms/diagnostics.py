"""Environment and asset diagnostics for SciTrans-LM.

This module inspects optional/required dependencies, model assets, and
configuration so users get actionable guidance instead of cryptic errors.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Dict, List

from .config import DEFAULT_GLOSSARY, GLOSSARY_DIR, LAYOUT_MODEL
from .keys import list_keys


@dataclass
class CheckResult:
    """Represents a diagnostic check with status and human-readable detail."""

    name: str
    status: str  # ok | warn | error
    detail: str


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _check_dependency(module: str, friendly: str, required: bool = False) -> CheckResult:
    available = _module_available(module)
    status = "ok" if available else ("error" if required else "warn")
    detail = f"{friendly} available" if available else f"{friendly} missing"
    return CheckResult(friendly, status, detail)


def _check_layout_model() -> CheckResult:
    if not LAYOUT_MODEL.exists():
        return CheckResult(
            "Layout model",
            "warn",
            "Placeholder created; run setup --yolo to download/train DocLayout weights.",
        )
    try:
        size = LAYOUT_MODEL.stat().st_size
    except OSError:
        size = 0
    if size <= 1024:
        return CheckResult(
            "Layout model",
            "warn",
            "Placeholder weights detected; run setup --yolo to improve layout detection.",
        )
    return CheckResult("Layout model", "ok", "DocLayout weights detected")


def _check_glossary() -> CheckResult:
    if not DEFAULT_GLOSSARY.exists():
        return CheckResult(
            "Default glossary",
            "warn",
            "Missing; run setup --glossary to regenerate the built-in terminology.",
        )
    try:
        lines = DEFAULT_GLOSSARY.read_text(encoding="utf-8").strip().splitlines()
    except OSError:
        lines = []
    if len(lines) < 5:
        return CheckResult(
            "Default glossary",
            "warn",
            "Too few entries; re-run setup --glossary or upload your own CSV.",
        )
    extra_files = list(Path(GLOSSARY_DIR).glob("*.csv"))
    detail = "Found built-in glossary"
    if len(extra_files) > 1:
        detail += f" + {len(extra_files) - 1} custom file(s)"
    return CheckResult("Default glossary", "ok", detail)


def _check_keys() -> CheckResult:
    keys = list_keys()
    missing = [svc for svc in ("openai", "deepl", "google", "deepseek", "perplexity") if not keys.get(svc)]
    if missing:
        return CheckResult(
            "API keys",
            "warn",
            "Missing: " + ", ".join(missing) + "; store with 'python -m scitrans_lm set-key <service>'.",
        )
    return CheckResult("API keys", "ok", "Found stored keys or environment variables")


def collect_diagnostics() -> List[CheckResult]:
    """Run a series of lightweight checks and return their results."""

    checks: List[CheckResult] = []
    checks.append(_check_dependency("fitz", "PyMuPDF", required=True))
    checks.append(_check_dependency("ultralytics", "ultralytics/YOLO"))
    checks.append(_check_dependency("torch", "PyTorch"))
    checks.append(_check_dependency("gradio", "Gradio"))
    checks.append(_check_dependency("requests", "Requests", required=True))
    checks.append(_check_dependency("sacrebleu", "SacreBLEU"))
    checks.append(_check_layout_model())
    checks.append(_check_glossary())
    checks.append(_check_keys())
    return checks


def summarize_checks(checks: List[CheckResult]) -> Dict[str, int]:
    summary = {"ok": 0, "warn": 0, "error": 0}
    for c in checks:
        if c.status in summary:
            summary[c.status] += 1
    return summary
