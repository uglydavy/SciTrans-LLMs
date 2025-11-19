
from __future__ import annotations
import re
from typing import Tuple

def parse_page_range(pages: str, total: int) -> Tuple[int, int]:
    """Parse '1-5' or 'all' into 0-based inclusive start,end."""
    pages = (pages or '').strip().lower()
    if not pages or pages == 'all':
        return (0, max(0, total-1))
    if '-' in pages:
        a, b = pages.split('-', 1)
        s = max(1, int(a))
        e = min(total, int(b))
        if s > e:
            s, e = e, s
        return (s-1, e-1)
    # single page
    p = max(1, min(total, int(pages)))
    return (p-1, p-1)

def detect_device() -> str:
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
