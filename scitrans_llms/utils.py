"""
Utility functions used across CLI, GUI, and pipeline.

This module provides small helper functions that are used throughout
the SciTrans-LLMs system for various tasks like page range parsing,
device detection, and geometric operations.

Functions:
    parse_page_range: Parse user-friendly page range strings
    detect_device: Detect available compute device (CUDA/CPU)
    boxes_intersect: Check if two bounding boxes intersect

Example:
    >>> from scitrans_llms.utils import parse_page_range, detect_device
    >>> start, end = parse_page_range("1-5", total=10)
    >>> print(f"Pages {start} to {end}")
    >>> device = detect_device()
    >>> print(f"Using device: {device}")
"""

from typing import Tuple


def parse_page_range(pages: str, total: int) -> Tuple[int, int]:
    """
    Parse user-friendly page range string into 0-based indices.
    
    Supports various formats:
    - "all" or empty: All pages
    - "5": Single page
    - "1-5": Range of pages
    - "5-1": Reversed range (automatically corrected)
    
    Args:
        pages: Page range string (e.g., "1-5", "all", "3")
        total: Total number of pages in document
        
    Returns:
        Tuple of (start_index, end_index) as 0-based inclusive indices
        
    Example:
        >>> parse_page_range("1-5", total=10)
        (0, 4)
        >>> parse_page_range("all", total=10)
        (0, 9)
        >>> parse_page_range("3", total=10)
        (2, 2)
    """
    pages = (pages or "").strip().lower()
    
    # Handle "all" or empty string
    if not pages or pages == "all":
        return (0, max(0, total - 1))
    
    # Handle range like "1-5"
    if "-" in pages:
        a, b = pages.split("-", 1)
        s = max(1, int(a))
        e = min(total, int(b))
        # Swap if reversed
        if s > e:
            s, e = e, s
        return (s - 1, e - 1)
    
    # Handle single page like "3"
    p = max(1, min(total, int(pages)))
    return (p - 1, p - 1)


def detect_device() -> str:
    """
    Detect available compute device for PyTorch/YOLO operations.
    
    Checks if CUDA is available and returns the appropriate device string.
    Falls back to CPU if CUDA is not available or PyTorch is not installed.
    
    Returns:
        Device string: "cuda:0" if CUDA available, "cpu" otherwise
        
    Example:
        >>> device = detect_device()
        >>> print(f"Using device: {device}")
        Using device: cuda:0
    """
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def boxes_intersect(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
    padding: float = 0.0
) -> bool:
    """
    Check if two bounding boxes intersect.
    
    Bounding boxes are represented as (x0, y0, x1, y1) tuples where:
    - (x0, y0) is the top-left corner
    - (x1, y1) is the bottom-right corner
    
    Args:
        a: First bounding box (x0, y0, x1, y1)
        b: Second bounding box (x0, y0, x1, y1)
        padding: Extra padding to add around boxes before checking (default: 0.0)
        
    Returns:
        True if boxes intersect, False otherwise
        
    Example:
        >>> box1 = (0, 0, 10, 10)
        >>> box2 = (5, 5, 15, 15)
        >>> boxes_intersect(box1, box2)
        True
        >>> box3 = (20, 20, 30, 30)
        >>> boxes_intersect(box1, box3)
        False
    """
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    
    # Apply padding
    ax0 -= padding
    ay0 -= padding
    ax1 += padding
    ay1 += padding
    bx0 -= padding
    by0 -= padding
    bx1 += padding
    by1 += padding
    
    # Check if boxes don't intersect (then invert)
    return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)
