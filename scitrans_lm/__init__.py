"""
Alias module for scitrans_llms.

This module exists to support `python -m scitrans_lm` as an alternative
to `python -m scitrans_llms`. Both work identically.

For all functionality, see scitrans_llms module.
"""

# Re-export everything from the main module
from scitrans_llms import *
from scitrans_llms import __version__, __all__

