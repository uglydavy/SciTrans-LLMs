#!/usr/bin/env python
"""Quick GUI launcher for SciTrans-LLMs"""

from scitran_llms.gui import launch

if __name__ == "__main__":
    print("Starting SciTrans-LLMs GUI...")
    print("Open your browser at http://127.0.0.1:7860")
    launch(port=7860)
