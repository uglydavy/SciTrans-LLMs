#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download sample PDFs for testing SciTrans-LLMs.

Downloads publicly available scientific papers from arXiv for testing
the translation pipeline.
"""

import sys
from pathlib import Path
import urllib.request
import ssl

# Sample papers from arXiv (open access)
SAMPLE_PAPERS = {
    "attention": {
        "url": "https://arxiv.org/pdf/1706.03762.pdf",
        "name": "Attention Is All You Need",
        "description": "The original Transformer paper",
    },
    "bert": {
        "url": "https://arxiv.org/pdf/1810.04805.pdf",
        "name": "BERT: Pre-training of Deep Bidirectional Transformers",
        "description": "BERT language model paper",
    },
    "gpt2": {
        "url": "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
        "name": "Language Models are Unsupervised Multitask Learners",
        "description": "GPT-2 paper from OpenAI",
    },
    "resnet": {
        "url": "https://arxiv.org/pdf/1512.03385.pdf",
        "name": "Deep Residual Learning for Image Recognition",
        "description": "ResNet paper",
    },
    "dropout": {
        "url": "https://arxiv.org/pdf/1207.0580.pdf",
        "name": "Improving neural networks with dropout",
        "description": "Dropout regularization paper",
    },
}


def download_file(url: str, dest: Path, name: str) -> bool:
    """Download a file with progress indicator."""
    print(f"  Downloading: {name}")
    print(f"    URL: {url}")
    
    try:
        # Create SSL context that doesn't verify (for corporate proxies)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        # Add headers to avoid 403
        request = urllib.request.Request(
            url,
            headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Accept': 'application/pdf',
            }
        )
        
        with urllib.request.urlopen(request, context=ctx, timeout=60) as response:
            total_size = int(response.headers.get('content-length', 0))
            data = response.read()
            
            dest.write_bytes(data)
            
            size_mb = len(data) / (1024 * 1024)
            print(f"    Saved: {dest.name} ({size_mb:.1f} MB)")
            return True
            
    except Exception as e:
        print(f"    Failed: {e}")
        return False


def main():
    """Download sample PDFs."""
    print("=" * 60)
    print("SciTrans-LLMs Sample PDF Downloader")
    print("=" * 60)
    
    # Create samples directory
    samples_dir = Path(__file__).parent.parent / "samples"
    samples_dir.mkdir(exist_ok=True)
    
    print(f"\nDownloading to: {samples_dir}\n")
    
    # Also create tests/data/pdfs for test fixtures
    test_data_dir = Path(__file__).parent.parent / "tests" / "data" / "pdfs"
    test_data_dir.mkdir(parents=True, exist_ok=True)
    
    successful = 0
    failed = 0
    
    for key, paper in SAMPLE_PAPERS.items():
        dest = samples_dir / f"{key}.pdf"
        
        if dest.exists():
            print(f"  Skipping {key}.pdf (already exists)")
            successful += 1
            continue
        
        if download_file(paper["url"], dest, paper["name"]):
            successful += 1
            
            # Also copy to test data dir
            test_dest = test_data_dir / f"{key}.pdf"
            if not test_dest.exists():
                test_dest.write_bytes(dest.read_bytes())
        else:
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Downloaded: {successful} papers")
    if failed > 0:
        print(f"Failed: {failed} papers")
    print("=" * 60)
    
    print(f"\nSample PDFs are in: {samples_dir}")
    print("\nTo translate a sample:")
    print(f"  scitrans translate -i {samples_dir}/attention.pdf -o attention_fr.pdf")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

