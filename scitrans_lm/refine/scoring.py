
from __future__ import annotations
from typing import List
def bleu(hyp: str, ref: str) -> float:
    try:
        import sacrebleu
    except Exception:
        raise RuntimeError("Please install sacrebleu to use BLEU scoring.")
    return sacrebleu.corpus_bleu([hyp], [[ref]]).score
