from __future__ import annotations


def bleu(hyp: str, ref: str) -> float:
    try:
        import sacrebleu
    except Exception as exc:
        raise RuntimeError("Please install sacrebleu to use BLEU scoring.") from exc
    return sacrebleu.corpus_bleu([hyp], [[ref]]).score
