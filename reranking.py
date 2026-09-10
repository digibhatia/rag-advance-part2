"""
reranking.py

Samiksha — Bi-Encoder Retrieval, Then Cross-Encoder Rerank
===============================================================
Explain it simply: a bi-encoder (the embedding model used everywhere so
far) reads the question and each document SEPARATELY, then compares
their vectors — fast, because every document's vector was computed once,
in advance, at ingestion time. A cross-encoder reads the question AND a
document TOGETHER, in a single pass, and outputs one relevance score —
slower (it can't be precomputed), but noticeably more accurate, because
it can actually reason about how the two texts relate to each other,
not just how close two independently-computed points are in space.

The standard pattern, and the one used here: use the fast bi-encoder
(hybrid_retrieval.py) to cheaply narrow thousands of chunks down to a
manageable candidate set, THEN spend the extra time running the slower,
more accurate cross-encoder on just that smaller set.
"""
from sentence_transformers import CrossEncoder

# A small, fast, well-established cross-encoder for this kind of reranking.
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model = None


def _get_model():
    """Lazy-load the model once, the first time it's actually needed —
    downloading/loading it on import would slow down every script that
    imports this module, even ones that never call rerank()."""
    global _model
    if _model is None:
        _model = CrossEncoder(CROSS_ENCODER_MODEL)
    return _model


def rerank(query: str, candidates: list, top_k: int = 3) -> list:
    """
    candidates: a list of (Document, fusion_score) pairs from hybrid_retrieval.
    Returns the top_k candidates re-scored and re-sorted by the cross-encoder,
    as a list of (Document, cross_encoder_score) pairs.
    """
    if not candidates:
        return []

    model = _get_model()
    pairs = [[query, doc.page_content] for doc, _ in candidates]
    cross_scores = model.predict(pairs)

    scored = list(zip([doc for doc, _ in candidates], cross_scores))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]
