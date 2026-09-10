"""
chunking.py

Samiksha — Chunking (carried over from Day 5's Disha lab)
=============================================================
Day 5 compared 5 chunking strategies. This lab isn't about chunking —
it's about what happens AFTER chunking (retrieval, reranking, evaluation)
— so we fix on one solid strategy (overlap) and move on. If you want to
re-introduce a strategy comparison here, Disha's chunking.py has all 5.
"""


def overlap_chunk(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Fixed-size chunks that share `overlap` characters with the next
    one, so an idea sitting on a chunk boundary isn't lost entirely."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Single entry point, kept for consistency with Disha's interface."""
    return overlap_chunk(text, chunk_size=chunk_size, overlap=overlap)
