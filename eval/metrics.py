"""
eval/metrics.py

Samiksha — Custom Retrieval & Citation Metrics
===================================================
Hit Rate, MRR, and Citation Correctness are straightforward to compute
directly in Python — no framework needed, since they're really just
"did the right filename show up, and where?" questions.

Faithfulness and Answer Relevance (see run_evaluation.py) are handled
differently, by Ragas — because judging whether an ANSWER is actually
supported by its context is a job for an LLM-as-judge, not a string
comparison.
"""


def hit_rate(retrieved_sources: list, expected_source: str) -> int:
    """1 if the expected source appears ANYWHERE in the retrieved chunks,
    else 0. Returns None for questions with no single expected source
    (e.g., 'should refuse' questions), so they're excluded from the average
    rather than counted as a miss."""
    if expected_source is None:
        return None
    return 1 if expected_source in retrieved_sources else 0


def reciprocal_rank(retrieved_sources: list, expected_source: str) -> float:
    """1 / (rank of the FIRST retrieved chunk from the expected source),
    or 0.0 if it never appears. A perfect top-1 hit scores 1.0; a hit
    buried at position 5 scores 0.2 — MRR rewards not just finding the
    right source, but finding it EARLY."""
    if expected_source is None:
        return None
    for i, source in enumerate(retrieved_sources, start=1):
        if source == expected_source:
            return 1 / i
    return 0.0


def citation_correctness(cited_sources: list, expected_source: str) -> int:
    """1 if the FINAL ANSWER's citation list includes the expected source,
    else 0. Distinct from hit_rate: a chunk can be retrieved (hit_rate=1)
    but the LLM might still cite the wrong source in its answer, or fail
    to cite anything at all — that's what this metric catches."""
    if expected_source is None:
        return None
    return 1 if expected_source in cited_sources else 0


def mean(values: list) -> float:
    """Average, ignoring any None entries (questions the metric didn't apply to)."""
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else 0.0
