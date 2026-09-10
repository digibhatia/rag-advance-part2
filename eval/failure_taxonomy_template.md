# Failure Taxonomy Report — Samiksha Evaluation Run

Fill this in after running `python eval/run_evaluation.py`. Copy the failures
it printed into the relevant category below, then work out a targeted fix
for each — not a generic "make it better" note.

**Run date:** ______________
**Config used (top_k, query_transform, use_rerank):** ______________

## Summary Metrics
- Hit Rate: ______
- MRR: ______
- Citation Correctness: ______
- Faithfulness (Ragas): ______
- Answer Relevance (Ragas): ______

## Failures by Category

### 1. Retrieval Failure
*The expected document never made it into the retrieved set at all.*

| Question | What happened | Suspected cause | Targeted fix |
|---|---|---|---|
| | | (chunking cut the fact awkwardly? query wording mismatch? embedding model limitation?) | |

### 2. Hallucination
*The answer contains a claim the retrieved context doesn't actually support.*

| Question | What happened | Suspected cause | Targeted fix |
|---|---|---|---|
| | | (context assembly dropped the relevant chunk? system prompt grounding rule too weak? model ignored the citation instruction?) | |

### 3. Context Overflow
*Relevant content was retrieved but didn't survive into the final context sent to the LLM.*

| Question | What happened | Suspected cause | Targeted fix |
|---|---|---|---|
| | | (max_chars budget too small? deduplication too aggressive, dropped a distinct fact as if it were a duplicate?) | |

### 4. Citation Error
*The right source was retrieved, but the final answer cited the wrong one — or none at all.*

| Question | What happened | Suspected cause | Targeted fix |
|---|---|---|---|
| | | (multiple sources in context, model picked the wrong one? citation instruction not explicit enough?) | |

## Overall Recommendation

Based on everything above, what is the SINGLE highest-priority fix to make
before the next evaluation run? Why that one first?

______________________________________________________________________
______________________________________________________________________
