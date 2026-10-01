# Samiksha — Advanced RAG: Hybrid Search, Re-ranking & Evaluation

*An upgrade of Day 5's Disha pipeline — same documents, now with hybrid
retrieval, cross-encoder reranking, query transformation, metadata
enrichment, and automated evaluation against a golden Q&A set.*

---

## 1. What Changed Since Disha (Day 5)

Disha proved the basic RAG loop works: ingest, chunk, embed, retrieve,
answer, cite. Samiksha asks the harder question Module 6 is built around:
**how do you know it's actually working well, and how do you make it work
better?**

| Disha (Day 5) | Samiksha (Day 6) |
|---|---|
| Vector search only | Hybrid: BM25 (keyword) + vector search, fused |
| One retrieval pass | Bi-encoder retrieval, then cross-encoder rerank |
| Question sent as-is | Query transformation: rewrite / multi-query / step-back |
| Metadata: source, chunk_index | + doc_type, version, effective_date |
| "Does it work?" judged by eye | Hit Rate, MRR, Faithfulness, Answer Relevance, Citation Correctness — measured |
| No structured way to find weaknesses | A failure taxonomy, generated from a real eval run |

Same 5 documents as Disha. Same three-script pattern. Genuinely more
pipeline underneath.

## 2. Architecture — the Full Flow

```
STEP 1 -- build_vectorstore.py  (run once)

    documents/*.{pdf,docx,html,csv,json}
        |
        |  loaders.py            -- same 5 format-specific loaders as Disha
        v
    raw text units
        |
        |  pii.py                -- redact emails/phone/PAN-style IDs
        |  metadata_enrichment.py -- NEW: extract doc_type, version,
        |                            effective_date per document
        v
    redacted, enriched text
        |
        |  chunking.py           -- overlap strategy (fixed for this lab)
        v
    chunks, tagged with source + chunk_index + doc_type + version + effective_date
        |
        |  LangChain + OpenAIEmbeddings
        v
    ./chroma_store  (ChromaDB vector store, persisted to disk)


STEP 2 -- app.py  (FastAPI backend, keep running in Terminal 1)

    /ask?q=...&query_transform=none&top_k=3&fetch_n=15&use_rerank=true

    1. query_transform.py    -- turn 1 question into 1+ search queries
    2. hybrid_retrieval.py   -- BM25 + vector search per query, fused
                                 with Reciprocal Rank Fusion (RRF)
    3. reranking.py          -- cross-encoder rescoring of the candidate
                                 pool, keep only the true top_k
    4. context.py            -- dedupe + pack (same as Disha)
    5. LLM call              -- citation-required system prompt
    6. return JSON            -- includes per-stage TIMINGS, so the
                                 latency/cost trade-off is visible, not
                                 just claimed


STEP 3 -- agent.py  (Terminal 2)

    Sends the same 5 domain questions as Disha, PLUS 3 bonus comparisons:
    query-transform techniques side by side, rerank on vs. off, and a
    quick top_k look.


STEP 4 -- eval/run_evaluation.py  (Terminal 2, after agent.py)

    Runs 10 golden Q&A pairs through the live pipeline, computes:
      - Hit Rate, MRR, Citation Correctness (custom, metrics.py)
      - Faithfulness, Answer Relevance (Ragas)
    Classifies every failure into: retrieval_failure / hallucination /
    context_overflow / citation_error, and points you at
    failure_taxonomy_template.md to write up targeted fixes.


STEP 5 (optional) -- topk_tuning.py

    Sweeps top_k across [1, 2, 3, 5, 8], reporting Hit Rate, average
    latency, and average context size for each — the trade-off, measured.
```

## 3. Setup — Step by Step

### Step 1 — Clone the repo
```
git clone <the link shared with you in the lab>
cd samiksha-advanced-rag
```

### Step 2 — Create your `.env` file
Copy `.env.example` to `.env` and add your key:
```
OPENAI_API_KEY="paste-the-key-shared-in-the-lab-here"
```

### Step 3 — Set up your environment and install dependencies
```
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
This lab installs more than Disha did — `rank_bm25` (keyword search),
`sentence-transformers` (the cross-encoder), and `ragas` + `datasets`
(evaluation). The first time `reranking.py` runs, it downloads the
cross-encoder model automatically — expect a short one-time pause.

### Step 4 — Build the vector store (run once)
```
python build_vectorstore.py
```
Watch the console output — it now reports the enriched metadata
(doc_type, version, effective_date) for every document as it's processed.

### Step 5 — Start Samiksha (Terminal 1)
```
uvicorn app:app --reload --port 8000
```
Leave this running. Visit `http://localhost:8000/docs` any time.

### Step 6 — Ask Samiksha your questions (Terminal 2)
```
python agent.py
```

### Step 7 — Run the evaluation (Terminal 2, after agent.py)
```
python eval/run_evaluation.py
```
Then copy `eval/failure_taxonomy_template.md` and fill it in with the
failures this run surfaced.

### Step 8 (optional) — Tune top_k
```
python topk_tuning.py
```

## 4. What to Notice — Specific Things to Try

- **Hybrid beats pure vector search on exact terms.** Ask a question that
  hinges on an exact ID — e.g., `curl "http://localhost:8000/ask?q=What+happened+in+incident+INC-10235%3F"`
  — and compare `use_rerank=true` (default) to disabling hybrid by editing
  `fetch_n=0` temporarily. The exact incident ID is exactly what BM25 is
  built to catch reliably.
- **Reranking changes the ORDER, not just the score.** Run `agent.py`'s
  Bonus 2 and compare the chunk order with `use_rerank=True` vs.
  `False` — the same candidate pool, reordered by a model that actually
  reads question and chunk together.
- **Query transformation widens the net.** Run `agent.py`'s Bonus 1 and
  compare `queries_used` across techniques — `multi_query` and
  `step_back` search for things `none` never would have tried.
- **The metadata is real, not decorative.** Run `inspect_vectorstore.py`
  and check that `doc_type`, `version`, and `effective_date` actually
  appear in the stored chunk metadata — then notice `app.py`'s
  `/ask` response includes them too.
- **Evaluation finds real gaps.** Run `eval/run_evaluation.py` and read
  the FAILURE TAXONOMY section closely — if Hit Rate or MRR isn't 1.0 on
  every question, that's real information about where the pipeline is
  weak, not a bug in the eval script.
- **top_k is a real trade-off, not a free lunch.** Run `topk_tuning.py`
  and watch Hit Rate plateau while latency and context size keep climbing
  — the "right" k is usually right where the plateau starts.

## 5. A Note on Confidence Thresholds

Disha's `MIN_CONFIDENCE` guardrail relied on Chroma's normalised
relevance scores. Cross-encoder scores from `reranking.py` are NOT
normalised the same way (they're raw logits, not 0–1 probabilities), so
Samiksha's "I don't know" guardrail only checks whether ANY candidates
survived retrieval at all — it does not threshold on the cross-encoder
score. This is a real, honest limitation worth discussing: a production
system would likely calibrate the cross-encoder's output (e.g., via a
sigmoid or a held-out validation set) before using it as a refusal
threshold. Worth trying as an extension.

## 6. A Note on Ragas

`eval/run_evaluation.py` wraps the Ragas call in a try/except. Ragas'
exact API and configuration requirements (which LLM/embeddings it uses
internally to judge Faithfulness and Answer Relevance) have changed
across versions — if it fails in your environment, check Ragas' current
documentation for your installed version. The Hit Rate / MRR / Citation
Correctness metrics don't depend on Ragas working, so the rest of the
evaluation report is unaffected either way.

## 7. Troubleshooting

| Problem | Likely cause |
|---|---|
| `ModuleNotFoundError` | Virtual environment not activated, or `pip install -r requirements.txt` not run |
| First `/ask` call is very slow | The cross-encoder model is downloading on first use — subsequent calls are fast |
| `app.py` fails with "collection not found" | Run `build_vectorstore.py` first, from this same folder |
| `eval/run_evaluation.py` can't import `metrics` | Run it as `python eval/run_evaluation.py` from the repo root, not from inside `eval/` |
| Ragas step fails | See Section 6 — the rest of the report still prints |
| Every answer is "I don't know" | Check `fetch_n` isn't set to 0; confirm `build_vectorstore.py` completed without errors |

## 8. Sample Assignment Outline (for Participants)

**Assignment: Extend the Failure Taxonomy**

1. Run `eval/run_evaluation.py` and record the baseline Hit Rate, MRR, and
   Citation Correctness.
2. Deliberately degrade ONE part of the pipeline — for example, set
   `use_rerank=False` as the default in `app.py`, or shrink
   `max_chars` in `context.py` to 500.
3. Re-run the evaluation. Which metric moved, and by how much?
4. Using `failure_taxonomy_template.md`, document at least 2 NEW failures
   this change introduced, with a specific suspected cause for each.
5. Revert your change, confirm the original metrics return.
6. **Written reflection (150–200 words):** Of the four failure categories
   (retrieval failure, hallucination, context overflow, citation error),
   which one did your deliberate change move the needle on, and why does
   that make sense given what you changed?

## 9. Where This Goes Next

This lab evaluates and tunes a single-pass pipeline. Later modules take
the pieces built here — retrieval, tool use, evaluation — and wrap them
inside autonomous agents that decide for themselves when to retrieve,
when to ask a clarifying question, and when to hand off to another agent
entirely.
