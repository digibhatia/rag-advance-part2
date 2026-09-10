"""
eval/run_evaluation.py

Samiksha — Automated RAG Evaluation
=======================================
Runs the golden Q&A set (golden_qa_set.json) through the LIVE pipeline,
computes Hit Rate, MRR, and Citation Correctness directly (metrics.py),
computes Faithfulness and Answer Relevance via Ragas, classifies each
failure into the failure taxonomy, and prints a report.

Run AFTER app.py is already running in another terminal:
    python eval/run_evaluation.py
"""
import json
import os
import sys
import requests

# Allow "from metrics import ..." whether this is run from the repo root
# or from inside eval/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import hit_rate, reciprocal_rank, citation_correctness, mean

BASE_URL = "http://localhost:8000/ask"

with open(os.path.join(os.path.dirname(__file__), "golden_qa_set.json")) as f:
    GOLDEN_SET = json.load(f)


def ask_samiksha(question: str, **params) -> dict:
    response = requests.get(BASE_URL, params={"q": question, **params}, timeout=60)
    response.raise_for_status()
    return response.json()


def classify_failure(item: dict, result: dict, hit, cite_ok) -> str | None:
    """Assigns each result to one bucket of the failure taxonomy, or
    returns None if nothing went wrong. This is a simple, explainable
    rule-based classifier — good enough to point you at what to look at,
    not a substitute for reading the actual failing cases yourself."""
    answer_lower = result["answer"].lower()
    refused = "don't know" in answer_lower or "cannot" in answer_lower or "not able to" in answer_lower

    if item.get("should_refuse"):
        return None if refused else "hallucination"  # answered when it should have refused

    if refused:
        return "retrieval_failure"  # had an answer to find, but the pipeline gave up

    if hit == 0:
        return "retrieval_failure"  # expected source never made it into the retrieved set

    if cite_ok == 0:
        return "citation_error"  # right source was retrieved, but the answer cited something else (or nothing)

    expected_phrase = item.get("expected_answer_contains")
    if expected_phrase and expected_phrase.lower() not in answer_lower:
        return "hallucination"  # cited correctly, but the specific claim isn't actually in the answer

    return None


def run_ragas_metrics(rows: list):
    """Faithfulness (does the answer only claim things the context
    supports?) and Answer Relevance (does the answer actually address
    the question?) both need an LLM-as-judge — that's what Ragas provides.
    Wrapped in try/except: Ragas' exact API/config needs can shift between
    versions, and this shouldn't block the rest of the report from printing."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy

        dataset = Dataset.from_list(rows)
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
        print(result)
    except Exception as e:
        print(f"Ragas evaluation could not complete: {e}")
        print("Common causes: Ragas needs its own LLM/embeddings configured, or a version")
        print("mismatch with the 'datasets' library. Check current Ragas docs — its API")
        print("has changed across versions. The metrics below don't depend on this working.")


def main():
    print("=" * 64)
    print("Samiksha — Running Evaluation Against the Golden Q&A Set")
    print("=" * 64)

    hit_rates, mrrs, citation_scores = [], [], []
    ragas_rows = []
    failures = []

    for item in GOLDEN_SET:
        question = item["question"]
        expected_source = item.get("expected_source")

        try:
            result = ask_samiksha(question)
        except requests.exceptions.ConnectionError:
            print("ERROR: Could not reach Samiksha. Is app.py running? "
                  "(uvicorn app:app --reload --port 8000)")
            return

        retrieved_sources = [c["source"] for c in result["retrieved_chunks"]]
        cited_sources = result["sources"]

        hit = hit_rate(retrieved_sources, expected_source)
        mrr = reciprocal_rank(retrieved_sources, expected_source)
        cite_ok = citation_correctness(cited_sources, expected_source)

        if hit is not None:
            hit_rates.append(hit)
        if mrr is not None:
            mrrs.append(mrr)
        if cite_ok is not None:
            citation_scores.append(cite_ok)

        failure_type = classify_failure(item, result, hit, cite_ok)
        if failure_type:
            failures.append({"question": question, "failure_type": failure_type, "answer": result["answer"]})

        ragas_rows.append({
            "question": question,
            "answer": result["answer"],
            "contexts": [c["text_preview"] for c in result["retrieved_chunks"]] or ["(no context retrieved)"],
            "ground_truth": item.get("expected_answer_contains") or "",
        })

        print(f"\nQ: {question}")
        print(f"  Hit Rate: {hit}  |  MRR: {mrr}  |  Citation Correct: {cite_ok}")
        if failure_type:
            print(f"  FAILURE DETECTED: {failure_type}")

    print("\n" + "=" * 64)
    print("RETRIEVAL & CITATION METRICS")
    print("=" * 64)
    print(f"Hit Rate            : {mean(hit_rates):.2f}")
    print(f"MRR                 : {mean(mrrs):.2f}")
    print(f"Citation Correctness: {mean(citation_scores):.2f}")

    print("\n" + "=" * 64)
    print("RAGAS METRICS (Faithfulness & Answer Relevance)")
    print("=" * 64)
    run_ragas_metrics(ragas_rows)

    print("\n" + "=" * 64)
    print("FAILURE TAXONOMY")
    print("=" * 64)
    if not failures:
        print("No failures detected against this golden set.")
    else:
        by_type = {}
        for f in failures:
            by_type.setdefault(f["failure_type"], []).append(f)
        for failure_type, items in by_type.items():
            print(f"\n{failure_type.upper()} ({len(items)}):")
            for item in items:
                print(f"  - {item['question']}")

    print("\nNext step: copy failure_taxonomy_template.md and fill it in with")
    print("targeted fixes for whatever failures showed up above.")


if __name__ == "__main__":
    main()
