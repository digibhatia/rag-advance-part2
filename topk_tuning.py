"""
topk_tuning.py

Samiksha — Top-K Tuning: the Quality / Latency / Cost Trade-off
====================================================================
Sweeps top_k across a range of values and reports, for each: retrieval
quality (Hit Rate against the golden set), average latency, and average
context size (a proxy for token cost) — so the trade-off Day 6 asks you
to reason about is a real, measured number, not just a claim.

Run AFTER app.py is already running in another terminal:
    python topk_tuning.py
"""
import json
import os
import time
import requests

BASE_URL = "http://localhost:8000/ask"
K_VALUES = [1, 2, 3, 5, 8]

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "eval", "golden_qa_set.json")
with open(GOLDEN_PATH) as f:
    # Only questions that have a single clear expected source are usable
    # for a Hit Rate measurement — "should refuse" questions are skipped here.
    GOLDEN_SET = [item for item in json.load(f) if item.get("expected_source")]


def main():
    print("=" * 72)
    print("Top-K Tuning — Quality vs. Latency vs. Context Size")
    print("=" * 72)
    print(f"{'k':>3} | {'Hit Rate':>9} | {'Avg latency (s)':>16} | {'Avg context chars':>18}")
    print("-" * 72)

    for k in K_VALUES:
        hits = []
        latencies = []
        context_sizes = []

        for item in GOLDEN_SET:
            start = time.time()
            response = requests.get(BASE_URL, params={"q": item["question"], "top_k": k}, timeout=60)
            elapsed = time.time() - start
            result = response.json()

            retrieved_sources = [c["source"] for c in result["retrieved_chunks"]]
            hits.append(1 if item["expected_source"] in retrieved_sources else 0)
            latencies.append(elapsed)
            context_sizes.append(sum(len(c["text_preview"]) for c in result["retrieved_chunks"]))

        avg_hit = sum(hits) / len(hits) if hits else 0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        avg_context = sum(context_sizes) / len(context_sizes) if context_sizes else 0
        print(f"{k:>3} | {avg_hit:>9.2f} | {avg_latency:>16.2f} | {avg_context:>18.0f}")

    print("\nExpect Hit Rate to rise (or plateau) as k increases, while latency and")
    print("context size keep rising — that's the trade-off itself, made visible.")
    print("A sensible k is usually just past where Hit Rate plateaus, not the largest k tried.")


if __name__ == "__main__":
    main()
