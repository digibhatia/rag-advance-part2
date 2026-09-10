"""
agent.py

Samiksha — Ask the Advanced Pipeline (Step 3 of 3)
=======================================================
5 hardcoded questions (same domain coverage as Disha), run through the
default advanced pipeline, followed by 3 bonus comparisons that make
query transformation, reranking, and top-k tuning visible side by side.

IMPORTANT: app.py must already be running in a separate terminal
(uvicorn app:app --reload --port 8000) before you run this file.

Run:
    python agent.py
"""
import requests

BASE_URL = "http://localhost:8000/ask"

QUESTIONS = [
    "What is the maximum indoor EIRP for Band 78 deployments?",
    "What service credit applies if uptime falls below the SLA guarantee?",
    "Who do I contact for a P1 escalation outside business hours?",
    "Which incidents in the log are still Open?",
    "Which vendor's equipment is NOT approved for indoor deployment, and why?",
]


def ask(question: str, **params) -> dict:
    response = requests.get(BASE_URL, params={"q": question, **params}, timeout=60)
    response.raise_for_status()
    return response.json()


def main():
    print("=" * 60)
    print("Samiksha — NCS Telco+ Advanced RAG Pipeline")
    print("=" * 60)

    for i, question in enumerate(QUESTIONS, start=1):
        print(f"\nQ{i}: {question}")
        try:
            result = ask(question)
        except requests.exceptions.ConnectionError:
            print("ERROR: Could not reach Samiksha. Is app.py running in another "
                  "terminal? (uvicorn app:app --reload --port 8000)")
            return

        for chunk in result["retrieved_chunks"]:
            print(f"    - {chunk['source']} [{chunk['doc_type']} v{chunk['version']}] "
                  f"score={chunk['score']}")
        print(f"\nA{i}: {result['answer']}")
        print(f"     (Sources: {', '.join(result['sources']) if result['sources'] else 'none'})")
        print(f"     (Timings: {result['timings']})")

    print("\n" + "=" * 60)
    print("BONUS 1 — Query transformation comparison")
    print("=" * 60)
    bonus_q = "What happens if we breach the uptime guarantee?"
    for technique in ["none", "rewrite", "multi_query", "step_back"]:
        result = ask(bonus_q, query_transform=technique)
        print(f"\n--- technique: {technique} ---")
        print(f"Queries used  : {result['queries_used']}")
        print(f"Chunks found  : {len(result['retrieved_chunks'])}")
        print(f"Timings       : {result['timings']}")

    print("\n" + "=" * 60)
    print("BONUS 2 — Cross-encoder reranking on vs. off")
    print("=" * 60)
    bonus_q2 = "What is the maximum indoor EIRP for Band 78 deployments?"
    for use_rerank in [True, False]:
        result = ask(bonus_q2, use_rerank=use_rerank)
        print(f"\n--- use_rerank={use_rerank} ---")
        for chunk in result["retrieved_chunks"]:
            print(f"    - {chunk['source']} score={chunk['score']}")
        print(f"Timings: {result['timings']}")

    print("\n" + "=" * 60)
    print("BONUS 3 — top_k quick look (see topk_tuning.py for the full sweep)")
    print("=" * 60)
    for k in [1, 3, 8]:
        result = ask(bonus_q2, top_k=k)
        print(f"\n--- top_k={k} ---")
        print(f"Chunks used: {len(result['retrieved_chunks'])}, Timings: {result['timings']}")

    print("\n" + "=" * 60)
    print("Done! Run eval/run_evaluation.py next for the full evaluation report.")
    print("=" * 60)


if __name__ == "__main__":
    main()
