"""
app.py

Samiksha — Advanced RAG Backend (Step 2 of 3)
==================================================
Full pipeline: query transform -> hybrid retrieve -> cross-encoder rerank
-> top-k selection -> context assembly -> answer -> cite (or "I don't know").

Run AFTER build_vectorstore.py:
    uvicorn app:app --reload --port 8000

Try it directly in a browser once running:
    http://localhost:8000/ask?q=What+is+the+maximum+indoor+EIRP+for+Band+78%3F
"""
import time

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from openai import OpenAI

from query_transform import transform_query
from hybrid_retrieval import HybridIndex
from reranking import rerank
from context import assemble_context


CHAT_MODEL = "gpt-5.4-mini"              # substitute your organisation's approved model
EMBED_MODEL = "text-embedding-3-small"    # must match build_vectorstore.py
COLLECTION_NAME = "samiksha_documents"
PERSIST_DIR = "./chroma_store"

SYSTEM_PROMPT = """
You are Samiksha, the NCS Telco+ Advanced Document Q&A Assistant.

Tone: Precise, calm, and technical — the voice of a senior engineer.

Grounding rule: Only answer using the excerpts provided. Every factual
claim MUST be followed by a citation in the form (Source: <filename>).
If the excerpts don't cover the question, say so plainly and do not guess.
"""

client = OpenAI()
embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR,
)

# Reconstruct the full chunk list straight from Chroma (single source of
# truth), so BM25 indexes the EXACT same chunks the vector store has.
_raw = vectorstore.get(include=["documents", "metadatas"])
_all_documents = [
    Document(page_content=text, metadata=meta)
    for text, meta in zip(_raw["documents"], _raw["metadatas"])
]
hybrid_index = HybridIndex(vectorstore, _all_documents)
print(f"Samiksha loaded {len(_all_documents)} chunks into the hybrid index.")

app = FastAPI(title="Samiksha - Advanced RAG Pipeline")


@app.get("/")
def root():
    return {"message": "Samiksha is running. Try /ask?q=your+question&query_transform=none&top_k=3"}


@app.get("/ask")
def ask(q: str, query_transform: str = "none", top_k: int = 3, fetch_n: int = 15, use_rerank: bool = True):
    """
    query_transform : "none" | "rewrite" | "multi_query" | "step_back"
    top_k           : how many chunks survive to the final answer (tune this — see README)
    fetch_n         : how many candidates hybrid retrieval pulls BEFORE reranking
    use_rerank      : whether the cross-encoder rerank step runs at all
    """
    timings = {}

    # 1. QUERY TRANSFORMATION — may turn one question into several queries.
    t0 = time.time()
    queries = transform_query(q, technique=query_transform)
    timings["query_transform_sec"] = round(time.time() - t0, 3)

    # 2. HYBRID RETRIEVAL — run BM25 + vector search for EVERY query variant,
    #    merge into one candidate pool, keeping each chunk's best score.
    t0 = time.time()
    candidate_pool = {}
    for query_variant in queries:
        results = hybrid_index.hybrid_search(query_variant, k=fetch_n, fetch_n=fetch_n)
        for doc, score in results:
            key = (doc.metadata.get("source"), doc.metadata.get("chunk_index"))
            if key not in candidate_pool or score > candidate_pool[key][1]:
                candidate_pool[key] = (doc, score)
    candidates = sorted(candidate_pool.values(), key=lambda pair: pair[1], reverse=True)[:fetch_n]
    timings["hybrid_retrieval_sec"] = round(time.time() - t0, 3)

    # 3. CROSS-ENCODER RERANK — slower, more accurate, run only on the
    #    already-narrowed candidate pool from step 2.
    t0 = time.time()
    if use_rerank and candidates:
        final_results = rerank(q, candidates, top_k=top_k)
    else:
        final_results = [(doc, score) for doc, score in candidates[:top_k]]
    timings["rerank_sec"] = round(time.time() - t0, 3)

    # GUARDRAIL: nothing survived retrieval at all -> refuse, no LLM call.
    if not final_results:
        return {
            "question": q,
            "answer": "I don't know — no relevant chunks were retrieved for this question.",
            "retrieved_chunks": [],
            "sources": [],
            "timings": timings,
        }

    # 4. CONTEXT ASSEMBLY — dedupe, then pack into a character budget.
    context_text, used_results = assemble_context(final_results)

    retrieved_chunks = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "doc_type": doc.metadata.get("doc_type", "Unknown"),
            "version": doc.metadata.get("version", "Unknown"),
            "score": round(float(score), 4),
            "text_preview": doc.page_content[:150] + ("..." if len(doc.page_content) > 150 else ""),
        }
        for doc, score in used_results
    ]

    user_prompt = (
        f"Question: {q}\n\n"
        f"Retrieved excerpts:\n{context_text}\n\n"
        "Answer using only the excerpts above. Cite each fact as (Source: <filename>)."
    )

    # 5. ANSWER + CITE
    t0 = time.time()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    timings["llm_call_sec"] = round(time.time() - t0, 3)

    return {
        "question": q,
        "query_transform": query_transform,
        "queries_used": queries,
        "top_k": top_k,
        "retrieved_chunks": retrieved_chunks,
        "answer": response.choices[0].message.content,
        "sources": sorted(set(doc.metadata.get("source", "unknown") for doc, _ in used_results)),
        "timings": timings,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
