"""
hybrid_retrieval.py

Samiksha — Hybrid Retrieval: BM25 + Vector Search
=====================================================
Day 1 (Sheet 3) covered why vector search beats keyword search for
meaning-based questions. But keyword search still wins in one specific
case: exact terms that matter precisely BECAUSE they're exact — a part
number, an error code, an incident ID, a clause number. "INC-10235"
should match "INC-10235" exactly; a pure vector search might rank a
semantically-similar-but-wrong incident just as high.

Hybrid retrieval runs BOTH searches and combines their rankings using
Reciprocal Rank Fusion (RRF) — a simple, well-established fusion method:
for each result, take 1 / (k + its rank position) in EACH ranking, sum
those two contributions, then sort by the combined score. A document
that ranks well in both searches wins; a document that ranks well in
only one still gets partial credit.
"""
from rank_bm25 import BM25Okapi


class HybridIndex:
    """Wraps a Chroma vectorstore with a parallel BM25 keyword index over
    the SAME chunks, so both can be queried and fused together."""

    def __init__(self, vectorstore, all_documents: list):
        """
        vectorstore: a LangChain Chroma vectorstore (already built and populated)
        all_documents: the same list of langchain_core.documents.Document
                        objects that were embedded into vectorstore — needed
                        so BM25 can be built over identical chunks/ids
        """
        self.vectorstore = vectorstore
        self.documents = all_documents
        tokenized = [doc.page_content.lower().split() for doc in all_documents]
        self.bm25 = BM25Okapi(tokenized)

    def _bm25_rank(self, query: str, top_n: int) -> list:
        """Returns the top_n Document objects ranked by BM25 score, best first."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.documents[i] for i in ranked_indices[:top_n]]

    def _vector_rank(self, query: str, top_n: int) -> list:
        """Returns the top_n Document objects ranked by vector similarity, best first."""
        return self.vectorstore.similarity_search(query, k=top_n)

    def hybrid_search(self, query: str, k: int = 5, fetch_n: int = 15, rrf_k: int = 60) -> list:
        """Run both searches, fuse with Reciprocal Rank Fusion, return the
        top k documents by combined score.

        rrf_k is RRF's own smoothing constant (60 is the commonly used
        default in the literature) — it controls how quickly a result's
        contribution shrinks as its rank gets worse."""
        bm25_results = self._bm25_rank(query, fetch_n)
        vector_results = self._vector_rank(query, fetch_n)

        scores = {}
        docs_by_key = {}

        def doc_key(doc):
            # Chunks are uniquely identified by source + chunk_index.
            return (doc.metadata.get("source"), doc.metadata.get("chunk_index"))

        for rank, doc in enumerate(bm25_results):
            key = doc_key(doc)
            scores[key] = scores.get(key, 0) + 1 / (rrf_k + rank + 1)
            docs_by_key[key] = doc

        for rank, doc in enumerate(vector_results):
            key = doc_key(doc)
            scores[key] = scores.get(key, 0) + 1 / (rrf_k + rank + 1)
            docs_by_key[key] = doc

        ranked_keys = sorted(scores.keys(), key=lambda key: scores[key], reverse=True)
        return [(docs_by_key[key], scores[key]) for key in ranked_keys[:k]]
