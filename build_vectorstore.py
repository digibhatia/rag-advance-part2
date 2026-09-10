"""
build_vectorstore.py

Samiksha — Build the Knowledge Base (Step 1 of 3)
=====================================================
Same ingest -> chunk -> embed -> index flow as Day 5's Disha lab, with
metadata enrichment added: every chunk is now stamped with doc_type,
version, and effective_date, in addition to source and chunk_index.

Run:
    python build_vectorstore.py
"""
import glob
import os
import shutil

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from loaders import load_document
from chunking import chunk_text
from pii import redact_pii
from metadata_enrichment import enrich_document

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "samiksha_documents"
PERSIST_DIR = "./chroma_store"


def build():
    print("=" * 64)
    print("Samiksha — Building the Knowledge Base (with metadata enrichment)")
    print("=" * 64)

    filepaths = sorted(glob.glob("documents/*"))
    if not filepaths:
        print("No documents found in documents/ — nothing to build.")
        return

    print(f"\nFound {len(filepaths)} source document(s):")
    for f in filepaths:
        print(f"  - {os.path.basename(f)}")

    all_documents = []
    total_pii_redactions = {}

    for filepath in filepaths:
        filename = os.path.basename(filepath)
        print(f"\nProcessing {filename} ...")

        raw_units = load_document(filepath)
        print(f"  Loaded as {len(raw_units)} source unit(s)")

        # Enrich once per file. For pdf/docx/html we pass the first unit's
        # text as a sample to search for the "Version: X | Effective
        # Date: Y" header line; json/csv read the file directly instead.
        sample_text = raw_units[0]["text"] if raw_units else ""
        doc_metadata = enrich_document(filepath, sample_text=sample_text)
        print(f"  Enriched metadata: {doc_metadata}")

        file_chunk_count = 0
        for unit in raw_units:
            redacted_text, pii_counts = redact_pii(unit["text"])
            for label, count in pii_counts.items():
                total_pii_redactions[label] = total_pii_redactions.get(label, 0) + count

            chunks = chunk_text(redacted_text)
            for i, chunk in enumerate(chunks):
                # Start from this unit's own metadata (e.g. page number,
                # or row_id for CSV), then layer the document-level
                # enrichment on top, then set the chunk's own index.
                metadata = dict(unit["metadata"])
                metadata.update(doc_metadata)
                metadata["chunk_index"] = i
                all_documents.append(Document(page_content=chunk, metadata=metadata))
                file_chunk_count += 1

        print(f"  -> {file_chunk_count} chunk(s) from this file")

    print(f"\nTotal chunks to embed: {len(all_documents)}")
    if total_pii_redactions:
        print(f"PII redacted during ingestion: {total_pii_redactions}")
    else:
        print("No PII patterns matched during ingestion.")

    print("\nEmbedding and storing in ChromaDB (this calls the OpenAI API)...")
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)

    Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )

    print("\n" + "=" * 64)
    print(f"Done. {len(all_documents)} chunks stored in {PERSIST_DIR}")
    print("Next step: start app.py (see README.md)")
    print("=" * 64)


if __name__ == "__main__":
    build()
