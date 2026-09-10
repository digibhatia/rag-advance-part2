"""
metadata_enrichment.py

Samiksha — Metadata Enrichment: doc_type, version, effective_date, source
=============================================================================
Chunk metadata in Day 5's Disha lab only carried `source` and
`chunk_index`. This module adds three more fields, extracted once per
source document, then stamped onto every chunk that document produces:

  doc_type        -- what kind of document this is (SOP, SLA, FAQ, log, spec)
  version         -- the document's stated version, if it has one
  effective_date  -- the document's stated effective date, if it has one

Not every format carries these fields the same way -- that's realistic,
not a bug. PDF/Word/HTML headers state Version and Effective Date in
plain text; JSON states them as real keys; the CSV incident log has no
"version" concept at all, so we derive a stand-in (the most recent
incident timestamp) instead of leaving the field blank.
"""
import re
import csv
import json

# Matches: "... Version: 3.2  |  Effective Date: 15 March 2026 | ..."
# Works whether Version/Effective Date are adjacent or separated by other
# fields, as long as Version appears before Effective Date on the line.
HEADER_PATTERN = re.compile(
    r"Version:\s*([\w.]+).*?Effective Date:\s*([^|]+?)(?:\s*\||\s*$)",
    re.IGNORECASE | re.DOTALL,
)


def infer_doc_type(filename: str) -> str:
    """Guess a human-readable document type from the filename — a simple
    stand-in for a real document classifier, good enough for this lab."""
    name = filename.lower()
    if "sop" in name:
        return "SOP"
    if "sla" in name:
        return "SLA"
    if "faq" in name:
        return "FAQ"
    if "log" in name or "incident" in name:
        return "Incident Log"
    if "spec" in name or "vendor" in name:
        return "Equipment Spec"
    return "Unknown"


def enrich_from_text(text: str, filename: str) -> dict:
    """For PDF / Word / HTML sources: look for a 'Version: X ... Effective
    Date: Y' style header line, common to every document template used
    in this lab."""
    match = HEADER_PATTERN.search(text)
    version = match.group(1).strip() if match else "Unknown"
    effective_date = match.group(2).strip() if match else "Unknown"
    return {
        "doc_type": infer_doc_type(filename),
        "version": version,
        "effective_date": effective_date,
    }


def enrich_from_json(filepath: str) -> dict:
    """JSON documents state these fields as real keys — no regex needed."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "doc_type": infer_doc_type(filepath.split("/")[-1]),
        "version": data.get("version", "Unknown"),
        "effective_date": data.get("effective_date", "Unknown"),
    }


def enrich_from_csv(filepath: str) -> dict:
    """CSV has no document-level version. We derive a stand-in
    'effective_date' from the most recent timestamp in the data itself —
    a realistic alternative for formats with no explicit versioning."""
    latest_timestamp = None
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = row.get("timestamp")
            if ts and (latest_timestamp is None or ts > latest_timestamp):
                latest_timestamp = ts
    return {
        "doc_type": infer_doc_type(filepath.split("/")[-1]),
        "version": "N/A (no document-level version; derived from data)",
        "effective_date": latest_timestamp or "Unknown",
    }


def enrich_document(filepath: str, sample_text: str = "") -> dict:
    """Single entry point — dispatches by file extension.
    `sample_text` is only needed for pdf/docx/html (pass the first loaded
    unit's text); json and csv read the file directly."""
    filename = filepath.split("/")[-1]
    ext = filepath.rsplit(".", 1)[-1].lower()

    if ext == "json":
        return enrich_from_json(filepath)
    if ext == "csv":
        return enrich_from_csv(filepath)
    return enrich_from_text(sample_text, filename)
