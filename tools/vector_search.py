"""
Mock Vector Search tool for dataset documentation / data dictionaries.

In production this would embed a query and do similarity search against a
real vector store (pgvector, Chroma, Pinecone, ...) holding column
descriptions, business glossaries, known data-quality caveats, etc. Here
it's a small in-memory keyword-overlap retriever so the graph's retrieval
node has something real to call and the interface is easy to swap out
later without touching any agent code.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class DocChunk:
    doc_id: str
    text: str
    tags: set[str]


class MockVectorStore:
    """Keyword-overlap 'vector search' stand-in. Swap `.search()`'s body
    for a real embedding + ANN lookup without changing the call site."""

    def __init__(self, corpus: list[DocChunk] | None = None):
        self.corpus = corpus or self._default_corpus()

    @staticmethod
    def _default_corpus() -> list[DocChunk]:
        return [
            DocChunk(
                doc_id="dd-001",
                text="Column 'revenue' is reported in USD, already net of refunds. "
                     "Values are NULL for rows created before 2021 migration.",
                tags={"revenue", "usd", "null", "migration"},
            ),
            DocChunk(
                doc_id="dd-002",
                text="Column 'region' uses ISO-3166 alpha-2 country codes, not full "
                     "region names. 'UNK' indicates unresolved geolocation.",
                tags={"region", "country", "geo", "unk"},
            ),
            DocChunk(
                doc_id="dd-003",
                text="Duplicate rows can occur for 'order_id' due to upstream retry "
                     "logic; always deduplicate on (order_id, updated_at) keeping latest.",
                tags={"duplicate", "order_id", "dedupe"},
            ),
            DocChunk(
                doc_id="dd-004",
                text="Date columns are stored as UTC. Convert to local time zone before "
                     "any day-of-week or business-hours analysis.",
                tags={"date", "utc", "timezone"},
            ),
        ]

    def search(self, query: str, top_k: int = 3) -> list[DocChunk]:
        """Naive overlap scoring. Real implementation: embed(query), ANN search."""
        query_terms = set(re.findall(r"[a-z0-9_]+", query.lower()))
        scored = []
        for chunk in self.corpus:
            overlap = len(query_terms & (chunk.tags | set(re.findall(r"[a-z0-9_]+", chunk.text.lower()))))
            if overlap > 0:
                scored.append((overlap, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]

    def add_column_docs(self, dataframe_columns: list[str]) -> None:
        """Convenience: auto-register a generic doc per column so retrieval
        always has *something* relevant even without curated docs."""
        for col in dataframe_columns:
            self.corpus.append(
                DocChunk(doc_id=f"auto-{col}", text=f"Column '{col}' is present in the dataset.", tags={col.lower()})
            )
