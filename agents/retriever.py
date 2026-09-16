"""
Retrieval node. Not an LLM agent — a direct tool call to the (mock) vector
store, run before planning so the Planner and Coder both get grounded
context about the dataset's known quirks/conventions.
"""
from __future__ import annotations

from state.schema import PipelineState
from tools.vector_search import MockVectorStore


def make_retriever_node(store: MockVectorStore):
    def retriever_node(state: PipelineState) -> dict:
        docs = store.search(state["user_query"], top_k=3)
        return {
            "retrieved_docs": [d.text for d in docs],
            "trace": [f"[retriever] found {len(docs)} relevant doc chunk(s)"],
        }

    return retriever_node
