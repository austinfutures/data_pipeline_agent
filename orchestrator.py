"""
Top-level entry point: MultiAgentDataPipeline.

Keeps a familiar surface (load_data / run_query) similar to the original
single-agent script, but backed by the LangGraph state machine.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from graph import build_graph, make_initial_state
from tools.llm_client import GeminiClient, LLMClient
from tools.vector_search import MockVectorStore


class MultiAgentDataPipeline:
    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        max_retries: int = 3,
    ):
        self.llm: LLMClient = llm or GeminiClient(api_key=api_key, model=model)
        self.dataframe: Optional[pd.DataFrame] = None
        self.max_retries = max_retries
        self.vector_store = MockVectorStore()
        self._compiled_graph = None

    def load_data(self, data_source: str | pd.DataFrame) -> None:
        self.dataframe = data_source.copy() if isinstance(data_source, pd.DataFrame) else pd.read_csv(data_source)
        self._compiled_graph = build_graph(self.llm, self.dataframe, self.vector_store)
        print(f"✓ Loaded data with shape: {self.dataframe.shape}")
        print(f"✓ Columns: {list(self.dataframe.columns)}\n")

    def run_query(self, user_query: str, verbose: bool = True) -> dict:
        if self.dataframe is None or self._compiled_graph is None:
            raise ValueError("No data loaded. Call load_data() first.")

        initial_state = make_initial_state(user_query, self.dataframe, self.max_retries)
        final_state = self._compiled_graph.invoke(initial_state)

        if verbose:
            print("=" * 60)
            for line in final_state["trace"]:
                print(line)
            print("=" * 60)
            print(f"Status: {final_state['status']}")
            print(f"Answer:\n{final_state['final_answer']}")

        return final_state
