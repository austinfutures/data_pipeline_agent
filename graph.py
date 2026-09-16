"""
Graph assembly: wires Retriever -> Planner -> Coder -> Executor -> Critic
into a LangGraph StateGraph with self-correction routing.

Flow:
    retriever -> planner -> coder -> executor --(runtime_error)--> coder   [retry]
                                          |--(unsafe_rejected)--> END (unsafe_terminated)
                                          |--(success)--> critic --(rejected)--> coder [retry]
                                                              |--(approved)--> advance_step
                                                                                   |--(more steps)--> coder
                                                                                   |--(done)--> finalize -> END
    retries exhausted at any retry point -> finalize (status=failed) -> END
"""
from __future__ import annotations

import pandas as pd
from langgraph.graph import StateGraph, END

from state.schema import PipelineState
from tools.llm_client import LLMClient
from tools.vector_search import MockVectorStore
from agents.retriever import make_retriever_node
from agents.planner import make_planner_node
from agents.coder import make_coder_node
from agents.executor import make_executor_node
from agents.critic import make_critic_node


def _advance_step_node(state: PipelineState) -> dict:
    """Move to the next plan step, resetting per-step retry/critic state."""
    next_index = state["current_step_index"] + 1
    return {
        "current_step_index": next_index,
        "retry_count": 0,
        "critic_verdict": None,
        "critic_feedback": None,
        "trace": [f"[graph] advancing to step index {next_index}"],
    }


def _increment_retry_node(state: PipelineState) -> dict:
    return {"retry_count": state["retry_count"] + 1}


def _finalize_node(state: PipelineState) -> dict:
    if state["status"] == "unsafe_terminated":
        answer = "Execution halted: generated code failed the safety check and was not run."
    elif state["retry_count"] >= state["max_retries"] and state.get("critic_verdict") != "approved":
        answer = "Analysis failed after exhausting retries. See execution_history for details."
        return {"status": "failed", "final_answer": answer, "trace": ["[graph] finalized as FAILED"]}
    else:
        last_success = next((e for e in reversed(state["execution_history"]) if e["status"] == "success"), None)
        answer = last_success["stdout"] if last_success else "No output produced."
    return {
        "status": "success" if state["status"] != "unsafe_terminated" else state["status"],
        "final_answer": answer,
        "trace": ["[graph] finalized"],
    }


def _route_after_executor(state: PipelineState) -> str:
    last = state["execution_history"][-1]
    if last["status"] == "unsafe_rejected":
        return "finalize"
    if last["status"] == "runtime_error":
        if state["retry_count"] + 1 >= state["max_retries"]:
            return "finalize"
        return "retry_to_coder"
    return "critic"  # success


def _route_after_critic(state: PipelineState) -> str:
    if state["critic_verdict"] == "approved":
        if state["current_step_index"] + 1 >= len(state["plan"]):
            return "finalize"
        return "advance_step"
    # rejected
    if state["retry_count"] + 1 >= state["max_retries"]:
        return "finalize"
    return "retry_to_coder"


def build_graph(llm: LLMClient, dataframe: pd.DataFrame, vector_store: MockVectorStore | None = None):
    store = vector_store or MockVectorStore()
    store.add_column_docs(list(dataframe.columns))

    graph = StateGraph(PipelineState)

    graph.add_node("retriever", make_retriever_node(store))
    graph.add_node("planner", make_planner_node(llm))
    graph.add_node("coder", make_coder_node(llm))
    graph.add_node("executor", make_executor_node(dataframe))
    graph.add_node("critic", make_critic_node(llm))
    graph.add_node("increment_retry", _increment_retry_node)
    graph.add_node("advance_step", _advance_step_node)
    graph.add_node("finalize", _finalize_node)

    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "planner")
    graph.add_edge("planner", "coder")
    graph.add_edge("coder", "executor")

    graph.add_conditional_edges(
        "executor",
        _route_after_executor,
        {"finalize": "finalize", "retry_to_coder": "increment_retry", "critic": "critic"},
    )
    graph.add_conditional_edges(
        "critic",
        _route_after_critic,
        {"finalize": "finalize", "retry_to_coder": "increment_retry", "advance_step": "advance_step"},
    )
    graph.add_edge("increment_retry", "coder")
    graph.add_edge("advance_step", "coder")
    graph.add_edge("finalize", END)

    return graph.compile()


def make_initial_state(user_query: str, dataframe: pd.DataFrame, max_retries: int = 3) -> PipelineState:
    return PipelineState(
        user_query=user_query,
        dataset_metadata={
            "shape": dataframe.shape,
            "columns": list(dataframe.columns),
            "dtypes": {c: str(t) for c, t in dataframe.dtypes.items()},
            "sample_rows": dataframe.head().to_string(),
        },
        plan=[],
        current_step_index=0,
        retrieved_docs=[],
        current_code="",
        critic_verdict=None,
        critic_feedback=None,
        execution_history=[],
        artifacts=[],
        retry_count=0,
        max_retries=max_retries,
        status="running",
        final_answer=None,
        trace=[],
    )
