"""
Shared execution state for the multi-agent data pipeline.

This TypedDict is the single object that flows through every node in the
LangGraph state graph. Each agent reads what it needs and writes back a
partial update; LangGraph merges these according to the reducers defined
below (mostly "append to list" for history-like fields).
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, Optional, TypedDict


class DatasetMetadata(TypedDict):
    shape: tuple[int, int]
    columns: list[str]
    dtypes: dict[str, str]
    sample_rows: str  # small head() preview, kept short deliberately


class ExecutionAttempt(TypedDict):
    iteration: int
    agent: str            # which agent produced this artifact ("coder", "critic", ...)
    code: str
    stdout: str
    stderr: str
    status: Literal["success", "runtime_error", "validation_failed", "unsafe_rejected"]
    error_message: Optional[str]


class Artifact(TypedDict):
    kind: Literal["table", "chart", "text"]
    name: str
    # For tables: a small serialized preview (CSV/markdown string), not the raw DataFrame,
    # so the state dict stays JSON-serializable and checkpointable.
    preview: str
    iteration_created: int


class PlanStep(TypedDict):
    step_number: int
    description: str
    expected_output: str  # what the coder should produce for this step


class PipelineState(TypedDict):
    # --- Input ---
    user_query: str
    dataset_metadata: DatasetMetadata

    # --- Planner output ---
    plan: list[PlanStep]
    current_step_index: int

    # --- Retrieval context (from vector search tool) ---
    retrieved_docs: list[str]

    # --- Coder output (current attempt, overwritten each retry) ---
    current_code: str

    # --- Critic verdict for the current attempt ---
    critic_verdict: Optional[Literal["approved", "rejected"]]
    critic_feedback: Optional[str]

    # --- Accumulating history (append-only via reducer) ---
    execution_history: Annotated[list[ExecutionAttempt], operator.add]
    artifacts: Annotated[list[Artifact], operator.add]

    # --- Retry / control flow ---
    retry_count: int
    max_retries: int

    # --- Terminal outcome ---
    status: Literal["running", "success", "failed", "unsafe_terminated"]
    final_answer: Optional[str]

    # --- Misc ---
    trace: Annotated[list[str], operator.add]  # human-readable step log
