"""
Executor node. Not an LLM call — invokes the sandbox tool on the Coder's
current code and records the outcome in execution_history. This is what
the graph's conditional edges branch on (success -> critic, failure ->
back to coder or terminate).
"""
from __future__ import annotations

import pandas as pd

from state.schema import PipelineState
from tools.sandbox import run_in_sandbox


def make_executor_node(dataframe: pd.DataFrame):
    def executor_node(state: PipelineState) -> dict:
        code = state["current_code"]
        outcome = run_in_sandbox(code, dataframe)

        if outcome.status == "unsafe_rejected":
            entry = {
                "iteration": state["retry_count"],
                "agent": "executor",
                "code": code,
                "stdout": "",
                "stderr": "",
                "status": "unsafe_rejected",
                "error_message": "; ".join(outcome.violations),
            }
            return {
                "execution_history": [entry],
                "status": "unsafe_terminated",
                "trace": [f"[executor] REJECTED unsafe code: {'; '.join(outcome.violations)}"],
            }

        if outcome.status in ("runtime_error", "timeout"):
            entry = {
                "iteration": state["retry_count"],
                "agent": "executor",
                "code": code,
                "stdout": outcome.stdout,
                "stderr": outcome.error_message,
                "status": "runtime_error",
                "error_message": outcome.error_message,
            }
            return {
                "execution_history": [entry],
                "trace": [f"[executor] runtime error: {outcome.error_message}"],
            }

        # success
        entry = {
            "iteration": state["retry_count"],
            "agent": "executor",
            "code": code,
            "stdout": outcome.stdout,
            "stderr": "",
            "status": "success",
            "error_message": None,
        }
        return {
            "execution_history": [entry],
            "trace": [f"[executor] code ran successfully"],
        }

    return executor_node
