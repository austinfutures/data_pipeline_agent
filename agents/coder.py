"""
Coder Agent.

Generates Pandas code for the current plan step. On retries, it receives
the previous attempt's code plus *why it failed* (sandbox traceback, or
critic rejection reason) so correction is targeted rather than a blind
regeneration.
"""
from __future__ import annotations

import re

from state.schema import PipelineState
from tools.llm_client import LLMClient

CODER_SYSTEM_PROMPT = """You are the Coder in a multi-agent data analysis system.
Write Python/Pandas code to accomplish ONE plan step against an in-memory DataFrame `df`.

RULES:
- `df`, `pd`, and `np` are already available. Do NOT import anything else.
- Do NOT read/write files, use `open`, `eval`, `exec`, `os`, `sys`, or network calls.
- Assign the step's final answer to a variable named `result` and print it.
- Wrap the code in <code>...</code> tags. No prose outside the tags.
- Keep it focused: implement only the current step, not the whole plan."""


def _build_user_prompt(state: PipelineState) -> str:
    step = state["plan"][state["current_step_index"]]
    meta = state["dataset_metadata"]

    feedback_block = ""
    if state.get("critic_feedback"):
        feedback_block += f"\nCRITIC FEEDBACK ON PREVIOUS ATTEMPT:\n{state['critic_feedback']}\n"

    last_error = None
    for entry in reversed(state.get("execution_history", [])):
        if entry["status"] in ("runtime_error", "unsafe_rejected") and entry["iteration"] == state["retry_count"]:
            last_error = entry
            break
    if last_error:
        feedback_block += f"\nPREVIOUS CODE:\n{last_error['code']}\n\nRUNTIME ERROR:\n{last_error['error_message']}\n"

    return f"""Plan step {step['step_number']}: {step['description']}
Expected output: {step['expected_output']}

Dataset columns: {meta['columns']}
Dtypes: {meta['dtypes']}
{feedback_block}
Write corrected/new code for this step now."""


def _extract_code(raw_text: str) -> str:
    match = re.search(r"<code>(.*?)</code>", raw_text, re.DOTALL)
    code = match.group(1).strip() if match else raw_text.strip()
    if code.startswith("python"):
        code = code[6:].strip()
    return code


def make_coder_node(llm: LLMClient):
    def coder_node(state: PipelineState) -> dict:
        raw = llm.generate(CODER_SYSTEM_PROMPT, _build_user_prompt(state))
        code = _extract_code(raw)
        step_desc = state["plan"][state["current_step_index"]]["description"]
        return {
            "current_code": code,
            "trace": [f"[coder] generated code for step '{step_desc}' (attempt {state['retry_count'] + 1})"],
        }

    return coder_node
