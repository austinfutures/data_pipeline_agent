"""
Critic / Validator Agent.

Runs only after the sandbox reports a *successful* execution — its job is
not to catch crashes (the executor already does that) but to judge
whether the output actually answers the plan step: right shape of
answer, sane values, no silent no-op (e.g. code ran but printed nothing
useful), consistent with retrieved documentation caveats.
"""
from __future__ import annotations

import re

from state.schema import PipelineState
from tools.llm_client import LLMClient

CRITIC_SYSTEM_PROMPT = """You are the Critic/Validator in a multi-agent data analysis system.
You receive a plan step, the code that was run, and its printed output.
Judge whether the output actually satisfies the step's expected_output.

Check for:
- Empty or clearly wrong output (e.g. all NaN, empty dataframe, obviously wrong types)
- Output that ignores documented data caveats (e.g. not deduplicating when docs say to)
- Silent no-ops (code executed but produced no meaningful result)

Respond with EXACTLY one of these two formats, nothing else:
APPROVED
or
REJECTED: <one sentence reason>"""


def _build_user_prompt(state: PipelineState) -> str:
    step = state["plan"][state["current_step_index"]]
    last_success = next(
        (e for e in reversed(state["execution_history"]) if e["status"] == "success"), None
    )
    docs = "\n".join(f"- {d}" for d in state.get("retrieved_docs", [])) or "(none)"
    return f"""Plan step: {step['description']}
Expected output: {step['expected_output']}

Documentation caveats to check against:
{docs}

Code that was run:
{last_success['code'] if last_success else '(none)'}

Printed output:
{last_success['stdout'] if last_success else '(none)'}
"""


def make_critic_node(llm: LLMClient):
    def critic_node(state: PipelineState) -> dict:
        raw = llm.generate(CRITIC_SYSTEM_PROMPT, _build_user_prompt(state)).strip()

        if raw.upper().startswith("APPROVED"):
            return {
                "critic_verdict": "approved",
                "critic_feedback": None,
                "trace": ["[critic] approved"],
            }

        match = re.search(r"REJECTED:?\s*(.*)", raw, re.IGNORECASE | re.DOTALL)
        reason = match.group(1).strip() if match else raw
        return {
            "critic_verdict": "rejected",
            "critic_feedback": reason,
            "trace": [f"[critic] rejected: {reason}"],
        }

    return critic_node
