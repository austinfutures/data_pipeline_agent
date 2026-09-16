"""
Planner Agent.

Takes the raw user query + dataset metadata + retrieved documentation and
produces an ordered list of concrete analysis steps. Keeping planning
separate from code generation means the Coder always works against an
explicit, inspectable plan rather than re-deriving intent from scratch
on every retry.
"""
from __future__ import annotations

import json
import re

from state.schema import PipelineState, PlanStep
from tools.llm_client import LLMClient

PLANNER_SYSTEM_PROMPT = """You are the Planner in a multi-agent data analysis system.
Given a user's analysis request, dataset schema, and relevant documentation notes,
break the request into a short ordered list of concrete steps a Python/Pandas coder
can implement one at a time.

Respond ONLY with a JSON array, no prose, no markdown fences. Each element:
{"step_number": <int>, "description": "<what to do>", "expected_output": "<what this step should produce>"}

Keep it to 1-4 steps. Prefer fewer, well-scoped steps over many trivial ones."""


def _build_user_prompt(state: PipelineState) -> str:
    meta = state["dataset_metadata"]
    docs = "\n".join(f"- {d}" for d in state.get("retrieved_docs", [])) or "(none retrieved)"
    return f"""User query: {state['user_query']}

Dataset shape: {meta['shape']}
Columns: {meta['columns']}
Dtypes: {meta['dtypes']}
Sample rows:
{meta['sample_rows']}

Relevant documentation:
{docs}
"""


def _parse_plan(raw_text: str) -> list[PlanStep]:
    # Strip accidental code fences defensively; some models add them despite instructions.
    cleaned = re.sub(r"^```(json)?|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(cleaned)
        return [
            PlanStep(
                step_number=item.get("step_number", i + 1),
                description=item["description"],
                expected_output=item.get("expected_output", ""),
            )
            for i, item in enumerate(data)
        ]
    except (json.JSONDecodeError, KeyError, TypeError):
        # Fallback: treat the whole query as a single step rather than failing the run.
        return [PlanStep(step_number=1, description=cleaned or "Analyze the data per the user query.", expected_output="Answer to the user's query.")]


def make_planner_node(llm: LLMClient):
    """Returns a LangGraph node function closing over the given LLM client."""

    def planner_node(state: PipelineState) -> dict:
        raw = llm.generate(PLANNER_SYSTEM_PROMPT, _build_user_prompt(state))
        plan = _parse_plan(raw)
        return {
            "plan": plan,
            "current_step_index": 0,
            "trace": [f"[planner] produced {len(plan)}-step plan"],
        }

    return planner_node
