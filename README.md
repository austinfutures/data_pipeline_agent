# Multi-Agent Autonomous Data Pipeline

Upgrade of a single-agent "generate Pandas code, `exec()` it, retry on
traceback" loop into a multi-agent LangGraph pipeline with real sandboxing,
retrieval, and a separate validation step.

## What changed from the original, and why

| Original | Here | Why |
|---|---|---|
| One agent does plan+code+judge output itself | Planner / Coder / Critic are separate nodes | A single LLM call rationalizing its own output is a weak validator; splitting the roles means the Critic sees only the step's contract and the result, not the reasoning that produced it |
| `exec(code, local_scope)` in-process | Subprocess sandbox: AST allowlist + restricted builtins + CPU/memory/wall-clock limits | The original gives generated code full access to the host process (filesystem, network, arbitrary imports). This is the most important fix in this upgrade — see `tools/sandbox.py` |
| Retry loop just resends the traceback | Retry passes traceback **and**, if applicable, Critic feedback, and is capped and routed by explicit graph edges | Makes the corrective signal specific instead of "try again," and makes the stop condition explicit state rather than a `for` loop counter |
| No context beyond `df.head()` | Mock vector-search tool retrieves data-dictionary notes (units, known dedup issues, timezone caveats) before planning | Keeps generated code from silently violating documented data-quality rules |
| No safety testing | `tests/eval_pipeline.py` asserts 7 known sandbox-escape patterns are blocked at all 3 layers (static scan, sandbox, full graph) | Safety claims are worth nothing until something actually asserts them |

## Architecture

```
retriever -> planner -> coder -> executor --(runtime error, retries left)--> coder
                                     |--(unsafe code)--> finalize [unsafe_terminated]
                                     |--(success)--> critic --(rejected, retries left)--> coder
                                                          |--(approved, more steps)--> coder (next step)
                                                          |--(approved, done)--> finalize [success]
              (retries exhausted at either point) -------------------------------> finalize [failed]
```

- **`state/schema.py`** — the `PipelineState` TypedDict threaded through every
  node. List fields (`execution_history`, `artifacts`, `trace`) use an
  `operator.add` reducer so nodes append rather than overwrite.
- **`tools/sandbox.py`** — the safety-critical module. Two independent layers:
  1. **Static**: AST walk rejecting disallowed imports (`os`, `sys`,
     `subprocess`, `socket`, `importlib`, ...), dunder/escape-hatch names
     (`__subclasses__`, `__globals__`, `eval`, `exec`, `open`, ...).
  2. **Runtime**: even if something got past the static layer, code runs in a
     **separate subprocess** with a restricted `__builtins__` (no `open`,
     `import`, `getattr`, etc. exposed at all), plus `RLIMIT_CPU` and
     `RLIMIT_AS` resource limits and a wall-clock `subprocess` timeout — this
     is what catches non-import-based abuse like infinite loops or memory
     bombs that the AST layer can't see.
  - Documented deployment note at the bottom of the file: for a real
    multi-tenant / internet-facing service, put this behind an OS-level
    sandbox too (container with dropped capabilities + seccomp, or a
    microVM). This module is defense-in-depth for a trusted internal tool,
    not a hard security boundary on its own.
- **`tools/vector_search.py`** — mock keyword-overlap retriever standing in
  for a real embedding store; swap `.search()`'s internals for pgvector/
  Chroma/Pinecone without touching any agent code.
- **`tools/llm_client.py`** — `GeminiClient` wraps the real API;
  `FakeLLMClient` returns scripted responses in order, which is what lets
  `tests/eval_pipeline.py` and the retry-logic test run deterministically
  with no API key and no network.
- **`agents/`** — one file per node: `retriever.py` (tool call, no LLM),
  `planner.py`, `coder.py`, `executor.py` (tool call, no LLM), `critic.py`.
- **`graph.py`** — builds the `StateGraph`, including the conditional-edge
  routing functions that implement the retry cap and step advancement.
- **`orchestrator.py`** — `MultiAgentDataPipeline`, a thin façade
  (`load_data` / `run_query`) over the compiled graph, for a familiar
  call site.

## Usage

```python
from orchestrator import MultiAgentDataPipeline

pipeline = MultiAgentDataPipeline(api_key="...", model="gemini-2.5-flash")
pipeline.load_data("sales.csv")
final_state = pipeline.run_query("What's the month-over-month revenue trend by region?")

print(final_state["status"])        # "success" | "failed" | "unsafe_terminated"
print(final_state["final_answer"])
print(final_state["execution_history"])  # every attempt, including failed ones
```

## Running the evaluation suite

```bash
pip install -r requirements.txt
python -m tests.eval_pipeline
```

No API key needed — it uses `FakeLLMClient` with scripted planner/coder/critic
responses. It asserts:

1. **Functional cases** — two scripted query/response pairs reach
   `status == "success"`.
2. **Safety cases** — seven known sandbox-escape code samples (file read,
   `os` import, `importlib`-mediated `os` access, `eval`-based escape, dunder
   `__subclasses__` traversal, raw socket, `subprocess`) are each asserted
   blocked at all three checkpoints: the static AST scan directly, the
   sandbox's end-to-end `run_in_sandbox`, and the full graph reaching
   `status == "unsafe_terminated"` (not silently swallowed as a generic
   failure).
3. **Retry case** — a deliberately broken first attempt (typo'd column name)
   is asserted to produce exactly one `runtime_error` history entry before
   the scripted corrected code succeeds, proving the coder↔executor retry
   edge actually engages instead of just being reachable in principle.

Separately (not part of the scripted eval, since it takes real wall-clock
time), `tools/sandbox.py`'s resource limits can be exercised directly against
an infinite loop or a memory-bomb script to confirm the *runtime* layer — not
just the AST layer — independently kills unsafe code that no import-based
static check could ever catch.

## Known limitations / things to harden further for production

- The subprocess sandbox is not a substitute for OS-level isolation
  (containers/seccomp/microVM) if this is ever exposed multi-tenant or to
  untrusted users — see the note in `tools/sandbox.py`.
- `MockVectorStore` is keyword-overlap, not embeddings; swap in a real vector
  DB when you have actual data-dictionary content to index.
- `PipelineState` is JSON-serializable by design (DataFrames are never put in
  state, only string previews) so it can be checkpointed with LangGraph's
  persistence layer if you want resumable runs — this repo doesn't wire up a
  checkpointer, but the state shape is ready for one.
- Chart/table artifact capture (`Artifact` in `state/schema.py`) is modeled in
  the schema but the Coder prompt doesn't yet instruct the model to register
  artifacts — wire that up if you need chart outputs, not just printed text.
