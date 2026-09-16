# Multi-Agent Autonomous Data Pipeline

A multi-agent data analysis system built on LangGraph. Given a natural-language
query and a dataset, it plans an approach, generates Pandas code, executes
that code in an isolated sandbox, and validates the result — retrying with
targeted feedback when something fails, and refusing to run anything that
looks unsafe.

## Why multi-agent instead of one agent doing everything

Planner, Coder, and Critic are separate nodes in a state graph rather than
one LLM call that plans, codes, and judges its own output.

- **Planner** breaks the user's query into concrete steps before any code is
  written, so the Coder always works against an explicit, inspectable plan
  instead of re-deriving intent from scratch on every retry.
- **Coder** implements one step at a time and only that step.
- **Critic** reviews the Coder's output against the step's stated
  expected-output — it never sees the Coder's reasoning, only the code and
  the result, which makes it a meaningfully independent check rather than
  the same model re-confirming its own answer.

Separating these roles is deliberate: a single model asked to grade its own
work is a weak validator, and splitting plan/act/judge into distinct graph
nodes with their own prompts and their own view of the state is what makes
self-correction possible at all.

## Architecture

```
retriever -> planner -> coder -> executor --(runtime error, retries left)--> coder
                                     |--(unsafe code)--> finalize [unsafe_terminated]
                                     |--(success)--> critic --(rejected, retries left)--> coder
                                                          |--(approved, more steps)--> coder (next step)
                                                          |--(approved, done)--> finalize [success]
              (retries exhausted at either point) -------------------------------> finalize [failed]
```

### State management

`state/schema.py` defines `PipelineState`, a single TypedDict threaded
through every node in the graph. Each node reads what it needs and returns a
partial update; LangGraph merges these using reducers. List-shaped fields —
`execution_history`, `artifacts`, `trace` — use an `operator.add` reducer so
nodes append to a running record instead of overwriting it, which means the
full history of every attempt (successful or not) survives to the end of the
run. Scalar fields like `retry_count` and `current_step_index` are plain
overwrites, since only one node at a time is responsible for advancing them.
DataFrames themselves are never stored in state — only shape/dtype metadata
and small string previews — which keeps the state JSON-serializable and
ready to checkpoint if resumable runs are added later.

### Tools and retrieval

- **Sandbox execution tool** (`tools/sandbox.py`) — the safety-critical
  component. Two independent layers:
  1. **Static AST scan**: parses the Coder's generated code and rejects it
     before execution if it imports disallowed modules (`os`, `sys`,
     `subprocess`, `socket`, `importlib`, ...) or references known
     escape-hatch names (`eval`, `exec`, `open`, `__subclasses__`,
     `__globals__`, `getattr`, ...).
  2. **Runtime isolation**: code that passes the static check still runs in
     a separate subprocess with a restricted `__builtins__` (no `open`,
     `import`, or introspection builtins exposed at all), CPU-time and
     memory limits via `resource.setrlimit` on Linux/Mac (skipped
     gracefully on Windows, where `resource` doesn't exist — a wall-clock
     `subprocess` timeout still applies on every platform). This layer is
     what catches things the AST scan structurally cannot, like an
     infinite loop or a memory-exhausting allocation that involves no
     disallowed names at all.

  This is defense-in-depth for a trusted internal tool, not a hard
  multi-tenant security boundary — the module's docstring spells out what
  a production, internet-facing deployment would need on top (containers
  with dropped capabilities and seccomp filtering, or a microVM).

- **Vector search tool** (`tools/vector_search.py`) — retrieves relevant
  data-dictionary notes (units, known deduplication issues, timezone
  caveats) before planning begins, so generated code doesn't silently
  violate documented data-quality rules. It's currently a keyword-overlap
  retriever rather than a real embedding index, built specifically so its
  `.search()` internals can be swapped for pgvector/Chroma/Pinecone without
  changing any agent code — the interface is real, the backing retrieval
  method is a placeholder.

- **LLM client** (`tools/llm_client.py`) — a small `generate(system, user)`
  interface. `GeminiClient` wraps the real Gemini API; `FakeLLMClient`
  returns scripted responses in order and requires no API key or network
  access, which is what makes the evaluation suite deterministic and
  runnable in CI.

### Failure handling and self-correction

Two distinct failure paths, routed by explicit conditional edges in the
graph rather than a bare `for` loop with a counter:

1. **Sandbox rejects the code as unsafe** → the run terminates immediately
   as `unsafe_terminated`. This is intentionally not treated as a retryable
   error — unsafe code isn't a mistake to correct, it's a stop condition.
2. **Code runs but throws at runtime, or the Critic rejects the result** →
   routes back to the Coder with the specific failure attached: the actual
   traceback for a runtime error, or the Critic's stated reason for a
   rejection. The Coder's next attempt is generated with that context, not
   a bare "try again." Retries are capped (`max_retries`, default 3) and
   tracked as state (`retry_count`), so the stop condition is explicit and
   inspectable rather than implicit in loop structure.

### Evaluation strategy

`tests/eval_pipeline.py` is a standalone script (no API key or network
required, via `FakeLLMClient`) that asserts three things:

1. **Functional correctness** — two scripted query/response sequences each
   reach `status == "success"` with the expected output.
2. **Safety** — seven known sandbox-escape code samples (direct file read,
   `os` import, `importlib`-mediated `os` access, `eval`-based escape,
   dunder `__subclasses__` traversal, raw socket creation, `subprocess`
   call) are each asserted blocked at all three checkpoints: the static AST
   scan directly, the sandbox's end-to-end `run_in_sandbox` call, and the
   full graph reaching `status == "unsafe_terminated"` rather than being
   silently absorbed as a generic failure. That's 21 separate assertions,
   all currently passing.
3. **Retry recovery** — a deliberately broken first attempt (a typo'd
   column name, producing a real `KeyError`) is asserted to generate
   exactly one `runtime_error` history entry before a corrected attempt
   succeeds, which proves the retry edge actually engages rather than just
   being reachable in the graph definition.

Separately, `tools/sandbox.py`'s resource limits have been exercised
directly against an infinite loop and a memory-allocating script to confirm
the runtime isolation layer — not just the AST layer — independently kills
code that no import-based static check could ever catch.

## Usage

```python
from orchestrator import MultiAgentDataPipeline

pipeline = MultiAgentDataPipeline(api_key="...", model="gemini-2.5-flash")
pipeline.load_data("sales.csv")
final_state = pipeline.run_query("What's the month-over-month revenue trend by region?")

print(final_state["status"])              # "success" | "failed" | "unsafe_terminated"
print(final_state["final_answer"])
print(final_state["execution_history"])   # every attempt, including failed ones
```

## Running the evaluation suite

```bash
pip install -r requirements.txt
python -m tests.eval_pipeline
```

No API key needed. Current output: 2/2 functional cases pass, 21/21 safety
assertions pass, retry recovery confirmed.

## Known limitations

- The subprocess sandbox is not a substitute for OS-level isolation
  (containers, seccomp, microVM) if this were ever exposed to untrusted
  multi-tenant users — see the deployment note in `tools/sandbox.py`.
- CPU/memory resource limits are POSIX-only and silently no-op on Windows;
  the wall-clock timeout still applies everywhere, but a memory-heavy
  script on Windows won't be capped until it hits that timeout.
- `MockVectorStore` is a keyword-overlap stand-in, not a real embedding
  index — swap in an actual vector DB when there's real documentation
  content to index.
- `PipelineState` is JSON-serializable by design specifically so it could
  be checkpointed with LangGraph's persistence layer for resumable runs,
  but no checkpointer is wired up yet.
- The `Artifact` type in `state/schema.py` models chart/table output, but
  the Coder's prompt doesn't yet instruct it to populate one — currently
  only printed text results are captured.
