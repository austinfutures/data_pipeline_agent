"""
Python Sandbox Execution Tool.

The original agent called `exec(code, {...})` directly in-process on
LLM-generated code. That gives generated code full access to the host
process: it can import os/sys/socket, read arbitrary files, spawn
subprocesses, or exhaust memory/CPU with no limit. This module replaces
that with a two-layer defense:

  1. STATIC LAYER  — an AST allowlist scan. Rejects disallowed imports,
     attribute access to dunder/escape-hatch names, exec/eval/compile,
     and other known sandbox-escape primitives *before* anything runs.
     This is a cheap, fast rejection of obviously unsafe code and is
     NOT itself a complete sandbox (AST checks alone are known to be
     bypassable in general Python) — hence layer 2.

  2. RUNTIME LAYER — execution in a separate subprocess with:
       - a restricted `__builtins__`
       - CPU time limit and memory limit via `resource.setrlimit`
         (POSIX only — Linux/Mac). On Windows, `resource` doesn't exist,
         so those two limits are silently skipped and only the
         wall-clock timeout below applies. See WINDOWS NOTE below.
       - wall-clock timeout (subprocess timeout, hard-kills the process
         on every OS, including Windows)
       - no network / filesystem access assumed (run this in a
         container or gVisor/firecracker sandbox in real production;
         see docstring at bottom for deployment notes)

This is still not a substitute for OS-level sandboxing (containers,
seccomp, a VM) in a real production deployment — see the note at the
bottom of this file. Treat this as defense-in-depth for a trusted
internal tool, not a hard multi-tenant security boundary.

WINDOWS NOTE: `resource.setrlimit` is POSIX-only. On Windows this module
still runs — the AST allowlist, subprocess isolation, restricted
builtins, and wall-clock timeout all work the same — but CPU-time and
memory limits are not enforced at the OS level. A malicious/broken
script that busy-loops will still be killed by the wall-clock timeout;
one that allocates a huge amount of memory will not be capped until it
hits the timeout or exhausts system memory. If you need real memory/CPU
caps on Windows, run this inside WSL2, a Linux container (Docker
Desktop), or add `psutil`-based external monitoring that kills the
child process if it exceeds a memory threshold.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

try:
    import resource  # POSIX only
    _HAS_RESOURCE = True
except ImportError:
    _HAS_RESOURCE = False  # Windows

# ---------------------------------------------------------------------------
# Layer 1: static AST allowlist
# ---------------------------------------------------------------------------

ALLOWED_IMPORTS = {"pandas", "numpy", "math", "statistics", "re", "json", "datetime"}

# Names that, if accessed as attributes or called, are common sandbox-escape
# vectors (getting to __builtins__, __subclasses__, os via os-like modules, etc.)
FORBIDDEN_NAMES = {
    "__import__", "eval", "exec", "compile", "open", "input",
    "__builtins__", "__globals__", "__subclasses__", "__bases__",
    "__mro__", "__loader__", "__spec__", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "breakpoint", "help",
}

FORBIDDEN_MODULES = {
    "os", "sys", "subprocess", "shutil", "socket", "requests", "urllib",
    "http", "ftplib", "telnetlib", "smtplib", "ctypes", "multiprocessing",
    "threading", "importlib", "pickle", "pty", "signal", "resource", "io",
}


class UnsafeCodeError(Exception):
    """Raised when generated code fails the static safety scan."""
    def __init__(self, reasons: list[str]):
        self.reasons = reasons
        super().__init__("; ".join(reasons))


class _SafetyVisitor(ast.NodeVisitor):
    def __init__(self):
        self.violations: list[str] = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in FORBIDDEN_MODULES or root not in ALLOWED_IMPORTS:
                self.violations.append(f"Disallowed import: '{alias.name}'")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        root = (node.module or "").split(".")[0]
        if root in FORBIDDEN_MODULES or root not in ALLOWED_IMPORTS:
            self.violations.append(f"Disallowed import: 'from {node.module} import ...'")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if node.id in FORBIDDEN_NAMES:
            self.violations.append(f"Disallowed name reference: '{node.id}'")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr in FORBIDDEN_NAMES or node.attr.startswith("__"):
            self.violations.append(f"Disallowed attribute access: '.{node.attr}'")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # Catches eval/exec/compile/__import__ even if aliased via Name/Attribute,
        # already handled above; this also blocks calling dunder-returning chains.
        self.generic_visit(node)


def static_safety_check(code: str) -> list[str]:
    """Parse code and return a list of violation strings (empty = safe)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]
    visitor = _SafetyVisitor()
    visitor.visit(tree)
    return visitor.violations


# ---------------------------------------------------------------------------
# Layer 2: runtime isolation (subprocess + resource limits)
# ---------------------------------------------------------------------------

_RUNNER_TEMPLATE = '''
import sys, json, io, contextlib

# Resource limits: CPU seconds and address-space bytes. POSIX only (Linux/Mac);
# on Windows `resource` doesn't exist, so this block is a no-op and the caller
# relies on the wall-clock subprocess timeout instead.
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, ({cpu_seconds}, {cpu_seconds}))
    resource.setrlimit(resource.RLIMIT_AS, ({mem_bytes}, {mem_bytes}))
except ImportError:
    pass

import pandas as pd
import numpy as np

df = pd.read_json(io.StringIO({df_json!r}), orient="split")

_stdout = io.StringIO()
result = None
try:
    with contextlib.redirect_stdout(_stdout):
        exec(compile({code!r}, "<agent_code>", "exec"), {{
            "df": df, "pd": pd, "np": np, "result": None,
            "__builtins__": {{
                "print": print, "range": range, "len": len, "str": str,
                "int": int, "float": float, "bool": bool, "list": list,
                "dict": dict, "set": set, "tuple": tuple, "sorted": sorted,
                "sum": sum, "min": min, "max": max, "abs": abs, "round": round,
                "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
                "isinstance": isinstance, "Exception": Exception, "ValueError": ValueError,
                "TypeError": TypeError, "KeyError": KeyError, "None": None,
                "True": True, "False": False,
            }}
        }}, {{}})
    out = {{"status": "success", "stdout": _stdout.getvalue()}}
except Exception as e:
    out = {{"status": "error", "stdout": _stdout.getvalue(), "error": f"{{type(e).__name__}}: {{e}}"}}

print("___RESULT_JSON___")
print(json.dumps(out))
'''


@dataclass
class SandboxResult:
    status: str  # "success" | "runtime_error" | "unsafe_rejected" | "timeout"
    stdout: str = ""
    error_message: str = ""
    violations: list[str] = field(default_factory=list)


def run_in_sandbox(
    code: str,
    dataframe,
    cpu_seconds: int = 5,
    mem_bytes: int = 512 * 1024 * 1024,
    wall_clock_timeout: int = 15,
) -> SandboxResult:
    """
    Execute `code` against `dataframe` in an isolated subprocess.

    Returns a SandboxResult; never raises for code-level errors (those are
    captured as status="runtime_error"). Only infrastructure failures
    (e.g. subprocess couldn't launch) raise.
    """
    violations = static_safety_check(code)
    if violations:
        return SandboxResult(status="unsafe_rejected", violations=violations)

    df_json = dataframe.to_json(orient="split")
    script = _RUNNER_TEMPLATE.format(
        cpu_seconds=cpu_seconds,
        mem_bytes=mem_bytes,
        df_json=df_json,
        code=code,
    )

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=wall_clock_timeout,
        )
    except subprocess.TimeoutExpired:
        return SandboxResult(status="timeout", error_message=f"Execution exceeded {wall_clock_timeout}s wall-clock limit.")
    finally:
        Path(script_path).unlink(missing_ok=True)

    raw = proc.stdout
    if "___RESULT_JSON___" not in raw:
        # Process crashed/OOM-killed before printing its result marker.
        return SandboxResult(
            status="runtime_error",
            stdout=raw,
            error_message=proc.stderr.strip() or "Process terminated without producing output (likely resource limit).",
        )

    pre, _, tail = raw.partition("___RESULT_JSON___")
    try:
        payload = json.loads(tail.strip().splitlines()[0])
    except (json.JSONDecodeError, IndexError):
        return SandboxResult(status="runtime_error", stdout=raw, error_message="Could not parse sandbox output.")

    if payload["status"] == "success":
        return SandboxResult(status="success", stdout=payload["stdout"])
    return SandboxResult(status="runtime_error", stdout=payload["stdout"], error_message=payload.get("error", "Unknown error"))


# ---------------------------------------------------------------------------
# Production deployment note
# ---------------------------------------------------------------------------
"""
DEPLOYMENT NOTE: subprocess + resource limits (where available) + an AST
allowlist raises the bar substantially over raw exec(), but on a shared or
internet-facing service you should still run this inside an OS-level sandbox:
  - a locked-down container (no network, read-only rootfs, dropped
    capabilities, seccomp-bpf syscall filtering), or
  - a microVM (Firecracker / gVisor), or
  - a managed code-execution API.
Treat this module's Layer 1+2 as defense-in-depth for a trusted internal
tool, not as a hard multi-tenant security boundary.
"""
