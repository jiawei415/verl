# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Function-tool: Python code interpreter backed by the ByteIntl seed-sandbox FaaS
via the `sandbox_fusion` SDK. Modeled after
    playground/AIIC_CODE/aiic_verl/utils/tools/sandbox_fusion.py

Referenced by:
    actor_rollout_ref.rollout.multi_turn.function_tool_path=examples/xujiawei/code_tool.py

Env vars:
    SANDBOX_ENDPOINT     -- endpoint URL (default: seed-sandbox.byteintl.net)
    SANDBOX_TIMEOUT      -- per-run wall time in seconds (default 3)
    SANDBOX_CLIENT_TIMEOUT   -- HTTP client timeout (default 30)
    SANDBOX_MAX_ATTEMPTS -- retry count for transient failures (default 2)
    SANDBOX_CONCURRENCY  -- global in-flight cap (default 500)
    SANDBOX_LOCAL_FALLBACK=1 -- run in local subprocess if SDK/endpoint unavailable (unsafe)

The string returned by this coroutine becomes the "tool" message content in the
multi-turn rollout. Truncation (512 chars) mirrors AIIC.
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import tempfile

from verl.tools.function_tool import function_tool


# ---------------- Config ----------------

DEFAULT_ENDPOINT = "https://seed-sandbox.byteintl.net/faas/sandbox/"
SANDBOX_ENDPOINT = os.environ.get("SANDBOX_ENDPOINT", DEFAULT_ENDPOINT)
RUN_TIMEOUT_S = float(os.environ.get("SANDBOX_TIMEOUT", "3"))
CLIENT_TIMEOUT_S = float(os.environ.get("SANDBOX_CLIENT_TIMEOUT", "30"))
MAX_ATTEMPTS = int(os.environ.get("SANDBOX_MAX_ATTEMPTS", "2"))
CONCURRENCY = int(os.environ.get("SANDBOX_CONCURRENCY", "500"))
MAX_OUTPUT_LEN = int(os.environ.get("SANDBOX_MAX_OUTPUT", "512"))
LOCAL_FALLBACK = os.environ.get("SANDBOX_LOCAL_FALLBACK", "").lower() in ("1", "true", "yes")

# Prepended so the model can call `final_answer(x)` -> stdout: \boxed{x}
_FINAL_ANSWER_PREFIX = 'def final_answer(result):\n    print(f"\\\\boxed{{{result}}}")\n\n'

_CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)


# ---------------- SDK bootstrap (import lazily) ----------------

try:
    from sandbox_fusion import RunCodeRequest, run_code_async, set_sandbox_endpoint
    from sandbox_fusion.models import CommandRunStatus, RunStatus

    set_sandbox_endpoint(SANDBOX_ENDPOINT)
    _SDK_OK = True
except Exception as _import_exc:  # noqa: BLE001
    _SDK_OK = False
    _SDK_IMPORT_ERROR = _import_exc


_SEMAPHORE: asyncio.Semaphore | None = None


def _semaphore() -> asyncio.Semaphore:
    global _SEMAPHORE
    if _SEMAPHORE is None:
        _SEMAPHORE = asyncio.Semaphore(CONCURRENCY)
    return _SEMAPHORE


# ---------------- Code prep ----------------


def _extract_code(code: str) -> str:
    m = _CODE_FENCE.search(code)
    return m.group(1).strip() if m else code.strip()


def _truncate(s: str, max_len: int = MAX_OUTPUT_LEN) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len // 2] + f"\n...[truncated to {max_len} chars]...\n" + s[-max_len // 2 :]


# ---------------- Runners ----------------


async def _run_sandbox(code: str) -> str:
    req = RunCodeRequest(
        code=code,
        language="python",
        compile_timeout=1.0,
        run_timeout=RUN_TIMEOUT_S,
    )
    try:
        async with _semaphore():
            resp = await run_code_async(req, client_timeout=CLIENT_TIMEOUT_S, max_attempts=MAX_ATTEMPTS)
    except Exception as exc:  # noqa: BLE001
        return f"[sandbox_error] {type(exc).__name__}: {exc}"

    run = resp.run_result
    stdout = (run.stdout if run else "") or ""
    stderr = (run.stderr if run else "") or ""

    if resp.status == RunStatus.Success:
        return _truncate(stdout.strip()) or "[empty stdout]"
    if run and run.status == CommandRunStatus.TimeLimitExceeded:
        return "Time limit exceeded"
    if stderr:
        return _truncate(stderr.strip().splitlines()[-1] if "\n" in stderr else stderr.strip())
    return f"[sandbox {resp.status}]"


def _run_local(code: str) -> str:
    """Unsafe local subprocess fallback. Enable via SANDBOX_LOCAL_FALLBACK=1."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        proc = subprocess.run(
            ["python3", path],
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return "Time limit exceeded"
    finally:
        os.unlink(path)
    stdout, stderr = proc.stdout.strip(), proc.stderr.strip()
    if proc.returncode == 0:
        return _truncate(stdout) or "[empty stdout]"
    return _truncate(stderr.splitlines()[-1] if stderr else "Code execution failed")


# ---------------- Tool entry ----------------


@function_tool(
    "code_interpreter",
    schema={
        "type": "function",
        "function": {
            "name": "code_interpreter",
            "description": "Execute a Python code snippet and return its stdout (or error message).",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python source to execute."},
                },
                "required": ["code"],
            },
        },
    },
)
async def code_interpreter(code: str) -> str:
    """Execute `code` via seed-sandbox (async) and return captured output."""
    code = _extract_code(code)
    code = _FINAL_ANSWER_PREFIX + code

    if _SDK_OK:
        return await _run_sandbox(code)
    if LOCAL_FALLBACK:
        return await asyncio.to_thread(_run_local, code)
    return f"[sandbox_unavailable] sandbox_fusion SDK not importable: {_SDK_IMPORT_ERROR!r}"
