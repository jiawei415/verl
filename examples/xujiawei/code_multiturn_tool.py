# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Function-tool: Python code executor with optional stdin for competitive-programming
multi-turn rollouts. Backed by the ByteIntl seed-sandbox FaaS via sandbox_fusion SDK.

Model calls this mid-rollout to sanity-check a candidate solution against provided
example I/O before submitting the final answer. Reward-side verification (private
tests, SJ checker, functional call) lives in `code_multiturn_reward.py`.

Referenced by:
    actor_rollout_ref.rollout.multi_turn.function_tool_path=examples/xujiawei/code_multiturn_tool.py

Env vars (shared w/ code_tool.py):
    SANDBOX_ENDPOINT     -- endpoint URL (default: seed-sandbox.byteintl.net)
    SANDBOX_TIMEOUT      -- per-run wall time in seconds (default 5)
    SANDBOX_CLIENT_TIMEOUT   -- HTTP client timeout (default 30)
    SANDBOX_MAX_ATTEMPTS -- retry count for transient failures (default 2)
    SANDBOX_CONCURRENCY  -- global in-flight cap (default 500)
    SANDBOX_MAX_OUTPUT   -- max chars of stdout/stderr returned to model (default 1024)
"""

from __future__ import annotations

import asyncio
import os
import re

from verl.tools.function_tool import function_tool


DEFAULT_ENDPOINT = "https://seed-sandbox.byteintl.net/faas/sandbox/"
SANDBOX_ENDPOINT = os.environ.get("SANDBOX_ENDPOINT", DEFAULT_ENDPOINT)
RUN_TIMEOUT_S = float(os.environ.get("SANDBOX_TIMEOUT", "5"))
CLIENT_TIMEOUT_S = float(os.environ.get("SANDBOX_CLIENT_TIMEOUT", "30"))
MAX_ATTEMPTS = int(os.environ.get("SANDBOX_MAX_ATTEMPTS", "2"))
CONCURRENCY = int(os.environ.get("SANDBOX_CONCURRENCY", "500"))
MAX_OUTPUT_LEN = int(os.environ.get("SANDBOX_MAX_OUTPUT", "1024"))

_CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)


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


def _extract_code(text: str) -> str:
    m = _CODE_FENCE.search(text)
    return m.group(1).strip() if m else text.strip()


def _truncate(s: str, max_len: int = MAX_OUTPUT_LEN) -> str:
    if not s:
        return ""
    if len(s) <= max_len:
        return s
    head = s[: max_len // 2]
    tail = s[-max_len // 2 :]
    return f"{head}\n...[truncated to {max_len} chars]...\n{tail}"


async def _run_sandbox(code: str, stdin: str, timeout_s: float) -> dict:
    req = RunCodeRequest(
        code=code,
        language="python",
        stdin=stdin,
        compile_timeout=1.0,
        run_timeout=timeout_s,
    )
    try:
        async with _semaphore():
            resp = await run_code_async(
                req, client_timeout=CLIENT_TIMEOUT_S, max_attempts=MAX_ATTEMPTS
            )
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "stdout": "", "stderr": f"[sandbox_error] {type(exc).__name__}: {exc}", "exit_code": -1}

    run = resp.run_result
    stdout = (run.stdout if run else "") or ""
    stderr = (run.stderr if run else "") or ""

    if resp.status == RunStatus.Success:
        return {"status": "ok", "stdout": stdout, "stderr": stderr, "exit_code": 0}
    if run and run.status == CommandRunStatus.TimeLimitExceeded:
        return {"status": "timeout", "stdout": stdout, "stderr": stderr, "exit_code": 124}
    return {
        "status": str(resp.status),
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": getattr(run, "return_code", 1) if run else 1,
    }


def _format_for_model(res: dict) -> str:
    """Compact, readable single-string return for the model."""
    lines = []
    stdout = _truncate(res.get("stdout", ""))
    stderr = _truncate(res.get("stderr", ""))
    lines.append(f"[status] {res.get('status')} (exit_code={res.get('exit_code')})")
    if stdout:
        lines.append(f"[stdout]\n{stdout}")
    else:
        lines.append("[stdout] (empty)")
    if stderr:
        lines.append(f"[stderr]\n{stderr}")
    return "\n".join(lines)


@function_tool(
    "execute_python",
    schema={
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": (
                "Run a Python code snippet and return its stdout, stderr, exit_code. "
                "Optionally pipe `stdin` to it. Use this to test a candidate solution "
                "against sample inputs before submitting the final answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python source to execute (may be wrapped in ```python fences).",
                    },
                    "stdin": {
                        "type": "string",
                        "description": "Optional string piped to the program as standard input.",
                        "default": "",
                    },
                },
                "required": ["code"],
            },
        },
    },
)
async def execute_python(code: str, stdin: str = "") -> str:
    code = _extract_code(code)
    if not _SDK_OK:
        return f"[sandbox_unavailable] sandbox_fusion SDK not importable: {_SDK_IMPORT_ERROR!r}"
    res = await _run_sandbox(code, stdin or "", RUN_TIMEOUT_S)
    return _format_for_model(res)
