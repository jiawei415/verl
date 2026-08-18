# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Rule-based reward for code_multiturn training (CF + LCB).

Handles three grading modes discovered during data prep:
  1. stdio + stdout diff       (CF non-SJ, LCB AtCoder/Codeforces)
  2. stdio + special judge     (CF ~20%, uses `generated_checker`)
  3. functional call           (LCB LeetCode, uses `fn_name` + `starter_code`)

Wired via:
    custom_reward_function.path=examples/xujiawei/code_multiturn_reward.py
    custom_reward_function.name=compute_score

Ground-truth schema (produced by verl/examples/data_preprocess/code_multiturn.py):
    {
      "official_tests": [{"input": str, "output": str, "testtype"?: str}, ...],
      "input_mode":        "stdio" | "file",
      "generated_checker": None | str (python source),
      "time_limit_s":      float,
      "memory_limit_mb":   float,
      "fn_name":           None | str,
      "language":          "python",
    }

Signature: `compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs)`.
Returns a dict `{score, acc, pred, n_passed, n_total, mode}`.
- `score` is the training reward (binary all-pass per Q9).
- `acc` is a 0/1 accuracy signal for logging (matches `score` here).

Env vars:
    SANDBOX_ENDPOINT       -- FaaS endpoint (default seed-sandbox.byteintl.net)
    SANDBOX_CLIENT_TIMEOUT -- HTTP client timeout (default 30)
    SANDBOX_MAX_ATTEMPTS   -- retry count (default 2)
    CODE_REWARD_TL_MULT    -- Python TL multiplier (default 3.0, CF-official style)
    CODE_REWARD_MAX_TESTS  -- cap tests per problem (default 40, protects rollout cost)
    CODE_REWARD_CONCURRENCY -- per-problem concurrent sandbox calls (default 8)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://seed-sandbox.byteintl.net/faas/sandbox/"
SANDBOX_ENDPOINT = os.environ.get("SANDBOX_ENDPOINT", DEFAULT_ENDPOINT)
CLIENT_TIMEOUT_S = float(os.environ.get("SANDBOX_CLIENT_TIMEOUT", "30"))
MAX_ATTEMPTS = int(os.environ.get("SANDBOX_MAX_ATTEMPTS", "2"))
TL_MULT = float(os.environ.get("CODE_REWARD_TL_MULT", "3.0"))
MAX_TESTS = int(os.environ.get("CODE_REWARD_MAX_TESTS", "40"))
PER_PROBLEM_CONCURRENCY = int(os.environ.get("CODE_REWARD_CONCURRENCY", "8"))

_CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)


try:
    from sandbox_fusion import RunCodeRequest, run_code_async, set_sandbox_endpoint
    from sandbox_fusion.models import CommandRunStatus, RunStatus

    set_sandbox_endpoint(SANDBOX_ENDPOINT)
    _SDK_OK = True
except Exception as _exc:  # noqa: BLE001
    _SDK_OK = False
    _SDK_IMPORT_ERROR = _exc


def _extract_code(text: str) -> str:
    m = _CODE_FENCE.search(text or "")
    if m:
        return m.group(1).strip()
    # Fallback: raw text (already code, no fences)
    return (text or "").strip()


async def _sandbox_run(code: str, stdin: str, run_timeout_s: float) -> dict[str, Any]:
    if not _SDK_OK:
        return {"status": "sdk_unavailable", "stdout": "", "stderr": str(_SDK_IMPORT_ERROR), "exit_code": -1}
    req = RunCodeRequest(
        code=code,
        language="python",
        stdin=stdin,
        compile_timeout=2.0,
        run_timeout=run_timeout_s,
    )
    try:
        resp = await run_code_async(req, client_timeout=CLIENT_TIMEOUT_S, max_attempts=MAX_ATTEMPTS)
    except Exception as exc:  # noqa: BLE001
        return {"status": "sandbox_error", "stdout": "", "stderr": f"{type(exc).__name__}: {exc}", "exit_code": -1}
    run = resp.run_result
    stdout = (run.stdout if run else "") or ""
    stderr = (run.stderr if run else "") or ""
    if resp.status == RunStatus.Success:
        return {"status": "ok", "stdout": stdout, "stderr": stderr, "exit_code": 0}
    if run and run.status == CommandRunStatus.TimeLimitExceeded:
        return {"status": "timeout", "stdout": stdout, "stderr": stderr, "exit_code": 124}
    return {"status": str(resp.status), "stdout": stdout, "stderr": stderr, "exit_code": 1}


def _normalize_output(s: str) -> str:
    """CF/LCB convention: strip trailing whitespace per line, drop trailing empty lines."""
    lines = [ln.rstrip() for ln in (s or "").splitlines()]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def _diff_ok(user: str, expected: str) -> bool:
    return _normalize_output(user) == _normalize_output(expected)


# ---------------- Mode 1: stdio + stdout diff ----------------


async def _judge_stdio(user_code: str, tc: dict, tl_s: float) -> bool:
    res = await _sandbox_run(user_code, tc.get("input", ""), tl_s)
    if res["status"] != "ok":
        return False
    return _diff_ok(res["stdout"], tc.get("output", ""))


# ---------------- Mode 2: stdio + special judge (checker) ----------------


_SJ_HARNESS_TEMPLATE = '''\
import sys, subprocess, tempfile, os, json

user_code = {user_code!r}
inp       = {inp!r}
expected  = {expected!r}
checker   = {checker!r}
run_tl    = {tl_s!r}

tmp = tempfile.mkdtemp()
with open(os.path.join(tmp, "user.py"), "w") as f:
    f.write(user_code)
with open(os.path.join(tmp, "input.txt"), "w") as f:
    f.write(inp)
with open(os.path.join(tmp, "correct.txt"), "w") as f:
    f.write(expected)
with open(os.path.join(tmp, "checker.py"), "w") as f:
    f.write(checker)

# Stage 1: run user solution
try:
    proc = subprocess.run(
        ["python3", os.path.join(tmp, "user.py")],
        input=inp, capture_output=True, text=True, timeout=run_tl,
    )
    user_out = proc.stdout
    if proc.returncode != 0:
        print(json.dumps({{"verdict": "RE", "stderr": proc.stderr[:500]}}))
        sys.exit(0)
except subprocess.TimeoutExpired:
    print(json.dumps({{"verdict": "TLE"}}))
    sys.exit(0)

with open(os.path.join(tmp, "user.txt"), "w") as f:
    f.write(user_out)

# Stage 2: run checker
try:
    ck = subprocess.run(
        ["python3", os.path.join(tmp, "checker.py"),
         os.path.join(tmp, "input.txt"),
         os.path.join(tmp, "correct.txt"),
         os.path.join(tmp, "user.txt")],
        capture_output=True, text=True, timeout=10,
    )
    tok = (ck.stdout or "").strip().split()
    ok = bool(tok) and tok[-1] in ("1",)
    print(json.dumps({{"verdict": "AC" if ok else "WA", "checker_stdout": ck.stdout[:200]}}))
except subprocess.TimeoutExpired:
    print(json.dumps({{"verdict": "CHECKER_TLE"}}))
'''


async def _judge_sj(user_code: str, tc: dict, checker: str, tl_s: float) -> bool:
    harness = _SJ_HARNESS_TEMPLATE.format(
        user_code=user_code,
        inp=tc.get("input", ""),
        expected=tc.get("output", ""),
        checker=checker,
        tl_s=tl_s,
    )
    # Give the harness itself a generous wall-clock: run_tl (user) + 10s (checker) + slack.
    outer_tl = tl_s + 15.0
    res = await _sandbox_run(harness, "", outer_tl)
    if res["status"] != "ok":
        return False
    try:
        last_line = (res["stdout"] or "").strip().splitlines()[-1]
        payload = json.loads(last_line)
        return payload.get("verdict") == "AC"
    except Exception:  # noqa: BLE001
        return False


# ---------------- Mode 3: functional call (LCB LeetCode) ----------------


_FN_HARNESS_TEMPLATE = '''\
import sys, json, io, contextlib

USER_CODE = {user_code!r}
FN_NAME   = {fn_name!r}
INPUT     = {inp!r}
EXPECTED  = {expected!r}

ns = {{}}
try:
    exec(USER_CODE, ns)
except Exception as e:
    print(json.dumps({{"verdict": "RE", "error": f"exec: {{type(e).__name__}}: {{e}}"}}))
    sys.exit(0)

# LCB functional input format: JSON-encoded list of positional args, one per line
# OR whitespace-separated tokens. Prefer JSON-list parsing, fall back to eval.
def _parse_args(raw):
    raw = raw.strip()
    if not raw:
        return []
    # Try list-of-args as JSON: e.g. "[1, [2,3]]"
    try:
        val = json.loads(raw)
        if isinstance(val, list):
            return val
        return [val]
    except Exception:
        pass
    # Try ast.literal_eval
    try:
        import ast
        val = ast.literal_eval(raw)
        if isinstance(val, list):
            return val
        return [val]
    except Exception:
        return [raw]

args = _parse_args(INPUT)

# Handle Solution class convention (common on LeetCode)
target = ns.get(FN_NAME)
if target is None:
    sol_cls = ns.get("Solution")
    if sol_cls is not None:
        obj = sol_cls()
        target = getattr(obj, FN_NAME, None)
if target is None:
    print(json.dumps({{"verdict": "FN_NOT_FOUND", "fn": FN_NAME}}))
    sys.exit(0)

try:
    with contextlib.redirect_stdout(io.StringIO()):
        result = target(*args)
except Exception as e:
    print(json.dumps({{"verdict": "RE", "error": f"call: {{type(e).__name__}}: {{e}}"}}))
    sys.exit(0)

# Compare against expected (try JSON parse, else string compare)
try:
    exp_val = json.loads(EXPECTED.strip())
except Exception:
    try:
        import ast
        exp_val = ast.literal_eval(EXPECTED.strip())
    except Exception:
        exp_val = EXPECTED.strip()

# Normalize list comparisons (order-sensitive)
ok = (result == exp_val)
print(json.dumps({{"verdict": "AC" if ok else "WA", "got": repr(result)[:200], "want": repr(exp_val)[:200]}}))
'''


async def _judge_functional(user_code: str, tc: dict, fn_name: str, tl_s: float) -> bool:
    harness = _FN_HARNESS_TEMPLATE.format(
        user_code=user_code,
        fn_name=fn_name,
        inp=tc.get("input", ""),
        expected=tc.get("output", ""),
    )
    res = await _sandbox_run(harness, "", tl_s + 5.0)
    if res["status"] != "ok":
        return False
    try:
        last_line = (res["stdout"] or "").strip().splitlines()[-1]
        payload = json.loads(last_line)
        return payload.get("verdict") == "AC"
    except Exception:  # noqa: BLE001
        return False


# ---------------- Dispatcher ----------------


async def _grade_all(user_code: str, ground_truth: dict) -> tuple[int, int, str]:
    tests = list(ground_truth.get("official_tests") or [])[:MAX_TESTS]
    if not tests:
        return 0, 0, "no_tests"

    checker = ground_truth.get("generated_checker") or None
    fn_name = ground_truth.get("fn_name") or None
    tl_s = float(ground_truth.get("time_limit_s") or 1.0) * TL_MULT

    if fn_name:
        mode = "functional"
        judge = lambda tc: _judge_functional(user_code, tc, fn_name, tl_s)  # noqa: E731
    elif checker:
        mode = "special_judge"
        judge = lambda tc: _judge_sj(user_code, tc, checker, tl_s)  # noqa: E731
    else:
        mode = "stdio_diff"
        judge = lambda tc: _judge_stdio(user_code, tc, tl_s)  # noqa: E731

    sem = asyncio.Semaphore(PER_PROBLEM_CONCURRENCY)

    async def _run_one(tc: dict) -> bool:
        async with sem:
            try:
                return await judge(tc)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[code_reward] judge error mode={mode}: {exc}")
                return False

    results = await asyncio.gather(*(_run_one(tc) for tc in tests), return_exceptions=False)
    passed = sum(1 for r in results if r)
    return passed, len(tests), mode


def _run_grade_sync(user_code: str, ground_truth: dict) -> tuple[int, int, str]:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        # Called from an already-running loop (unlikely w/ verl reward path but be defensive).
        fut = asyncio.run_coroutine_threadsafe(_grade_all(user_code, ground_truth), loop)
        return fut.result()
    return asyncio.run(_grade_all(user_code, ground_truth))


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Rule-based binary all-pass reward. See module docstring."""
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except Exception:
            return {"score": 0.0, "acc": 0.0, "pred": "[INVALID_GT]", "n_passed": 0, "n_total": 0, "mode": "error"}

    user_code = _extract_code(solution_str or "")
    if not user_code:
        return {"score": 0.0, "acc": 0.0, "pred": "[NO_CODE]", "n_passed": 0, "n_total": 0, "mode": "error"}

    passed, total, mode = _run_grade_sync(user_code, ground_truth)
    all_pass = float(total > 0 and passed == total)  # binary
    return {
        "score": all_pass,
        "acc": all_pass,
        "pred": user_code[:120],
        "n_passed": passed,
        "n_total": total,
        "mode": mode,
    }
