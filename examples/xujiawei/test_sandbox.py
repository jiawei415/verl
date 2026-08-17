"""Smoke test for code_interpreter function-tool.

Usage (uses ByteIntl seed-sandbox by default):
    python3 examples/xujiawei/test_sandbox.py

Or point at a private sandbox:
    SANDBOX_ENDPOINT=http://your-host:8080/faas/sandbox/ python3 examples/xujiawei/test_sandbox.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import code_tool  # noqa: E402

from verl.tools.function_tool import get_function_tool  # noqa: E402


async def run(case, code):
    print(f"\n=== {case} ===")
    print(f"code:\n{code}")
    out = await code_tool.code_interpreter(code=code)
    print(f"tool_response:\n{out}")


async def main():
    print(f"SANDBOX_ENDPOINT = {code_tool.SANDBOX_ENDPOINT}")
    print(f"SDK loaded: {code_tool._SDK_OK}")
    print(f"registered tool: {get_function_tool('code_interpreter').name}")

    await run("basic arithmetic", "print(1 + 1)")
    await run("markdown fence + final_answer", "```python\nfinal_answer(2 + 2)\n```")
    await run("runtime error", "1/0")
    await run("import + sympy", "from sympy import symbols, solve\nx = symbols('x')\nfinal_answer(solve(x**2 - 9, x))")
    await run("timeout", "while True: pass")


if __name__ == "__main__":
    asyncio.run(main())
