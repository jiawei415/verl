# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Adapt the math parquets under $HDFS/hf_datasets/math for multi-turn RL:
prepend a system message telling the model to use the code_interpreter tool
and wrap the final answer in \\boxed{}. Everything else (reward_model,
extra_info, data_source) stays identical, so the reward router keeps working.

Input:   <SRC_DIR>/{dapo_train, gsm8k_test, math500_test, aime24_test, aime25_test}.parquet
Output:  <DST_DIR>/<same files>
"""

import argparse
import os
import shutil
import tempfile

import pandas as pd


DATA_ROOT = "/mnt/hdfs/byte_data_seed/hdd_hldy/user/xujiawei.415/hf_datasets"
DEFAULT_SRC = f"{DATA_ROOT}/math"
DEFAULT_DST = f"{DATA_ROOT}/math_multiturn"

FILES = [
    "dapo_train.parquet",
    "gsm8k_test.parquet",
    "math500_test.parquet",
    "aime24_test.parquet",
    "aime25_test.parquet",
]

SYSTEM_PROMPT = (
    "You are a math problem solver that solves problems by iteratively writing "
    "and running short Python 3 snippets. You MUST use code at least once per "
    "problem to verify or compute your answer -- direct reasoning alone is not "
    "acceptable. Prefer symbolic or numeric checks even for easy-looking problems.\n\n"
    "## Running code\n\n"
    "Write code inside a fenced block:\n\n"
    "```python\n"
    "# your code here\n"
    "```\n\n"
    "Every ```python ... ``` block is executed in a sandbox. Its stdout, stderr, "
    "and exit code are returned as the next user turn. You may write intermediate "
    "`print(...)` statements to inspect values. Do NOT use `<tool_call>` XML or "
    "JSON envelopes -- ONLY plain ```python fenced blocks.\n\n"
    "## Final answer\n\n"
    "Your code should end with `final_answer(x)`, which prints `\\boxed{x}` to "
    "stdout. Once you see `\\boxed{...}` in the tool result and it matches your "
    "reasoning, stop and repeat the final answer as `\\boxed{...}` in plain text.\n\n"
    "## Workflow\n\n"
    "1. Read the problem, write a short plan.\n"
    "2. Write a ```python block; end it with `final_answer(...)`.\n"
    "3. Inspect the tool result. If wrong or unclear, revise and write a new "
    "   ```python block. Repeat until the boxed answer is correct.\n"
    "4. Finish by saying `The answer is \\boxed{...}` in plain text.\n\n"
    "## Example\n\n"
    "```python\n"
    "from sympy import symbols, solve\n"
    "x = symbols('x')\n"
    "final_answer(solve(x**2 - 9, x))\n"
    "```\n"
    "Tool returns: `\\boxed{[-3, 3]}`\n"
    "Then you say: The answer is \\boxed{[-3, 3]}."
)


def rewrite_prompt(prompt):
    """Prepend a system message if not already present."""
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    msgs = [dict(m) if hasattr(m, "items") else m for m in prompt]
    if msgs and msgs[0].get("role") == "system":
        # Already has a system prompt (unlikely in our data) — overwrite it.
        msgs[0]["content"] = SYSTEM_PROMPT
    else:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + msgs
    return msgs


def _atomic_write_parquet(df, dst):
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = tmp.name
    df.to_parquet(tmp_path, index=False)
    shutil.copy(tmp_path, dst)
    os.remove(tmp_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", default=DEFAULT_SRC)
    parser.add_argument("--dst_dir", default=DEFAULT_DST)
    args = parser.parse_args()

    print(f"src: {args.src_dir}")
    print(f"dst: {args.dst_dir}")

    for fname in FILES:
        src = os.path.join(args.src_dir, fname)
        dst = os.path.join(args.dst_dir, fname)
        print(f"-- {fname}")
        df = pd.read_parquet(src)
        df["prompt"] = [rewrite_prompt(p) for p in df["prompt"].tolist()]
        _atomic_write_parquet(df, dst)
        print(f"   wrote {dst}  rows={len(df)}")


if __name__ == "__main__":
    main()
