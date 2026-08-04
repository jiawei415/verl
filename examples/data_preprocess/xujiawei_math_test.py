# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Build four MATH-family test parquets in verl format:

  1. gsm8k_test.parquet    -- from HF gsm8k main test split (raw {question, answer})
  2. math500_test.parquet  -- from MATH-500 (jsonl {problem, solution})
  3. aime24_test.parquet   -- passthrough from DPAO_filter/test/aime24.parquet
  4. aime25_test.parquet   -- passthrough from DPAO_filter/test/aime25.parquet

Each output has columns: data_source, prompt, ability, reward_model, extra_info.
"""

import argparse
import json
import os
import re
import shutil
import tempfile

import pandas as pd


DATA_ROOT = "/mnt/hdfs/byte_data_seed/hdd_hldy/user/xujiawei.415/hf_datasets"
GSM8K_SRC = f"{DATA_ROOT}/gsm8k/main/test-00000-of-00001.parquet"
MATH500_SRC = f"{DATA_ROOT}/MATH-500/test.jsonl"
AIME24_SRC = f"{DATA_ROOT}/DPAO_filter/test/aime24.parquet"
AIME25_SRC = f"{DATA_ROOT}/DPAO_filter/test/aime25.parquet"

GSM8K_INSTR = "Let's think step by step and output the final answer within \\boxed{}."
MATH_INSTR = "Let's think step by step and output the final answer within \\boxed{}."


def extract_gsm8k_answer(answer_raw):
    m = re.search(r"#### (\-?[0-9\.\,]+)", answer_raw)
    assert m is not None, f"no #### in: {answer_raw[:200]}"
    return m.group(0).split("#### ")[1].replace(",", "")


def _last_boxed(s):
    idx = s.rfind("\\boxed")
    if idx < 0:
        idx = s.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    depth = 0
    start = None
    while i < len(s):
        if s[i] == "{":
            depth += 1
            if start is None:
                start = i
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start + 1 : i]
        i += 1
    return None


def build_gsm8k():
    df = pd.read_parquet(GSM8K_SRC)
    rows = []
    for i, row in enumerate(df.itertuples(index=False)):
        q = row.question + " " + GSM8K_INSTR
        gt = extract_gsm8k_answer(row.answer)
        rows.append(
            {
                "data_source": "openai/gsm8k",
                "prompt": [{"role": "user", "content": q}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": gt},
                "extra_info": {"split": "test", "index": i},
            }
        )
    return pd.DataFrame(rows)


def build_math500():
    rows = []
    with open(MATH500_SRC, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            ex = json.loads(line)
            problem = ex["problem"]
            solution = ex["solution"]
            gt = _last_boxed(solution)
            assert gt is not None, f"no boxed in MATH-500 row {i}"
            rows.append(
                {
                    "data_source": "HuggingFaceH4/MATH-500",
                    "prompt": [{"role": "user", "content": problem + " " + MATH_INSTR}],
                    "ability": "math",
                    "reward_model": {"style": "rule", "ground_truth": gt},
                    "extra_info": {"split": "test", "index": i},
                }
            )
    return pd.DataFrame(rows)


def passthrough(src, data_source_override=None):
    df = pd.read_parquet(src)
    if data_source_override is not None:
        df["data_source"] = data_source_override
    df = df.reset_index(drop=True)
    # Force extra_info to a common schema {split, index} so all test parquets can concat.
    df["extra_info"] = [{"split": "test", "index": i} for i in range(len(df))]
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="/mnt/hdfs/byte_data_seed/hdd_hldy/user/xujiawei.415/hf_datasets/math",
        help="Directory to write the four test parquets.",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    builders = [
        ("gsm8k_test.parquet", build_gsm8k),
        ("math500_test.parquet", build_math500),
        ("aime24_test.parquet", lambda: passthrough(AIME24_SRC, "deepscaler/aime24")),
        ("aime25_test.parquet", lambda: passthrough(AIME25_SRC, "deepscaler/aime25")),
    ]
    for fname, fn in builders:
        print(f"Building {fname} ...", flush=True)
        df = fn()
        out = os.path.join(args.output_dir, fname)
        # HDFS FUSE mount does not support parquet's random writes; stage locally then copy.
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp_path = tmp.name
        df.to_parquet(tmp_path, index=False)
        shutil.copy(tmp_path, out)
        os.remove(tmp_path)
        print(f"  wrote {out}  rows={len(df)}", flush=True)


if __name__ == "__main__":
    main()
