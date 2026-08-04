# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Build a deduped MATH training set from DAPO-Math-17k.

Source is already stored in verl format (columns: data_source, prompt, ability,
reward_model, extra_info). Rows are heavily duplicated (~1.79M rows for ~17k
unique problems); we hash the prompt content to keep exactly one row per
unique problem.
"""

import argparse
import hashlib
import json
import os
import shutil
import tempfile

import pandas as pd


DEFAULT_SRC = (
    "/mnt/hdfs/byte_data_seed/hdd_hldy/user/xujiawei.415"
    "/hf_datasets/DAPO-Math-17k/data/dapo-math-17k.parquet"
)
DEFAULT_OUT = "/mnt/hdfs/byte_data_seed/hdd_hldy/user/xujiawei.415/hf_datasets/math/dapo_train.parquet"

BOXED_INSTR = "Let's think step by step and output the final answer within \\boxed{}."

# DAPO-Math-17k prompts are wrapped with a fixed prefix/suffix; strip them so the
# problem body can be re-appended with the unified boxed instruction.
DAPO_PREFIX = (
    "Solve the following math problem step by step. "
    "The last line of your response should be of the form "
    "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
)
DAPO_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'


def strip_dapo_template(content):
    if content.startswith(DAPO_PREFIX):
        content = content[len(DAPO_PREFIX):]
    if content.endswith(DAPO_SUFFIX):
        content = content[: -len(DAPO_SUFFIX)]
    return content.rstrip()


def prompt_hash(prompt):
    """Stable hash over the prompt list-of-dicts."""
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if isinstance(prompt, (list, tuple)):
        normalized = [dict(m) if hasattr(m, "items") else m for m in prompt]
    else:
        normalized = prompt
    key = json.dumps(normalized, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=DEFAULT_SRC, help="Source parquet path.")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUT,
        help="Output parquet path.",
    )
    args = parser.parse_args()

    print(f"Reading {args.src} ...", flush=True)
    df = pd.read_parquet(args.src)
    print(f"  raw rows: {len(df)}", flush=True)

    df["_h"] = df["prompt"].apply(prompt_hash)
    df = df.drop_duplicates(subset="_h").drop(columns="_h").reset_index(drop=True)
    print(f"  unique rows: {len(df)}", flush=True)

    # Strip DAPO's original prompt template and append the unified boxed instruction.
    def rewrite_prompt(prompt):
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        msgs = [dict(m) if hasattr(m, "items") else m for m in prompt]
        for m in msgs:
            body = strip_dapo_template(m["content"])
            m["content"] = body + " " + BOXED_INSTR
        return msgs

    df["prompt"] = [rewrite_prompt(p) for p in df["prompt"].tolist()]

    # Rewrite extra_info.index to a sequential int for stable ordering.
    def reindex(row_idx, extra):
        base = dict(extra) if hasattr(extra, "items") else {}
        base["index"] = row_idx
        base["split"] = "train"
        return base

    df["extra_info"] = [reindex(i, e) for i, e in enumerate(df["extra_info"].tolist())]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    # HDFS FUSE mount does not support parquet's random writes; stage locally then copy.
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = tmp.name
    df.to_parquet(tmp_path, index=False)
    shutil.copy(tmp_path, args.output)
    os.remove(tmp_path)
    print(f"Wrote {args.output}  rows={len(df)}", flush=True)


if __name__ == "__main__":
    main()
