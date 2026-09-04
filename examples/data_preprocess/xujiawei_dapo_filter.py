"""Rebuild verl-format math train parquets from DAPO family sources.

Currently supports two sources under $HDFS_ROOT/hf_datasets/:
  - DAPO-Math-17k/data/dapo-math-17k.parquet   -> math/dapo_train.parquet
                                                math_multiturn/dapo_train.parquet
  - DPAO_filter/train/*                        -> math/dapofilter_train.parquet
                                                math_multiturn/dapofilter_train.parquet

Both source layouts store prompts as `[{'role': 'user', 'content': '<question> ...'}]`
with `reward_model.ground_truth`. DAPO-Math-17k is ~100x replicated by problem
index; we dedupe on `extra_info.index` first (one row per unique problem).

All outputs tag ``data_source='math_dapo'`` so verl's existing router picks up
`math_dapo.compute_score` without any wiring changes; per-source separation is
via filename only.

Usage:
    python examples/data_preprocess/xujiawei_dapo_filter.py                  # both sources
    python examples/data_preprocess/xujiawei_dapo_filter.py --sources dapo   # DAPO-17k only
    python examples/data_preprocess/xujiawei_dapo_filter.py --limit 100
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import shutil
import tempfile

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


HDFS_ROOT = "/mnt/hdfs/byte_data_seed/hdd_hldy/user/xujiawei.415"

SOURCES = {
    "dapo": {
        "src_glob": f"{HDFS_ROOT}/hf_datasets/DAPO-Math-17k/data/*.parquet",
        "out_single": f"{HDFS_ROOT}/hf_datasets/math/dapo_train.parquet",
        "out_multi": f"{HDFS_ROOT}/hf_datasets/math_multiturn/dapo_train.parquet",
        "dedupe_by_index": True,
    },
    "dapofilter": {
        "src_glob": f"{HDFS_ROOT}/hf_datasets/DPAO_filter/train/*",
        "out_single": f"{HDFS_ROOT}/hf_datasets/math/dapofilter_train.parquet",
        "out_multi": f"{HDFS_ROOT}/hf_datasets/math_multiturn/dapofilter_train.parquet",
        "dedupe_by_index": False,
    },
}

REWARD_STYLE = "rule-lighteval/MATH_v2"
DATA_SOURCE = "math_dapo"

SYSTEM_PROMPT_MULTITURN = """You are a math problem solver that solves problems by iteratively writing and running short Python 3 snippets. You MUST use code at least once per problem to verify or compute your answer -- direct reasoning alone is not acceptable. Prefer symbolic or numeric checks even for easy-looking problems.

## Running code

Write code inside a fenced block:

```python
# your code here
```

Every ```python ... ``` block is executed in a sandbox. Its stdout, stderr, and exit code are returned as the next user turn. You may write intermediate `print(...)` statements to inspect values. Do NOT use `<tool_call>` XML or JSON envelopes -- ONLY plain ```python fenced blocks.

## Final answer

Your code should end with `final_answer(x)`, which prints `\\boxed{x}` to stdout. Once you see `\\boxed{...}` in the tool result and it matches your reasoning, stop and repeat the final answer as `\\boxed{...}` in plain text.

## Workflow

1. Read the problem, write a short plan.
2. Write a ```python block; end it with `final_answer(...)`.
3. Inspect the tool result. If wrong or unclear, revise and write a new    ```python block. Repeat until the boxed answer is correct.
4. Finish by saying `The answer is \\boxed{...}` in plain text.

## Example

```python
from sympy import symbols, solve
x = symbols('x')
final_answer(solve(x**2 - 9, x))
```
Tool returns: `\\boxed{[-3, 3]}`
Then you say: The answer is \\boxed{[-3, 3]}."""


def _load_src(src_glob: str, limit: int | None = None) -> pd.DataFrame:
    files = sorted(glob.glob(src_glob))
    if not files:
        raise RuntimeError(f"no parquet matching {src_glob}")
    logger.info(f"loading src: {files}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if limit is not None:
        df = df.iloc[:limit].reset_index(drop=True)
    return df


def _extract_question(prompt_field) -> str:
    """Both sources store prompt as an ndarray/list of ``{'role', 'content'}`` dicts."""
    if isinstance(prompt_field, str):
        return prompt_field
    if prompt_field is None:
        return ""
    try:
        first = prompt_field[0]
    except (TypeError, IndexError):
        return ""
    if isinstance(first, dict):
        return first.get("content", "") or ""
    return str(first)


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the first row per `extra_info.index` (DAPO-Math-17k replicates ~100x)."""
    seen: set = set()
    keep_mask = []
    for r in df["extra_info"]:
        key = r.get("index") if isinstance(r, dict) else None
        if key is None or key in seen:
            keep_mask.append(False)
        else:
            seen.add(key)
            keep_mask.append(True)
    out = df[keep_mask].reset_index(drop=True)
    logger.info(f"dedupe by extra_info.index: {len(df)} -> {len(out)} rows")
    return out


def _build_single_row(row: pd.Series, idx: int) -> dict:
    question = _extract_question(row.get("prompt"))
    gt = row.get("reward_model") or {}
    ground_truth = gt.get("ground_truth") if isinstance(gt, dict) else None
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": question}],
        "ability": row.get("ability") or "math",
        "reward_model": {"ground_truth": ground_truth, "style": REWARD_STYLE},
        "extra_info": {"index": idx, "split": "train"},
    }


def _build_multi_row(row: pd.Series, idx: int) -> dict:
    question = _extract_question(row.get("prompt"))
    gt = row.get("reward_model") or {}
    ground_truth = gt.get("ground_truth") if isinstance(gt, dict) else None
    return {
        "data_source": DATA_SOURCE,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT_MULTITURN},
            {"role": "user", "content": question},
        ],
        "ability": row.get("ability") or "math",
        "reward_model": {"ground_truth": ground_truth, "style": REWARD_STYLE},
        "extra_info": {"index": idx, "split": "train"},
    }


def _atomic_write(df: pd.DataFrame, dst: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = tmp.name
    df.to_parquet(tmp_path, index=False)
    shutil.copy(tmp_path, dst)
    os.remove(tmp_path)


def _process_source(name: str, cfg: dict, limit: int | None) -> None:
    logger.info(f"--- source={name} ---")
    src = _load_src(cfg["src_glob"], limit=limit)
    logger.info(f"src rows={len(src)}")
    if cfg["dedupe_by_index"]:
        src = _dedupe(src)

    logger.info("building single-turn rows")
    single_df = pd.DataFrame([_build_single_row(r, i) for i, r in src.iterrows()])
    _atomic_write(single_df, cfg["out_single"])
    logger.info(f"wrote {cfg['out_single']}  rows={len(single_df)}")

    logger.info("building multi-turn rows")
    multi_df = pd.DataFrame([_build_multi_row(r, i) for i, r in src.iterrows()])
    _atomic_write(multi_df, cfg["out_multi"])
    logger.info(f"wrote {cfg['out_multi']}  rows={len(multi_df)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sources",
        default="all",
        help="Comma-separated source keys or `all`. Valid keys: " + ", ".join(SOURCES),
    )
    ap.add_argument("--limit", type=int, default=None, help="Optional row cap for sanity checks.")
    args = ap.parse_args()

    keys = list(SOURCES) if args.sources == "all" else [s.strip() for s in args.sources.split(",")]
    for k in keys:
        if k not in SOURCES:
            raise ValueError(f"unknown source {k!r}; expected one of {list(SOURCES)}")

    for k in keys:
        _process_source(k, SOURCES[k], args.limit)

    logger.info("done")


if __name__ == "__main__":
    main()
