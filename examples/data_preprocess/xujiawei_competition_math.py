"""Split competition_math (Hendrycks MATH, 12500) into per-level verl parquets.

Emits under `math/`:
  - competition_math_lv1.parquet
  - competition_math_lv2.parquet
  - competition_math_lv3.parquet
  - competition_math_lv4.parquet
  - competition_math_lv5.parquet
  - competition_math_lv234.parquet   (concat of 2+3+4)

Ground truth is extracted from the last ``\\boxed{...}`` in the solution field.
Rows without a parseable boxed answer are dropped and logged.

Schema matches DAPO-Math-17k verl layout so `math_dapo.compute_score` picks up
via `data_source='math_dapo'`; per-file separation is by filename, not by
data_source, so existing router / reward wiring stays untouched.

Usage:
    python examples/data_preprocess/xujiawei_competition_math.py
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

HDFS_ROOT = "/mnt/hdfs/byte_data_seed/hdd_hldy/user/xujiawei.415"
SRC_PARQUET = f"{HDFS_ROOT}/hf_datasets/competition_math/data/train-00000-of-00001-7320a6f3aba8ebd2.parquet"
DST_DIR = f"{HDFS_ROOT}/hf_datasets/math"

REWARD_STYLE = "rule-lighteval/MATH_v2"
DATA_SOURCE = "math_dapo"
USER_SUFFIX = "\n\nLet's think step by step and output the final answer within \\boxed{}."


def _extract_boxed(solution: str) -> str | None:
    """Return the content of the *last* ``\\boxed{...}`` in ``solution``.

    Handles nested braces by scanning from the ``\\boxed{`` marker and matching
    balanced ``{``/``}`` -- ``regex.findall`` alone fails on things like
    ``\\boxed{\\frac{a}{b}}`` because ``.*?`` stops at the first ``}``.
    """
    idx = solution.rfind("\\boxed{")
    if idx < 0:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(solution):
        c = solution[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return solution[start:i].strip()
        i += 1
    return None


def _build_row(row: pd.Series, idx: int) -> dict | None:
    solution = row.get("solution") or ""
    gt = _extract_boxed(solution)
    if gt is None:
        return None
    problem = row.get("problem") or ""
    user_content = problem.rstrip() + USER_SUFFIX
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": user_content}],
        "ability": "math",
        "reward_model": {"ground_truth": gt, "style": REWARD_STYLE},
        "extra_info": {
            "index": idx,
            "split": "train",
            "level": row.get("level"),
            "type": row.get("type"),
        },
    }


def _atomic_write(df: pd.DataFrame, dst: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = tmp.name
    df.to_parquet(tmp_path, index=False)
    shutil.copy(tmp_path, dst)
    os.remove(tmp_path)


def main():
    logger.info(f"loading {SRC_PARQUET}")
    src = pd.read_parquet(SRC_PARQUET)
    logger.info(f"src rows={len(src)}")

    # Build rows level by level so index restarts within each file (useful for
    # downstream `extra_info.index` bookkeeping).
    per_level: dict[str, list[dict]] = {}
    dropped = 0
    for lvl in sorted(src["level"].dropna().unique()):
        subset = src[src["level"] == lvl].reset_index(drop=True)
        rows = []
        for i, r in subset.iterrows():
            built = _build_row(r, i)
            if built is None:
                dropped += 1
                continue
            rows.append(built)
        per_level[lvl] = rows
        logger.info(f"{lvl}: kept={len(rows)} / total={len(subset)}")
    logger.info(f"total dropped (no boxed): {dropped}")

    # Write per-level files -- skip the tiny `Level ?` bucket.
    for lvl, rows in per_level.items():
        if lvl == "Level ?" or not rows:
            continue
        n = lvl.split()[-1]                     # "Level 3" -> "3"
        dst = os.path.join(DST_DIR, f"competition_math_lv{n}.parquet")
        _atomic_write(pd.DataFrame(rows), dst)
        logger.info(f"wrote {dst}  rows={len(rows)}")

    # Combined Level 2+3+4 file.
    combo: list[dict] = []
    for lvl in ("Level 2", "Level 3", "Level 4"):
        combo.extend(per_level.get(lvl, []))
    # Re-index the combined file so `extra_info.index` is unique across the split.
    for i, row in enumerate(combo):
        row["extra_info"] = {**row["extra_info"], "index": i}
    dst = os.path.join(DST_DIR, "competition_math_lv234.parquet")
    _atomic_write(pd.DataFrame(combo), dst)
    logger.info(f"wrote {dst}  rows={len(combo)}")

    logger.info("done")


if __name__ == "__main__":
    main()
