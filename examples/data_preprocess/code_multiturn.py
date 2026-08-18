"""Preprocess open-r1/codeforces (verifiable) train + LiveCodeBench test into
verl multiturn parquet, plus a stats.json to inform the sandbox decision.

Outputs:
- ${OUT_DIR}/train.parquet   (from CF verifiable train)
- ${OUT_DIR}/test.parquet    (from LCB, release_date >= 2024-08-01)
- ${OUT_DIR}/stats.json      (distributions for sandbox choice)

Usage:
  python verl/examples/data_preprocess/code_multiturn.py \
      --cf_dir /mnt/hdfs/.../hf_datasets/codeforces \
      --lcb_dir /mnt/hdfs/.../hf_datasets/livecodebench \
      --out_dir /mnt/hdfs/.../hf_datasets/code_multiturn \
      --lcb_release release_v6 \
      --lcb_cutoff 2024-08-01
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import pickle
import shutil
import sys
import tempfile
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset


MAX_TESTS_PER_PROBLEM = 20      # cap official_tests / private_tests per problem
MAX_TEST_IO_BYTES = 8 * 1024    # cap each individual test input/output string
MAX_TOTAL_GT_BYTES = 512 * 1024  # drop problems whose total tests exceed this after cap


def _trim_tests(tests: list, keep_n: int, io_max_bytes: int) -> list:
    """Keep first `keep_n` tests, and drop tests whose input/output exceeds
    `io_max_bytes` characters. Truncating would produce false negatives against
    genuine solutions, so we skip oversized cases entirely."""
    trimmed = []
    for t in tests[:keep_n]:
        inp = t.get("input", "")
        out = t.get("output", "")
        if len(inp) > io_max_bytes or len(out) > io_max_bytes:
            continue
        row = {"input": inp, "output": out}
        if "testtype" in t:
            row["testtype"] = t["testtype"]
        trimmed.append(row)
    return trimmed


def _write_parquet_to_hdfs(df: pd.DataFrame, hdfs_path: Path) -> None:
    """Write via HF `datasets` (arrow-native) to avoid pyarrow chunked-read bugs on
    nested list-of-struct columns. Also stage to /tmp first because FUSE HDFS
    open_output_stream sometimes fails with 'Device or resource busy'."""
    hdfs_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    rows = df.to_dict(orient="records")
    Dataset.from_list(rows).to_parquet(str(tmp_path))
    shutil.copy2(tmp_path, hdfs_path)
    tmp_path.unlink(missing_ok=True)

SYSTEM_PROMPT = """You are a competitive-programming assistant that solves problems by iteratively writing and testing Python 3 code.

## Tool: running code

Whenever you want to execute Python code, write it inside a fenced block:

```python
# your code here
```

Every ```python ... ``` block is sent to a sandbox and executed as a full Python 3 script. The sandbox's stdout, stderr, and exit code are returned to you as the next user turn. If your program needs input, put the input string in a ```stdin ... ``` block immediately before the code block.

- You may write intermediate `print(...)` statements to inspect variables.
- Use this to test your candidate solution against the provided examples BEFORE submitting.
- One ```python block per turn is the recommended pattern; multiple blocks are allowed but executed independently.

## Final submission

Your final answer is the **last** ```python ... ``` block you produce before you stop or run out of turns. The grader will run that code against the private test cases. So:

1. Reason about the problem, draft an approach.
2. Write a ```python block containing a complete standalone solution.
3. Look at the tool result. If any example fails, revise and write a new ```python block. Repeat.
4. When confident, write the final ```python block. That is your submission.

## Style rules

- Every code block MUST be a complete, self-contained Python 3 script (imports + main logic).
- Read from `sys.stdin` / `input()` and write to `sys.stdout` / `print(...)` for stdio problems.
- For functional problems (function signature provided in the prompt), define the function; the grader will import it and call it with the test inputs.
- Do NOT use `<tool_call>` XML, JSON envelopes, or any special markers -- ONLY plain ```python fenced blocks.
"""


PROMPT_TEMPLATE_CF = """You are given a competitive programming problem from Codeforces.

## Problem
{title}

## Description
{description}

## Input format
{input_format}

## Output format
{output_format}

## Examples
{examples}

## Notes
{note}

## Constraints
- Time limit: {time_limit_s:.1f} s
- Memory limit: {memory_limit_mb:.0f} MB
- Input mode: {input_mode}

Solve the problem in Python 3. Follow the workflow in the system prompt: iterate with ```python blocks, use the sample inputs to sanity-check, and produce your final solution as the last ```python block."""

PROMPT_TEMPLATE_LCB_STDIO = """You are given a competitive programming problem.

## Problem
{title}

## Description
{content}

Solve it in Python 3, reading from stdin and writing to stdout. Follow the workflow in the system prompt: iterate with ```python blocks; your final submission is the last ```python block."""

PROMPT_TEMPLATE_LCB_FUNCTIONAL = """You are given a coding problem.

## Problem
{title}

## Description
{content}

## Starter code
```python
{starter_code}
```

Complete the function above (or define your own with the same signature) in Python 3. Follow the workflow in the system prompt: iterate with ```python blocks; your final submission is the last ```python block."""


# ---------------- Codeforces (open-r1/codeforces verifiable) ----------------


def load_cf(cf_dir: Path) -> dict[str, Dataset]:
    train_files = sorted(glob.glob(str(cf_dir / "verifiable" / "train-*.parquet")))
    test_files = sorted(glob.glob(str(cf_dir / "verifiable" / "test-*.parquet")))
    if not train_files:
        raise RuntimeError(f"no CF train parquet under {cf_dir}/verifiable/")
    return {
        "train": load_dataset("parquet", data_files=train_files, split="train"),
        "test": load_dataset("parquet", data_files=test_files, split="train"),
    }


def cf_format_examples(examples) -> str:
    if not examples:
        return "(none)"
    parts = []
    for i, ex in enumerate(examples, 1):
        parts.append(f"### Example {i}\nInput:\n```\n{ex['input']}\n```\nOutput:\n```\n{ex['output']}\n```")
    return "\n\n".join(parts)


def cf_build_prompt(row) -> str:
    return PROMPT_TEMPLATE_CF.format(
        title=row.get("title") or "",
        description=row.get("description") or "",
        input_format=row.get("input_format") or "",
        output_format=row.get("output_format") or "",
        examples=cf_format_examples(row.get("examples") or []),
        note=row.get("note") or "(none)",
        time_limit_s=float(row.get("time_limit") or 1.0),
        memory_limit_mb=float(row.get("memory_limit") or 256.0),
        input_mode=row.get("input_mode") or "stdio",
    )


def cf_row_to_verl(row, idx: int, split: str) -> dict | None:
    rating = row.get("rating")
    if rating is None or not (800 <= int(rating) <= 2600):
        return None
    if not row.get("executable", False):
        return None
    official_tests = row.get("official_tests") or []
    if len(official_tests) < 3:
        return None

    prompt = cf_build_prompt(row)
    trimmed_tests = _trim_tests(
        [{"input": t["input"], "output": t["output"]} for t in official_tests],
        MAX_TESTS_PER_PROBLEM, MAX_TEST_IO_BYTES,
    )
    if len(trimmed_tests) < 3:
        return None
    ground_truth = {
        "official_tests": trimmed_tests,
        "input_mode": row.get("input_mode") or "stdio",
        "generated_checker": row.get("generated_checker") or None,
        "time_limit_s": float(row.get("time_limit") or 1.0),
        "memory_limit_mb": float(row.get("memory_limit") or 256.0),
        "fn_name": None,
        "language": "python",
    }
    gt_json = json.dumps(ground_truth)
    if len(gt_json) > MAX_TOTAL_GT_BYTES:
        return None
    return {
        "data_source": "codeforces",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": gt_json},
        "extra_info": {
            "split": split,
            "index": idx,
            "source_id": row.get("id"),
            "rating": int(rating),
            "tags": list(row.get("tags") or []),
            "contest_id": row.get("contest_id"),
            "has_special_judge": bool(row.get("generated_checker")),
            "n_official_tests": len(trimmed_tests),
            "n_generated_tests": int(row.get("generated_tests") or 0),
        },
    }


# ---------------- LiveCodeBench (code_generation_lite) ----------------


LCB_RELEASE_FILES = {
    "release_v1": ["test.jsonl"],
    "release_v2": ["test.jsonl", "test2.jsonl"],
    "release_v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
    "release_v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
    "release_v5": [
        "test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl",
    ],
    "release_v6": [
        "test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl",
        "test6.jsonl",
    ],
    "release_latest": [
        "test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl",
        "test6.jsonl",
    ],
}


def load_lcb(lcb_dir: Path, release: str) -> Dataset:
    files = [str(lcb_dir / f) for f in LCB_RELEASE_FILES[release]]
    missing = [f for f in files if not Path(f).exists()]
    if missing:
        raise RuntimeError(f"LCB files missing: {missing}")
    return load_dataset("json", data_files=files, split="train")


def _lcb_decode_tests(blob):
    """LCB test_cases fields are stored as either:
    - JSON string of list[{input, output, testtype}]  (public_test_cases)
    - base64(zlib(pickle(json_string)))               (private_test_cases)
    """
    if not blob:
        return []
    if isinstance(blob, list):
        return blob
    if isinstance(blob, str):
        try:
            return json.loads(blob)
        except Exception:
            pass
        try:
            dec = pickle.loads(zlib.decompress(base64.b64decode(blob)))
            if isinstance(dec, str):
                return json.loads(dec)
            if isinstance(dec, list):
                return dec
        except Exception:
            return []
    return []


def lcb_row_to_verl(row, idx: int, cutoff_iso: str) -> dict | None:
    date = row.get("contest_date") or row.get("date") or ""
    if hasattr(date, "isoformat"):
        date = date.isoformat()
    date = str(date)
    if not date or date[:10] < cutoff_iso:
        return None

    public = _lcb_decode_tests(row.get("public_test_cases"))
    private = _lcb_decode_tests(row.get("private_test_cases"))

    all_tests = list(public) + list(private)
    if not all_tests:
        return None

    metadata = row.get("metadata") or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except Exception:
            metadata = {}
    fn_name = metadata.get("func_name") or None
    starter_code = row.get("starter_code") or ""
    is_functional = fn_name is not None or bool(starter_code.strip())

    title = row.get("question_title") or row.get("title") or ""
    content = row.get("question_content") or row.get("content") or ""
    if is_functional:
        prompt = PROMPT_TEMPLATE_LCB_FUNCTIONAL.format(
            title=title, content=content, starter_code=starter_code,
        )
    else:
        prompt = PROMPT_TEMPLATE_LCB_STDIO.format(title=title, content=content)

    normalized_tests = []
    for t in all_tests:
        if isinstance(t, dict):
            normalized_tests.append(
                {"input": t.get("input", ""), "output": t.get("output", ""),
                 "testtype": t.get("testtype", "stdin")}
            )
    if not normalized_tests:
        return None

    normalized_tests = _trim_tests(normalized_tests, MAX_TESTS_PER_PROBLEM, MAX_TEST_IO_BYTES)
    if not normalized_tests:
        return None

    ground_truth = {
        "official_tests": normalized_tests,
        "input_mode": "stdio",
        "generated_checker": None,
        "time_limit_s": 6.0,
        "memory_limit_mb": 512.0,
        "fn_name": fn_name,
        "language": "python",
    }
    gt_json = json.dumps(ground_truth)
    if len(gt_json) > MAX_TOTAL_GT_BYTES:
        return None
    return {
        "data_source": "livecodebench",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": gt_json},
        "extra_info": {
            "split": "test",
            "index": idx,
            "source_id": row.get("question_id") or row.get("id"),
            "platform": row.get("platform"),
            "difficulty": row.get("difficulty"),
            "contest_date": date,
            "is_functional": is_functional,
            "n_public_tests": len(public),
            "n_private_tests": len(private),
        },
    }


# ---------------- Stats ----------------


def _hist(values, bins=None):
    values = [v for v in values if v is not None]
    if not values:
        return {}
    if bins is None:
        return dict(Counter(values).most_common(20))
    counts, edges = np.histogram(values, bins=bins)
    return {f"[{edges[i]:.0f}, {edges[i+1]:.0f})": int(counts[i]) for i in range(len(counts))}


def cf_stats(ds: Dataset, tag: str) -> dict:
    n = len(ds)
    input_modes = Counter(ds["input_mode"])
    sj = sum(1 for x in ds["generated_checker"] if x)
    exec_ok = sum(1 for x in ds["executable"] if x)
    ratings = [int(x) for x in ds["rating"] if x is not None]
    n_tests = [len(x) for x in ds["official_tests"]]
    return {
        "n": n,
        "input_mode": dict(input_modes),
        "special_judge_ratio": round(sj / max(n, 1), 4),
        "executable_ratio": round(exec_ok / max(n, 1), 4),
        "rating_hist": _hist(ratings, bins=[0, 800, 1200, 1600, 2000, 2400, 2800, 4000]),
        "n_official_tests_hist": _hist(n_tests, bins=[0, 3, 5, 10, 20, 50, 100, 10000]),
        "tag": tag,
    }


def lcb_stats(ds: Dataset, cutoff_iso: str) -> dict:
    raw_dates = ds["contest_date"] if "contest_date" in ds.column_names else []
    dates = []
    for x in raw_dates:
        if x is None:
            continue
        if hasattr(x, "isoformat"):
            dates.append(x.isoformat()[:10])
        else:
            dates.append(str(x)[:10])
    filtered_idx = [i for i, d in enumerate(dates) if d >= cutoff_iso]
    platforms = Counter(ds["platform"]) if "platform" in ds.column_names else {}
    difficulties = Counter(ds["difficulty"]) if "difficulty" in ds.column_names else {}
    return {
        "n_all": len(ds),
        "n_after_cutoff": len(filtered_idx),
        "cutoff": cutoff_iso,
        "platform": dict(platforms),
        "difficulty": dict(difficulties),
        "date_min": min(dates) if dates else None,
        "date_max": max(dates) if dates else None,
    }


# ---------------- Main ----------------


def convert(ds: Dataset, fn, *args, **kwargs) -> list[dict]:
    out = []
    for i, row in enumerate(ds):
        v = fn(row, i, *args, **kwargs)
        if v is not None:
            out.append(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cf_dir", required=True, type=Path)
    ap.add_argument("--lcb_dir", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--lcb_release", default="release_latest")
    ap.add_argument("--lcb_cutoff", default="2024-08-01")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[cf] loading {args.cf_dir}")
    cf = load_cf(args.cf_dir)
    print(f"[cf] train={len(cf['train'])} test={len(cf['test'])}")

    print(f"[lcb] loading {args.lcb_dir} release={args.lcb_release}")
    lcb = load_lcb(args.lcb_dir, args.lcb_release)
    print(f"[lcb] rows={len(lcb)}")

    print("[stats] computing")
    stats = {
        "codeforces_train_raw": cf_stats(cf["train"], "cf_train_raw"),
        "codeforces_test_raw": cf_stats(cf["test"], "cf_test_raw"),
        "livecodebench_raw": lcb_stats(lcb, args.lcb_cutoff),
    }

    print("[cf] convert train")
    train_rows = convert(cf["train"], cf_row_to_verl, "train")
    print(f"[cf] train kept={len(train_rows)}")

    print("[lcb] convert test")
    test_rows = convert(lcb, lcb_row_to_verl, args.lcb_cutoff)
    print(f"[lcb] test kept={len(test_rows)}")

    stats["train_after_filter"] = {"n": len(train_rows)}
    stats["test_after_filter"] = {"n": len(test_rows)}

    train_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows)

    train_path = args.out_dir / "train.parquet"
    test_path = args.out_dir / "test.parquet"
    _write_parquet_to_hdfs(train_df, train_path)
    _write_parquet_to_hdfs(test_df, test_path)
    print(f"[out] {train_path} ({len(train_df)} rows)")
    print(f"[out] {test_path} ({len(test_df)} rows)")

    stats_path = args.out_dir / "stats.json"
    stats_json = json.dumps(stats, indent=2, default=str)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        tmp.write(stats_json)
        tmp_stats = Path(tmp.name)
    shutil.copy2(tmp_stats, stats_path)
    tmp_stats.unlink(missing_ok=True)
    print(f"[out] {stats_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
