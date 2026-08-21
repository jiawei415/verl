"""
Preprocess Search-R1 dataset (PeterJinGo/nq_hotpotqa_train) into verl format.

- Train: full set from HF `train.parquet`.
- Val:   `test.parquet` grouped by `data_source`, first N (default 500) rows per
         sub-dataset (nq / triviaqa / popqa / hotpotqa / 2wikimultihopqa /
         musique / bamboogle) so val stays under ~3.5k rows.

Output verl-format parquets to
`/mnt/hdfs/byte_data_seed/hdd_hldy/user/xujiawei.415/hf_datasets/search_multiturn/`
staged via a local tmp dir (HDFS FUSE can flake on direct parquet writes).

Usage:
    python examples/data_preprocess/xujiawei_search_r1.py
    python examples/data_preprocess/xujiawei_search_r1.py --val_per_subset 200
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import tempfile

import pandas as pd
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


HDFS_ROOT = "/mnt/hdfs/byte_data_seed/hdd_hldy/user/xujiawei.415"
DEFAULT_DST = f"{HDFS_ROOT}/hf_datasets/search_multiturn"
HF_REPO = "PeterJinGo/nq_hotpotqa_train"

SYSTEM_CONTENT = "You are a helpful and harmless assistant."
USER_CONTENT_PREFIX = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack "
    "some knowledge, you can call a search engine by <tool_call> query </tool_call> "
    "and it will return the top searched results between <tool_response> and "
    "</tool_response>. You can search as many times as your want. If you find no "
    "further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. For example, "
    "<answer> Beijing </answer>. Question: "
)


def _build_row(row, split_name: str, row_index: int) -> pd.Series:
    """Transform one raw HF row into verl format."""
    question = row.get("question", "")
    user_content = USER_CONTENT_PREFIX.rstrip("\n") + question
    prompt = [
        {"role": "system", "content": SYSTEM_CONTENT},
        {"role": "user", "content": user_content},
    ]

    reward_model_data = row.get("reward_model")
    if isinstance(reward_model_data, dict) and "ground_truth" in reward_model_data:
        ground_truth = reward_model_data.get("ground_truth")
    else:
        ground_truth = row.get("golden_answers", [])

    raw_data_source = str(row.get("data_source", ""))
    data_source_tagged = "searchR1_" + raw_data_source

    tools_kwargs = {
        "search": {
            "create_kwargs": {
                "ground_truth": ground_truth,
                "question": question,
                "data_source": data_source_tagged,
            }
        }
    }
    extra_info = {
        "index": row_index,
        "need_tools_kwargs": True,
        "question": question,
        "split": split_name,
        "tools_kwargs": tools_kwargs,
        "raw_data_source": raw_data_source,
    }

    return pd.Series(
        {
            "data_source": data_source_tagged,
            "prompt": prompt,
            "ability": row.get("ability"),
            "reward_model": reward_model_data,
            "extra_info": extra_info,
            "metadata": row.get("metadata"),
        }
    )


def _process_split(df_raw: pd.DataFrame, split_name: str) -> pd.DataFrame:
    logger.info(f"Processing split={split_name}, rows={len(df_raw)}")
    df_raw = df_raw.reset_index(drop=True)
    return df_raw.apply(
        lambda r: _build_row(r, split_name=split_name, row_index=int(r.name)), axis=1
    )


def _sample_val(df: pd.DataFrame, per_subset: int) -> pd.DataFrame:
    if "data_source" not in df.columns:
        raise ValueError("expected `data_source` column in raw test parquet")
    parts = []
    for name, group in df.groupby("data_source"):
        head = group.head(per_subset)
        logger.info(f"  val subset {name}: {len(head)} / {len(group)}")
        parts.append(head)
    return pd.concat(parts, ignore_index=True)


def _atomic_write(df: pd.DataFrame, dst: str) -> None:
    """Stage locally then move to HDFS to sidestep FUSE random-write quirks."""
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = tmp.name
    df.to_parquet(tmp_path, index=False)
    shutil.copy(tmp_path, dst)
    os.remove(tmp_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo_id", default=HF_REPO)
    parser.add_argument("--dst_dir", default=DEFAULT_DST)
    parser.add_argument(
        "--val_per_subset",
        type=int,
        default=500,
        help="Rows kept per sub-dataset in the val split.",
    )
    args = parser.parse_args()

    os.makedirs(args.dst_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dl:
        for split in ("train", "test"):
            fname = f"{split}.parquet"
            logger.info(f"Downloading {fname} from {args.hf_repo_id}")
            local_path = hf_hub_download(
                repo_id=args.hf_repo_id,
                filename=fname,
                repo_type="dataset",
                local_dir=tmp_dl,
                local_dir_use_symlinks=False,
            )
            df_raw = pd.read_parquet(local_path)
            logger.info(f"  loaded rows={len(df_raw)}, columns={list(df_raw.columns)}")

            if split == "test":
                df_raw = _sample_val(df_raw, args.val_per_subset)

            df_out = _process_split(df_raw, split_name=split)
            out_path = os.path.join(args.dst_dir, fname)
            _atomic_write(df_out, out_path)
            logger.info(f"Wrote {out_path}  rows={len(df_out)}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
