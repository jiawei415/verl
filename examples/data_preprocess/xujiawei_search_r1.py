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

# --- With-system-prompt (WSP) template ------------------------------------------
# Longer, tutorial-style system message plus a single-turn worked example, mirroring
# the code / math multi-turn scripts so base models get a strong format anchor
# before RL exploration.
SYSTEM_CONTENT_WSP = """You are a helpful assistant that answers questions by iteratively reasoning and searching Wikipedia.

## Workflow

1. Read the question. Write a short reasoning inside <think> and </think>.
2. If you lack knowledge, call the search engine by writing exactly:
     <tool_call> your query string </tool_call>
   The system will return the top passages between <tool_response> and </tool_response>.
3. After each tool response, add another <think>...</think> block to analyze the results and plan the next step.
4. Repeat steps 2-3 as many times as you need. Prefer short, focused queries over long paragraphs.
5. Once you have enough information, output the final answer inside <answer> and </answer>. Keep the answer short -- a single entity, name, number, or short phrase. Do NOT explain.

## Tag rules

- `<tool_call>` contains a bare query string, NOT JSON, NOT a function schema.
- Every `<tool_call>` must be closed with `</tool_call>`; then STOP and wait for the `<tool_response>` block.
- The final answer appears exactly ONCE, at the very end, inside `<answer>` and `</answer>`.
- Do NOT invent tool responses. Wait for the actual `<tool_response>` block.

## Example

Question: When did Albert Einstein win the Nobel Prize in Physics?
<think> I need to look up when Einstein won the Nobel Prize. </think>
<tool_call> Albert Einstein Nobel Prize Physics year </tool_call>
<tool_response>
[1] "Einstein was awarded the 1921 Nobel Prize in Physics for his discovery of the photoelectric effect..."
</tool_response>
<think> The result says 1921. </think>
<answer> 1921 </answer>"""

USER_CONTENT_PREFIX_WSP = "Question: "

# --- WSP with Search-R1 original tags -------------------------------------------
# Same layout as WSP but uses the Search-R1 paper's <search>...</search> and
# <information>...</information> tags, which are closer to base pretrain
# distribution (wiki article + citation style) than the Qwen SFT-flavoured
# <tool_call>/<tool_response> tags.
SYSTEM_CONTENT_WSP_SR = """You are a helpful assistant that answers questions by iteratively reasoning and searching Wikipedia.

## Workflow

1. Read the question. Write a short reasoning inside <think> and </think>.
2. If you lack knowledge, call the search engine by writing exactly:
     <search> your query string </search>
   The system will return the top passages between <information> and </information>.
3. After each information block, add another <think>...</think> block to analyze the results and plan the next step.
4. Repeat steps 2-3 as many times as you need. Prefer short, focused queries over long paragraphs.
5. Once you have enough information, output the final answer inside <answer> and </answer>. Keep the answer short -- a single entity, name, number, or short phrase. Do NOT explain.

## Tag rules

- `<search>` contains a bare query string, NOT JSON, NOT a function schema.
- Every `<search>` must be closed with `</search>`; then STOP and wait for the `<information>` block.
- The final answer appears exactly ONCE, at the very end, inside `<answer>` and `</answer>`.
- Do NOT invent information blocks. Wait for the actual `<information>` block.

## Example

Question: When did Albert Einstein win the Nobel Prize in Physics?
<think> I need to look up when Einstein won the Nobel Prize. </think>
<search> Albert Einstein Nobel Prize Physics year </search>
<information>
[1] "Einstein was awarded the 1921 Nobel Prize in Physics for his discovery of the photoelectric effect..."
</information>
<think> The result says 1921. </think>
<answer> 1921 </answer>"""


def _build_row(row, split_name: str, row_index: int, wsp: bool = False, tag_style: str = "tool_call") -> pd.Series:
    """Transform one raw HF row into verl format.

    Args:
        wsp: If True, use the with-system-prompt template (long tutorial system
            content + worked single-turn example, user turn only carries the
            question). If False, keep the Search-R1 official layout (no system
            message, protocol lives in the user turn).
        tag_style: Which tag pair to teach in the WSP system prompt. One of:
            - ``tool_call`` (default): `<tool_call>` / `<tool_response>`,
              matches SearchR1ToolParser out of the box.
            - ``search``: `<search>` / `<information>`, the Search-R1 paper's
              original tags -- closer to base pretrain distribution but the
              tool parser + tool_agent_loop need matching updates before RL.
            Only takes effect when ``wsp=True``.
    """
    question = row.get("question", "")
    if wsp:
        user_content = USER_CONTENT_PREFIX_WSP + question
        if tag_style == "search":
            sys_content = SYSTEM_CONTENT_WSP_SR
        elif tag_style == "tool_call":
            sys_content = SYSTEM_CONTENT_WSP
        else:
            raise ValueError(f"unknown tag_style={tag_style!r}, expected tool_call | search")
        prompt = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": user_content},
        ]
    else:
        user_content = USER_CONTENT_PREFIX.rstrip("\n") + question
        # Search-R1 official protocol targets base models with no system prompt:
        # the entire tool-use spec lives inside the user turn, matching
        # https://github.com/PeterGriffinJin/Search-R1/blob/main/scripts/data_process/nq_search.py
        prompt = [
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


def _process_split(df_raw: pd.DataFrame, split_name: str, wsp: bool = False, tag_style: str = "tool_call") -> pd.DataFrame:
    logger.info(f"Processing split={split_name}, wsp={wsp}, tag_style={tag_style}, rows={len(df_raw)}")
    df_raw = df_raw.reset_index(drop=True)
    return df_raw.apply(
        lambda r: _build_row(r, split_name=split_name, row_index=int(r.name), wsp=wsp, tag_style=tag_style), axis=1
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
    parser.add_argument(
        "--with_system_prompt",
        action="store_true",
        help="Use the with-system-prompt (WSP) template and write to "
        "{train,test}_wsp[_sr].parquet instead of overwriting the default files.",
    )
    parser.add_argument(
        "--tag_style",
        choices=("tool_call", "search"),
        default="tool_call",
        help="Which tag pair the WSP system prompt teaches: "
        "`tool_call` -> <tool_call>/<tool_response> (default, matches "
        "SearchR1ToolParser); `search` -> <search>/<information> "
        "(Search-R1 paper's original tags, closer to base pretrain "
        "distribution). Only meaningful with --with_system_prompt.",
    )
    args = parser.parse_args()

    if args.tag_style != "tool_call" and not args.with_system_prompt:
        parser.error("--tag_style only has an effect with --with_system_prompt")

    os.makedirs(args.dst_dir, exist_ok=True)
    if args.with_system_prompt:
        suffix = "_wsp_sr" if args.tag_style == "search" else "_wsp"
    else:
        suffix = ""
    with tempfile.TemporaryDirectory() as tmp_dl:
        for split in ("train", "test"):
            src_fname = f"{split}.parquet"
            logger.info(f"Downloading {src_fname} from {args.hf_repo_id}")
            local_path = hf_hub_download(
                repo_id=args.hf_repo_id,
                filename=src_fname,
                repo_type="dataset",
                local_dir=tmp_dl,
                local_dir_use_symlinks=False,
            )
            df_raw = pd.read_parquet(local_path)
            logger.info(f"  loaded rows={len(df_raw)}, columns={list(df_raw.columns)}")

            if split == "test":
                df_raw = _sample_val(df_raw, args.val_per_subset)

            df_out = _process_split(df_raw, split_name=split, wsp=args.with_system_prompt, tag_style=args.tag_style)
            out_fname = f"{split}{suffix}.parquet"
            out_path = os.path.join(args.dst_dir, out_fname)
            _atomic_write(df_out, out_path)
            logger.info(f"Wrote {out_path}  rows={len(df_out)}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
