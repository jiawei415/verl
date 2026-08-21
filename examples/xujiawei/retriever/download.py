"""
Download Search-R1 retrieval artifacts (Wiki-18 corpus + HNSW64 e5 index) from
HuggingFace and stage them on HDFS. Also downloads the e5 encoder to
`$MODEL_ROOT/e5-base-v2`.

Strategy: HF -> local /tmp cache -> rsync to HDFS. This sidesteps HDFS FUSE
issues on large sequential writes and gives us a resumable copy.

Total footprint:
- HNSW64 index parts: ~30-50 GB (2 shards, joined into `e5_HNSW64.index`)
- wiki-18-corpus:     ~35 GB (.gz), ~76 GB decompressed
- e5-base-v2:         ~440 MB

Usage:
    export https_proxy=http://sys-proxy-rd-relay.byted.org:8118 \
           http_proxy=http://sys-proxy-rd-relay.byted.org:8118
    python examples/xujiawei/retriever/download.py
"""

from __future__ import annotations

import argparse
import gzip
import logging
import os
import shutil
import subprocess

from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


HDFS_ROOT = "/mnt/hdfs/byte_data_seed/hdd_hldy/user/xujiawei.415"
DEFAULT_STAGE = "/tmp/hf_stage/searchR1"
DEFAULT_SEARCHR1_DST = f"{HDFS_ROOT}/searchR1"
DEFAULT_MODEL_DST = f"{HDFS_ROOT}/hf_models/e5-base-v2"

INDEX_REPO = "PeterJinGo/wiki-18-e5-index-HNSW64"     # part_aa / part_ab
CORPUS_REPO = "PeterJinGo/wiki-18-corpus"             # wiki-18.jsonl.gz
ENCODER_REPO = "intfloat/e5-base-v2"


def _rsync(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(dst.rstrip("/"))), exist_ok=True)
    cmd = ["rsync", "-aW", "--partial", "--info=progress2", src.rstrip("/") + "/", dst.rstrip("/") + "/"]
    logger.info("+ %s", " ".join(cmd))
    subprocess.check_call(cmd)


def _download_repo(repo_id: str, dst: str, repo_type: str = "dataset") -> str:
    os.makedirs(dst, exist_ok=True)
    logger.info(f"Downloading {repo_id} -> {dst}")
    return snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=dst,
        local_dir_use_symlinks=False,
    )


def _join_index_parts(stage_dir: str) -> str:
    """Concatenate HNSW index shards into a single `.index` file."""
    parts = sorted(f for f in os.listdir(stage_dir) if f.startswith("part_"))
    if not parts:
        raise FileNotFoundError(f"no `part_*` files in {stage_dir}")
    out = os.path.join(stage_dir, "e5_HNSW64.index")
    logger.info(f"Joining {len(parts)} index shards -> {out}")
    with open(out, "wb") as w:
        for p in parts:
            src = os.path.join(stage_dir, p)
            logger.info(f"  + {p} ({os.path.getsize(src) / 1e9:.2f} GB)")
            with open(src, "rb") as r:
                shutil.copyfileobj(r, w, length=64 * 1024 * 1024)
    for p in parts:
        os.remove(os.path.join(stage_dir, p))
    return out


def _decompress_corpus(stage_dir: str) -> str:
    gz = os.path.join(stage_dir, "wiki-18.jsonl.gz")
    out = os.path.join(stage_dir, "wiki-18.jsonl")
    if not os.path.isfile(gz):
        raise FileNotFoundError(gz)
    logger.info(f"Decompressing {gz} -> {out}")
    with gzip.open(gz, "rb") as r, open(out, "wb") as w:
        shutil.copyfileobj(r, w, length=64 * 1024 * 1024)
    # keep the .gz too as a backup? cheap enough (~35 GB), leave for now.
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage_dir", default=DEFAULT_STAGE, help="Local staging directory (needs ~200 GB free).")
    parser.add_argument("--dst_searchr1", default=DEFAULT_SEARCHR1_DST)
    parser.add_argument("--dst_model", default=DEFAULT_MODEL_DST)
    parser.add_argument("--skip_index", action="store_true")
    parser.add_argument("--skip_corpus", action="store_true")
    parser.add_argument("--skip_encoder", action="store_true")
    parser.add_argument("--keep_stage", action="store_true", help="Do NOT remove the local staging dir after sync.")
    args = parser.parse_args()

    os.makedirs(args.stage_dir, exist_ok=True)

    # ---- 1. HNSW64 index ----
    if not args.skip_index:
        idx_stage = os.path.join(args.stage_dir, "index")
        _download_repo(INDEX_REPO, idx_stage)
        _join_index_parts(idx_stage)
        _rsync(idx_stage, os.path.join(args.dst_searchr1, "wiki-18-e5-index-HNSW64"))

    # ---- 2. Corpus ----
    if not args.skip_corpus:
        cor_stage = os.path.join(args.stage_dir, "corpus")
        _download_repo(CORPUS_REPO, cor_stage)
        _decompress_corpus(cor_stage)
        _rsync(cor_stage, args.dst_searchr1)

    # ---- 3. E5 encoder ----
    if not args.skip_encoder:
        enc_stage = os.path.join(args.stage_dir, "e5-base-v2")
        _download_repo(ENCODER_REPO, enc_stage, repo_type="model")
        _rsync(enc_stage, args.dst_model)

    # ---- 4. cleanup local stage ----
    if not args.keep_stage:
        logger.info(f"Removing stage dir {args.stage_dir}")
        shutil.rmtree(args.stage_dir, ignore_errors=True)

    logger.info("All Search-R1 artifacts staged on HDFS.")
    logger.info(f"  index / corpus: {args.dst_searchr1}")
    logger.info(f"  encoder:        {args.dst_model}")


if __name__ == "__main__":
    main()
