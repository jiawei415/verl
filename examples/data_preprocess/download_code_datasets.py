"""Download Codeforces (open-r1/codeforces, verifiable config) + LiveCodeBench
(livecodebench/code_generation_lite, latest) HF datasets.

Snapshot to local NVME first (fast, parallel), then rsync into HDFS mount.

Rationale:
- /mnt/hdfs is FUSE-mounted; many small file writes are slow and can fail.
- /tmp is a 3.5T NVME, plenty of headroom for HF snapshots.

Usage: python verl/examples/data_preprocess/download_code_datasets.py [--hdfs_dst ...] [--local_cache ...]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_HDFS_DST = "/mnt/hdfs/byte_data_seed/hdd_hldy/user/xujiawei.415/hf_datasets"
DEFAULT_LOCAL_CACHE = "/tmp/hf_cache"

# (repo_id, allow_patterns_or_None, local_subdir, hdfs_subdir)
DATASETS = [
    (
        "open-r1/codeforces",
        ["verifiable/*", "README.md", "*.json", ".gitattributes"],
        "open-r1__codeforces",
        "codeforces",
    ),
    (
        "livecodebench/code_generation_lite",
        None,  # full snapshot (still small, MBs)
        "livecodebench__code_generation_lite",
        "livecodebench",
    ),
]


def snapshot(repo_id: str, allow_patterns, local_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    print(f"[dl] {repo_id} -> {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
        max_workers=8,
    )
    print(f"[ok ] {repo_id}")


def rsync_to_hdfs(src: Path, dst: Path) -> None:
    print(f"[cp ] {src} -> {dst}")
    dst.mkdir(parents=True, exist_ok=True)
    # rsync -a preserves timestamps + is idempotent (skip already-copied files).
    # HDFS FUSE lacks some POSIX ops; fall back to cp -r if rsync fails.
    try:
        subprocess.check_call(["rsync", "-a", "--info=progress2", f"{src}/", f"{dst}/"])
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[cp ] rsync unavailable/failed, falling back to cp -r")
        for item in src.iterdir():
            target = dst / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
    print(f"[ok ] {dst}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdfs_dst", default=DEFAULT_HDFS_DST)
    ap.add_argument("--local_cache", default=DEFAULT_LOCAL_CACHE)
    ap.add_argument("--skip_hdfs_copy", action="store_true", help="Only fetch to local NVME.")
    ap.add_argument("--only", nargs="*", choices=[d[0] for d in DATASETS], help="Subset of repos.")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_HOME", args.local_cache)

    local_root = Path(args.local_cache)
    hdfs_root = Path(args.hdfs_dst)
    local_root.mkdir(parents=True, exist_ok=True)

    selected = [d for d in DATASETS if not args.only or d[0] in args.only]
    for repo_id, patterns, local_sub, hdfs_sub in selected:
        local_dir = local_root / local_sub
        snapshot(repo_id, patterns, local_dir)
        if not args.skip_hdfs_copy:
            rsync_to_hdfs(local_dir, hdfs_root / hdfs_sub)

    print(f"[done] local_cache={local_root} hdfs_dst={hdfs_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
