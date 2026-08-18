#!/bin/bash
# Download open-r1/codeforces (verifiable config) + livecodebench/code_generation_lite
# to /mnt/hdfs/byte_data_seed/hdd_hldy/user/xujiawei.415/hf_datasets/{codeforces,livecodebench}
#
# Stages to local NVME (/tmp/hf_cache) first, then rsync to HDFS mount.
#
# Usage: bash verl/examples/data_preprocess/download_code_datasets.sh [hdfs_dst] [local_cache]

set -euo pipefail

HDFS_DST="${1:-/mnt/hdfs/byte_data_seed/hdd_hldy/user/xujiawei.415/hf_datasets}"
LOCAL_CACHE="${2:-/tmp/hf_cache}"

export http_proxy="${http_proxy:-http://sys-proxy-rd-relay.byted.org:8118}"
export https_proxy="${https_proxy:-${http_proxy}}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME="${LOCAL_CACHE}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[dl] hdfs_dst   : ${HDFS_DST}"
echo "[dl] local_cache: ${LOCAL_CACHE}"

python "${SCRIPT_DIR}/download_code_datasets.py" \
    --hdfs_dst "${HDFS_DST}" \
    --local_cache "${LOCAL_CACHE}"

echo "[dl] all done."
