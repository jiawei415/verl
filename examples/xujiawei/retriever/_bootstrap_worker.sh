#!/usr/bin/env bash
# One-shot bootstrap on a fresh retriever worker.
# 1. Ensure fastapi/starlette + faiss-cpu deps
# 2. Rsync + start server via start_retriever.sh (which touches .ready and health-checks)
set -eu
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118 http_proxy=http://sys-proxy-rd-relay.byted.org:8118
pip install --user "fastapi==0.115.6" "starlette==0.41.3" faiss-cpu 2>&1 | tail -3
cd /mnt/bn/jiawei-llm2/mlx/users/xujiawei.415/playground/VERL/verl
nohup bash examples/xujiawei/start_retriever.sh > /tmp/start_retriever.log 2>&1 </dev/null &
disown
echo "starter_pid=$!"
