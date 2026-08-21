#!/usr/bin/env bash
# End-to-end retriever deployment:
#   1. Launch a 1-GPU H20 mlx worker (alias search-retriever).
#   2. Wait for it to be Ready.
#   3. Bootstrap: install deps, rsync HDFS→/tmp, start retrieval_server.py,
#      fork keep_alive.py to hold GPU util above the 30% cull threshold.
#   4. Poll /health until the server responds.
#   5. Print the RETRIEVAL_URL for the training side.
#
# Env inputs (all optional):
#   ALIAS          worker alias (default search-retriever)
#   WORKDIR        --workdir passed to mlx (default this repo root)
#   MAX_WAIT_MIN   overall wait cap in minutes (default 20)
#
# Usage:
#   bash examples/xujiawei/retriever/deploy_retriever_worker.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

ALIAS=${ALIAS:-search-retriever}
WORKDIR=${WORKDIR:-$REPO_ROOT}
MAX_WAIT_MIN=${MAX_WAIT_MIN:-20}
LAUNCH_LOG=${LAUNCH_LOG:-/tmp/deploy_retriever_launch.log}

echo "[deploy_retriever] alias=$ALIAS workdir=$WORKDIR"

# ---- 1. Launch in the background ----
: > "$LAUNCH_LOG"
echo "[deploy_retriever] launching mlx worker..."
nohup script -qc "mlx worker launch \
    --gpu 1 --cpu 23 --memory 237 \
    --type NVIDIA-H20 --resourcetype arnold --usergroup seed_tob \
    --cluster soil-hl \
    --queuename nvidia-h20.hpccluster-ydmbubdyq6f9dwd0mqr4.ai \
    --alias $ALIAS \
    --workdir $WORKDIR \
    -- sleep infinity" "$LAUNCH_LOG" > /dev/null 2>&1 &
disown
echo "[deploy_retriever] launcher backgrounded (log=$LAUNCH_LOG)"

# ---- 2. Extract our worker id straight from the launcher TUI log ----
# The TUI writes lines like `worker-0 (4174418) Pending`, then `... Ready`.
# We just need the numeric id — first `worker-0 (<num>)` occurrence wins.
deadline=$(( $(date +%s) + MAX_WAIT_MIN * 60 ))
WID=""
while (( $(date +%s) < deadline )); do
    if [[ -s "$LAUNCH_LOG" ]]; then
        WID=$(awk 'match($0, /worker-0 \(([0-9]+)\)/, m) { print m[1]; exit }' "$LAUNCH_LOG" || true)
        if [[ -n "$WID" ]]; then
            echo "[deploy_retriever] launched worker id=$WID (from TUI log)"
            break
        fi
    fi
    sleep 2
done
[[ -z "$WID" ]] && { echo "[deploy_retriever] TIMEOUT: no worker id in $LAUNCH_LOG"; exit 1; }

# ---- 3. Wait for that worker to publish a pod IP (real IPv6, not a timestamp) ----
POD_IP=""
while (( $(date +%s) < deadline )); do
    cand=$(mlx worker list -l 200 2>/dev/null | awk -v w="$WID" '$1 == w {print $6}')
    # Accept only if col6 is a pure hex+colon string (IPv6). Rejects
    # timestamps like `2026-08-20T12:04:28` that appear when pod IP is absent.
    if [[ "$cand" =~ ^[0-9a-fA-F:]+$ && "$cand" == *:*:*:* ]]; then
        POD_IP="$cand"
        echo "[deploy_retriever] worker Ready: id=$WID pod=$POD_IP"
        break
    fi
    echo "[deploy_retriever] worker $WID still scheduling (col6=${cand:-empty})..."
    sleep 10
done
[[ -z "$POD_IP" ]] && { echo "[deploy_retriever] TIMEOUT: no pod IP for $WID"; exit 1; }

# ---- 4. Run bootstrap on the worker (installs + rsync + start server + keep_alive) ----
echo "[deploy_retriever] bootstrapping on worker $WID..."
timeout 60 mlx worker login "$WID" -- bash \
    "$WORKDIR/examples/xujiawei/retriever/_bootstrap_worker.sh"

# ---- 5. Poll /health until the server responds ----
url="http://[$POD_IP]:8000"
echo "[deploy_retriever] polling health at $url/health ..."
until_ts=$(( $(date +%s) + 900 ))     # up to 15 min for rsync + FAISS load
while (( $(date +%s) < until_ts )); do
    if timeout 10 mlx worker login "$WID" -- bash -c \
        'curl --noproxy "*" -sf --max-time 3 http://[::1]:8000/health' >/dev/null 2>&1; then
        echo "[deploy_retriever] /health responded"
        break
    fi
    sleep 15
    echo "[deploy_retriever]   still loading ..."
done

# Final cross-worker check (via the node IPv6)
if ! timeout 15 curl --noproxy '*' -sf --max-time 5 "$url/health" >/dev/null 2>&1; then
    echo "[deploy_retriever] WARNING: /health via $url unreachable from this host"
    echo "[deploy_retriever] (may still work from another mlx worker — cross-worker IPv6 confirmed earlier)"
fi

# ---- 6. Emit training-side env ----
cat <<EOF

================================================================================
  Retriever worker ready.
  worker_id     : $WID
  pod IPv6      : $POD_IP
  RETRIEVAL_URL : http://[$POD_IP]:8000/retrieve

  Use on training side:
      export RETRIEVAL_URL="http://[$POD_IP]:8000/retrieve"
      export no_proxy="*"

  Sanity check from any mlx worker:
      bash $SCRIPT_DIR/test_cross_worker.sh $POD_IP 8000
================================================================================
EOF
