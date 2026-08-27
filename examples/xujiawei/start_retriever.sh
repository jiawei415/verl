#!/usr/bin/env bash
# Start the local Wiki-18 dense retrieval server on a chosen GPU set.
# Idempotent: rsync HDFS -> /tmp only on first run (or when .ready is missing).
#
# Env inputs (all optional):
#   RETRIEVER_GPU_IDS    comma-separated CUDA IDs for the server. Default "7"
#   RETRIEVAL_PORT       HTTP port. Default 8000
#   RETRIEVAL_HOST       Bind host. Default 0.0.0.0
#   HDFS_SEARCHR1_DIR    HDFS root of index/corpus. Default $HDFS_PATH/searchR1
#   MODEL_ROOT           HDFS models root; encoder read from $MODEL_ROOT/e5-base-v2
#   LOCAL_SEARCHR1_DIR   Local staging dir. Default /tmp/searchR1
#   RETRIEVER_LOG        Log path. Default /tmp/retriever.log
#   TOPK                 Default top-K. Default 3
#   FORCE_RESYNC=1       Remove .ready and re-rsync
#   SKIP_PIP=1           Skip fastapi/starlette/faiss-cpu install
#   KEEP_ALIVE=0         Disable GPU keep-alive burst (default on for mlx workers)
#   FOREGROUND=1         Don't return after health check — tail $RETRIEVER_LOG
#                        so log streams to the terminal. Ctrl-C exits tail;
#                        server + keep_alive keep running in the background.
#
# Usage:
#   bash examples/xujiawei/start_retriever.sh
#   RETRIEVER_GPU_IDS=6,7 bash examples/xujiawei/start_retriever.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HDFS_PATH=${HDFS_PATH:-/mnt/hdfs/byte_data_seed/hdd_hldy/user/xujiawei.415}
MODEL_ROOT=${MODEL_ROOT:-$HDFS_PATH/hf_models}

RETRIEVER_GPU_IDS=${RETRIEVER_GPU_IDS:-0}
RETRIEVAL_PORT=${RETRIEVAL_PORT:-8000}
RETRIEVAL_HOST=${RETRIEVAL_HOST:-::}    # `::` = dual-stack IPv4+IPv6, `0.0.0.0` = IPv4 only
HDFS_SEARCHR1_DIR=${HDFS_SEARCHR1_DIR:-$HDFS_PATH/searchR1}
LOCAL_SEARCHR1_DIR=${LOCAL_SEARCHR1_DIR:-/tmp/searchR1}
RETRIEVER_LOG=${RETRIEVER_LOG:-/tmp/retriever.log}
TOPK=${TOPK:-3}

INDEX_FILE=$LOCAL_SEARCHR1_DIR/wiki-18-e5-index-HNSW64/e5_HNSW64.index
CORPUS_FILE=$LOCAL_SEARCHR1_DIR/wiki-18.jsonl
ENCODER_DIR=$MODEL_ROOT/e5-base-v2

# ---- Stage runtime data ----
if [[ "${FORCE_RESYNC:-0}" == "1" ]]; then
    rm -f "$LOCAL_SEARCHR1_DIR/.ready"
fi
if [[ ! -f "$LOCAL_SEARCHR1_DIR/.ready" ]]; then
    echo "[start_retriever] staging $HDFS_SEARCHR1_DIR -> $LOCAL_SEARCHR1_DIR"
    mkdir -p "$LOCAL_SEARCHR1_DIR"
    rsync -aW --partial --info=progress2 \
        "$HDFS_SEARCHR1_DIR/wiki-18.jsonl" "$LOCAL_SEARCHR1_DIR/"
    rsync -aW --partial --info=progress2 \
        "$HDFS_SEARCHR1_DIR/wiki-18-e5-index-HNSW64/" "$LOCAL_SEARCHR1_DIR/wiki-18-e5-index-HNSW64/"
    touch "$LOCAL_SEARCHR1_DIR/.ready"
else
    echo "[start_retriever] runtime files already staged (.ready)"
fi

for f in "$INDEX_FILE" "$CORPUS_FILE"; do
    [[ -f "$f" ]] || { echo "MISSING $f" >&2; exit 1; }
done
[[ -d "$ENCODER_DIR" ]] || { echo "MISSING encoder dir $ENCODER_DIR" >&2; exit 1; }

# ---- Install deps (idempotent, --user; skip if SKIP_PIP=1) ----
if [[ "${SKIP_PIP:-0}" != "1" ]]; then
    echo "[start_retriever] ensuring fastapi/starlette/faiss-cpu installed"
    pip install --user --quiet fastapi==0.115.6 starlette==0.41.3 faiss-cpu \
        || { echo "[start_retriever] pip install failed" >&2; exit 1; }
fi

# ---- Kill any previous server on this port ----
if fuser -sk "${RETRIEVAL_PORT}/tcp" 2>/dev/null; then
    echo "[start_retriever] killed previous server on :$RETRIEVAL_PORT"
    sleep 1
fi

# ---- Launch ----
echo "[start_retriever] launching on GPUs=$RETRIEVER_GPU_IDS, port=$RETRIEVAL_PORT"
FAISS_GPU_FLAG=${FAISS_GPU_FLAG:-}    # empty = CPU (HNSW must be CPU); set to "--faiss_gpu" for Flat index
CUDA_VISIBLE_DEVICES=$RETRIEVER_GPU_IDS nohup python3 "$SCRIPT_DIR/retriever/retrieval_server.py" \
    --index_path "$INDEX_FILE" \
    --corpus_path "$CORPUS_FILE" \
    --retriever_model "$ENCODER_DIR" \
    --topk "$TOPK" \
    ${FAISS_GPU_FLAG} \
    --host "$RETRIEVAL_HOST" \
    --port "$RETRIEVAL_PORT" \
    > "$RETRIEVER_LOG" 2>&1 &
PID=$!
echo "[start_retriever] pid=$PID, log=$RETRIEVER_LOG"

# ---- Health check ----
for i in $(seq 1 120); do
    if curl --noproxy '*' -sf "http://[::1]:${RETRIEVAL_PORT}/health" >/dev/null 2>&1; then
        echo "[start_retriever] ready: http://[::1]:${RETRIEVAL_PORT}/retrieve"
        # ---- Keep-alive: keep GPU util > 30% so mlx doesn't cull the worker. ----
        if [[ "${KEEP_ALIVE:-1}" == "1" ]]; then
            KEEP_ALIVE_LOG=${KEEP_ALIVE_LOG:-/tmp/keep_alive.log}
            KEEP_ALIVE_BURST=${KEEP_ALIVE_BURST:-32}
            KEEP_ALIVE_BATCH=${KEEP_ALIVE_BATCH:-8}
            echo "[start_retriever] launching keep_alive (log=$KEEP_ALIVE_LOG)"
            CUDA_VISIBLE_DEVICES=$RETRIEVER_GPU_IDS nohup python3 -u "$SCRIPT_DIR/retriever/keep_alive.py" \
                --model_path "$ENCODER_DIR" \
                --batch "$KEEP_ALIVE_BATCH" \
                --burst_iters "$KEEP_ALIVE_BURST" \
                > "$KEEP_ALIVE_LOG" 2>&1 </dev/null &
            disown
        fi
        # ---- Foreground: stream logs to the terminal (Ctrl-C exits tail; server keeps running). ----
        if [[ "${FOREGROUND:-0}" == "1" ]]; then
            echo "[start_retriever] FOREGROUND=1 -> tailing $RETRIEVER_LOG (Ctrl-C to detach; server stays up)"
            exec tail -n +1 -F "$RETRIEVER_LOG"
        fi
        exit 0
    fi
    sleep 5
done
echo "[start_retriever] TIMEOUT waiting for health (see $RETRIEVER_LOG)" >&2
exit 1
