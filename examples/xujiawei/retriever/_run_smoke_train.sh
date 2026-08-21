#!/usr/bin/env bash
# Val + 10 train steps smoke test for Search-R1 multi-turn GRPO.
# Runs on the current worker; assumes 8 H20 GPUs available and a reachable
# retrieval server at RETRIEVAL_URL.
set -eu

export https_proxy=http://sys-proxy-rd-relay.byted.org:8118 http_proxy=http://sys-proxy-rd-relay.byted.org:8118
# Never proxy the retrieval URL (byted proxy would rewrite it).
export no_proxy="*"

export RETRIEVAL_URL=${RETRIEVAL_URL:-http://[2605:340:cd51:4900:3f7e:2291:e203:597f]:8000/retrieve}
export MODEL_NAME=${MODEL_NAME:-Qwen3-4B-Instruct-2507}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-search_r1_smoke}

cd /mnt/bn/jiawei-llm2/mlx/users/xujiawei.415/playground/VERL/verl

# Sanity: retriever reachable
if ! curl --noproxy '*' -sf --max-time 5 "${RETRIEVAL_URL%/retrieve}/health" >/dev/null; then
    echo "[smoke] retriever /health unreachable at $RETRIEVAL_URL"
    exit 1
fi
echo "[smoke] retriever OK, launching training"

# Kick training in background so nohup pattern survives login timeout.
nohup bash examples/xujiawei/run_grpo_search_multiturn.sh \
    trainer.val_before_train=True \
    trainer.total_training_steps=10 \
    trainer.total_epochs=1 \
    trainer.test_freq=100 \
    trainer.save_freq=100 \
    trainer.log_val_generations=10 \
    > /tmp/search_train.log 2>&1 </dev/null &
disown
echo "train_pid=$!"
sleep 3
tail -5 /tmp/search_train.log 2>/dev/null || true
