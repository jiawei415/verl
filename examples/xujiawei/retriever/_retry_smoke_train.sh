s   #!/usr/bin/env bash
set -eu
pkill -9 -f main_ppo 2>/dev/null || true
pkill -9 -f vllm 2>/dev/null || true
pkill -9 -f ray 2>/dev/null || true
sleep 3
rm -f /tmp/search_train.log

export https_proxy=http://sys-proxy-rd-relay.byted.org:8118 http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy="*"
export RETRIEVAL_URL=${RETRIEVAL_URL:-http://[2605:340:cd51:4900:3f7e:2291:e203:597f]:8000/retrieve}
export MODEL_NAME=${MODEL_NAME:-Qwen3-4B-Instruct-2507}
export ROLLOUT_TP=4   # 2 vLLM replicas x TP=4, less port contention
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-search_r1_smoke}   # fresh dir, avoid stale 7B ckpt resume

cd /mnt/bn/jiawei-llm2/mlx/users/xujiawei.415/playground/VERL/verl
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
