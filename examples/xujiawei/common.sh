#!/usr/bin/env bash
# Shared environment/knobs for xujiawei training scripts.
# Sourced by run_*.sh; do NOT execute directly.

export VLLM_USE_V1=1
export WANDB_OFFICIAL=1
export WANDB_API_KEY=2e430da03653e9b9961aaa2a0facadd7fe45204a
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118

# ---- HDFS paths ----
HDFS_PATH=${HDFS_PATH:-/mnt/hdfs/byte_data_seed/hdd_hldy/user/xujiawei.415}
MODEL_ROOT=${MODEL_ROOT:-$HDFS_PATH/hf_models}
DATA_PATH=${DATA_PATH:-$HDFS_PATH/hf_datasets}
CKPT_PATH=${CKPT_PATH:-$HDFS_PATH/checkpoints}

# ---- Cluster ----
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# ---- Data files ----
# TRAIN_FILES / VAL_FILES: space-separated list of paths relative to $DATA_PATH.
# Example: TRAIN_FILES="math/dapo_train.parquet"
TRAIN_FILES=${TRAIN_FILES:-"math/dapo_train.parquet"}
VAL_FILES=${VAL_FILES:-"math/gsm8k_test.parquet math/math500_test.parquet math/aime24_test.parquet math/aime25_test.parquet"}

read -ra _train_arr <<< "$TRAIN_FILES"
read -ra _val_arr   <<< "$VAL_FILES"
train_files="[$(printf "'$DATA_PATH/%s', " "${_train_arr[@]}")"; train_files="${train_files%, }]"
val_files="[$(printf "'$DATA_PATH/%s', " "${_val_arr[@]}")";     val_files="${val_files%, }]"

# ---- Shared training knobs ----
train_batch_size=${TRAIN_BATCH_SIZE:-64}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-64}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-8192}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}
max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))
actor_lr=${ACTOR_LR:-1e-6}
total_epochs=${TOTAL_EPOCHS:-15}

# ---- Logging defaults (algo scripts may override) ----
project_name=${PROJECT_NAME:-verl_test}
experiment_name=${EXPERIMENT_NAME:-verl_exp}

# ---- Model / rollout / checkpoint defaults (algo scripts may override) ----
# MODEL_NAME: short name under $MODEL_ROOT (e.g. Qwen3-8B-Base). If set, wins
# over MODEL_PATH default. MODEL_PATH still overrides both if set explicitly.
MODEL_NAME=${MODEL_NAME:-Qwen3-8B-Base}
MODEL_PATH=${MODEL_PATH:-$MODEL_ROOT/$MODEL_NAME}
rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.75}
rollout_n=${ROLLOUT_N:-8}
val_n=${VAL_N:-4}
max_turns=${MAX_TURNS:-4}
entropy_coeff=${ENTROPY_COEFF:-0}
save_freq=${SAVE_FREQ:-20}
test_freq=${TEST_FREQ:-5}

# ---- Single min-p knob (applies to training-logit masking, train rollout
# sampling, and val rollout sampling). Default 0 => disabled everywhere.
# Enable with e.g. `MIN_P=6.144212353328e-06` (= exp(-12)) to use min-p
# sampling as the sole tail filter. Val temperature/top_p are forced to
# 1.0/1.0 in COMMON_ROLLOUT so min-p is the only cutoff when enabled.
min_p=${MIN_P:-0.0}

# ---- Shared Hydra arg arrays (algo scripts extend or override) ----
COMMON_DATA=(
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=${KL_COEF:-0.0}
    data.train_files="$train_files"
    data.val_files="$val_files"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
    # Bypass-mode policy loss: ratio uses rollout_log_probs as `old`, but the
    # training-side old_log_probs are still computed for diagnostics (Σπ²,
    # entropy) and surfaced via `rollout_corr/*` metrics (kl/ppl/chi²).
    algorithm.rollout_correction.bypass_mode=${ROLLOUT_CORR_BYPASS:-True}
    algorithm.rollout_correction.loss_type=${ROLLOUT_CORR_LOSS_TYPE:-ppo_clip}
)

COMMON_TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.max_actor_ckpt_to_keep=2
    trainer.val_before_train=${VAL_BEFORE_TRAIN:-True}
    trainer.log_val_generations=${LOG_VAL_GENERATIONS:-50}
    trainer.total_epochs=${total_epochs}
)

# Helper: call `setup_output "$project_name" "$experiment_name"` in the algo script
# after project_name/experiment_name are set to derive output_path + wandb/tb dirs.
setup_output() {
    local proj="$1"
    local exp="$2"
    output_path=${OUTPUT_PATH:-$CKPT_PATH/$proj/$exp}
    export WANDB_DIR=$output_path
    export TENSORBOARD_DIR=$output_path
}

# Build shared Hydra arg arrays. Call this from the algo script AFTER setting:
#   MODEL_PATH, project_name, experiment_name, save_freq, test_freq,
#   rollout_tp, rollout_gpu_mem_util  (and after setup_output).
# Produces: MODEL, COMMON_ACTOR, COMMON_ROLLOUT, COMMON_REF, TRAINER_BASE.
build_common_arrays() {
    MODEL=(
        actor_rollout_ref.model.path="$MODEL_PATH"
        actor_rollout_ref.model.use_remove_padding=True
        actor_rollout_ref.model.enable_gradient_checkpointing=True
    )
    COMMON_ACTOR=(
        actor_rollout_ref.actor.optim.lr=${actor_lr}
        actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
        actor_rollout_ref.actor.use_dynamic_bsz=True
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
        actor_rollout_ref.actor.use_kl_loss=False
        actor_rollout_ref.actor.kl_loss_coef=0.0
        actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
        # DAPO-style asymmetric PPO clip (low=0.2, high=0.28, dual-clip c=10).
        # Applies to both single-turn and multi-turn scripts.
        actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW:-0.2}
        actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}
        actor_rollout_ref.actor.clip_ratio_c=${CLIP_RATIO_C:-10.0}
        # FSDP CPU offload for large models / long context. Set FSDP_OFFLOAD=0 to disable.
        actor_rollout_ref.actor.fsdp_config.param_offload=${FSDP_OFFLOAD:-True}
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=${FSDP_OFFLOAD:-True}
        # Emit variance-proxy metrics (`gradient_variance_proxy_*`) every step;
        # requires Σπ² from the actor forward (small extra compute).
        actor_rollout_ref.actor.calculate_sum_pi_squared=${CALCULATE_SUM_PI_SQUARED:-True}
        # min-p gradient masking on training logits (same value as sampling min-p
        # by default -- MIN_P=0 disables).
        actor_rollout_ref.actor.train_min_p=${min_p}
        # Worker-side mirror of bypass-mode settings. Ray workers read this at
        # init time; the driver-side `apply_bypass_mode` mutation is a no-op
        # for already-instantiated workers.
        actor_rollout_ref.actor.policy_loss.loss_mode=${ROLLOUT_CORR_LOSS_MODE:-bypass_mode}
        actor_rollout_ref.actor.policy_loss.rollout_correction.bypass_mode=${ROLLOUT_CORR_BYPASS:-True}
        actor_rollout_ref.actor.policy_loss.rollout_correction.loss_type=${ROLLOUT_CORR_LOSS_TYPE:-ppo_clip}
    )
    COMMON_ROLLOUT=(
        actor_rollout_ref.rollout.name=vllm
        actor_rollout_ref.rollout.mode=${ROLLOUT_MODE:-sync}
        actor_rollout_ref.rollout.max_model_len=${max_num_tokens}
        actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
        actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
        actor_rollout_ref.rollout.n=${rollout_n}
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
        actor_rollout_ref.rollout.calculate_log_probs=True
        # Train rollout sampling: temperature=1, top_p=1, top_k=-1. min-p is the
        # sole tail filter when enabled (MIN_P=0 disables).
        actor_rollout_ref.rollout.temperature=${TRAIN_TEMPERATURE:-1.0}
        actor_rollout_ref.rollout.top_p=${TRAIN_TOP_P:-1.0}
        actor_rollout_ref.rollout.top_k=${TRAIN_TOP_K:--1}
        actor_rollout_ref.rollout.min_p=${min_p}
        # Val rollout sampling: mirrors train defaults (temp/top_p/top_k = 1/1/-1).
        # Override with VAL_N / VAL_TEMPERATURE / VAL_TOP_P / VAL_TOP_K.
        actor_rollout_ref.rollout.val_kwargs.n=${val_n}
        actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE:-1.0}
        actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P:-1.0}
        actor_rollout_ref.rollout.val_kwargs.top_k=${VAL_TOP_K:--1}
        actor_rollout_ref.rollout.val_kwargs.min_p=${min_p}
    )
    COMMON_REF=(
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
        actor_rollout_ref.ref.fsdp_config.param_offload=True
    )
    TRAINER_BASE=(
        "${COMMON_TRAINER[@]}"
        trainer.default_local_dir="$output_path"
        trainer.project_name=${project_name}
        trainer.experiment_name=${experiment_name}
        trainer.save_freq=${save_freq}
        trainer.test_freq=${test_freq}
    )
}
