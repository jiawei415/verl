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
train_batch_size=${TRAIN_BATCH_SIZE:-128}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-128}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-2048}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}
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
entropy_coeff=${ENTROPY_COEFF:-0}
save_freq=${SAVE_FREQ:-20}
test_freq=${TEST_FREQ:-5}

# ---- Shared Hydra arg arrays (algo scripts extend or override) ----
COMMON_DATA=(
    algorithm.use_kl_in_reward=False
    data.train_files="$train_files"
    data.val_files="$val_files"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
    # Rollout correction: MONITOR-ONLY diagnostics + bypass_mode loss.
    # - bypass_mode=True  => loss uses rollout log_prob as the "old" log_prob
    #   (skip training-side old_log_prob in ratio computation). Training-side
    #   old_log_prob is still computed when needed (e.g. OTB Σπ²) but only for
    #   diagnostics, not for the policy loss.
    # - monitor_only=True => IS weights + RS masks computed but NOT applied,
    #   so `rollout_corr/*` metrics show up without changing training math.
    algorithm.rollout_correction.rollout_is=${ROLLOUT_IS:-token}
    algorithm.rollout_correction.rollout_is_threshold=${ROLLOUT_IS_THRESHOLD:-2.0}
    algorithm.rollout_correction.rollout_is_batch_normalize=${ROLLOUT_IS_BATCH_NORM:-False}
    algorithm.rollout_correction.rollout_rs="'${ROLLOUT_RS:-token_k1,seq_sum_k1,seq_mean_k1,seq_max_k1}'"
    algorithm.rollout_correction.rollout_rs_threshold=${ROLLOUT_RS_THRESHOLD:-1000.0}
    algorithm.rollout_correction.bypass_mode=${ROLLOUT_CORR_BYPASS:-True}
    algorithm.rollout_correction.loss_type=${ROLLOUT_CORR_LOSS_TYPE:-ppo_clip}
    algorithm.rollout_correction.monitor_only=${ROLLOUT_CORR_MONITOR_ONLY:-True}
)

COMMON_TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.max_actor_ckpt_to_keep=2
    trainer.log_val_generations=10
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
        actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
        actor_rollout_ref.actor.fsdp_config.param_offload=False
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    )
    COMMON_ROLLOUT=(
        actor_rollout_ref.rollout.name=vllm
        actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
        actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
        actor_rollout_ref.rollout.n=${rollout_n}
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
        actor_rollout_ref.rollout.calculate_log_probs=True
        actor_rollout_ref.rollout.val_kwargs.n=4
        actor_rollout_ref.rollout.val_kwargs.temperature=1.0
        actor_rollout_ref.rollout.val_kwargs.top_p=0.95
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
