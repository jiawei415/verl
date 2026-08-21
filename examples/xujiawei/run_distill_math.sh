#!/usr/bin/env bash
# On-policy distillation | text | vLLM rollout | FSDP training | NVIDIA GPUs

set -xeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Distillation on 8 GPUs: split student(4) + teacher(4).
: "${NGPUS_PER_NODE:=4}"
: "${PPO_MAX_TOKEN_LEN_PER_GPU:=16384}"
: "${ROLLOUT_GPU_MEM_UTIL:=0.85}"
: "${ROLLOUT_N:=1}"
: "${SAVE_FREQ:=200}"
source "$SCRIPT_DIR/common.sh"

# ---- Distillation-specific knobs ----
STUDENT_MODEL=${STUDENT_MODEL:-$MODEL_ROOT/Qwen3-4B-Base}
TEACHER_MODEL=${TEACHER_MODEL:-$MODEL_ROOT/Qwen3-8B-Base}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-4}

distillation_loss_mode=${DISTILLATION_LOSS_MODE:-k1}
use_policy_gradient=${USE_POLICY_GRADIENT:-True}
distillation_topk=${DISTILLATION_TOPK:-32}

teacher_tp=${TEACHER_TP:-2}
teacher_gpu_mem_util=${TEACHER_GPU_MEM_UTIL:-0.85}

MODEL_PATH="$STUDENT_MODEL"   # unify with common builder
setup_output "$project_name" "$experiment_name"
build_common_arrays

########################### parameter arrays ###########################
DATA=("${COMMON_DATA[@]}" algorithm.adv_estimator=grpo data.shuffle=False)

ACTOR=(
    "${COMMON_ACTOR[@]}"
    actor_rollout_ref.actor.use_torch_compile=True
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

ROLLOUT=(
    "${COMMON_ROLLOUT[@]}"
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens}
)

TRAINER=("${TRAINER_BASE[@]}" trainer.val_before_train=False)

EXTRA=(
    distillation.enabled=True
    distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE}
    distillation.nnodes=${NNODES}
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL"
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${teacher_tp}
    distillation.teacher_models.teacher_model.inference.name=vllm
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${teacher_gpu_mem_util}
    distillation.teacher_models.teacher_model.inference.max_model_len=${max_num_tokens}
    distillation.distillation_loss.loss_mode=${distillation_loss_mode}
    distillation.distillation_loss.topk=${distillation_topk}
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=${use_policy_gradient}
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

########################### launch ###########################
source "$SCRIPT_DIR/launch.sh" "$@"
