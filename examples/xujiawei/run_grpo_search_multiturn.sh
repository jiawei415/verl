#!/usr/bin/env bash
# GRPO multi-turn on Search-R1 (NQ + HotpotQA train, 7-domain val).
#
# Uses a @function_tool from examples/xujiawei/search_tool.py (search backed
# by a local FAISS + e5-base retrieval server, see start_retriever.sh).
# Reward: official EM verifier via search_r1 route in the reward router.
#
# Data prep: python verl/examples/data_preprocess/xujiawei_search_r1.py
# Retriever: bash examples/xujiawei/start_retriever.sh  (or deploy_retriever_worker.sh)

set -xeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Multi-turn / long-response defaults (before common.sh so they win) ----
: "${MAX_PROMPT_LENGTH:=1024}"
: "${MAX_RESPONSE_LENGTH:=8192}"
: "${PPO_MAX_TOKEN_LEN_PER_GPU:=16384}"
: "${TRAIN_BATCH_SIZE:=64}"
: "${PPO_MINI_BATCH_SIZE:=64}"
: "${ROLLOUT_N:=8}"
: "${ROLLOUT_TP:=4}"
: "${ROLLOUT_GPU_MEM_UTIL:=0.6}"

# ---- Data ----
: "${TRAIN_FILES:=search_multiturn/train.parquet}"
: "${VAL_FILES:=search_multiturn/test.parquet}"

# ---- Naming ----
: "${PROJECT_NAME:=verl_search_multiturn}"
: "${EXPERIMENT_NAME:=grpo_qwen3-4b-base_search_r1}"

# ---- Model: use base (bare-query Search-R1 protocol). Instruction-tuned
# models emit Hermes JSON inside <tool_call>, which the search_r1 parser
# does not decode — use format=hermes if you must train an Instruct model.
: "${MODEL_NAME:=Qwen3-4B-Base}"

source "$SCRIPT_DIR/common.sh"

# ---- Multi-turn / algo knobs ----
adv_estimator=${ADV_ESTIMATOR:-grpo}
max_turns=${MAX_TURNS:-4}
clip_ratio_low=${CLIP_RATIO_LOW:-0.2}
clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}
val_n=${VAL_N:-4}
val_top_p=${VAL_TOP_P:-1.0}
val_temperature=${VAL_TEMPERATURE:-1.0}

FUNCTION_TOOL_PATH=${FUNCTION_TOOL_PATH:-$SCRIPT_DIR/search_tool.py}
MULTITURN_FORMAT=${MULTITURN_FORMAT:-search_r1}

# Search tool env forwarded via python (already read at import time in search_tool.py).
export RETRIEVAL_URL=${RETRIEVAL_URL:-http://localhost:8000/retrieve}
export RETRIEVAL_TOPK=${RETRIEVAL_TOPK:-3}

setup_output "$project_name" "$experiment_name"
build_common_arrays

########################### parameter arrays ###########################
DATA=(
    "${COMMON_DATA[@]}"
    algorithm.adv_estimator=${adv_estimator}
    algorithm.kl_ctrl.kl_coef=0.0
    data.return_raw_chat=True
)

ACTOR=(
    "${COMMON_ACTOR[@]}"
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

ROLLOUT=(
    "${COMMON_ROLLOUT[@]}"
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens}
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
    actor_rollout_ref.rollout.multi_turn.enable=True
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${max_turns}
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${max_turns}
    actor_rollout_ref.rollout.multi_turn.function_tool_path=${FUNCTION_TOOL_PATH}
    actor_rollout_ref.rollout.multi_turn.format=${MULTITURN_FORMAT}
    actor_rollout_ref.rollout.val_kwargs.n=${val_n}
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature}
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p}
)

REF=("${COMMON_REF[@]}")

TRAINER=(
    "${TRAINER_BASE[@]}"
    trainer.val_before_train=True
    trainer.log_val_generations=50
)

########################### launch ###########################
source "$SCRIPT_DIR/launch.sh" "$@"
