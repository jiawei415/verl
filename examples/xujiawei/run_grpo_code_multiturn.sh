#!/usr/bin/env bash
# GRPO multi-turn on Codeforces (open-r1/codeforces verifiable) + LiveCodeBench eval.
#
# Uses a @function_tool from examples/xujiawei/code_multiturn_tool.py (execute_python
# with stdin support) backed by seed-sandbox FaaS. Reward from examples/xujiawei/
# code_multiturn_reward.py (rule-based binary all-pass, 3 branches: stdio diff /
# special judge / functional call).
#
# Data prep: run OfflineRL/scripts/download_code_datasets.sh then
#            python verl/examples/data_preprocess/code_multiturn.py before this.
#
# Sandbox: SANDBOX_ENDPOINT must point at seed-sandbox
#          (default: https://seed-sandbox.byteintl.net/faas/sandbox/).

set -xeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Multi-turn responses are long; bump before sourcing common.sh ----
: "${MAX_PROMPT_LENGTH:=4096}"
: "${MAX_RESPONSE_LENGTH:=8192}"
: "${PPO_MAX_TOKEN_LEN_PER_GPU:=16384}"
: "${TRAIN_BATCH_SIZE:=64}"
: "${PPO_MINI_BATCH_SIZE:=64}"
: "${ROLLOUT_N:=8}"
: "${ROLLOUT_TP:=4}"
: "${ROLLOUT_GPU_MEM_UTIL:=0.5}"

# ---- Data ----
: "${TRAIN_FILES:=code_multiturn/train.parquet}"
: "${VAL_FILES:=code_multiturn/test.parquet}"

# ---- Naming ----
: "${PROJECT_NAME:=verl_code_multiturn}"
: "${EXPERIMENT_NAME:=grpo_qwen3-8b-base_cf_lcb}"

source "$SCRIPT_DIR/common.sh"

# ---- Multi-turn / algo knobs ----
adv_estimator=${ADV_ESTIMATOR:-grpo}
max_turns=${MAX_TURNS:-4}
clip_ratio_low=${CLIP_RATIO_LOW:-0.2}
clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}
val_n=${VAL_N:-1}
val_top_p=${VAL_TOP_P:-0.9}

FUNCTION_TOOL_PATH=${FUNCTION_TOOL_PATH:-$SCRIPT_DIR/code_multiturn_tool.py}
MULTITURN_FORMAT=${MULTITURN_FORMAT:-code_fence}
export CODE_FENCE_TOOL_NAME=${CODE_FENCE_TOOL_NAME:-execute_python}

# Reward: rule-based, sandbox-backed. Same seed-sandbox endpoint as the tool
# to preserve train/val environment consistency.
CUSTOM_REWARD_PATH=${CUSTOM_REWARD_PATH:-$SCRIPT_DIR/code_multiturn_reward.py}

setup_output "$project_name" "$experiment_name"
build_common_arrays

########################### parameter arrays ###########################
DATA=(
    "${COMMON_DATA[@]}"
    algorithm.adv_estimator=${adv_estimator}
    algorithm.kl_ctrl.kl_coef=0.0
    data.return_raw_chat=True
    custom_reward_function.path=${CUSTOM_REWARD_PATH}
    custom_reward_function.name=compute_score
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
