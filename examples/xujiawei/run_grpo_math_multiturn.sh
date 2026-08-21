#!/usr/bin/env bash
# GRPO multi-turn with code-interpreter tool | text | vLLM async rollout | FSDP training | NVIDIA GPUs
#
# Uses a @function_tool loaded from examples/xujiawei/code_tool.py that talks to a
# sandbox_fusion HTTP server. Set SANDBOX_FUSION_URL to point at a running instance
# (default http://localhost:8080/run_code).

set -xeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Multi-turn responses are long; bump defaults before sourcing common.
: "${ROLLOUT_TP:=4}"
: "${ROLLOUT_GPU_MEM_UTIL:=0.5}"
: "${ROLLOUT_MODE:=async}"
: "${SAVE_FREQ:=30}"
# Point to the tool-use-aware dataset (system prompt injected).
: "${TRAIN_FILES:=math_multiturn/dapo_train.parquet}"
: "${VAL_FILES:=math_multiturn/gsm8k_test.parquet math_multiturn/math500_test.parquet math_multiturn/aime24_test.parquet math_multiturn/aime25_test.parquet}"
source "$SCRIPT_DIR/common.sh"

# ---- Multi-turn / algo-specific knobs ----
adv_estimator=${ADV_ESTIMATOR:-grpo}

FUNCTION_TOOL_PATH=${FUNCTION_TOOL_PATH:-$SCRIPT_DIR/code_tool.py}
MULTITURN_FORMAT=${MULTITURN_FORMAT:-code_fence}
export CODE_FENCE_TOOL_NAME=${CODE_FENCE_TOOL_NAME:-code_interpreter}

setup_output "$project_name" "$experiment_name"
build_common_arrays

########################### parameter arrays ###########################
DATA=(
    "${COMMON_DATA[@]}"
    algorithm.adv_estimator=${adv_estimator}
    data.return_raw_chat=True
    # Uncomment to enable tool-use reward shaping.
    # custom_reward_function.path=${SCRIPT_DIR}/tool_bonus_reward.py
    # custom_reward_function.name=compute_score
)

ACTOR=("${COMMON_ACTOR[@]}")

ROLLOUT=(
    "${COMMON_ROLLOUT[@]}"
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
    actor_rollout_ref.rollout.multi_turn.enable=True
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${max_turns}
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${max_turns}
    actor_rollout_ref.rollout.multi_turn.function_tool_path=${FUNCTION_TOOL_PATH}
    actor_rollout_ref.rollout.multi_turn.format=${MULTITURN_FORMAT}
)

REF=("${COMMON_REF[@]}")

TRAINER=("${TRAINER_BASE[@]}")

########################### launch ###########################
source "$SCRIPT_DIR/launch.sh" "$@"
