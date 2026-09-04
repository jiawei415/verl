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
: "${ROLLOUT_TP:=4}"
: "${ROLLOUT_GPU_MEM_UTIL:=0.6}"

# ---- Data ----
: "${TRAIN_FILES:=search_multiturn/train.parquet}"
: "${VAL_FILES:=search_multiturn/test.parquet}"

# ---- Naming ----
: "${EXPERIMENT_NAME:=grpo_search_multiturn}"

source "$SCRIPT_DIR/common.sh"

# ---- Multi-turn / algo knobs ----
adv_estimator=${ADV_ESTIMATOR:-grpo}

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
    data.return_raw_chat=True
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
