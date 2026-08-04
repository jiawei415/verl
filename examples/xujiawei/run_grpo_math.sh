#!/usr/bin/env bash
# GRPO | text | vLLM rollout | FSDP training | NVIDIA GPUs

set -xeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# ---- GRPO-specific knobs (mirrors OTB) ----
adv_estimator=${ADV_ESTIMATOR:-grpo}

setup_output "$project_name" "$experiment_name"
build_common_arrays

########################### parameter arrays ###########################
DATA=("${COMMON_DATA[@]}" algorithm.adv_estimator=${adv_estimator})

ACTOR=("${COMMON_ACTOR[@]}")

ROLLOUT=("${COMMON_ROLLOUT[@]}")

REF=("${COMMON_REF[@]}")

TRAINER=("${TRAINER_BASE[@]}")

########################### launch ###########################
source "$SCRIPT_DIR/launch.sh" "$@"
