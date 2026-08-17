#!/usr/bin/env bash
# OTB (Optimal Token Baseline) | text | vLLM rollout | FSDP training | NVIDIA GPUs

set -xeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# ---- OTB-specific knobs ----
adv_estimator=${ADV_ESTIMATOR:-optimal_token_baseline}

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
