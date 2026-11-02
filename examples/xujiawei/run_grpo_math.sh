#!/usr/bin/env bash
# GRPO / DPPO | text | vLLM rollout | FSDP training | NVIDIA GPUs
# LOSS_MODE=dppo_tv bash examples/xujiawei/run_grpo_math.sh
# LOSS_MODE=dppo_kl bash examples/xujiawei/run_grpo_math.sh
# DPPO retains GRPO advantages; CLIP_RATIO_LOW/HIGH are divergence thresholds.

set -xeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configure before sourcing common.sh, which constructs driver-side settings.
loss_mode=${LOSS_MODE:-${ROLLOUT_CORR_LOSS_MODE:-bypass_mode}}
case "$loss_mode" in
    dppo_tv|dppo_binary_tv) dppo_clip_default=0.15 ;;
    dppo_kl|dppo_binary_kl) dppo_clip_default=0.05 ;;
esac
if [[ -n ${dppo_clip_default:-} ]]; then
    # Stable-RL uses rollout probabilities for both divergence and truncated IS.
    # Additional IS/RS belongs to the separate rollout-correction objective.
    if [[ ${ROLLOUT_CORR_BYPASS:-True} != True || ${ROLLOUT_IS:-null} != null || ${ROLLOUT_RS:-null} != null ]]; then
        echo "DPPO requires ROLLOUT_CORR_BYPASS=True, ROLLOUT_IS=null and ROLLOUT_RS=null" >&2
        exit 1
    fi
    ROLLOUT_CORR_BYPASS=True
    CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-$dppo_clip_default}
    CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-$dppo_clip_default}
    CLIP_RATIO_C=${CLIP_RATIO_C:-10.0}
fi
ROLLOUT_CORR_LOSS_MODE=$loss_mode
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
