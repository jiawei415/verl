#!/usr/bin/env bash
# Shared launcher. Sourced from run_*.sh AFTER all Hydra arg arrays are defined.
# Missing arrays (e.g. distill has no REF) are safely expanded to nothing under `set -u`.

# Resolve the final Hydra overrides before touching W&B. The preflight checks
# the actual project and actor environment, including command-line overrides.
# The temporary file contains only non-secret Ray environment overrides.
_wandb_runtime_override=""
_wandb_config_only=0
for _wandb_arg in "$@"; do
    case "$_wandb_arg" in
        --help|--hydra-help|--cfg|--cfg=*|--info|--info=*|--version) _wandb_config_only=1 ;;
    esac
done
if (( ! _wandb_config_only )); then
    _wandb_trace=0
    case "$-" in *x*) _wandb_trace=1; set +x ;; esac
    _wandb_env_file=$(mktemp "${TMPDIR:-/tmp}/verl-wandb-env.XXXXXX")
    if ! timeout 180 python3 "$SCRIPT_DIR/wandb_preflight.py" --env-file "$_wandb_env_file" -- \
        ${DATA[@]+"${DATA[@]}"} \
        ${MODEL[@]+"${MODEL[@]}"} \
        ${ACTOR[@]+"${ACTOR[@]}"} \
        ${ROLLOUT[@]+"${ROLLOUT[@]}"} \
        ${REF[@]+"${REF[@]}"} \
        ${TRAINER[@]+"${TRAINER[@]}"} \
        ${EXTRA[@]+"${EXTRA[@]}"} "$@"; then
        rm -f "$_wandb_env_file"
        echo "W&B preflight failed; training was not started." >&2
        exit 1
    fi
    # Read the generated Hydra override without evaluating it.
    _wandb_runtime_override=$(cat "$_wandb_env_file")
    rm -f "$_wandb_env_file"
    if (( _wandb_trace )); then set -x; fi
fi

python3 -m verl.trainer.main_ppo \
    ${DATA[@]+"${DATA[@]}"} \
    ${MODEL[@]+"${MODEL[@]}"} \
    ${ACTOR[@]+"${ACTOR[@]}"} \
    ${ROLLOUT[@]+"${ROLLOUT[@]}"} \
    ${REF[@]+"${REF[@]}"} \
    ${TRAINER[@]+"${TRAINER[@]}"} \
    ${EXTRA[@]+"${EXTRA[@]}"} \
    "$@" \
    ${_wandb_runtime_override:+"$_wandb_runtime_override"}
