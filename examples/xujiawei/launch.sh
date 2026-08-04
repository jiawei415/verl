#!/usr/bin/env bash
# Shared launcher. Sourced from run_*.sh AFTER all Hydra arg arrays are defined.
# Missing arrays (e.g. distill has no REF) are safely expanded to nothing under `set -u`.

python3 -m verl.trainer.main_ppo \
    ${DATA[@]+"${DATA[@]}"} \
    ${MODEL[@]+"${MODEL[@]}"} \
    ${ACTOR[@]+"${ACTOR[@]}"} \
    ${ROLLOUT[@]+"${ROLLOUT[@]}"} \
    ${REF[@]+"${REF[@]}"} \
    ${TRAINER[@]+"${TRAINER[@]}"} \
    ${EXTRA[@]+"${EXTRA[@]}"} \
    "$@"
