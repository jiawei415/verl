#!/usr/bin/env bash
# Source this file in a fresh trial. Keep the image's Pydantic/FastAPI/protobuf.
setup_wandb_runtime() {
    local sdk_dir=${WANDB_SDK_DIR:-/tmp/verl-wandb-runtime-0.23.1}
    local uv_bin
    if command -v uv >/dev/null 2>&1; then
        uv_bin=$(command -v uv)
    else
        # Bootstrap uv locally; do not alter the base Python environment.
        local bootstrap=${TMPDIR:-/tmp}/verl-wandb-uv
        if [[ ! -x "$bootstrap/bin/uv" ]]; then
            python3 -m pip install --target "$bootstrap" uv || return
        fi
        uv_bin="$bootstrap/bin/uv"
    fi
    UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/verl-wandb-uv-cache} "$uv_bin" pip install \
        --python "$(command -v python3)" --target "$sdk_dir" --no-deps \
        'wandb==0.23.1' 'sentry-sdk==2.73.0' || return
    export PYTHONPATH="$sdk_dir${PYTHONPATH:+:$PYTHONPATH}"
    python3 - <<'PY'
import wandb
assert wandb.__version__ == "0.23.1", (wandb.__version__, wandb.__file__)
print("W&B runtime:", wandb.__version__, wandb.__file__)
PY
}
setup_wandb_runtime
