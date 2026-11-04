#!/usr/bin/env bash
# Source from common.sh: W&B backend, credentials, proxies and local output paths.

# run_*.sh enable xtrace; suppress it while loading credentials.
_wandb_restore_xtrace=0
case "$-" in *x*) _wandb_restore_xtrace=1; set +x ;; esac
export WANDB_BACKEND=${WANDB_BACKEND:-official}
if [[ "$WANDB_BACKEND" == byted ]]; then
    export WANDB_OFFICIAL=0 WANDB_MODE=online
    unset WANDB_API_KEY WANDB_ENTITY WANDB_BASE_URL WANDB_HTTP_PROXY WANDB_HTTPS_PROXY WANDB_DISABLED
    _wandb_internal_proxy=""
    IFS=',' read -ra _wandb_entries <<< "${no_proxy:-},${NO_PROXY:-},.byted.org,.bytedance.net,localhost,127.0.0.1"
    for _wandb_entry in "${_wandb_entries[@]}"; do
        [[ -z $_wandb_entry ]] && continue
        case ",${_wandb_internal_proxy}," in
            *",${_wandb_entry},"*) ;;
            *) _wandb_internal_proxy="${_wandb_internal_proxy:+${_wandb_internal_proxy},}${_wandb_entry}" ;;
        esac
    done
    export no_proxy="$_wandb_internal_proxy" NO_PROXY="$_wandb_internal_proxy"
    unset _wandb_internal_proxy _wandb_entries _wandb_entry
elif [[ "$WANDB_BACKEND" == official ]]; then
    export WANDB_OFFICIAL=1
    # Official W&B credentials are supplied by the caller or SDK login.
    # export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
    # export https_proxy=http://sys-proxy-rd-relay.byted.org:8118

    # Direct mode bypasses inherited proxies. A dedicated relay is opt-in.
    export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.wandb.ai}
    export WANDB_MODE=${WANDB_MODE:-online}
    _wandb_no_proxy=""
    if [[ -n ${WANDB_HTTPS_PROXY:-} ]]; then
        export WANDB_HTTP_PROXY="$WANDB_HTTPS_PROXY"
        _wandb_extra="localhost,127.0.0.1,::1"
    else
        unset WANDB_HTTP_PROXY WANDB_HTTPS_PROXY
        _wandb_extra="wandb.ai,.wandb.ai,coreweave.com,.coreweave.com"
    fi
    IFS=',' read -ra _wandb_exclusions <<< "${no_proxy:-},${NO_PROXY:-},${_wandb_extra}"
    for _wandb_host in "${_wandb_exclusions[@]}"; do
        [[ -z $_wandb_host ]] && continue
        if [[ -n ${WANDB_HTTPS_PROXY:-} ]]; then
            # Preserve internal-network bypasses, but never bypass the W&B relay.
            case "${_wandb_host,,}" in
                '*'|ai|.ai|com|.com|wandb.ai|*.wandb.ai|coreweave.com|*.coreweave.com|googleapis.com|*.googleapis.com) continue ;;
            esac
        fi
        case ",${_wandb_no_proxy}," in
            *",${_wandb_host},"*) ;;
            *) _wandb_no_proxy="${_wandb_no_proxy:+${_wandb_no_proxy},}${_wandb_host}" ;;
        esac
    done
    export no_proxy="$_wandb_no_proxy"
    unset _wandb_no_proxy _wandb_exclusions _wandb_host _wandb_extra
    export NO_PROXY="$no_proxy"
else
    echo "Unsupported WANDB_BACKEND: $WANDB_BACKEND (use byted or official)" >&2
    return 1
fi
if (( _wandb_restore_xtrace )); then set -x; fi
unset _wandb_restore_xtrace

setup_wandb_output() {
    local exp="$1"
    # Keep ckpt / tb on HDFS, but wandb needs a local-fs dir because its
    # multi-process daemon uses unix sockets that HDFS FUSE doesn't support
    # (symptom: `wandb.errors.UsageError: Error communicating with wandb
    # process`). Override via WANDB_DIR / WANDB_LOCAL_ROOT.
    # local wandb_root=${WANDB_LOCAL_ROOT:-/tmp/wandb}
    # local wandb_dir="$wandb_root/${proj}/${exp}"
    mkdir -p "/tmp/wandb/${exp}"
    export WANDB_DIR="/tmp/wandb/${exp}"
    export WANDB_CACHE_DIR="/tmp/wandb-cache"
    export WANDB_CONFIG_DIR="/tmp/wandb-config"
}
