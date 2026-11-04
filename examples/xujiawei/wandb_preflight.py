"""Check the final training config and W&B write access before model allocation."""

import argparse
import json
import os
import re
import time
import uuid
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

# These are safe to print in Hydra/Ray logs. Never put credentials here.
ENV_KEYS = (
    "WANDB_BACKEND", "WANDB_OFFICIAL", "WANDB_BASE_URL", "WANDB_ENTITY", "WANDB_MODE", "WANDB_DIR",
    "WANDB_CACHE_DIR", "WANDB_CONFIG_DIR", "WANDB_HTTP_PROXY", "WANDB_HTTPS_PROXY",
    "no_proxy", "NO_PROXY", "PYTHONPATH",
)


def relay_exclusions(value):
    result = []
    targets = ("api.wandb.ai", "files.wandb.ai", "storage.googleapis.com", "forge.coreweave.com")
    for entry in (value + ",localhost,127.0.0.1,::1").split(","):
        entry = entry.strip()
        domain = entry.lower().split(":", 1)[0].lstrip(".")
        if not entry or entry == "*":
            continue
        if domain and any(host == domain or host.endswith("." + domain) for host in targets):
            continue
        if entry not in result:
            result.append(entry)
    return ",".join(result)


def effective_environment(config):
    runtime = OmegaConf.select(config, "ray_kwargs.ray_init.runtime_env.env_vars", default={})
    env = dict(runtime or {})
    for key in ENV_KEYS:
        # Empty proxy values are not accepted by wandb.Settings; use a real
        # localhost exclusion + explicit proxy, or omit the SDK proxy setting.
        if key not in env and key in os.environ:
            env[key] = os.environ[key]
    proxy = OmegaConf.select(config, "trainer.wandb_proxy") or env.get("WANDB_HTTPS_PROXY")
    if proxy:
        env["WANDB_HTTPS_PROXY"] = proxy
        env["WANDB_HTTP_PROXY"] = proxy
        exclusions = env.get("no_proxy", "") + "," + env.get("NO_PROXY", "")
        env["no_proxy"] = env["NO_PROXY"] = relay_exclusions(exclusions)
    return env


def runtime_override(env):
    if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) for key in env):
        raise ValueError("Ray environment keys must be valid environment variable names")
    values = ",".join(f"{key}:{json.dumps(value)}" for key, value in env.items())
    return "++ray_kwargs.ray_init.runtime_env.env_vars={" + values + "}"


def check_write(config, env):
    import wandb

    if env.get("WANDB_BACKEND") == "byted":
        if wandb.__version__ != "0.14.10" or not hasattr(wandb, "TrackingApi"):
            raise RuntimeError("byted backend requires byted-wandb==0.14.10")
        name = f"preflight_{config.trainer.experiment_name}_{uuid.uuid4().hex[:8]}"
        with wandb.init(project=config.trainer.project_name, name=name,
                        config={"purpose": "training-preflight"}) as run:
            run_id = run.id
            run.log({"preflight/write": 1}, step=0)
        for attempt in range(30):
            saved = wandb.TrackingApi().run(project=config.trainer.project_name, run_id=run_id)
            rows = saved.scan_history(name=["preflight/write"])
            if 1 in rows.get("preflight/write", []):
                print(f"Byted W&B write/readback passed: {saved.link}", flush=True)
                return
            if attempt < 29:
                time.sleep(2)
        raise RuntimeError(f"Byted W&B metrics did not sync: {run_id}")

    if wandb.__version__ != "0.23.1":
        raise RuntimeError(
            f"W&B {wandb.__version__} loaded from {wandb.__file__}; "
            "source examples/xujiawei/setup_wandb.sh first"
        )
    mode = env.get("WANDB_MODE", "online")
    if mode != "online":
        raise RuntimeError(f"W&B preflight requires WANDB_MODE=online, got {mode!r}")
    settings = {"init_timeout": 45}
    proxy = env.get("WANDB_HTTPS_PROXY")
    if proxy:
        settings.update(https_proxy=proxy, http_proxy=proxy)
    name = f"preflight_{config.trainer.experiment_name}_{uuid.uuid4().hex[:8]}"
    with wandb.init(
        project=config.trainer.project_name,
        entity=env.get("WANDB_ENTITY"),
        name=name,
        job_type="connectivity-test",
        tags=["preflight", "not-training"],
        config={"training_experiment": config.trainer.experiment_name},
        settings=wandb.Settings(**settings),
    ) as run:
        path = f"{run.entity}/{run.project}/{run.id}"
        run.log({"preflight/write": 1}, step=0)
        print(f"W&B preflight run: {run.url}", flush=True)
    for attempt in range(6):
        saved = wandb.Api(timeout=15).run(path)
        rows = list(saved.scan_history(keys=["_step", "preflight/write"]))
        if saved.state == "finished" and any(row["preflight/write"] == 1 for row in rows):
            print("W&B write/readback passed; starting training.", flush=True)
            return
        if attempt < 5:
            time.sleep(2)
    raise RuntimeError(f"W&B run did not finish syncing: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", required=True)
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    overrides = args.overrides[1:] if args.overrides[:1] == ["--"] else args.overrides
    config_dir = Path(__file__).resolve().parents[2] / "verl/trainer/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=overrides)
    backends = config.trainer.logger
    if isinstance(backends, str):
        backends = [backends]
    if not ({"wandb", "tracking"} & set(backends)):
        Path(args.env_file).write_text("")
        return
    env = effective_environment(config)
    # Do not serialize a caller-supplied secret into the generated override.
    if any(key.upper().endswith(("_KEY", "_TOKEN", "_PASSWORD")) for key in env):
        raise ValueError("Keep credentials in the launch environment, not Ray/Hydra command-line overrides")
    os.environ.update(env)
    for key in ("WANDB_HTTP_PROXY", "WANDB_HTTPS_PROXY"):
        if not os.environ.get(key):
            os.environ.pop(key, None)
            env.pop(key, None)
    if os.environ.get("WANDB_PREFLIGHT", "1") != "0":
        check_write(config, env)
    Path(args.env_file).write_text(runtime_override(env))


if __name__ == "__main__":
    main()
