# Merlin Tracking (byted-wandb)

Set `WANDB_BACKEND=byted` before running `run_grpo_math.sh`. `common.sh`
sources `wandb_env.sh`, which selects the internal backend, clears external
W&B credentials/proxies, and configures local SDK directories. The default
backend remains `official`; official credentials must be provided externally.

In the Merlin training image, install `byted-wandb==0.14.10` and
`setuptools==80.9.0` into a separate Python target directory with `uv pip
install --target ... --no-deps`, then prepend it to `PYTHONPATH`. The image
must already provide the SDK dependencies (including ZTI authentication).
Do not source `setup_wandb.sh`, which installs the official SDK.

The shared launcher performs an online write/readback preflight before
allocating the models. It propagates the backend and Python path to Ray.
Merlin supplies the internal identity; no external W&B API key or relay is
required. Runs appear in Merlin Tracking, not wandb.ai.

Query using the internal API in an authenticated internal environment:

```python
import wandb
run = wandb.TrackingApi().run(project="verl_research", run_id="YOUR_RUN_ID")
data = run.scan_history(name=["actor/pg_loss", "training/global_step"])
```

Use the project name, not its ID. `scan_history` returns a mapping of column
names to lists, including `step`. Request at most 10 metrics at a time.
The original `wandb.Api()` is for the external backend.
