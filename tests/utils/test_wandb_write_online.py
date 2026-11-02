"""Opt-in live W&B write/read smoke test using the math script's environment.

Run from the verl directory:
    RUN_WANDB_WRITE_TEST=1 python3 -m pytest tests/utils/test_wandb_write_online.py -q -s

Creates one tagged run in jiawei415/verl_research; does not delete it.
"""

import json
import os
import subprocess
import time
import uuid
from pathlib import Path

import pytest


@pytest.mark.skipif(os.environ.get("RUN_WANDB_WRITE_TEST") != "1", reason="Writes to live W&B")
def test_wandb_write_and_readback(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    # Capture rather than print environment values, which include credentials.
    result = subprocess.run(
        [
            "bash", "-c",
            'set +x; source examples/xujiawei/common.sh; '
            'python3 -c \'import os,json; print(json.dumps(dict(os.environ)))\'',
        ],
        cwd=root, capture_output=True, text=True, check=True,
    )
    for key, value in json.loads(result.stdout).items():
        if key.startswith("WANDB_") or key.lower().endswith("proxy"):
            monkeypatch.setenv(key, value)
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))

    import wandb

    print(f"W&B SDK: {wandb.__version__} ({wandb.__file__})")
    run_id = uuid.uuid4().hex[:12]
    with wandb.init(
        entity=os.environ.get("WANDB_ENTITY", "jiawei415"),
        project=os.environ.get("WANDB_TEST_PROJECT", "verl_research"),
        id=run_id,
        name=f"smoke_wandb_write_{run_id}",
        job_type="connectivity-test",
        tags=["smoke-test", "wandb-write-test"],
        config={"purpose": "Verify common.sh W&B write and read access", "training": False},
        settings=wandb.Settings(init_timeout=60),
    ) as run:
        path = f"{run.entity}/{run.project}/{run.id}"
        print(f"Test run: {run.url}")
        for step, value in enumerate([0.0, 0.5, 1.0]):
            run.log({"smoke/value": value}, step=step)
        run.summary["smoke/write_complete"] = True

    # Allow bounded time for the history and finished state to become visible.
    for attempt in range(10):
        saved = wandb.Api(timeout=30).run(path)
        rows = list(saved.scan_history(keys=["_step", "smoke/value"]))
        if saved.state == "finished" and len(rows) == 3:
            break
        if attempt < 9:
            time.sleep(2)
    assert saved.state == "finished"
    assert saved.summary["smoke/write_complete"] is True
    assert [(row["_step"], row["smoke/value"]) for row in rows] == [(0, 0.0), (1, 0.5), (2, 1.0)]
    print(f"Verified finished run and all 3 history points: {path}")
