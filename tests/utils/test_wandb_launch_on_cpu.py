"""No-network checks for W&B environment, preflight gating and relay policy."""

import importlib.util
import json
import os
import socket
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra.core.override_parser.overrides_parser import OverridesParser
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "examples/xujiawei"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preflight = load("wandb_preflight")
relay = load("wandb_relay")


def shell_environment(extra):
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("WANDB_") or key.lower().endswith("proxy"):
            env.pop(key)
    env.update(extra)
    result = subprocess.run(
        ["bash", "-c", 'set -eu; source examples/xujiawei/common.sh; source examples/xujiawei/common.sh; '
         'python3 -c \'import os,json; print(json.dumps({k:v for k,v in os.environ.items() '
         'if k in ("no_proxy","NO_PROXY","WANDB_HTTPS_PROXY","WANDB_API_KEY","WANDB_OFFICIAL","WANDB_BACKEND","WANDB_BASE_URL","WANDB_ENTITY")}))\''],
        cwd=ROOT, env=env, capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def test_direct_preserves_exclusions_without_duplicates():
    env = shell_environment({"no_proxy": "localhost,.internal", "NO_PROXY": "localhost", "WANDB_API_KEY": "test-key"})
    assert env["no_proxy"] == env["NO_PROXY"]
    entries = env["no_proxy"].split(",")
    assert len(entries) == len(set(entries))
    assert {"localhost", ".internal", "wandb.ai"} <= set(entries)
    assert env["WANDB_API_KEY"] == "test-key"


def test_proxy_removes_bypass_and_keeps_caller_key():
    env = shell_environment({"no_proxy": "*,.ai,wandb.ai", "WANDB_HTTPS_PROXY": "http://127.0.0.1:18881"})
    assert env["no_proxy"] == "localhost,127.0.0.1,::1"
    assert env["WANDB_HTTPS_PROXY"] == "http://127.0.0.1:18881"


def test_proxy_keeps_internal_network_exclusions():
    env = shell_environment({"no_proxy": "*,.internal,10.0.0.0/8", "WANDB_HTTPS_PROXY": "http://127.0.0.1:18881"})
    assert ".internal" in env["no_proxy"].split(",")
    assert "10.0.0.0/8" in env["no_proxy"].split(",")
    filtered = preflight.relay_exclusions(env["no_proxy"] + ",.ai,api.wandb.ai:443")
    assert ".internal" in filtered.split(",")
    assert ".ai" not in filtered.split(",")
    assert "api.wandb.ai:443" not in filtered.split(",")


def test_runtime_override_matches_preflight(monkeypatch):
    monkeypatch.setenv("WANDB_HTTPS_PROXY", "http://127.0.0.1:18881")
    monkeypatch.setenv("PYTHONPATH", "/tmp/sdk")
    config = OmegaConf.create({"trainer": {"wandb_proxy": "http://127.0.0.1:18882"}})
    env = preflight.effective_environment(config)
    assert env["WANDB_HTTPS_PROXY"] == "http://127.0.0.1:18882"
    assert env["PYTHONPATH"] == "/tmp/sdk"
    encoded = preflight.runtime_override(env)
    assert OverridesParser.create().parse_override(encoded).value() == env


def test_failed_preflight_never_launches_trainer(tmp_path):
    stub = tmp_path / "python3"
    marker = tmp_path / "training_started"
    stub.write_text('#!/bin/bash\nif [[ "$1" == *wandb_preflight.py ]]; then exit 9; fi\ntouch "$MARKER"\n')
    stub.chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}", MARKER=str(marker))
    result = subprocess.run(["bash", "-c", f'SCRIPT_DIR="{SCRIPTS}"; source "$SCRIPT_DIR/launch.sh"'], env=env,
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "preflight failed" in result.stderr
    assert not marker.exists()


@pytest.mark.parametrize("host,port,allowed", [
    ("api.wandb.ai", "443", True), ("storage.googleapis.com", "443", True),
    ("evilwandb.ai", "443", False), ("api.wandb.ai.evil.com", "443", False),
    ("127.0.0.1", "443", False), ("api.wandb.ai", "22", False),
])
def test_relay_destination_policy(host, port, allowed):
    assert relay.allowed_target(host, port) is allowed


def test_relay_denies_non_wandb_connection():
    with relay.ProxyServer(("127.0.0.1", 0), relay.ConnectProxy) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with socket.create_connection(server.server_address, timeout=2) as client:
                client.sendall(b"CONNECT example.com:443 HTTP/1.1\r\nHost: example.com\r\n\r\n")
                assert client.recv(1024).startswith(b"HTTP/1.1 403")
        finally:
            server.shutdown()


def test_ssh_forward_is_loopback_and_detects_disconnect(tmp_path):
    args = SimpleNamespace(ssh_key=tmp_path / "key", ssh_user="tiger", remote_port=18881)
    command = relay.ssh_command(args, "::1", 10060, 45678, tmp_path)
    assert "127.0.0.1:18881:127.0.0.1:45678" in command
    assert "ExitOnForwardFailure=yes" in command
    assert "ServerAliveCountMax=3" in command
    assert "-g" not in command


def test_relay_reconnects_and_closes_child(monkeypatch, tmp_path):
    children = []

    class Child:
        def __init__(self, command):
            self.returncode = 255 if not children else None
            self.terminated = False
            children.append(self)

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = 0

        def wait(self, timeout):
            return self.returncode

    class Stop:
        stopped = False

        def is_set(self):
            return self.stopped

        def wait(self, seconds):
            if len(children) >= 2:
                self.stopped = True
            return self.stopped

    monkeypatch.setattr(relay, "discover_pod", lambda *args: ("::1", 10060))
    monkeypatch.setattr(relay.subprocess, "Popen", Child)
    args = SimpleNamespace(merlin_cli="unused", control_plane="cn-seed", job_run_id="job", trial_id="123",
                           ssh_key=tmp_path / "key", ssh_user="tiger", remote_port=18881)
    relay.keep_connected(args, 12345, tmp_path, Stop())
    assert len(children) == 2
    assert children[-1].terminated


def test_preflight_override_composes_with_training_config(monkeypatch):
    from hydra import compose, initialize_config_dir

    monkeypatch.setenv("WANDB_HTTPS_PROXY", "http://127.0.0.1:18881")
    with initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=["trainer.project_name=actual_target"])
        override = preflight.runtime_override(preflight.effective_environment(config))
        final = compose(config_name="ppo_trainer", overrides=["trainer.project_name=actual_target", override])
    assert final.trainer.project_name == "actual_target"
    assert final.ray_kwargs.ray_init.runtime_env.env_vars.WANDB_HTTPS_PROXY == "http://127.0.0.1:18881"


def test_byted_clears_official_config_and_deduplicates():
    env = shell_environment({"WANDB_BACKEND": "byted", "WANDB_OFFICIAL": "1",
                             "WANDB_API_KEY": "test-key", "WANDB_ENTITY": "external",
                             "WANDB_BASE_URL": "https://api.wandb.ai",
                             "WANDB_HTTPS_PROXY": "http://127.0.0.1:18881",
                             "no_proxy": ".internal"})
    assert env["WANDB_OFFICIAL"] == "0"
    assert env["WANDB_BACKEND"] == "byted"
    assert not {"WANDB_API_KEY", "WANDB_ENTITY", "WANDB_BASE_URL", "WANDB_HTTPS_PROXY"} & env.keys()
    entries = env["no_proxy"].split(",")
    assert len(entries) == len(set(entries))
    assert {".internal", ".byted.org", ".bytedance.net"} <= set(entries)


def test_byted_preflight_uses_tracking_api(monkeypatch):
    import sys
    calls = []
    reads = []
    sleeps = []
    monkeypatch.setattr(preflight.time, "sleep", sleeps.append)

    class Run:
        id = "test-run"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def log(self, values, step):
            calls.append((values, step))

    class Api:
        def run(self, project, run_id):
            assert (project, run_id) == ("test-project", "test-run")
            def scan_history(name):
                reads.append(name)
                return {name[0]: [1]} if len(reads) == 8 else {}
            return SimpleNamespace(link="internal-test-link", scan_history=scan_history)

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(
        __version__="0.14.10", init=lambda **kwargs: Run(), TrackingApi=Api))
    config = OmegaConf.create({"trainer": {"project_name": "test-project", "experiment_name": "test"}})
    preflight.check_write(config, {"WANDB_BACKEND": "byted"})
    assert calls == [({"preflight/write": 1}, 0)]
    assert sleeps == [2] * 7
