"""Run on a W&B-accessible host; forward one Merlin trial's loopback port.

Keep this process running (e.g. in tmux). An existing SSH key is required.
The process never exports keys, changes trials, or starts training.
"""

import argparse
import fcntl
import json
import os
import select
import signal
import socket
import socketserver
import subprocess
import threading
from pathlib import Path


def allowed_target(host, port):
    return port == "443" and (
        host == "wandb.ai" or host.endswith(".wandb.ai") or host == "storage.googleapis.com"
    )


class ConnectProxy(socketserver.StreamRequestHandler):
    def handle(self):
        self.connection.settimeout(30)
        try:
            parts = self.rfile.readline(8192).decode("ascii").split()
            if len(parts) != 3:
                return
            method, target, _ = parts
            for _ in range(100):
                if self.rfile.readline(8192) in (b"\r\n", b"\n", b""):
                    break
            else:
                return
            host, _, port = target.rpartition(":")
            if method != "CONNECT" or not allowed_target(host, port):
                self.wfile.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
                return
            try:
                upstream = socket.create_connection((host, 443), timeout=20)
            except OSError:
                self.wfile.write(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
                return
            with upstream:
                self.wfile.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                self.wfile.flush()
                peers = [self.connection, upstream]
                while True:
                    ready, _, _ = select.select(peers, [], [], 300)
                    if not ready:
                        return
                    for src in ready:
                        data = src.recv(65536)
                        if not data:
                            return
                        dst = upstream if src is self.connection else self.connection
                        dst.sendall(data)
        except (OSError, UnicodeError):
            # Client disconnects are expected; never log tunneled payloads.
            return


class ProxyServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def discover_pod(cli, control_plane, job, trial):
    result = subprocess.run(
        [cli, "--control-plane", control_plane, "job-v2", "run-trial-pods", "list", "--json",
         json.dumps({"filter": {"trial_sid_any": [trial]}, "options": {"include_debug_endpoints": True}})],
        capture_output=True, text=True, timeout=45, check=True,
    )
    pods = json.loads(result.stdout).get("data", [])
    pods = [pod for pod in pods if pod.get("job_run_sid") == job and not pod.get("stop_at")]
    if len(pods) != 1:
        raise RuntimeError(f"Expected one active pod in {job}/{trial}, found {len(pods)}; this recipe is single-node")
    pod = pods[0]
    endpoints = [item for item in pod.get("debug_endpoints", []) if item.get("type") == "ssh"]
    if len(endpoints) != 1:
        raise RuntimeError("Pod has no unique SSH endpoint yet")
    endpoint = endpoints[0]
    # Merlin reports the SSH port separately and host without a port (IPv4/IPv6).
    return pod["host"], int(endpoint["port"])


def ssh_command(args, host, ssh_port, local_port, state_dir):
    return [
        "ssh", "-T", "-i", str(args.ssh_key), "-p", str(ssh_port),
        "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=15",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", f"UserKnownHostsFile={state_dir / 'known_hosts'}",
        "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=3",
        "-R", f"127.0.0.1:{args.remote_port}:127.0.0.1:{local_port}",
        f"{args.ssh_user}@{host}",
        "printf 'W&B tunnel ready\\n'; exec sleep 2147483647",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-run-id", required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--ssh-key", required=True, type=Path)
    parser.add_argument("--ssh-user", default="tiger")
    parser.add_argument("--merlin-cli", default=str(Path.home() / ".merlin-cli/bin/merlin-cli"))
    parser.add_argument("--control-plane", default="cn-seed")
    parser.add_argument("--remote-port", default=18881, type=int)
    args = parser.parse_args()
    args.ssh_key = args.ssh_key.expanduser().resolve()
    if not args.ssh_key.is_file() or args.ssh_key.stat().st_mode & 0o077:
        parser.error("--ssh-key must be an existing private key readable only by its owner (chmod 600)")
    if not args.trial_id.isdigit() or not 1024 <= args.remote_port <= 65535:
        parser.error("trial-id must be numeric; remote-port must be in 1024..65535")
    state_dir = Path(f"/tmp/verl-wandb-relay-{os.getuid()}-{args.trial_id}")
    state_dir.mkdir(mode=0o700, exist_ok=True)
    lock = (state_dir / "lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("A relay manager for this trial is already running")
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    with ProxyServer(("127.0.0.1", 0), ConnectProxy) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            keep_connected(args, server.server_address[1], state_dir, stop)
        finally:
            server.shutdown()
            lock.close()


def keep_connected(args, local_port, state_dir, stop):
    child = None
    try:
        while not stop.is_set():
            try:
                host, port = discover_pod(args.merlin_cli, args.control_plane, args.job_run_id, args.trial_id)
                print(f"Connecting trial {args.trial_id}; remote W&B proxy http://127.0.0.1:{args.remote_port}", flush=True)
                child = subprocess.Popen(ssh_command(args, host, port, local_port, state_dir))
                while child.poll() is None and not stop.wait(1):
                    pass
                if stop.is_set():
                    break
                print(f"SSH exited ({child.returncode}); reconnecting in 10s", flush=True)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                print(f"Relay connection failed: {type(exc).__name__}: {exc}; retry in 10s", flush=True)
            stop.wait(10)
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    main()
