"""Evaluate a starVLA checkpoint on the AFFO-16 grid (tasks x conditions x episodes): one policy server, many clients.

Starts the policy server on the queue's card (starvla environment), runs eval_client.py workers for slices of
every (task, condition) cell, at most `clients` at a time, then writes summary.json (success rate, Wilson interval,
per-cell counts). Finished episodes are skipped on a restart. Settings: configs/rc365/base_policy_eval.yaml,
overridden by `key=value` arguments.

Run through the queue (scripts/rc365/run_eval_base_gpu.sh); in the robocasa365 environment:
  "$RC365_PYTHON" scripts/rc365/eval_policy.py --ckpt <steps_N_pytorch_model.pt> --run <name> [clients=24 episodes=20]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from affordance_vla.config import load_config, resolved  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
W = Path(os.environ.get("W", "/mnt/nas/xinyizheng/affordance_vla"))
STARVLA = Path(os.environ.get("STARVLA", W / "repos/starVLA_rc365"))
STARVLA_PYTHON = os.environ.get("STARVLA_PYTHON", "/data/xinyizheng/envs/starvla/bin/python")


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval of a success rate."""
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def summarise(rows: list[dict]) -> dict:
    cells: dict[str, dict] = {}
    for row in rows:
        cell = cells.setdefault(f"{row['task']}/{row['condition']}", {"n": 0, "wins": 0, "steps": 0})
        cell["n"] += 1
        cell["wins"] += int(row["success"])
        cell["steps"] += row["steps"]
    for cell in cells.values():
        cell["rate"] = cell["wins"] / cell["n"]
        cell["wilson95"] = [round(x, 3) for x in wilson(cell["wins"], cell["n"])]
    by_condition: dict[str, dict] = {}
    for key, cell in cells.items():
        total = by_condition.setdefault(key.split("/")[1], {"n": 0, "wins": 0})
        total["n"] += cell["n"]
        total["wins"] += cell["wins"]
    for total in by_condition.values():
        total["rate"] = total["wins"] / total["n"]
        total["wilson95"] = [round(x, 3) for x in wilson(total["wins"], total["n"])]
    return {"cells": cells, "by_condition": by_condition}


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start_server(ckpt: str, port: int, log: Path) -> subprocess.Popen:
    env = {**os.environ, "PYTHONPATH": str(STARVLA), "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    server = subprocess.Popen(
        [STARVLA_PYTHON, "deployment/model_server/server_policy.py", "--ckpt_path", ckpt, "--port", str(port), "--use_bf16"],
        cwd=STARVLA, env=env, stdout=log.open("w"), stderr=subprocess.STDOUT,
    )
    for _ in range(240):
        if server.poll() is not None:
            raise SystemExit(f"policy server died, see {log}")
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return server
        time.sleep(5)
    server.terminate()
    raise SystemExit(f"policy server did not open port {port} in 20 minutes, see {log}")


def jobs(cfg: dict, done: set[tuple[str, str, int]]) -> list[tuple[str, str, int, int]]:
    """(task, condition, start, stop) slices of `chunk` episodes that still have episodes to run."""
    out = []
    for task in cfg["tasks"]:
        for condition in cfg["conditions"]:
            for start in range(0, cfg["episodes"], cfg["chunk"]):
                stop = min(start + cfg["chunk"], cfg["episodes"])
                if any((task, condition, i) not in done for i in range(start, stop)):
                    out.append((task, condition, start, stop))
    return out


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO / "configs/rc365/base_policy_eval.yaml"))
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("overrides", nargs="*", help="key=value overrides of the config")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    out = W / "runs/robocasa365_eval" / args.run
    out.mkdir(parents=True, exist_ok=True)
    meta = {**resolved(cfg, args.config, args.overrides, REPO), "ckpt": args.ckpt}
    (out / "config.json").write_text(json.dumps(meta, indent=2))
    results = out / "episodes.jsonl"

    port = free_port()
    server = start_server(args.ckpt, port, out / "server.log")
    running: list[tuple[subprocess.Popen, tuple]] = []
    try:
        pending = jobs(cfg, {(r["task"], r["condition"], r["index"]) for r in read_rows(results)})
        print(f"{len(pending)} slices to run, {cfg['clients']} clients", flush=True)
        env = {**os.environ, "PYTHONPATH": f"{STARVLA}:{REPO}"}
        while pending or running:
            running = [(p, j) for p, j in running if p.poll() is None]
            while pending and len(running) < cfg["clients"]:
                task, condition, start, stop = pending.pop(0)
                log = (out / "client_logs" / f"{task}_{condition}_{start}.log")
                log.parent.mkdir(exist_ok=True)
                proc = subprocess.Popen(
                    [sys.executable, str(REPO / "scripts/rc365/eval_client.py"), "--task", task, "--condition", condition,
                     "--start", str(start), "--stop", str(stop), "--port", str(port), "--base-seed", str(cfg["base_seed"]),
                     "--n-act", str(cfg["n_act"]), "--videos-per-cell", str(cfg["videos_per_cell"]), "--out", str(out)],
                    env=env, stdout=log.open("w"), stderr=subprocess.STDOUT,
                )
                running.append((proc, (task, condition, start)))
            time.sleep(5)
    finally:
        for proc, _ in running:
            proc.terminate()
        server.terminate()
        server.wait()
    summary = summarise(read_rows(results))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["by_condition"], indent=2))


if __name__ == "__main__":
    main()
