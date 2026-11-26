#!/usr/bin/env bash
# Fetch RoboCasa365 target/human LeRobot bundles for a few tasks (CPU + network, no GPU).
# The official downloader (robocasa.scripts.download_datasets) cannot resume and asks nothing about size;
# this one reads the Box link from box_links_ds.json, prints the size first, then downloads with curl (resumable).
#
# Usage on the server:  bash scripts/data/fetch_robocasa365.sh [--dry-run] [--split target|pretrain] [TASK ...]
# Default split target; default tasks: CoffeeSetupMug (pick-place a mug), OpenDrawer (drawer), TurnOffStove (knob).
# The human demonstrations only (the MimicGen bundles under .../mg/... are skipped).
# Output: $W/data/robocasa365/v1.0/<split>/atomic/<Task>/<date>/lerobot (RoboCasa dataset root from macros_private.py)
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_DIR/envs/server/robocasa365_env.sh"
DRY=0; SPLIT=target
while [[ "${1:-}" == --* ]]; do
  case "$1" in --dry-run) DRY=1; shift;; --split) SPLIT="$2"; shift 2;; *) echo "unknown option $1" >&2; exit 2;; esac
done
TASKS=("$@"); [ ${#TASKS[@]} -gt 0 ] || TASKS=(CoffeeSetupMug OpenDrawer TurnOffStove)
export DRY_RUN=$DRY SPLIT TASKS_LIST="${TASKS[*]}"
mkdir -p "$W/logs"
"$RC365_PYTHON" - <<'PY'
import json, os, subprocess, tarfile
from pathlib import Path
import robocasa
from robocasa.macros import DATASET_BASE_PATH

links = json.load(open(Path(robocasa.__path__[0]) / "models/assets/box_links/box_links_ds.json"))
root = Path(DATASET_BASE_PATH) / "v1.0"
for task in os.environ["TASKS_LIST"].split():
    split = os.environ["SPLIT"]
    key = next((k for k in links if k.startswith(f"{split}/") and f"/{task}/" in k and "/mg/" not in k), None)
    if key is None:
        raise SystemExit(f"no {split} human dataset for {task}")
    date = key.split("/")[3]
    dest = root / Path(key).parent                     # .../<Task>/<date>; the tar holds "lerobot/"
    url = f"https://utexas.box.com/shared/static/{links[key].rstrip('/').split('/')[-1]}.tar"
    head = subprocess.run(["curl", "-sL", "--max-time", "60", "-r", "0-0", "-D", "-", "-o", "/dev/null", url],
                          capture_output=True, text=True).stdout
    size = int([l for l in head.splitlines() if l.lower().startswith("content-range")][-1].split("/")[-1])
    print(f"{task} {date}: {size / 1e6:.0f} MB -> {dest}/lerobot", flush=True)
    if os.environ["DRY_RUN"] == "1" or (dest / "lerobot").is_dir():
        continue
    dest.mkdir(parents=True, exist_ok=True)
    tar = dest / "lerobot.tar"
    subprocess.run(["curl", "-fL", "--retry", "8", "--retry-delay", "10", "-C", "-", "-o", str(tar), url], check=True)
    assert tar.stat().st_size == size, f"{tar}: {tar.stat().st_size} != {size}"
    with tarfile.open(tar) as t:
        t.extractall(dest)
    tar.unlink()
    print(f"done {task}", flush=True)
PY
