#!/usr/bin/env bash
# Fetch RoboCasa365 PRETRAIN-split LeRobot demo bundles (CPU + network, no GPU). Never touches the target split.
# Like scripts/data/fetch_robocasa365.sh (Box link from box_links_ds.json, size first, curl resume) but for the
# pretrain split and either source: mg (MimicGen-generated, 10k demos per task, 5k for OpenCabinet) or human (100 demos).
#
# Usage on the server:  bash scripts/rc365/fetch_pretrain_demos.sh --source mg|human [--dry-run] [--max-gb N] [TASK ...]
# Default tasks: the eight atomic tasks of AFFO-16. --max-gb (default 100): stop before downloading if the sum is larger.
# Output: $W/data/robocasa365/v1.0/pretrain/atomic/<Task>/<date>[/mg/demo/<stamp>]/lerobot   (the registry's own paths)
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_DIR/envs/server/robocasa365_env.sh"
SOURCE=""; DRY=0; MAXGB=100; TASKS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --source) SOURCE="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    --max-gb) MAXGB="$2"; shift 2;;
    *) TASKS+=("$1"); shift;;
  esac
done
[ "$SOURCE" = mg ] || [ "$SOURCE" = human ] || { echo "--source mg|human is required" >&2; exit 2; }
[ ${#TASKS[@]} -gt 0 ] || TASKS=(CoffeeSetupMug PickPlaceCounterToCabinet PickPlaceCounterToStove PickPlaceSinkToCounter OpenDrawer OpenCabinet TurnOffStove TurnOnSinkFaucet)
export SOURCE DRY_RUN=$DRY MAX_GB=$MAXGB TASKS_LIST="${TASKS[*]}"
mkdir -p "$W/logs"
"$RC365_PYTHON" - <<'PY' 2> >(grep -v "WARNING" >&2)
import json, os, subprocess, tarfile
from pathlib import Path
import robocasa
from robocasa.macros import DATASET_BASE_PATH
from robocasa.utils.dataset_registry import ATOMIC_TASK_DATASETS

links = json.load(open(Path(robocasa.__path__[0]) / "models/assets/box_links/box_links_ds.json"))
root = Path(DATASET_BASE_PATH)
key_field = "mg_path" if os.environ["SOURCE"] == "mg" else "human_path"
items = []
for task in os.environ["TASKS_LIST"].split():
    folder = ATOMIC_TASK_DATASETS[task]["pretrain"][key_field]          # e.g. v1.0/pretrain/atomic/<Task>/<date>[/mg/...]
    key = f"{folder.removeprefix('v1.0/')}/lerobot.tar"
    url = f"https://utexas.box.com/shared/static/{links[key].rstrip('/').split('/')[-1]}.tar"
    head = subprocess.run(["curl", "-sL", "--max-time", "60", "-r", "0-0", "-D", "-", "-o", "/dev/null", url],
                          capture_output=True, text=True).stdout
    size = int([l for l in head.splitlines() if l.lower().startswith("content-range")][-1].split("/")[-1])
    items.append((task, root / folder, url, size))
    print(f"{task}: {size / 1e9:.2f} GB -> {root / folder}/lerobot", flush=True)
total = sum(i[3] for i in items) / 1e9
print(f"TOTAL {os.environ['SOURCE']}: {total:.1f} GB", flush=True)
if os.environ["DRY_RUN"] == "1":
    raise SystemExit(0)
if total > float(os.environ["MAX_GB"]):
    raise SystemExit(f"total {total:.1f} GB exceeds --max-gb {os.environ['MAX_GB']}; not downloading")
for task, dest, url, size in items:
    if (dest / "lerobot").is_dir():
        print(f"skip {task} (exists)", flush=True)
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
