#!/usr/bin/env bash
# Set up LIBERO-PRO on the server (CPU and network only, no GPU, no queue). Safe to run again: finished
# steps are skipped.
#   1. the repository as an archive from codeload.github.com (git clone hangs on this server), ~375 MB
#   2. the BDDL and start-state files of the perturbed suites from Hugging Face (676 files, ~10 MB)
#   3. those files copied into the fork, as its README asks
#   4. a LIBERO config that points at the fork, used by libero_pro_env.sh
# Usage on the server:  bash scripts/remote/setup_libero_pro.sh
set -euo pipefail

W="${W:-/mnt/nas/xinyizheng/affordance_vla}"
FORK="$W/repos/LIBERO-PRO"
FILES="$W/data/libero_pro"
HF="https://huggingface.co"
DATASET="zhouxueyang/LIBERO-Pro"
mkdir -p "$W/repos/_dl" "$FILES" "$W/cache/libero_pro_config"

if [ ! -d "$FORK" ]; then
  archive="$W/repos/_dl/LIBERO-PRO-master.zip"
  curl -fL --http2 --retry 5 --retry-delay 5 --connect-timeout 20 --max-time 1800 -C - -o "$archive" \
    "https://codeload.github.com/Zxy-MLlab/LIBERO-PRO/zip/refs/heads/master"
  python3 -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$archive" "$W/repos/_dl/unpacked"
  mv "$W/repos/_dl/unpacked/LIBERO-PRO-master" "$FORK"
  rmdir "$W/repos/_dl/unpacked"
fi

curl -fsS --connect-timeout 20 --max-time 120 -o "$FILES/listing.json" "$HF/api/datasets/$DATASET/tree/main?recursive=true"
python3 - "$FILES" "$HF/datasets/$DATASET/resolve/main/" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

out, base = Path(sys.argv[1]), sys.argv[2]
files = [entry for entry in json.loads((out / "listing.json").read_text()) if entry.get("type") == "file"]
failed = []
for entry in files:
    target = out / entry["path"]
    if target.exists() and target.stat().st_size == entry.get("size"):
        continue
    target.parent.mkdir(parents=True, exist_ok=True)
    code = subprocess.run(["curl", "-fsSL", "--retry", "4", "--retry-delay", "3", "--connect-timeout", "20",
                           "--max-time", "180", "-o", str(target), base + entry["path"]]).returncode
    if code != 0 or target.stat().st_size != entry.get("size"):
        failed.append(entry["path"])
print(f"{len(files) - len(failed)} of {len(files)} files present")
if failed:
    sys.exit(f"could not fetch: {failed[:5]} ...")
PY

for kind in bddl_files init_files; do
  for folder in "$FILES/$kind"/libero_*; do
    cp -rn "$folder" "$FORK/libero/libero/$kind/"
  done
done

cat > "$W/cache/libero_pro_config/config.yaml" <<YAML
assets: $FORK/libero/libero/assets
bddl_files: $FORK/libero/libero/bddl_files
benchmark_root: $FORK/libero/libero
datasets: $W/data/libero
init_states: $FORK/libero/libero/init_files
YAML
echo "LIBERO-PRO ready: $(ls "$FORK/libero/libero/bddl_files" | grep -c -E '_(swap|task|object|lan)$') perturbed suites"
