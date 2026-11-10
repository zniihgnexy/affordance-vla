#!/usr/bin/env bash
# Download what the starVLA LIBERO recipe needs onto the A6000 server (estar): the four LIBERO suites in
# LeRobot format and the Qwen3-VL-2B weights. Everything goes to /data; no GPU is used. Resumable.
#
# Measured from this server on 2026-11-10: huggingface.co 0.05-0.1 MB/s and unstable, hf-mirror.com
# ~0.3 MB/s with dropped connections (`hf download` then reports success with files missing),
# modelscope.cn ~25 MB/s. So both the model (official Qwen release) and the datasets come from ModelScope.
#
# Usage on the server:  nohup bash download_assets.sh > $W/logs/download_assets.log 2>&1 &
set -uo pipefail

W="${W:-/data/xinyizheng/affordance3d_vla}"
PY="${PY:-$HOME/anaconda3/bin/python}"
MODEL="${MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
DATASET_OWNER="${DATASET_OWNER:-adamzhuo}"
MAX_TRIES="${MAX_TRIES:-12}"
mkdir -p "$W/data/libero_lerobot" "$W/models" "$W/logs"

# ---- model weights from ModelScope: list files, fetch each with resume, check the size ----
model_dir="$W/models/$(basename "$MODEL")"
mkdir -p "$model_dir"
curl -fsS -m 60 "https://modelscope.cn/api/v1/models/$MODEL/repo/files?Revision=master&Recursive=true" \
  | "$PY" -c 'import json, sys
for f in json.load(sys.stdin)["Data"]["Files"]:
    if f.get("Type") != "tree": print(f["Size"], f["Path"], sep="\t")' > "$model_dir/.modelscope_files.tsv" \
  || echo "FAILED: ModelScope file list"
while IFS=$'\t' read -r size path; do
  target="$model_dir/$path"
  mkdir -p "$(dirname "$target")"
  for try in $(seq "$MAX_TRIES"); do
    [ -f "$target" ] && [ "$(stat -c %s "$target")" = "$size" ] && break
    curl -fsSL --retry 3 -C - -o "$target" "https://modelscope.cn/models/$MODEL/resolve/master/$path" || sleep 5
  done
  [ "$(stat -c %s "$target" 2>/dev/null)" = "$size" ] && echo "ok: $path ($size B)" || echo "FAILED: model file $path"
done < "$model_dir/.modelscope_files.tsv"

# ---- LIBERO in LeRobot format ----
# ModelScope has a re-upload of the IPEC-COMMUNITY release under the same names (adamzhuo/...). Sampled
# files were byte-identical to the Hugging Face originals (sha256, 2026-11-10); every file is verified
# against the sha256 in the ModelScope listing by the fetcher.
for suite in libero_spatial libero_object libero_goal libero_10; do
  repo="${suite}_no_noops_1.0.0_lerobot"
  if "$PY" "$(dirname "$0")/fetch_modelscope_dataset.py" "$DATASET_OWNER/$repo" "$W/data/libero_lerobot/$repo" \
      --workers 16; then echo "ok: $repo"; else echo "FAILED: $repo"; fi
  # starVLA reads the camera/state/action layout from meta/modality.json in each dataset folder
  cp "$W/repos/starVLA/examples/simBenchmarks/LIBERO/train_files/modality.json" \
     "$W/data/libero_lerobot/$repo/meta/" || echo "FAILED: modality.json for $suite"
done
echo "ALL-DOWNLOADS-FINISHED"
