#!/usr/bin/env bash
# Download what the starVLA LIBERO recipe needs onto the A6000 server (estar): the four LIBERO suites in
# LeRobot format and the Qwen3-VL-2B weights. Everything goes to /data; no GPU is used. Resumable.
#
# Measured from this server on 2026-11-10: huggingface.co 0.05-0.1 MB/s and unstable, hf-mirror.com
# ~0.3 MB/s, modelscope.cn ~25 MB/s. So the model comes from ModelScope (same Qwen release) and the
# datasets, which are not on ModelScope, come from hf-mirror with retries.
#
# Usage on the server:  nohup bash download_assets.sh > $W/logs/download_assets.log 2>&1 &
set -uo pipefail

W="${W:-/data/xinyizheng/affordance3d_vla}"
HF="${HF:-$HOME/anaconda3/bin/hf}"
PY="${PY:-$HOME/anaconda3/bin/python}"
MODEL="${MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
MAX_TRIES="${MAX_TRIES:-12}"
export HF_HOME="$W/cache/hf" HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
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

# ---- LIBERO in LeRobot format from hf-mirror, retried until complete ----
for suite in libero_spatial libero_object libero_goal libero_10; do
  repo="${suite}_no_noops_1.0.0_lerobot"
  done_flag=1
  for try in $(seq "$MAX_TRIES"); do
    if "$HF" download "IPEC-COMMUNITY/$repo" --repo-type dataset --local-dir "$W/data/libero_lerobot/$repo" \
        --max-workers 16 > "$W/logs/download_${suite}.log" 2>&1; then done_flag=0; break; fi
    echo "retry $try for $repo"; sleep 10
  done
  [ "$done_flag" = 0 ] && echo "ok: $repo" || echo "FAILED: $repo"
  # starVLA reads the camera/state/action layout from meta/modality.json in each dataset folder
  cp "$W/repos/starVLA/examples/simBenchmarks/LIBERO/train_files/modality.json" \
     "$W/data/libero_lerobot/$repo/meta/" || echo "FAILED: modality.json for $suite"
done
echo "ALL-DOWNLOADS-FINISHED"
