#!/usr/bin/env bash
# E1 of Pipeline v3 as ONE job for the shared GPU queue on the A6000 server (estar): instruction-to-cell
# similarity maps of the frozen VLM, and the grid plan head scored the same way (probe_semantic_prior.py).
# Nothing is trained.
# Settings come from the environment (defaults in brackets):
#   RUN (required)       outputs go to $W/runs/semantic_probe/$RUN
#   RUNS [released spatial_released_seed7]   runs under runs/libero_pro whose first frames are probed
#   PREDICTOR [g1]       run under runs/plan_predictor holding the grid head
#   DATA [.../dataset10] plan dataset whose offsets.json gives the grasp points
#   CAMERA_LOG [...]     a rollout log with the camera calibration
# Submit (never run directly; the queue sets CUDA_VISIBLE_DEVICES):
#   bash scripts/server/submit_queue_job.sh 171 semantic_probe_e1 22000 6 -- \
#       env RUN=e1 bash ~/affordance-vla/scripts/vla/run_semantic_probe_gpu.sh
set -uo pipefail

W="${W:-/mnt/nas/xinyizheng/affordance_vla}"
REPO="${REPO:-$HOME/affordance-vla}"
STARVLA="${STARVLA:-$W/repos/starVLA}"
STARVLA_PYTHON="${STARVLA_PYTHON:-/data/xinyizheng/envs/starvla/bin/python}"
CKPT="${CKPT:-$W/models/Qwen3-VL-PI-LIBERO-4in1/checkpoints/steps_100000_pytorch_model.pt}"
BASE_VLM="${BASE_VLM:-$W/models/Qwen3-VL-4B-Instruct}"
RUN="${RUN:-}"
RUNS="${RUNS:-released spatial_released_seed7}"
PREDICTOR="${PREDICTOR:-g1}"
DATA="${DATA:-$W/runs/plan_following/dataset10}"
CAMERA_LOG="${CAMERA_LOG:-$W/runs/plan_following/demos/pick_up_the_salad_dressing_and_place_it_in_the_basket_demo0.npz}"

[[ "$RUN" =~ ^[A-Za-z0-9_-]+$ && "$PREDICTOR" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "RUN and PREDICTOR: letters, digits, _ and -" >&2; exit 2; }
[[ "$RUNS" =~ ^[A-Za-z0-9_\ -]+$ ]] || { echo "RUNS holds run names separated by spaces" >&2; exit 2; }
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "CUDA_VISIBLE_DEVICES is not set: submit this through the queue" >&2; exit 2; }
read -r -a run_names <<< "$RUNS"
run_dirs=()
for name in "${run_names[@]}"; do run_dirs+=("$W/runs/libero_pro/$name"); done
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$STARVLA" || exit 1
echo "=== semantic probe $RUN on ${RUNS} with head $PREDICTOR ($(date '+%F %T'), card ${GPU_INDEX:-?}) ==="
PYTHONPATH="$STARVLA:$REPO" "$STARVLA_PYTHON" "$REPO/scripts/vla/probe_semantic_prior.py" --ckpt "$CKPT" \
  --base-vlm "$BASE_VLM" --runs "${run_dirs[@]}" --predictor "$W/runs/plan_predictor/$PREDICTOR" \
  --camera-log "$CAMERA_LOG" --data "$DATA" --out "$W/runs/semantic_probe/$RUN" || { echo "PROBE-FAILED"; exit 1; }
echo "=== finished ($(date '+%F %T')), failed=0 ==="
