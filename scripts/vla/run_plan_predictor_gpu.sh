#!/usr/bin/env bash
# Train the plan predictor as ONE job for the shared GPU queue on the A6000 server (estar):
# a readout of the frozen VLM that predicts the next gripper event, the contact point and the approach
# direction (train_plan_predictor.py). The job also tests it without the simulator.
# Settings come from the environment (defaults in brackets):
#   RUN (required)       outputs go to $W/runs/plan_predictor/$RUN
#   DATA [.../dataset10] training set from build_plan_dataset.py
#   STEPS [3000]
#   HEAD [regress]       regress: the contact's coordinates are regressed; grid: every image cell is scored
#   CAMERA_LOG [...]     a rollout log whose camera calibration the grid head uses
#   PRO [released]       run under runs/libero_pro whose first frames are the benchmark test; empty = none
#   TRAIN_ARGS []        further options of train_plan_predictor.py, e.g. "--evaluate-only" to test an
#                        existing predictor again and rewrite metrics.json and predictions.npz
# Submit (never run directly; the queue sets CUDA_VISIBLE_DEVICES):
#   bash scripts/server/submit_queue_job.sh 123 plan_predictor_p1 22000 6 "cards=5 1 6 7" -- \
#       env RUN=p1 bash ~/affordance-vla/scripts/vla/run_plan_predictor_gpu.sh
set -uo pipefail

W="${W:-/mnt/nas/xinyizheng/affordance_vla}"
REPO="${REPO:-$HOME/affordance-vla}"
STARVLA="${STARVLA:-$W/repos/starVLA}"
STARVLA_PYTHON="${STARVLA_PYTHON:-/data/xinyizheng/envs/starvla/bin/python}"
CKPT="${CKPT:-$W/models/Qwen3-VL-PI-LIBERO-4in1/checkpoints/steps_100000_pytorch_model.pt}"
BASE_VLM="${BASE_VLM:-$W/models/Qwen3-VL-4B-Instruct}"
RUN="${RUN:-}"
DATA="${DATA:-$W/runs/plan_following/dataset10}"
STEPS="${STEPS:-3000}"
HEAD="${HEAD:-regress}"
CAMERA_LOG="${CAMERA_LOG:-$W/runs/plan_following/demos/pick_up_the_salad_dressing_and_place_it_in_the_basket_demo0.npz}"
PRO="${PRO-released}"
TRAIN_ARGS="${TRAIN_ARGS:-}"

[[ "$RUN" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "RUN is required: letters, digits, _ and -" >&2; exit 2; }
[[ "$PRO" =~ ^[A-Za-z0-9_-]*$ && "$STEPS" =~ ^[1-9][0-9]*$ ]] || { echo "PRO or STEPS is malformed" >&2; exit 2; }
[[ "$TRAIN_ARGS" =~ ^[A-Za-z0-9\ ._+-]*$ ]] || { echo "TRAIN_ARGS may hold only option names and numbers" >&2; exit 2; }
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "CUDA_VISIBLE_DEVICES is not set: submit this through the queue" >&2; exit 2; }
case "$HEAD" in regress|grid) ;; *) echo "HEAD must be regress or grid" >&2; exit 2 ;; esac
read -r -a train_args <<< "$TRAIN_ARGS"
head_args=(--head "$HEAD")
[ "$HEAD" = grid ] && head_args+=(--camera-log "$CAMERA_LOG")
pro_args=()
[ -n "$PRO" ] && pro_args=(--pro "$W/runs/libero_pro/$PRO")
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$STARVLA" || exit 1
echo "=== plan predictor $RUN ($HEAD head): $STEPS steps on $(basename "$DATA") ($(date '+%F %T'), card ${GPU_INDEX:-?}) ==="
PYTHONPATH="$STARVLA:$REPO" "$STARVLA_PYTHON" "$REPO/scripts/vla/train_plan_predictor.py" --ckpt "$CKPT" \
  --base-vlm "$BASE_VLM" --data "$DATA" --out "$W/runs/plan_predictor/$RUN" --steps "$STEPS" "${head_args[@]}" \
  ${pro_args[@]+"${pro_args[@]}"} ${train_args[@]+"${train_args[@]}"} || { echo "TRAIN-FAILED"; exit 1; }
echo "=== finished ($(date '+%F %T')), failed=0 ==="
