#!/usr/bin/env bash
# LIBERO-PRO evaluation as ONE job for the shared GPU queue on the A6000 server (estar).
# A policy server runs on the queue's card; one simulator process per task runs on the CPU and sends
# its queries to it. Policies are trained on standard LIBERO only; nothing here is trained.
# Settings come from the environment (defaults in brackets):
#   NAME (required)     outputs go to $W/runs/libero_pro/$NAME/<suite>/task<k>_ep<n>.npz
#   ADAPTER []          run under runs/plan_following whose adapter.pt is served with a plan input;
#                       empty = the released checkpoint in starVLA's own server, without any plan
#   PLAN [none]         none | oracle | zero | predicted   (all but none need ADAPTER)
#   PREDICTOR []        run under runs/plan_predictor whose predictor.pt makes the plan (PLAN=predicted)
#   CAMERA_LOG [...]    a rollout log whose camera calibration a grid predictor uses
#   DATA [.../dataset10]  training set whose offsets.json the oracle plan reads
#   SUITES ["libero_object_swap libero_object_task"]   position and task perturbation of LIBERO-Object
#   TASKS ["0 1 2 3 4 5 6 7 8 9"]   TRIALS [20] start states 0..TRIALS-1 per task (the paper uses 50)
#   IMAGES_FOR [2]      episodes per task whose logs keep the camera images (the rest keep states only)
#   MAX_SIMS [24]       simulator processes running at the same time (one per suite and task)
# The instruction of a task is LIBERO's own; where the perturbation renames the object, the new name
# is put into it (rollout_libero_logged.py, benchmark_instruction).
# Before the first use (CPU):  bash scripts/remote/setup_libero_pro.sh
# Submit (never run directly; the queue sets CUDA_VISIBLE_DEVICES):
#   bash scripts/remote/submit_queue_job.sh 120 pro_released 16000 3 "cards=5 1" -- \
#       env NAME=released bash ~/affordance-vla/scripts/remote/run_libero_pro_gpu.sh
set -uo pipefail

W="${W:-/mnt/nas/xinyizheng/affordance_vla}"
REPO="${REPO:-$HOME/affordance-vla}"
STARVLA="${STARVLA:-$W/repos/starVLA}"
STARVLA_PYTHON="${STARVLA_PYTHON:-/data/xinyizheng/envs/starvla/bin/python}"
CKPT="${CKPT:-$W/models/Qwen3-VL-PI-LIBERO-4in1/checkpoints/steps_100000_pytorch_model.pt}"
BASE_VLM="${BASE_VLM:-$W/models/Qwen3-VL-4B-Instruct}"
NAME="${NAME:-}"
ADAPTER="${ADAPTER:-}"
PLAN="${PLAN:-none}"
PREDICTOR="${PREDICTOR:-}"
CAMERA_LOG="${CAMERA_LOG:-$W/runs/plan_following/demos/pick_up_the_salad_dressing_and_place_it_in_the_basket_demo0.npz}"
DATA="${DATA:-$W/runs/plan_following/dataset10}"
SUITES="${SUITES:-libero_object_swap libero_object_task}"
TASKS="${TASKS:-0 1 2 3 4 5 6 7 8 9}"
TRIALS="${TRIALS:-20}"
IMAGES_FOR="${IMAGES_FOR:-2}"
MAX_SIMS="${MAX_SIMS:-24}"
SERVER_START_TRIES=240   # x 5 s: loading 16 GB from the NAS takes a few minutes

[[ "$NAME" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "NAME is required: letters, digits, _ and -" >&2; exit 2; }
[[ "$ADAPTER" =~ ^[A-Za-z0-9_-]*$ ]] || { echo "ADAPTER is a run name: letters, digits, _ and -" >&2; exit 2; }
[[ "$SUITES" =~ ^[a-z0-9_]+(\ [a-z0-9_]+)*$ ]] || { echo "SUITES must be suite names separated by spaces" >&2; exit 2; }
[[ "$TASKS" =~ ^[0-9]+(\ [0-9]+)*$ ]] || { echo "TASKS must be task indices separated by spaces" >&2; exit 2; }
[[ "$TRIALS" =~ ^[1-9][0-9]*$ && "$IMAGES_FOR" =~ ^[0-9]+$ && "$MAX_SIMS" =~ ^[1-9][0-9]*$ ]] \
  || { echo "TRIALS, IMAGES_FOR or MAX_SIMS is malformed" >&2; exit 2; }
SUITES="$(tr ' ' '\n' <<< "$SUITES" | awk '!seen[$0]++' | tr '\n' ' ')"; SUITES="${SUITES% }"   # no suite twice
TASKS="$(tr ' ' '\n' <<< "$TASKS" | awk '!seen[$0]++' | tr '\n' ' ')"; TASKS="${TASKS% }"
[[ "$PREDICTOR" =~ ^[A-Za-z0-9_-]*$ ]] || { echo "PREDICTOR is a run name: letters, digits, _ and -" >&2; exit 2; }
case "$PLAN" in none|oracle|zero|predicted) ;; *) echo "PLAN must be none, oracle, zero or predicted" >&2; exit 2 ;; esac
if [ -z "$ADAPTER" ] && [ "$PLAN" != none ]; then echo "PLAN=$PLAN needs ADAPTER" >&2; exit 2; fi
if [ -n "$ADAPTER" ] && [ "$PLAN" = none ]; then echo "an adapter is served with a plan: set PLAN" >&2; exit 2; fi
if [ "$PLAN" = predicted ] && [ -z "$PREDICTOR" ]; then echo "PLAN=predicted needs PREDICTOR" >&2; exit 2; fi
if [ "$PLAN" != predicted ] && [ -n "$PREDICTOR" ]; then echo "PREDICTOR is only used with PLAN=predicted" >&2; exit 2; fi
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "CUDA_VISIBLE_DEVICES is not set: submit this through the queue" >&2; exit 2; }
OUT="$W/runs/libero_pro/$NAME"
mkdir -p "$OUT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# The released checkpoint was trained before starVLA's canonical DiT forward (see its LIBERO README).
OVERRIDES=("framework.qwenvl.base_vlm=$BASE_VLM"
           "framework.action_model.diffusion_model_cfg.use_canonical_forward=false")
cd "$STARVLA" || exit 1

echo "=== LIBERO-PRO: $NAME, adapter=${ADAPTER:-none (released policy)}, plan=$PLAN, $TRIALS trials per task"
echo "    suites: $SUITES ($(date '+%F %T'), card ${GPU_INDEX:-?}) ==="
PORT="$("$STARVLA_PYTHON" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')"
server_args=(--ckpt_path "$CKPT" --port "$PORT" --seed 7)
for override in "${OVERRIDES[@]}"; do server_args+=(--config_override "$override"); done
if [ -n "$ADAPTER" ]; then
  [ "$PLAN" = predicted ] && server_args+=(--predictor "$W/runs/plan_predictor/$PREDICTOR/predictor.pt"
                                           --camera-log "$CAMERA_LOG")
  PYTHONPATH="$STARVLA:$REPO" "$STARVLA_PYTHON" "$REPO/scripts/remote/serve_plan_policy.py" "${server_args[@]}" \
    --adapter "$W/runs/plan_following/$ADAPTER/adapter.pt" > "$OUT/server.log" 2>&1 &
else
  PYTHONPATH="$STARVLA" "$STARVLA_PYTHON" deployment/model_server/server_policy.py "${server_args[@]}" --use_bf16 \
    > "$OUT/server.log" 2>&1 &
fi
SERVER_PID=$!
pids=(); labels=()
cleanup() { kill "$SERVER_PID" ${pids[@]+"${pids[@]}"} 2>/dev/null; }  # also when the queue stops the job
trap cleanup EXIT
ready=0
for _ in $(seq "$SERVER_START_TRIES"); do
  kill -0 "$SERVER_PID" 2>/dev/null || break
  if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then ready=1; break; fi
  sleep 5
done
if [ "$ready" -ne 1 ]; then echo "SERVER-FAILED (see $OUT/server.log)"; tail -n 30 "$OUT/server.log"; exit 1; fi
echo "policy server ready on port $PORT ($(date '+%T'))"

# shellcheck disable=SC1091
source "$REPO/scripts/remote/libero_pro_env.sh"
plan_args=(--plan "$PLAN")
[ "$PLAN" = oracle ] && plan_args+=(--plan-offsets "$DATA/offsets.json")
read -r -a episodes <<< "$(seq -s ' ' 0 $((TRIALS - 1)))"
failed=0
for suite in $SUITES; do
  mkdir -p "$OUT/$suite"
  rm -f "$OUT/$suite"/task*_summary.json  # an earlier run with this NAME must not be counted again
  for task in $TASKS; do
    while [ "$(jobs -rp | wc -l)" -gt "$MAX_SIMS" ]; do sleep 5; done  # the policy server is one of the jobs
    CUDA_VISIBLE_DEVICES='' PYTHONPATH="$STARVLA:$REPO:$PYTHONPATH" "$LIBERO_PYTHON" \
      "$REPO/scripts/remote/rollout_libero_logged.py" --out "$OUT/$suite" --suite "$suite" --task-id "$task" \
      --tag "task$task" --port "$PORT" --episodes "${episodes[@]}" --images-for "$IMAGES_FOR" "${plan_args[@]}" \
      > "$OUT/$suite/task$task.log" 2>&1 &
    pids+=("$!"); labels+=("$suite/task$task")
  done
done
for k in "${!pids[@]}"; do
  wait "${pids[$k]}" || { echo "ROLLOUT-FAILED: ${labels[$k]} (see $OUT/${labels[$k]}.log)"; failed=1; }
done
pids=()  # every simulator has been waited for; only the server is left for the exit trap

expected=$(( $(wc -w <<< "$TASKS") * TRIALS ))
"$STARVLA_PYTHON" -I - "$OUT" "$expected" $SUITES <<'PY'
import json
import sys
from pathlib import Path

out, expected = Path(sys.argv[1]), int(sys.argv[2])
for suite in sys.argv[3:]:
    rows = [row for path in sorted((out / suite).glob("task*_summary.json")) for row in json.loads(path.read_text())]
    goal = sum(row["success"] for row in rows)
    note = "" if len(rows) == expected else f"  INCOMPLETE: {expected} episodes were asked for"
    print(f"[{suite}] task goal reached {goal}/{len(rows)}" + (f" = {goal / len(rows):.2f}" if rows else "") + note)
PY
echo "=== finished ($(date '+%F %T')), failed=$failed ==="
exit "$failed"
