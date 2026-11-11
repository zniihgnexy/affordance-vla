#!/usr/bin/env bash
# Plan-following experiment as ONE job for the shared GPU queue on the A6000 server (estar):
# does the released starVLA action head follow a plan it is given?
#   stage train  train_plan_following.py: plan encoder + low-rank updates on the frozen released checkpoint
#   stage eval   serve_plan_policy.py on the queue's card + closed-loop LIBERO rollouts that send the
#                oracle plan (or an empty plan) with every query; the simulator runs on the CPU
# Settings come from the environment (defaults in brackets):
#   RUN [shift]                name of the run; outputs go to $W/runs/plan_following/$RUN
#   DATA [.../dataset]         training set from build_plan_dataset.py
#   COUNTERFACTUAL [shift]     none | shift | shift+other
#   SUPERVISION [path]         path | actions
#   STEPS [1500]  STAGES ["train eval"]  PLAN [oracle]  SETS ["standard shift"]  (ask needs offsets of all objects)
#   EPISODES ["0 1"]           start states of the perturbed sets (ask, shift); the standard set keeps 0-4
#   SUFFIX []                  appended to the rollout folder: $OUT/rollouts_$PLAN$SUFFIX (e.g. _more)
#                              A second folder for the same adapter must hold other rollouts than the first:
#                              leave "standard" out of SETS and give EPISODES that were not run before
#                              (the notebook refuses a rollout that appears in two folders).
#
# Before submitting (CPU, no queue):  replay_libero_states.sh, then build_plan_dataset.py.
# Submit (never run directly; the queue sets CUDA_VISIBLE_DEVICES):
#   bash scripts/remote/submit_queue_job.sh 100 plan_shift 26000 3 "cards=1 5 6 7" -- \
#       env RUN=shift COUNTERFACTUAL=shift bash ~/affordance-vla/scripts/remote/run_plan_following_gpu.sh
set -uo pipefail

W="${W:-/mnt/nas/xinyizheng/affordance_vla}"
REPO="${REPO:-$HOME/affordance-vla}"
STARVLA="${STARVLA:-$W/repos/starVLA}"
STARVLA_PYTHON="${STARVLA_PYTHON:-/data/xinyizheng/envs/starvla/bin/python}"
CKPT="${CKPT:-$W/models/Qwen3-VL-PI-LIBERO-4in1/checkpoints/steps_100000_pytorch_model.pt}"
BASE_VLM="${BASE_VLM:-$W/models/Qwen3-VL-4B-Instruct}"
RUN="${RUN:-shift}"
DATA="${DATA:-$W/runs/plan_following/dataset}"
COUNTERFACTUAL="${COUNTERFACTUAL:-shift}"
SUPERVISION="${SUPERVISION:-path}"
STEPS="${STEPS:-1500}"
STAGES="${STAGES:-train eval}"
PLAN="${PLAN:-oracle}"
SETS="${SETS:-standard shift}"
EPISODES="${EPISODES:-0 1}"
SUFFIX="${SUFFIX:-}"
OUT="$W/runs/plan_following/$RUN"
TASK="pick up the salad dressing and place it in the basket"
SERVER_START_TRIES=240   # x 5 s: loading 16 GB from the NAS takes a few minutes

[[ "$RUN" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "RUN may contain only letters, digits, _ and -" >&2; exit 2; }
[[ "$SUFFIX" =~ ^[A-Za-z0-9_-]*$ ]] || { echo "SUFFIX may contain only letters, digits, _ and -" >&2; exit 2; }
[[ "$EPISODES" =~ ^[0-9]+(\ [0-9]+)*$ ]] || { echo "EPISODES must be start indices separated by spaces" >&2; exit 2; }
read -r -a episodes <<< "$EPISODES"
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "CUDA_VISIBLE_DEVICES is not set: submit this through the queue" >&2; exit 2; }
mkdir -p "$OUT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# The released checkpoint was trained before starVLA's canonical DiT forward (see its LIBERO README).
OVERRIDES=("framework.qwenvl.base_vlm=$BASE_VLM" "framework.action_model.diffusion_model_cfg.use_canonical_forward=false")
failed=0
cd "$STARVLA" || exit 1

if [[ " $STAGES " == *" train "* ]]; then
  echo "=== train: $RUN, counterfactual=$COUNTERFACTUAL, supervision=$SUPERVISION, $STEPS steps ($(date '+%F %T'), card ${GPU_INDEX:-?}) ==="
  PYTHONPATH="$STARVLA:$REPO" "$STARVLA_PYTHON" "$REPO/scripts/remote/train_plan_following.py" --ckpt "$CKPT" \
    --base-vlm "$BASE_VLM" --data "$DATA" --out "$OUT" --counterfactual "$COUNTERFACTUAL" \
    --supervision "$SUPERVISION" --steps "$STEPS" || { echo "TRAIN-FAILED"; exit 1; }
fi

if [[ " $STAGES " == *" eval "* ]]; then
  ROLLOUTS="$OUT/rollouts_$PLAN$SUFFIX"
  mkdir -p "$ROLLOUTS"
  echo "=== eval: $RUN with plan=$PLAN, sets: $SETS, starts of the perturbed sets: $EPISODES ($(date '+%F %T')) ==="
  PORT="$("$STARVLA_PYTHON" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')"
  server_args=(--ckpt_path "$CKPT" --adapter "$OUT/adapter.pt" --port "$PORT" --seed 7)
  for override in "${OVERRIDES[@]}"; do server_args+=(--config_override "$override"); done
  PYTHONPATH="$STARVLA:$REPO" "$STARVLA_PYTHON" "$REPO/scripts/remote/serve_plan_policy.py" "${server_args[@]}" \
    > "$ROLLOUTS/server.log" 2>&1 &
  SERVER_PID=$!
  pids=(); tags=()
  # on any exit (also when the queue stops the job) stop the server and every simulator still running
  cleanup() { kill "$SERVER_PID" ${pids[@]+"${pids[@]}"} 2>/dev/null; }
  trap cleanup EXIT
  ready=0
  for _ in $(seq "$SERVER_START_TRIES"); do
    kill -0 "$SERVER_PID" 2>/dev/null || break
    if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then ready=1; break; fi
    sleep 5
  done
  if [ "$ready" -ne 1 ]; then
    echo "SERVER-FAILED (see $ROLLOUTS/server.log)"; tail -n 30 "$ROLLOUTS/server.log"; exit 1
  fi
  echo "plan policy server ready on port $PORT ($(date '+%T'))"

  # shellcheck disable=SC1091
  source "$REPO/scripts/remote/libero_env.sh"
  plan_args=(--plan "$PLAN")
  [ "$PLAN" = oracle ] && plan_args+=(--plan-offsets "$DATA/offsets.json")
  rollout() {  # rollout <tag> [options of rollout_libero_logged.py ...]; simulator side, CPU only
    local tag="$1"; shift
    CUDA_VISIBLE_DEVICES='' PYTHONPATH="$STARVLA:$REPO:$PYTHONPATH" "$LIBERO_PYTHON" \
      "$REPO/scripts/remote/rollout_libero_logged.py" --out "$ROLLOUTS" --task "$TASK" --port "$PORT" \
      --tag "$tag" "${plan_args[@]}" "$@" > "$ROLLOUTS/$tag.log" 2>&1 &
    pids+=("$!"); tags+=("$tag")
  }
  # with the default starts: the same 21 rollouts as the pipeline demo of 2026-11-10 (run_vla_demo_gpu.sh)
  if [[ " $SETS " == *" standard "* ]]; then rollout standard --episodes 0 1 2 3 4; fi
  if [[ " $SETS " == *" ask "* ]]; then
    for object in cream_cheese milk alphabet_soup ketchup tomato_sauce; do
      rollout "ask_${object}" --episodes "${episodes[@]}" --target "${object}_1" \
        --instruction "pick up the ${object//_/ } and place it in the basket"
    done
  fi
  if [[ " $SETS " == *" shift "* ]]; then
    rollout shift_4cm --episodes "${episodes[@]}" --shift salad_dressing_1 0.03 0.03
    rollout shift_9cm --episodes "${episodes[@]}" --shift salad_dressing_1 -0.08 0.05
    rollout shift_16cm --episodes "${episodes[@]}" --shift salad_dressing_1 -0.10 0.12
  fi
  for k in "${!pids[@]}"; do
    if wait "${pids[$k]}"; then grep -F "[${tags[$k]} ep" "$ROLLOUTS/${tags[$k]}.log"
    else echo "ROLLOUT-FAILED: ${tags[$k]} (see $ROLLOUTS/${tags[$k]}.log)"; failed=1; fi
  done
  pids=()  # every simulator has been waited for; only the server is left for the exit trap
fi

echo "=== finished ($(date '+%F %T')), failed=$failed ==="
exit "$failed"
