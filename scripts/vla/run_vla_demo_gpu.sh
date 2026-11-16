#!/usr/bin/env bash
# GPU part of the VLA pipeline demo (report/2026-11-10_vla_pipeline_demo.ipynb), as ONE job for the shared
# queue on the A6000 server (estar). It uses the released starVLA checkpoint Qwen3-VL-PI-LIBERO-4in1:
#   stage 1  probe_vla_demo.py: model facts, flow sampling, open-loop predictions, flow-matching loss
#   stage 2  starVLA's policy server on the queue's card + closed-loop LIBERO rollouts with per-step logs
#            (the simulator runs on the CPU with software rendering and never opens a GPU context)
# Inference only, bf16 weights: about 15 GB on the card. Nothing is trained.
#
# Before submitting, replay one demonstration on the CPU (no GPU, no queue); stage 1 reads its log:
#   source envs/server/libero_env.sh
#   "$LIBERO_PYTHON" scripts/eval/rollout_libero_logged.py --tag demo_replay \
#       --out $W/runs/vla_pipeline_demo/rollouts \
#       --replay-demo $W/data/libero/libero_object/pick_up_the_salad_dressing_and_place_it_in_the_basket_demo.hdf5
#
# Submit (never run directly; the queue sets CUDA_VISIBLE_DEVICES):
#   bash scripts/server/submit_queue_job.sh 100 vla_demo_gpu 22000 3 "cards=0 1" -- \
#       bash ~/affordance-vla/scripts/vla/run_vla_demo_gpu.sh
# Outputs: $W/runs/vla_pipeline_demo/{probe,rollouts}; logs next to them.
set -uo pipefail

W="${W:-/mnt/nas/xinyizheng/affordance_vla}"
REPO="${REPO:-$HOME/affordance-vla}"
STARVLA="${STARVLA:-$W/repos/starVLA}"
STARVLA_PYTHON="${STARVLA_PYTHON:-/data/xinyizheng/envs/starvla/bin/python}"
CKPT="${CKPT:-$W/models/Qwen3-VL-PI-LIBERO-4in1/checkpoints/steps_100000_pytorch_model.pt}"
BASE_VLM="${BASE_VLM:-$W/models/Qwen3-VL-4B-Instruct}"
DEMO_LOG="${DEMO_LOG:-$W/runs/vla_pipeline_demo/rollouts/demo_replay_ep0.npz}"   # from the CPU replay (see below)
OUT="${OUT:-$W/runs/vla_pipeline_demo}"
STAGES="${STAGES:-probe rollouts}"
TASK="pick up the salad dressing and place it in the basket"
SERVER_START_TRIES=240   # × 5 s: loading 16 GB from the NAS takes a few minutes

[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "CUDA_VISIBLE_DEVICES is not set: submit this through the queue" >&2; exit 2; }
mkdir -p "$OUT/probe" "$OUT/rollouts"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# The released checkpoint was trained before starVLA's canonical DiT forward (see its LIBERO README).
OVERRIDES=("framework.qwenvl.base_vlm=$BASE_VLM" "framework.action_model.diffusion_model_cfg.use_canonical_forward=false")
failed=0
cd "$STARVLA" || exit 1

if [[ " $STAGES " == *" probe "* ]]; then
  echo "=== stage 1: probe ($(date '+%F %T'), card ${GPU_INDEX:-?}) ==="
  PYTHONPATH="$STARVLA" "$STARVLA_PYTHON" "$REPO/scripts/vla/probe_vla_demo.py" --ckpt "$CKPT" \
    --base-vlm "$BASE_VLM" --log "$DEMO_LOG" --task "$TASK" --out "$OUT/probe" || { echo "PROBE-FAILED"; failed=1; }
fi

if [[ " $STAGES " == *" rollouts "* ]]; then
  echo "=== stage 2: policy server + rollouts ($(date '+%F %T')) ==="
  PORT="$("$STARVLA_PYTHON" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')"
  server_args=(--ckpt_path "$CKPT" --port "$PORT" --use_bf16 --seed 7)
  for override in "${OVERRIDES[@]}"; do server_args+=(--config_override "$override"); done
  PYTHONPATH="$STARVLA" "$STARVLA_PYTHON" deployment/model_server/server_policy.py "${server_args[@]}" \
    > "$OUT/rollouts/server.log" 2>&1 &
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
    echo "SERVER-FAILED (see $OUT/rollouts/server.log)"; tail -n 30 "$OUT/rollouts/server.log"; exit 1
  fi
  echo "policy server ready on port $PORT ($(date '+%T'))"

  # shellcheck disable=SC1091
  source "$REPO/envs/server/libero_env.sh"
  # Simulator side: CPU only. Each call is its own process; they run in parallel because software rendering
  # (~0.3 s per step) is the bottleneck, while the policy server answers one request at a time.
  rollout() {  # rollout <tag> [options of rollout_libero_logged.py ...]
    local tag="$1"; shift
    CUDA_VISIBLE_DEVICES='' PYTHONPATH="$STARVLA:$PYTHONPATH" "$LIBERO_PYTHON" \
      "$REPO/scripts/eval/rollout_libero_logged.py" --out "$OUT/rollouts" --task "$TASK" --port "$PORT" \
      --tag "$tag" "$@" > "$OUT/rollouts/$tag.log" 2>&1 &
    pids+=("$!"); tags+=("$tag")
  }
  # (a) the task as trained and evaluated: five of LIBERO's initial states
  rollout standard --episodes 0 1 2 3 4
  # (b) same scenes, but the instruction names another object that is on the floor (home-made, LIBERO-PRO style)
  for object in cream_cheese milk alphabet_soup ketchup tomato_sauce; do
    rollout "ask_${object}" --episodes 0 1 --target "${object}_1" \
      --instruction "pick up the ${object//_/ } and place it in the basket"
  done
  # (c) the trained instruction, but the target starts somewhere else (home-made layout shift, metres)
  rollout shift_4cm --episodes 0 1 --shift salad_dressing_1 0.03 0.03
  rollout shift_9cm --episodes 0 1 --shift salad_dressing_1 -0.08 0.05
  rollout shift_16cm --episodes 0 1 --shift salad_dressing_1 -0.10 0.12
  for k in "${!pids[@]}"; do
    if wait "${pids[$k]}"; then grep -F "[${tags[$k]} ep" "$OUT/rollouts/${tags[$k]}.log"
    else echo "ROLLOUT-FAILED: ${tags[$k]} (see $OUT/rollouts/${tags[$k]}.log)"; failed=1; fi
  done
  pids=()  # every simulator has been waited for; only the server is left for the exit trap
fi

echo "=== finished ($(date '+%F %T')), failed=$failed ==="
exit "$failed"
