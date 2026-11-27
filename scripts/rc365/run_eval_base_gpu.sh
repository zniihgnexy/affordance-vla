#!/usr/bin/env bash
# AFFO-16 evaluation of a base-policy checkpoint as ONE job for the shared GPU queue (A6000 server): a policy server on
# the queue's card and many CPU simulator clients (software rendering) over the grid of configs/rc365/base_policy_eval.yaml.
# Restartable: finished episodes (episodes.jsonl) are skipped.
#   CKPT   steps_N_pytorch_model.pt of a MERGED checkpoint (scripts/rc365/merge_lora_checkpoint.py; the run folder next
#          to checkpoints/ holds config.yaml and dataset_statistics.json)
#   RUN    [eval_<run folder name>]  outputs go to $W/runs/robocasa365_eval/$RUN (an existing folder is continued)
#   CONFIG [configs/rc365/base_policy_eval.yaml]  evaluation config (e.g. configs/rc365/base_checks_stuck.yaml)
#   OVERRIDES  key=value config overrides, space separated, e.g. "episodes=5 clients=8"
#   BENCH [0]  1: first measure how many clients the server carries (bench_policy_server.py), logged to bench.log
# Submit (never run directly), from the repo root:
#   bash scripts/server/submit_queue_job.sh 164 rc365_eval_base 40000 1 "cards=0 1 2 3 5 6 7" -- \
#       env CKPT=<merged ckpt> RUN=<name> bash $PWD/scripts/rc365/run_eval_base_gpu.sh
# Output: $OUT/{config.json,episodes.jsonl,summary.json,server.log,client_logs/,videos/}
set -uo pipefail
W="${W:-/mnt/nas/xinyizheng/affordance_vla}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
STARVLA="${STARVLA:-$W/repos/starVLA_rc365}"
CKPT="${CKPT:?CKPT is required}"
RUN="${RUN:-eval_$(basename "$(dirname "$(dirname "$CKPT")")")}"
OUT="$W/runs/robocasa365_eval/$RUN"
mkdir -p "$OUT"
source "$REPO/envs/server/robocasa365_env.sh"      # RC365_PYTHON, MUJOCO_GL=osmesa, NUMBA_OPT=0
export PYTHONPATH="$STARVLA:$REPO" HF_HUB_OFFLINE=1

if [ "${BENCH:-0}" = 1 ]; then
  port=$((5600 + RANDOM % 300))
  PYTHONPATH="$STARVLA" /data/xinyizheng/envs/starvla/bin/python "$STARVLA/deployment/model_server/server_policy.py" \
    --ckpt_path "$CKPT" --port "$port" --use_bf16 > "$OUT/bench_server.log" 2>&1 &
  server=$!
  for _ in $(seq 1 240); do (echo > "/dev/tcp/127.0.0.1/$port") 2> /dev/null && break; sleep 5; done
  "$RC365_PYTHON" "$REPO/scripts/rc365/bench_policy_server.py" --port "$port" > "$OUT/bench.log" 2>&1
  kill "$server" 2> /dev/null; wait "$server" 2> /dev/null
fi
# shellcheck disable=SC2086
"$RC365_PYTHON" "$REPO/scripts/rc365/eval_policy.py" --ckpt "$CKPT" --run "$RUN" ${CONFIG:+--config "$REPO/$CONFIG"} ${OVERRIDES:-} >> "$OUT/eval.log" 2>&1
