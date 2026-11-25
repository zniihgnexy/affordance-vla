#!/usr/bin/env bash
# Evaluate a starVLA checkpoint on RoboCasa365 target kitchens as ONE job for the shared GPU queue (A6000 server):
# a policy server on the queue's card (starvla env) and one CPU simulator client per task (robocasa365 env,
# software rendering), all clients against the same server. Episode horizons are RoboCasa v1.0.1's own per task.
# Settings from the environment (defaults in brackets):
#   CKPT      path of steps_N_pytorch_model.pt (the run folder next to it must hold config.yaml and dataset_statistics.json)
#   RUN [eval_$(basename run folder)]   outputs go to $W/runs/robocasa365_eval/$RUN
#   TASKS ["CoffeeSetupMug OpenDrawer TurnOffStove"]  EPISODES [10]  N_ACT [8]  (actions executed per query)
# Submit (never run directly), from the repo root:
#   bash scripts/server/submit_queue_job.sh 164 rc365_eval 20000 1 "cards=6 7" -- \
#       env CKPT=$W/runs/robocasa365_starvla/<run>/checkpoints/steps_300_pytorch_model.pt \
#       bash ~/wt-affo-10/scripts/rc365/run_eval_gpu.sh
# Output: $OUT/{server.log,client_<Task>.log,videos/<Task>/,summary.txt}; a client log ends with the success rate
# and "N eps done in X s" (all N episodes of that task, three tasks running at the same time).
set -uo pipefail
W="${W:-/mnt/nas/xinyizheng/affordance_vla}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
STARVLA="${STARVLA:-$W/repos/starVLA_rc365}"
CKPT="${CKPT:?CKPT is required}"
RUN="${RUN:-eval_$(basename "$(dirname "$(dirname "$CKPT")")")}"
TASKS="${TASKS:-CoffeeSetupMug OpenDrawer TurnOffStove}"
EPISODES="${EPISODES:-10}"; N_ACT="${N_ACT:-8}"
PORT="${PORT:-$((5600 + RANDOM % 300))}"
OUT="$W/runs/robocasa365_eval/$RUN"
[ ! -e "$OUT" ] || { echo "$OUT exists; choose another RUN" >&2; exit 1; }
mkdir -p "$OUT"
source "$REPO/envs/server/robocasa365_env.sh"     # RC365_PYTHON, MUJOCO_GL=osmesa, NUMBA_OPT=0
export PYTHONPATH="$STARVLA" HF_HUB_OFFLINE=1
cd "$STARVLA" || exit 1

PYTHONPATH="$STARVLA" /data/xinyizheng/envs/starvla/bin/python deployment/model_server/server_policy.py \
  --ckpt_path "$CKPT" --port "$PORT" --use_bf16 > "$OUT/server.log" 2>&1 &
SERVER=$!
for _ in $(seq 1 120); do
  (echo > "/dev/tcp/127.0.0.1/$PORT") 2> /dev/null && break
  kill -0 "$SERVER" 2> /dev/null || { echo "policy server died, see $OUT/server.log" >&2; exit 1; }
  sleep 5
done
date > "$OUT/summary.txt"; echo "ckpt=$CKPT port=$PORT episodes=$EPISODES n_act=$N_ACT git=$(git -C "$REPO" rev-parse HEAD)" >> "$OUT/summary.txt"

pids=()
for task in $TASKS; do
  horizon="$("$RC365_PYTHON" -c "from robocasa.utils.dataset_registry import ATOMIC_TASK_DATASETS as A, COMPOSITE_TASK_DATASETS as C; print({**A, **C}['$task']['horizon'])" 2> /dev/null | tail -1)"
  "$RC365_PYTHON" -m examples.simBenchmarks.Robocasa_365.eval_files.simulation_env \
    --args.pretrained-path "$CKPT" --args.env-name "robocasa/$task" --args.port "$PORT" \
    --args.n-episodes "$EPISODES" --args.n-envs 1 --args.max-episode-steps "$horizon" --args.n-action-steps "$N_ACT" \
    --args.video-out-path "$OUT/videos/$task" > "$OUT/client_$task.log" 2>&1 &
  pids+=($!)
  echo "$task horizon=$horizon client_pid=$!" >> "$OUT/summary.txt"
done
for pid in "${pids[@]}"; do wait "$pid"; done
kill "$SERVER" 2> /dev/null; wait "$SERVER" 2> /dev/null
for task in $TASKS; do grep -a "success rate\|eps done in" "$OUT/client_$task.log" | sed "s/^/$task: /" >> "$OUT/summary.txt"; done
date >> "$OUT/summary.txt"
