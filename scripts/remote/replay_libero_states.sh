#!/usr/bin/env bash
# Log every demonstration of LIBERO tasks from its recorded simulator states (CPU only, no GPU, no queue).
#
# For each task file <task>_demo.hdf5 in the suite's raw-data folder this runs rollout_libero_logged.py
# --replay-states in PARTS parallel processes and writes one log per demonstration:
#   $OUT/<task>_demo<k>.npz   (images, gripper and object positions, contact flags, actions)
# Software rendering takes about 0.3 s per step, so 50 demonstrations of ~130 steps need about 35 CPU
# minutes; the parts run side by side, at most MAX_JOBS simulators at a time. Existing logs are kept: a
# part is skipped if all its files exist (logs are written under a temporary name and renamed).
#
# Usage on the server:  bash replay_libero_states.sh [task-slug ...]
#   task slugs are file names without _demo.hdf5; default: every complete file in the folder
set -uo pipefail

W="${W:-/mnt/nas/xinyizheng/affordance_vla}"
REPO="${REPO:-$HOME/affordance-vla}"
SUITE="${SUITE:-libero_object}"
RAW="${RAW:-$W/data/libero/$SUITE}"
OUT="${OUT:-$W/runs/plan_following/demos}"
PARTS="${PARTS:-5}"
MAX_JOBS="${MAX_JOBS:-48}"
DEMOS_PER_TASK="${DEMOS_PER_TASK:-50}"

# shellcheck disable=SC1091
source "$REPO/scripts/remote/libero_env.sh"
mkdir -p "$OUT/logs"
if [ $# -gt 0 ]; then tasks=("$@"); else
  tasks=()
  for file in "$RAW"/*_demo.hdf5; do tasks+=("$(basename "$file" _demo.hdf5)"); done
fi

for task in "${tasks[@]}"; do  # check everything before the first simulator starts
  [[ "$task" =~ ^[A-Za-z0-9_]+$ ]] || { echo "no usable task file in $RAW (got: $task)" >&2; exit 2; }
  [ -f "$RAW/${task}_demo.hdf5" ] || { echo "missing: $RAW/${task}_demo.hdf5" >&2; exit 1; }
done

pids=(); labels=()
trap 'kill ${pids[@]+"${pids[@]}"} 2>/dev/null' EXIT INT TERM  # an interrupted run leaves no simulator behind
for task in "${tasks[@]}"; do
  file="$RAW/${task}_demo.hdf5"
  for part in $(seq 0 $((PARTS - 1))); do
    demos=(); todo=0
    for demo in $(seq "$part" "$PARTS" $((DEMOS_PER_TASK - 1))); do
      demos+=("$demo")
      [ -f "$OUT/${task}_demo${demo}.npz" ] || todo=1
    done
    [ "$todo" -eq 1 ] || continue
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do sleep 2; done
    CUDA_VISIBLE_DEVICES='' "$LIBERO_PYTHON" "$REPO/scripts/remote/rollout_libero_logged.py" --out "$OUT" \
      --suite "$SUITE" --task "${task//_/ }" --tag "$task" --replay-states "$file" --demos "${demos[@]}" \
      --summary-suffix "_part$part" > "$OUT/logs/${task}_part$part.log" 2>&1 &
    pids+=("$!"); labels+=("${task}_part$part")
  done
done
failed=0
for k in "${!pids[@]}"; do
  wait "${pids[$k]}" || { echo "REPLAY-FAILED: ${labels[$k]} (see $OUT/logs/${labels[$k]}.log)"; failed=1; }
done
pids=()  # everything has been waited for
echo "REPLAY-FINISHED failed=$failed logs=$(find "$OUT" -maxdepth 1 -name '*_demo*.npz' -not -name '.*' | wc -l)"
exit "$failed"
