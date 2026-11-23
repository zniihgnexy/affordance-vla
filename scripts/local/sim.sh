#!/usr/bin/env bash
# Start the local MuJoCo simulator (scripts/local/sim.py) in the simulator env, under mjpython on macOS.
#   bash scripts/local/sim.sh scene  --suite libero_object_swap --task 2 --episode 0
#   bash scripts/local/sim.sh map    --suite libero_10 --task 4
#   bash scripts/local/sim.sh grasp  --suite libero_object_swap --task 2 --object ketchup_1
#   bash scripts/local/sim.sh replay runs/libero_pro/predicted_g1_hold_s10/libero_object_swap/task2_ep0.npz
#   bash scripts/local/sim.sh list   runs/libero_pro/predicted_g1_hold_s10
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck disable=SC1091
source "$REPO/envs/local/sim_env.sh"
cd "$REPO"
exec "$SIM_MJPYTHON" "$REPO/scripts/local/sim.py" "$@"
