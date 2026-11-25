#!/usr/bin/env bash
# Step 0 (1) of Pipeline v4: leave-one-mug-mesh-out transfer check, nearest point vs NDF
# (scripts/data/mug_transfer_check.py; settings in configs/mug_transfer/). Usage, from the repository root:
#
#   bash scripts/data/run_mug_transfer_check.sh collect    # CPU: held grasps, surfaces, up directions (libero env)
#   bash scripts/server/submit_queue_job.sh 164 mug_ndf 8000 1 cards=0 -- \
#       bash $PWD/scripts/data/run_mug_transfer_check.sh ndf     # GPU, only through the queue (starvla env)
#   bash scripts/data/run_mug_transfer_check.sh evaluate   # CPU: table and figures (libero env)
#
# Prerequisites: the replayed demonstrations listed in the config, the NDF repository and weights under $W
# (repos/ndf_robot, models/ndf/multi_category_weights.pth; see docs/vla/2026-11-25_VLA_branch_第0步_mug搬运检查.md).
# CONFIG [configs/mug_transfer/mug_mesh.yaml]; further arguments (e.g. --set ndf.iterations=100) go to the script.
set -uo pipefail

W="${W:-/mnt/nas/xinyizheng/affordance_vla}"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONFIG="${CONFIG:-configs/mug_transfer/mug_mesh.yaml}"
STARVLA_PYTHON="${STARVLA_PYTHON:-/data/xinyizheng/envs/starvla/bin/python}"
stage="${1:-}"; shift || true
cd "$REPO" || exit 1
export W PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
case "$stage" in
  collect|evaluate)
    # shellcheck disable=SC1091
    source envs/server/libero_env.sh
    CUDA_VISIBLE_DEVICES='' "$LIBERO_PYTHON" scripts/data/mug_transfer_check.py --config "$CONFIG" --stage "$stage" "$@" ;;
  ndf)
    "$STARVLA_PYTHON" scripts/data/mug_transfer_check.py --config "$CONFIG" --stage ndf "$@" ;;
  *) echo "usage: $0 collect|ndf|evaluate [script options]" >&2; exit 2 ;;
esac
