#!/usr/bin/env bash
# One LIBERO-Object task file (50 demos, ~664 MB) and its BDDL scene description, used by
# report/2026-11-10_vla_datasets_sim_report.ipynb. Data goes to the main checkout's data/ folder
# (shared by all worktrees) and is not versioned. Resumable: re-run to continue a partial download.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
MAIN="$(cd "$(git -C "$REPO" rev-parse --git-common-dir)/.." && pwd)"
OUT="$MAIN/data/libero/libero_object"
TASK=pick_up_the_salad_dressing_and_place_it_in_the_basket
mkdir -p "$OUT"

curl -fL --retry 5 -C - -o "$OUT/${TASK}_demo.hdf5" \
  "https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/main/libero_object/${TASK}_demo.hdf5"
curl -fsL --retry 3 -o "$OUT/${TASK}.bddl" \
  "https://raw.githubusercontent.com/Lifelong-Robot-Learning/LIBERO/master/libero/libero/bddl_files/libero_object/${TASK}.bddl"
echo "done: $OUT"
