#!/usr/bin/env bash
# Fetch what the local simulator needs into the git-ignored third_party/ and runs/ (about 750 MB in all):
#   1. LIBERO-PRO at the commit the server uses, from GitHub (its archive holds a file name that is not
#      UTF-8, so it is unpacked with Python's zipfile, as on the server);
#   2. LIBERO-PRO's perturbed task files (libero_*_{swap,task,object,lan}), which come from its Hugging Face
#      dataset, from the server's copy;
#   3. LIBERO's config.yaml, pointing at the checkout;
#   4. image-free copies of evaluation logs (scripts/data/export_sim_logs.py, run on the server), the
#      contact offsets of the plan datasets, and the 3D interaction maps of L (runs/interaction_labels).
# Usage, from the repository root:
#   bash scripts/local/fetch_sim_data.sh                # RUNS="..." picks other evaluation runs
#   VERIFY=1 bash scripts/local/fetch_sim_data.sh       # also compare every LIBERO-PRO file with the server's
# The server is reached as $SIM_SERVER (default estar, an ssh alias); large downloads come from GitHub because
# the server's link to this machine is slow (~150 kB/s measured).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SERVER="${SIM_SERVER:-estar}"
W_REMOTE="${W_REMOTE:-/mnt/nas/xinyizheng/affordance_vla}"
REPO_REMOTE="${REPO_REMOTE:-affordance-vla}"
COMMIT="eafdb809426b13153aa1e4c42d6601844217dfec"  # LIBERO-PRO master on 2026-03-23, unchanged since
RUNS="${RUNS:-released predicted_g1_hold_s10 oracle_plan_s10 spatial_released_seed7 libero10_released libero10_released_task23}"
DATASETS="dataset10 dataset_spatial"
LABELS="object libero10"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=25 -o ServerAliveInterval=15 "$SERVER")
THIRD="$REPO/third_party"
CHECKOUT="$THIRD/LIBERO-PRO"
mkdir -p "$THIRD/_downloads" "$THIRD/libero_config" "$THIRD/libero_datasets" "$REPO/runs"

if [ ! -d "$CHECKOUT/libero" ]; then
  archive="$THIRD/_downloads/LIBERO-PRO-$COMMIT.zip"
  curl -fL --retry 5 -C - -o "$archive" "https://codeload.github.com/Zxy-MLlab/LIBERO-PRO/zip/$COMMIT"
  rm -rf "$THIRD/_downloads/unpacked"
  python3 -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
    "$archive" "$THIRD/_downloads/unpacked"
  mv "$THIRD/_downloads/unpacked/LIBERO-PRO-$COMMIT" "$CHECKOUT"
  rm -rf "$THIRD/_downloads/unpacked" "$archive"
fi

"${SSH[@]}" "cd $W_REMOTE/data/libero_pro && tar -cf - bddl_files init_files" | tar -xf - -C "$THIRD/_downloads"
for kind in bddl_files init_files; do
  for folder in "$THIRD/_downloads/$kind"/libero_*; do
    cp -Rn "$folder" "$CHECKOUT/libero/libero/$kind/"
  done
  rm -rf "${THIRD:?}/_downloads/$kind"
done

root="$CHECKOUT/libero/libero"
cat > "$THIRD/libero_config/config.yaml" <<YAML
assets: $root/assets
bddl_files: $root/bddl_files
benchmark_root: $root
datasets: $THIRD/libero_datasets
init_states: $root/init_files
YAML

# shellcheck disable=SC2086
"${SSH[@]}" "cd $REPO_REMOTE && /data/xinyizheng/envs/starvla/bin/python scripts/data/export_sim_logs.py \
  --root $W_REMOTE/runs/libero_pro --out $W_REMOTE/cache/sim_logs/libero_pro --runs $RUNS"
# shellcheck disable=SC2086
"${SSH[@]}" "cd $W_REMOTE/cache/sim_logs && tar -cf - $(printf 'libero_pro/%s ' $RUNS)" | tar -xf - -C "$REPO/runs"
for dataset in $DATASETS; do
  mkdir -p "$REPO/runs/plan_following/$dataset"
  "${SSH[@]}" "cd $W_REMOTE/runs/plan_following/$dataset && tar -cf - offsets.json \$(ls phrases.json 2> /dev/null)" \
    | tar -xf - -C "$REPO/runs/plan_following/$dataset"
done

for labels in $LABELS; do
  mkdir -p "$REPO/runs/interaction_labels/$labels"
  "${SSH[@]}" "cd $W_REMOTE/runs/interaction_labels/$labels && tar -cf - maps.npz summary.json" \
    | tar -xf - -C "$REPO/runs/interaction_labels/$labels"
done

if [ "${VERIFY:-0}" = 1 ]; then
  files='find . -type f ! -name "*.pyc" ! -path "*/__pycache__/*" ! -path "*.egg-info*" ! -name .DS_Store -print0'
  normalise() { sed -E 's/^([0-9a-f]{32}) +\*?/\1 /' | LC_ALL=C sort; }  # "hash path", one order on both sides
  "${SSH[@]}" "cd $W_REMOTE/repos/LIBERO-PRO && $files | xargs -0 md5sum" | normalise > "$THIRD/_downloads/server.md5"
  (cd "$CHECKOUT" && eval "$files" | xargs -0 md5 -r) | normalise > "$THIRD/_downloads/local.md5"
  if cmp -s "$THIRD/_downloads/server.md5" "$THIRD/_downloads/local.md5"; then
    echo "LIBERO-PRO: all $(wc -l < "$THIRD/_downloads/local.md5" | tr -d ' ') files match the server's"
  else
    echo "LIBERO-PRO differs from the server's copy" >&2; exit 1
  fi
fi
echo "simulator data ready: $CHECKOUT, $(find "$REPO/runs/libero_pro" -name '*.npz' | wc -l | tr -d ' ') logs in runs/libero_pro"
