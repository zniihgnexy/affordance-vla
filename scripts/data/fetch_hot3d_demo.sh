#!/usr/bin/env bash
# Download what scripts/data/hot3d_contact_demo.py needs from HOT3D-Clips on Hugging Face (not gated, no sign-up):
# two Aria clips with a held white mug (about 190 MB) and the scanned meshes of the two mugs, the bowl and the model list.
# Files already there are kept.
#
# Usage (any machine with curl; the server's NAS is the default target):  bash scripts/data/fetch_hot3d_demo.sh
# Licence: the hand annotations are CC BY-NC-SA, the sequences and the models CC BY-SA
# (https://www.projectaria.com/datasets/hot3d/license/).
set -euo pipefail

HOT3D_DIR="${HOT3D_DIR:-/mnt/nas/xinyizheng/affordance_vla/data/hot3d}"
BASE="https://huggingface.co/datasets/bop-benchmark/hot3d/resolve/main"
CLIPS="${CLIPS:-clip-003345 clip-003346}"   # train_aria, sequence P0015_dbe00981
MODELS="${MODELS:-obj_000002 obj_000008 obj_000009}"   # bowl, mug_patterned, mug_white

mkdir -p "$HOT3D_DIR/clips" "$HOT3D_DIR/object_models"
fetch() { [ -s "$2" ] || curl -fsSL --retry 5 -C - -o "$2" "$BASE/$1"; }
for clip in $CLIPS; do fetch "train_aria/$clip.tar" "$HOT3D_DIR/clips/$clip.tar"; done
for model in $MODELS; do fetch "object_models/$model.glb" "$HOT3D_DIR/object_models/$model.glb"; done
fetch object_models/models_info.json "$HOT3D_DIR/object_models/models_info.json"
fetch clip_splits.json "$HOT3D_DIR/clip_splits.json"
du -sh "$HOT3D_DIR"
