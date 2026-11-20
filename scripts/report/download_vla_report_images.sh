#!/usr/bin/env bash
# Official project images shown in the gallery of report/experiments/2026-11-10_vla_datasets_sim_report.ipynb.
# They belong to the benchmark/dataset projects and are fetched from their pages, not stored in git.
set -euo pipefail

OUT="$(cd "$(dirname "$0")/../.." && pwd)/report/assets/external_vla"
mkdir -p "$OUT"
fetch() { [ -s "$OUT/$1" ] || curl -fL --retry 3 -A "Mozilla/5.0" -o "$OUT/$1" "$2"; }

GH=https://raw.githubusercontent.com
fetch libero.png "$GH/Lifelong-Robot-Learning/LIBERO/master/images/fig1.png"
fetch libero_plus.jpg "$GH/sylvestf/LIBERO-plus/main/static/images/libero-plus.jpg"
fetch libero_pro.png "$GH/Zxy-MLlab/LIBERO-OOD/master/images/overall.png"
fetch libero_occ.png https://arxiv.org/html/2606.10862v1/x1.png
fetch robotwin2_randomization.png "$GH/RoboTwin-Platform/RoboTwin/main/assets/files/domain_randomization.png"
fetch robocasa.webp "$GH/robocasa/robocasa/main/docs/images/readme.webp"
fetch simplerenv.png "$GH/simpler-env/SimplerEnv/main/images/teaser.png"
fetch calvin.png "$GH/mees/calvin/main/media/teaser.png"
fetch vlabench.png "$GH/OpenMOSS/VLABench/main/docs/images/Figure1_overview.png"
fetch maniskill.jpg "$GH/haosulab/ManiSkill/main/figures/teaser.jpg"
fetch droid.jpg https://droid-dataset.github.io/droid/assets/index/droid_teaser.jpg
fetch oxe.png "$GH/google-deepmind/open_x_embodiment/main/imgs/teaser.png"
echo "done: $OUT"
