#!/usr/bin/env bash
# Activate the HOT3D demo environment (source this file). CPU only.
export W="${W:-/mnt/nas/xinyizheng/affordance_vla}"
# shellcheck disable=SC1090
source ~/anaconda3/etc/profile.d/conda.sh
conda activate "${ENV_PREFIX:-/data/xinyizheng/envs/hot3d}"
export HOT3D_DIR="${HOT3D_DIR:-$W/data/hot3d}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
