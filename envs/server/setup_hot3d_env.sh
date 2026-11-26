#!/usr/bin/env bash
# Create the HOT3D demo environment on the A6000 server (estar): CPU only, no simulator.
#
# - reads HOT3D-Clips (webdataset tars from Hugging Face), the UmeTrack hand model and the object meshes
# - hand_tracking_toolkit (Meta, Apache-2.0) is pinned to a commit; MANO / smplx / chumpy are NOT installed, the
#   UmeTrack hand annotation needs no MANO licence
# - pypi.org is slow from this server, so pip uses the Aliyun mirror and the CPU torch wheel is fetched with curl
#
# Usage on the server:  bash envs/server/setup_hot3d_env.sh        (safe to re-run)
# To use the env:       source envs/server/hot3d_env.sh
set -euo pipefail

W="${W:-/mnt/nas/xinyizheng/affordance_vla}"
ENV_PREFIX="${ENV_PREFIX:-/data/xinyizheng/envs/hot3d}"
HTK_COMMIT="${HTK_COMMIT:-1aed9608616f62b6468c9c1a1042d60e9c210c70}"  # facebookresearch/hand_tracking_toolkit, 2026-11-17
PIP_INDEX="${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple/}"
TORCH_CPU_WHEEL="${TORCH_CPU_WHEEL:-https://mirrors.aliyun.com/pytorch-wheels/cpu/torch-2.2.2%2Bcpu-cp310-cp310-linux_x86_64.whl}"
WHEELHOUSE="$W/cache/wheelhouse"

export CONDA_PKGS_DIRS="$W/cache/conda_pkgs" PIP_CACHE_DIR="$W/cache/pip"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$WHEELHOUSE" "$W/logs"

# shellcheck disable=SC1090
source ~/anaconda3/etc/profile.d/conda.sh
[ -x "$ENV_PREFIX/bin/python" ] || conda create -y -p "$ENV_PREFIX" --solver libmamba -c conda-forge \
  --override-channels python=3.10
conda activate "$ENV_PREFIX"

pip_install() { python -m pip install --index-url "$PIP_INDEX" "$@"; }
pip_install --upgrade pip
if ! python -c "import torch" 2> /dev/null; then
  wheel="$WHEELHOUSE/torch-2.2.2+cpu-cp310-cp310-linux_x86_64.whl"
  [ -s "$wheel" ] || curl -fsSL --retry 5 -C - -o "$wheel" "$TORCH_CPU_WHEEL"
  pip_install "$wheel"
fi
pip_install -r "$(dirname "$0")/requirements_hot3d_env.txt"
# the toolkit's own dependencies are already in the requirements; its source comes from GitHub at a fixed commit
python -m pip install --no-deps "https://github.com/facebookresearch/hand_tracking_toolkit/archive/${HTK_COMMIT}.zip"
python -c "import hand_tracking_toolkit, trimesh, cv2, torch; print('hot3d env ok, torch', torch.__version__)"
