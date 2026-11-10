#!/usr/bin/env bash
# Create the starVLA training environment on the A6000 server (estar).
#
# - conda prefix env on /data (the /home disk is nearly full), package caches on /data as well
# - Python 3.10 and the pins from starVLA's requirements.txt (torch 2.6.0 / CUDA 12.4 wheels)
# - no GPU is used by this script
#
# Usage on the server:  bash setup_starvla_env.sh            (safe to re-run; it resumes)
set -euo pipefail

W="${W:-/data/xinyizheng/affordance3d_vla}"
ENV_PREFIX="${ENV_PREFIX:-/data/xinyizheng/envs/starvla}"
STARVLA="${STARVLA:-$W/repos/starVLA}"
PIP_INDEX="${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple/}"  # pypi.org is slow from this server
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl}"

export CONDA_PKGS_DIRS="$W/cache/conda_pkgs" PIP_CACHE_DIR="$W/cache/pip"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"

# shellcheck disable=SC1090
source ~/anaconda3/etc/profile.d/conda.sh
[ -x "$ENV_PREFIX/bin/python" ] || conda create -y -p "$ENV_PREFIX" -c conda-forge --override-channels python=3.10
conda activate "$ENV_PREFIX"

pip_install() { python -m pip install --index-url "$PIP_INDEX" "$@"; }

pip_install --upgrade pip
pip_install torch==2.6.0 torchvision==0.21.0     # the PyPI Linux wheels are the CUDA 12.4 build
pip_install -r "$STARVLA/requirements.txt"
pip_install -e "$STARVLA" --no-deps

# flash-attn: prebuilt wheel (there is no nvcc on this server, so it cannot be compiled here).
# starVLA runs without it when the attention implementation is "sdpa", so a failure here is not fatal.
python -c "import flash_attn" 2>/dev/null || python -m pip install "$FLASH_ATTN_WHEEL" \
  || echo "WARNING: flash-attn not installed; use attn_implementation=sdpa"

python - <<'EOF'
import torch, transformers, deepspeed, accelerate
print("torch", torch.__version__, "| cuda build", torch.version.cuda, "| transformers", transformers.__version__,
      "| deepspeed", deepspeed.__version__, "| accelerate", accelerate.__version__)
EOF
python -m pip freeze > "$W/logs/starvla_env_freeze.txt"
echo "ENV-READY: $ENV_PREFIX (freeze: $W/logs/starvla_env_freeze.txt)"
