#!/usr/bin/env bash
# Create the starVLA training environment on the A6000 server (estar).
#
# - conda prefix env on /data (the /home disk is nearly full), package caches on /data as well
# - Python 3.10 and the pins from starVLA's requirements.txt (torch 2.6.0 / CUDA 12.4 wheels)
# - no GPU is used by this script
#
# Network note (measured 2026-11-10): on this server HTTP/1.1 downloads are throttled to ~0.8 MB/s while
# HTTP/2 reaches ~20 MB/s. pip only speaks HTTP/1.1, so the large wheels (torch and its CUDA libraries,
# ~3.5 GB) are fetched with curl first and installed from a local wheelhouse.
#
# Usage on the server:  bash setup_starvla_env.sh            (safe to re-run; it resumes)
set -euo pipefail

W="${W:-/data/xinyizheng/affordance3d_vla}"
ENV_PREFIX="${ENV_PREFIX:-/data/xinyizheng/envs/starvla}"
STARVLA="${STARVLA:-$W/repos/starVLA}"
PIP_INDEX="${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple/}"  # pypi.org is slow from this server
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.21.0}"
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl}"
WHEELHOUSE="$W/cache/wheelhouse"

export CONDA_PKGS_DIRS="$W/cache/conda_pkgs" PIP_CACHE_DIR="$W/cache/pip"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$WHEELHOUSE"

# shellcheck disable=SC1090
source ~/anaconda3/etc/profile.d/conda.sh
[ -x "$ENV_PREFIX/bin/python" ] || conda create -y -p "$ENV_PREFIX" -c conda-forge --override-channels python=3.10
conda activate "$ENV_PREFIX"

pip_install() { python -m pip install --index-url "$PIP_INDEX" "$@"; }

# prefetch <project> <version>: curl the cp310 / py3 Linux x86_64 wheel into the wheelhouse and check its sha256
prefetch() {
  local project="$1" version="$2" href file url sha
  href="$(curl -fsS -m 60 "${PIP_INDEX%/}/${project}/" | grep -oE 'href="[^"]+"' | sed 's/^href="//; s/"$//' \
    | grep -E "/${project//-/_}-${version}-(cp310-cp310|py3-none)-[^/]*x86_64[^/]*\.whl" | head -1)"
  [ -n "$href" ] || { echo "no wheel found for $project==$version" >&2; return 1; }
  file="$(basename "${href%%#*}")"
  case "$href" in http*) url="${href%%#*}" ;; *) url="${PIP_INDEX%/}/${project}/${href%%#*}" ;; esac
  sha="${href##*#sha256=}"
  if [ ! -f "$WHEELHOUSE/$file" ] || ! echo "$sha  $WHEELHOUSE/$file" | sha256sum -c --status; then
    curl -fsSL --retry 5 -C - -o "$WHEELHOUSE/$file" "$url"
    echo "$sha  $WHEELHOUSE/$file" | sha256sum -c --status || { echo "sha256 mismatch: $file" >&2; return 1; }
  fi
  echo "$WHEELHOUSE/$file"
}

pip_install --upgrade pip
if ! python -c "import torch, torchvision" 2>/dev/null; then
  torch_wheel="$(prefetch torch "$TORCH_VERSION")"
  heavy=("$torch_wheel" "$(prefetch torchvision "$TORCHVISION_VERSION")")
  # torch pins its CUDA libraries exactly (Requires-Dist: nvidia-...-cu12 (==x.y); platform ...)
  while read -r project version; do
    heavy+=("$(prefetch "$project" "$version")")
  done < <(unzip -p "$torch_wheel" '*.dist-info/METADATA' \
            | sed -nE 's/^Requires-Dist: (nvidia-[a-z0-9-]+|triton) ?\(?==([0-9.]+)\)?.*/\1 \2/p')
  python -m pip install --no-deps "${heavy[@]}"
  pip_install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"   # only the small dependencies remain
fi
pip_install -r "$STARVLA/requirements.txt"
pip_install -e "$STARVLA" --no-deps

# flash-attn: prebuilt wheel (there is no nvcc on this server, so it cannot be compiled here).
# starVLA runs without it when the attention implementation is "sdpa", so a failure here is not fatal.
if ! python -c "import flash_attn" 2>/dev/null; then
  flash_file="$WHEELHOUSE/$(basename "$FLASH_ATTN_WHEEL")"
  { curl -fsSL --retry 5 -C - -o "$flash_file" "$FLASH_ATTN_WHEEL" && python -m pip install --no-deps "$flash_file"; } \
    || echo "WARNING: flash-attn not installed; use attn_implementation=sdpa"
fi

python - <<'EOF'
import torch, transformers, deepspeed, accelerate
print("torch", torch.__version__, "| cuda build", torch.version.cuda, "| transformers", transformers.__version__,
      "| deepspeed", deepspeed.__version__, "| accelerate", accelerate.__version__)
EOF
python -m pip freeze > "$W/logs/starvla_env_freeze.txt"
echo "ENV-READY: $ENV_PREFIX (freeze: $W/logs/starvla_env_freeze.txt)"
