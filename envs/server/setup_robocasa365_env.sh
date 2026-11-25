#!/usr/bin/env bash
# Create the RoboCasa365 simulator environment on the A6000 server (estar): MuJoCo + robosuite + RoboCasa.
# CPU and network only (no GPU, no queue). Safe to run again: finished steps are skipped.
#
# - code is fetched as archives of pinned commits from codeload.github.com (git clone hangs on this server)
# - the kitchen assets (~11 GB, six zips from the official utexas Box links in
#   robocasa/models/assets/box_links/box_links_assets.json) are fetched with curl, resumable, and unpacked
#   into the checkout, which lives on the NAS; the official download script asks for confirmation and
#   cannot resume
# - rendering is software (OSMesa from conda-forge), as in the LIBERO env; the starVLA example's own
#   recipe uses EGL, i.e. a GPU context, which this project must not open outside the queue
#
# Usage on the server:  bash envs/server/setup_robocasa365_env.sh
# To use the env:       source envs/server/robocasa365_env.sh
set -euo pipefail

W="${W:-/mnt/nas/xinyizheng/affordance_vla}"
ENV_PREFIX="${ENV_PREFIX:-/data/xinyizheng/envs/robocasa365}"
ROBOCASA_SHA="${ROBOCASA_SHA:-7bfc60c9d26c02506d4265cbe0e86b35e6e25d42}"   # robocasa/robocasa main, 2026-11-18 (v1.0.1)
ROBOSUITE_SHA="${ROBOSUITE_SHA:-5ce6643f3092639d08f7b0f90ed1c6a84f50552c}" # ARISE-Initiative/robosuite master
ROBOCASA="$W/repos/robocasa365"
ROBOSUITE="$W/repos/robosuite"
PIP_INDEX="${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple/}"          # pypi.org is slow from this server
TORCH_CPU_WHEEL="${TORCH_CPU_WHEEL:-https://mirrors.aliyun.com/pytorch-wheels/cpu/torch-2.6.0%2Bcpu-cp311-cp311-linux_x86_64.whl}"
WHEELHOUSE="$W/cache/wheelhouse"
ASSET_ZIPS="$W/cache/robocasa_assets"
BOX="https://utexas.box.com/shared/static"

export CONDA_PKGS_DIRS="$W/cache/conda_pkgs" PIP_CACHE_DIR="$W/cache/pip"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$WHEELHOUSE" "$ASSET_ZIPS" "$W/repos/_dl" "$W/logs"

fetch_repo() {   # fetch_repo <owner/name> <sha> <destination>
  [ ! -d "$3" ] || return 0
  local zip="$W/repos/_dl/$(basename "$3")-$2.zip" tmp="$W/repos/_dl/unpack_$(basename "$3")"
  curl -fL --http2 --retry 5 --retry-delay 5 --connect-timeout 20 --max-time 1800 -C - -o "$zip" \
    "https://codeload.github.com/$1/zip/$2"
  rm -rf "$tmp"
  python3 -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$zip" "$tmp"
  mv "$tmp"/* "$3"
  rmdir "$tmp"
}
fetch_repo ARISE-Initiative/robosuite "$ROBOSUITE_SHA" "$ROBOSUITE"
fetch_repo robocasa/robocasa "$ROBOCASA_SHA" "$ROBOCASA"

# shellcheck disable=SC1090
source ~/anaconda3/etc/profile.d/conda.sh
# mesalib < 25 still ships libOSMesa (removed from Mesa 25.1)
[ -x "$ENV_PREFIX/bin/python" ] || conda create -y -p "$ENV_PREFIX" --solver libmamba -c conda-forge \
  --override-channels python=3.11 "mesalib<25"
conda activate "$ENV_PREFIX"
ls "$ENV_PREFIX"/lib/libOSMesa.so* > /dev/null || { echo "libOSMesa is missing from $ENV_PREFIX/lib" >&2; exit 1; }

pip_install() { python -m pip install --index-url "$PIP_INDEX" "$@"; }
pip_install --upgrade pip
if ! python -c "import torch" 2> /dev/null; then   # lerobot (pinned by robocasa) needs torch; CPU build, curl because pip is throttled
  wheel="$WHEELHOUSE/torch-2.6.0+cpu-cp311-cp311-linux_x86_64.whl"
  curl -fsSL --retry 5 -C - -o "$wheel" "$TORCH_CPU_WHEEL"
  pip_install "$wheel"
fi
python -c "import robosuite, robocasa" 2> /dev/null || pip_install -e "$ROBOSUITE" -e "$ROBOCASA"
# the starVLA RoboCasa365 evaluation client (examples/simBenchmarks/Robocasa_365/eval_files) also imports these
python -c "import tyro, websockets, msgpack_numpy" 2> /dev/null || pip_install tyro websockets msgpack msgpack-numpy

# Assets: "<Box id> <name> <folder it creates, relative to robocasa/models/assets> <unpack dir>". Each zip holds
# its own top-level folder; the zips fixtures and fixtures_lightwheel share the top-level folder fixtures, so
# the folder column names one that only its own zip creates (the repository itself already ships a
# partial fixtures/, so a check on fixtures/ alone would skip both).
ASSETS="$ROBOCASA/robocasa/models/assets"
while read -r id name folder dest; do
  [ ! -d "$ASSETS/$folder" ] || continue
  curl -fL --http2 --retry 8 --retry-delay 10 --connect-timeout 20 -C - -o "$ASSET_ZIPS/$name.zip" "$BOX/$id.zip"
  python -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
    "$ASSET_ZIPS/$name.zip" "$ASSETS/$dest"
  [ -d "$ASSETS/$folder" ] || { echo "$name.zip did not create $folder" >&2; exit 1; }
  echo "unpacked $name"
done <<LIST
4i85ileasdvstmlln5sbvzptz7keuoy1 textures textures .
ebaad09k82tmfmlq6ohdkmrh8izl9vn5 generative_textures generative_textures .
zt9vbo38yb9f1alw9iuahck55hoa65y6 fixtures fixtures/accessories/stools .
idbncsadpnaz1jfl4i6m8qejawk7p9pi fixtures_lightwheel fixtures/windows .
03eionyo8fk3a9dsksq9jb8du5lqfw8h objaverse objects/objaverse objects
nwi1vrn5pgbo95kushkasa3nx1i012ff aigen_objs objects/aigen_objs objects
vckqvvkh1z8t69k8qcpcmee6k66stii4 objects_lightwheel objects/lightwheel objects
LIST

# private macros: the dataset root goes to the NAS
cp -n "$ROBOCASA/robocasa/macros.py" "$ROBOCASA/robocasa/macros_private.py"
grep -q '^DATASET_BASE_PATH = "'"$W"'/data/robocasa365"' "$ROBOCASA/robocasa/macros_private.py" || \
  sed -i "s|^DATASET_BASE_PATH = .*|DATASET_BASE_PATH = \"$W/data/robocasa365\"|" "$ROBOCASA/robocasa/macros_private.py"
mkdir -p "$W/data/robocasa365"
python -m pip freeze > "$W/logs/robocasa365_env_freeze.txt"
echo "ROBOCASA365-ENV-READY: $ENV_PREFIX (freeze: $W/logs/robocasa365_env_freeze.txt)"
