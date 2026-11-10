#!/usr/bin/env bash
# Create the LIBERO simulator environment on the A6000 server (estar): MuJoCo + robosuite + LIBERO.
#
# - separate from the starVLA env: LIBERO needs older pins (numpy 1.24, gym 0.25, torch < 2.6 because its
#   initial states are pickles that newer torch.load refuses), and starVLA's evaluation runs the simulator
#   and the policy as two processes anyway
# - rendering is done in software (OSMesa from conda-forge), so the simulator never touches a GPU
# - only what LIBERO's benchmark and environment code imports is installed, not its training stack
# - LIBERO asks for its paths interactively on first import; the config is written here instead
#
# Usage on the server:  bash setup_libero_env.sh            (safe to re-run; it resumes)
# To use the env:       source scripts/remote/libero_env.sh  (sets the rendering and config variables)
set -euo pipefail

W="${W:-/mnt/nas/xinyizheng/affordance_vla}"
ENV_PREFIX="${ENV_PREFIX:-/data/xinyizheng/envs/libero}"
LIBERO="${LIBERO:-$W/repos/LIBERO}"
PIP_INDEX="${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple/}"  # pypi.org is slow from this server
TORCH_CPU_WHEEL="${TORCH_CPU_WHEEL:-https://mirrors.aliyun.com/pytorch-wheels/cpu/torch-2.2.2%2Bcpu-cp310-cp310-linux_x86_64.whl}"
WHEELHOUSE="$W/cache/wheelhouse"
LIBERO_CONFIG_DIR="$W/cache/libero_config"

export CONDA_PKGS_DIRS="$W/cache/conda_pkgs" PIP_CACHE_DIR="$W/cache/pip"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$WHEELHOUSE" "$LIBERO_CONFIG_DIR" "$W/logs"

# shellcheck disable=SC1090
source ~/anaconda3/etc/profile.d/conda.sh
# mesalib < 25 still ships libOSMesa (removed from Mesa 25.1). The libmamba solver is used because the
# classic one needs many minutes on the full conda-forge index.
[ -x "$ENV_PREFIX/bin/python" ] || conda create -y -p "$ENV_PREFIX" --solver libmamba -c conda-forge \
  --override-channels python=3.10 "mesalib<25"
conda activate "$ENV_PREFIX"
ls "$ENV_PREFIX"/lib/libOSMesa.so* > /dev/null || { echo "libOSMesa is missing from $ENV_PREFIX/lib" >&2; exit 1; }

pip_install() { python -m pip install --index-url "$PIP_INDEX" "$@"; }
pip_install --upgrade pip
if ! python -c "import torch" 2> /dev/null; then
  wheel="$WHEELHOUSE/torch-2.2.2+cpu-cp310-cp310-linux_x86_64.whl"   # CPU build; curl because pip is throttled
  curl -fsSL --retry 5 -C - -o "$wheel" "$TORCH_CPU_WHEEL"
  pip_install "$wheel"
fi
pip_install -r "$(dirname "$0")/requirements_libero_env.txt"
# LIBERO itself is used from its checkout through PYTHONPATH (set in libero_env.sh), as starVLA's own
# eval script does: its top-level folder has no __init__.py, so an editable install cannot be imported.

python - "$LIBERO" "$LIBERO_CONFIG_DIR" "$W/data/libero" << 'EOF'
import sys
from pathlib import Path

import yaml

libero, config_dir, datasets = (Path(arg) for arg in sys.argv[1:4])
root = libero / "libero" / "libero"
config = {"benchmark_root": str(root), "bddl_files": str(root / "bddl_files"), "init_states": str(root / "init_files"),
          "datasets": str(datasets), "assets": str(root / "assets")}
(config_dir / "config.yaml").write_text(yaml.safe_dump(config))
print("wrote", config_dir / "config.yaml")
EOF

# shellcheck disable=SC1091
source "$(dirname "$0")/libero_env.sh"
python - "$W/logs/libero_env_first_frame.png" << 'EOF'
import sys
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np
import robosuite
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

suite = benchmark.get_benchmark_dict()["libero_object"]()
task = suite.get_task(0)
env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
                         camera_heights=256, camera_widths=256)
env.seed(7)
env.reset()
obs = env.set_init_state(suite.get_task_init_states(0)[0])
frame = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
iio.imwrite(sys.argv[1], frame)
print(f"mujoco {mujoco.__version__} | robosuite {robosuite.__version__} | task: {task.language} | "
      f"frame {frame.shape}, mean {frame.mean():.1f} -> {sys.argv[1]}")
env.close()
EOF
python -m pip freeze > "$W/logs/libero_env_freeze.txt"
echo "LIBERO-ENV-READY: $ENV_PREFIX (freeze: $W/logs/libero_env_freeze.txt)"
