#!/usr/bin/env bash
# Create the local simulator environment (CPU only; made on macOS / Apple Silicon): a conda env with
# Python 3.10 and the pinned packages of requirements_sim.txt, the versions of the server's simulator env.
#   bash envs/local/setup_sim_env.sh          (safe to re-run)
# The env is called affordance-vla-sim unless SIM_ENV says otherwise. Then fetch the data with
# scripts/local/fetch_sim_data.sh and start the simulator with scripts/local/sim.sh.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SIM_ENV="${SIM_ENV:-affordance-vla-sim}"
CONDA="${CONDA_EXE:-conda}"
if ! "$CONDA" env list | awk '{print $1}' | grep -qx "$SIM_ENV"; then
  "$CONDA" create -y -n "$SIM_ENV" -c conda-forge --override-channels --solver libmamba python=3.10
fi
"$CONDA" run -n "$SIM_ENV" python -m pip install -r "$HERE/requirements_sim.txt"
# robosuite warns at every import until it has a private macro file
"$CONDA" run -n "$SIM_ENV" python -c "import pathlib, shutil, robosuite
folder = pathlib.Path(robosuite.__file__).parent
target = folder / 'macros_private.py'
target.exists() or shutil.copy(folder / 'macros.py', target)"
echo "simulator env $SIM_ENV ready"
