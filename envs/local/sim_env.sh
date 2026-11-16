# Source this to use the local simulator env: sets SIM_PYTHON, and SIM_MJPYTHON (the interpreter the
# MuJoCo viewer needs on macOS; plain python elsewhere). The env is built by envs/local/setup_sim_env.sh.
SIM_ENV="${SIM_ENV:-affordance-vla-sim}"
SIM_PREFIX="${SIM_PREFIX:-$("${CONDA_EXE:-conda}" env list | awk -v name="$SIM_ENV" '$1 == name {print $NF}')}"
if [ -z "$SIM_PREFIX" ] || [ ! -x "$SIM_PREFIX/bin/python" ]; then
  echo "no conda env $SIM_ENV; create it with: bash envs/local/setup_sim_env.sh" >&2
  return 1 2> /dev/null || exit 1
fi
export SIM_PYTHON="$SIM_PREFIX/bin/python"
if [ "$(uname)" = Darwin ]; then export SIM_MJPYTHON="$SIM_PREFIX/bin/mjpython"; else export SIM_MJPYTHON="$SIM_PYTHON"; fi
