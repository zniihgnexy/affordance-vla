# Source this file to run LIBERO simulator code on the A6000 server (estar):
#   source envs/server/libero_env.sh && "$LIBERO_PYTHON" my_script.py
# It only sets variables; the env itself is built by setup_libero_env.sh.
W="${W:-/mnt/nas/xinyizheng/affordance_vla}"
LIBERO_ENV_PREFIX="${LIBERO_ENV_PREFIX:-/data/xinyizheng/envs/libero}"
export LIBERO_PYTHON="$LIBERO_ENV_PREFIX/bin/python"
export LIBERO_CONFIG_PATH="$W/cache/libero_config"   # holds config.yaml; without it LIBERO prompts on import
export PYTHONPATH="${LIBERO:-$W/repos/LIBERO}${PYTHONPATH:+:$PYTHONPATH}"   # LIBERO is imported from its checkout
# Software rendering: the simulator must not open a context on a GPU (cards are shared and queued).
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH="$LIBERO_ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"   # libOSMesa lives here
export NUMBA_CACHE_DIR="$W/cache/numba"               # robosuite's numba cache; site-packages may be read-only
