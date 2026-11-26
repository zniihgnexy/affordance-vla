# Source this file to run RoboCasa365 simulator code on the A6000 server (estar):
#   source envs/server/robocasa365_env.sh && "$RC365_PYTHON" my_script.py
# It only sets variables; the env itself is built by setup_robocasa365_env.sh.
W="${W:-/mnt/nas/xinyizheng/affordance_vla}"
RC365_ENV_PREFIX="${RC365_ENV_PREFIX:-/data/xinyizheng/envs/robocasa365}"
export RC365_PYTHON="$RC365_ENV_PREFIX/bin/python"
export ROBOCASA_DIR="$W/repos/robocasa365"
# Software rendering: the simulator must not open a context on a GPU (cards are shared and queued).
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH="$RC365_ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"   # libOSMesa lives here
export NUMBA_CACHE_DIR="$W/cache/numba_rc365"
# numba 0.61.2 / llvmlite 0.44 segfault in the LLVM optimiser while compiling the object-placement sampler
# (robocasa/utils/placement_samplers.py) on this server (Xeon 8352V); without optimisation it compiles and caches.
export NUMBA_OPT=0
