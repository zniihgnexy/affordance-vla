#!/usr/bin/env bash
# Simulator environment for LIBERO-PRO, a fork of LIBERO with perturbed evaluation suites (arXiv 2510.03827).
# Source this instead of libero_env.sh: the fork is imported in place of LIBERO and reads its own task
# and start-state files. Suites: libero_{spatial,object,goal,10}_{swap,task,object,lan}, i.e. position,
# task, object appearance and language perturbations. Set up once with setup_libero_pro.sh.
W="${W:-/mnt/nas/xinyizheng/affordance_vla}"
export LIBERO="${LIBERO_PRO:-$W/repos/LIBERO-PRO}"
# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/libero_env.sh"
export LIBERO_CONFIG_PATH="$W/cache/libero_pro_config"
