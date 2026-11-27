#!/usr/bin/env bash
# CPU only: run check_train_eval_alignment.py on the human pretrain demonstrations (the AFFO-16 config by default).
set -euo pipefail
W="${W:-/mnt/nas/xinyizheng/affordance_vla}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
STARVLA="${STARVLA:-$W/repos/starVLA_rc365}"
export PYTHONPATH="$STARVLA:$REPO" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$STARVLA"
/data/xinyizheng/envs/starvla/bin/python "$REPO/scripts/rc365/check_train_eval_alignment.py" \
  --config "${CONFIG:-$REPO/configs/rc365/qwenoft_lora_rc365_8tasks.yaml}" --data-root "$W/data/robocasa365" "$@"
