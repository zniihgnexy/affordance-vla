#!/usr/bin/env bash
# CPU only: build starVLA's per-dataset caches (step index, statistics) of a training mixture and report its size, so
# that no GPU time is spent on it. Run after the demonstrations are on disk, before the training job.
#   MIX [affo16_rc365_pretrain_8tasks]  CONFIG [configs/rc365/qwenoft_lora_rc365_8tasks.yaml]
#   bash scripts/rc365/prepare_train_data.sh      # log: $W/logs/affo16_prepare_<MIX>.log is the caller's to redirect
set -euo pipefail
W="${W:-/mnt/nas/xinyizheng/affordance_vla}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
STARVLA="${STARVLA:-$W/repos/starVLA_rc365}"
MIX="${MIX:-affo16_rc365_pretrain_8tasks}"
CONFIG="${CONFIG:-$REPO/configs/rc365/qwenoft_lora_rc365_8tasks.yaml}"
overlay="$STARVLA/examples/simBenchmarks/AFFO16_rc365/train_files/data_registry"
mkdir -p "$overlay" && cp "$REPO/configs/rc365/data_registry_affo16/data_config.py" "$overlay/data_config.py"
export PYTHONPATH="$STARVLA:$REPO" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$STARVLA"
/data/xinyizheng/envs/starvla/bin/python "$REPO/scripts/rc365/prepare_train_data.py" --config "$CONFIG" \
  --data-root "$W/data/robocasa365" --mix "$MIX"
