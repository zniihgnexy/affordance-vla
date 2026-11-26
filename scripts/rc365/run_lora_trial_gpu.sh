#!/usr/bin/env bash
# AFFO-16 LoRA trial: a few hundred steps of run_train_base_gpu.sh at several batch sizes, one after the other on the
# queue's ONE card, to measure peak memory and step time. A batch size that does not fit does not stop the next one.
#   BATCHES ["8 16"]  STEPS [200]  PREFIX [affo16_lora_trial]  MIX [affo16_rc365_pretrain_coffee_mg]
# Submit (never run directly), from the repo root:
#   bash scripts/server/submit_queue_job.sh 164 rc365_lora_trial 40000 1 "cards=0 1 2 3 5 6 7" -- \
#       env PREFIX=<name> bash $PWD/scripts/rc365/run_lora_trial_gpu.sh
# Output per batch size: $W/runs/robocasa365_starvla/${PREFIX}_b<B>/{train.log,PEAK_MEMORY.txt}
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
for batch in ${BATCHES:-8 16}; do
  env RUN="${PREFIX:-affo16_lora_trial}_b$batch" BATCH="$batch" STEPS="${STEPS:-200}" SAVE="${STEPS:-200}" RESUME=0 \
      MIX="${MIX:-affo16_rc365_pretrain_coffee_mg}" bash "$HERE/run_train_base_gpu.sh"
done
