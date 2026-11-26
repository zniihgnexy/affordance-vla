#!/usr/bin/env bash
# AFFO-16 base policy: QwenOFT + LoRA on the pretrain split of eight RoboCasa365 tasks, ONE card, as a job for the
# shared GPU queue (A6000 server). Same launch as run_train_gpu.sh, through scripts/rc365/train_lora_launcher.py.
# Settings from the environment (defaults in brackets):
#   RUN [qwenoft_lora_rc365_8tasks]  outputs go to $W/runs/robocasa365_starvla/$RUN
#   STEPS [48000]  BATCH [16]  SAVE [4000]  MIX [affo16_rc365_pretrain_8tasks]  (mixtures: configs/rc365/data_registry_affo16)
#   CONFIG [configs/rc365/qwenoft_lora_rc365_8tasks.yaml]   EXTRA  more starVLA key=value overrides, e.g. "--framework.lora.rank 32"
#   RESUME [1]  1: an existing $RUN folder is continued from its latest checkpoint (a queue restart); 0: refuse to reuse it
# Submit (never run directly; the queue sets CUDA_VISIBLE_DEVICES), from the repo root:
#   bash scripts/server/submit_queue_job.sh 164 rc365_base 40000 1 "cards=0" -- \
#       env RUN=<run> bash $PWD/scripts/rc365/run_train_base_gpu.sh
# Output: $OUT/{config.yaml,dataset_statistics.json,checkpoints/,train.log,PEAK_MEMORY.txt,versions.txt};
# train.log has "timing/model" and "timing/data" every 10 steps.
set -uo pipefail
W="${W:-/mnt/nas/xinyizheng/affordance_vla}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
STARVLA="${STARVLA:-$W/repos/starVLA_rc365}"
PY_ENV="${PY_ENV:-/data/xinyizheng/envs/starvla}"
RUN="${RUN:-qwenoft_lora_rc365_8tasks}"
STEPS="${STEPS:-48000}"; BATCH="${BATCH:-16}"; SAVE="${SAVE:-4000}"
MIX="${MIX:-affo16_rc365_pretrain_8tasks}"
CONFIG="${CONFIG:-$REPO/configs/rc365/qwenoft_lora_rc365_8tasks.yaml}"
RESUME="${RESUME:-1}"
OUT="$W/runs/robocasa365_starvla/$RUN"
if [ -e "$OUT" ] && [ "$RESUME" != 1 ]; then echo "$OUT exists; choose another RUN or RESUME=1" >&2; exit 1; fi
mkdir -p "$OUT"

# registry overlay: the pretrain mixtures (a new benchmark folder; no starVLA file is changed)
overlay="$STARVLA/examples/simBenchmarks/AFFO16_rc365/train_files/data_registry"
mkdir -p "$overlay" && cp "$REPO/configs/rc365/data_registry_affo16/data_config.py" "$overlay/data_config.py"

export WANDB_MODE=disabled HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$STARVLA:$REPO"
export PATH="$PY_ENV/bin:$PATH"
cd "$STARVLA" || exit 1
echo "git: $(git -C "$REPO" rev-parse HEAD)  mix=$MIX steps=$STEPS batch=$BATCH config=$CONFIG  $(date)" >> "$OUT/versions.txt"
cp "$CONFIG" "$OUT/config_source.yaml" 2> /dev/null || true
{
  date; nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
  accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml --num_processes 1 \
    "$REPO/scripts/rc365/train_lora_launcher.py" starVLA/training/train_starvla.py \
    --config_yaml "$CONFIG" \
    --framework.qwenvl.base_vlm "$W/models/Qwen3-VL-4B-Instruct" \
    --datasets.vla_data.data_root_dir "$W/data/robocasa365" \
    --datasets.vla_data.data_mix "$MIX" \
    --datasets.vla_data.per_device_batch_size "$BATCH" \
    --trainer.max_train_steps "$STEPS" --trainer.save_interval "$SAVE" \
    --run_root_dir "$W/runs/robocasa365_starvla" --run_id "$RUN" ${EXTRA:-}
  echo "exit=$?"; date
} >> "$OUT/train.log" 2>&1
grep -a "PEAK_MEMORY" "$OUT/train.log" | tail -1 > "$OUT/PEAK_MEMORY.txt" || echo "no PEAK_MEMORY line (see train.log)" > "$OUT/PEAK_MEMORY.txt"
