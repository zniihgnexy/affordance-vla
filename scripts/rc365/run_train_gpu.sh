#!/usr/bin/env bash
# starVLA QwenPI on three RoboCasa365 tasks, ONE card, as a job for the shared GPU queue (A6000 server).
# Pipeline check (AFFO-10): measures step time and peak memory of the frozen-VLM and the full fine-tune.
# Settings from the environment (defaults in brackets):
#   RUN [qwenpi_rc365_3tasks_frozen]  outputs go to $W/runs/robocasa365_starvla/$RUN
#   STEPS [300]  BATCH [8]  SAVE [300]  (checkpoint every SAVE steps)
#   FREEZE [qwen_vl_interface]  "" = train the whole model (full fine-tune)
# Submit (never run directly; the queue sets CUDA_VISIBLE_DEVICES), from the repo root:
#   bash scripts/server/submit_queue_job.sh 164 rc365_frozen 30000 1 "cards=6 7" -- \
#       env RUN=qwenpi_rc365_3tasks_frozen bash ~/wt-affo-10/scripts/rc365/run_train_gpu.sh
# Output: $OUT/{config.yaml,checkpoints/,train.log,PEAK_MEMORY.txt}; train.log has "timing/model" every 10 steps.
set -uo pipefail
W="${W:-/mnt/nas/xinyizheng/affordance_vla}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
STARVLA="${STARVLA:-$W/repos/starVLA_rc365}"
PY_ENV="${PY_ENV:-/data/xinyizheng/envs/starvla}"
RUN="${RUN:-qwenpi_rc365_3tasks_frozen}"
STEPS="${STEPS:-300}"; BATCH="${BATCH:-8}"; SAVE="${SAVE:-300}"
FREEZE="${FREEZE-qwen_vl_interface}"
OUT="$W/runs/robocasa365_starvla/$RUN"
[ ! -e "$OUT" ] || { echo "$OUT exists; choose another RUN" >&2; exit 1; }
mkdir -p "$OUT"

# registry overlay: the three-task mixture (a new benchmark folder; no starVLA file is changed)
overlay="$STARVLA/examples/simBenchmarks/AFFO10_rc365/train_files/data_registry"
mkdir -p "$overlay" && cp "$REPO/configs/rc365/data_registry/data_config.py" "$overlay/data_config.py"

export WANDB_MODE=disabled HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$STARVLA"
export PATH="$PY_ENV/bin:$PATH"
cd "$STARVLA" || exit 1
echo "git: $(git -C "$REPO" rev-parse HEAD)  starVLA snapshot: $(basename "$(ls "$W"/repos/_dl/starVLA-*.zip 2>/dev/null | head -1)")" > "$OUT/versions.txt"
{
  date; nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
  accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml --num_processes 1 \
    "$REPO/scripts/rc365/peak_memory_launcher.py" starVLA/training/train_starvla.py \
    --config_yaml "$REPO/configs/rc365/qwenpi_rc365_3tasks.yaml" \
    --framework.qwenvl.base_vlm "$W/models/Qwen3-VL-4B-Instruct" \
    --datasets.vla_data.data_root_dir "$W/data/robocasa365" \
    --datasets.vla_data.per_device_batch_size "$BATCH" \
    --trainer.max_train_steps "$STEPS" --trainer.save_interval "$SAVE" \
    --trainer.freeze_modules "$FREEZE" \
    --run_root_dir "$W/runs/robocasa365_starvla" --run_id "$RUN"
  echo "exit=$?"; date
} > "$OUT/train.log" 2>&1
grep -a "PEAK_MEMORY" "$OUT/train.log" > "$OUT/PEAK_MEMORY.txt" || echo "no PEAK_MEMORY line (see train.log)" > "$OUT/PEAK_MEMORY.txt"
