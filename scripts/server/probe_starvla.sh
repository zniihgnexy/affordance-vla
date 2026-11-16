#!/usr/bin/env bash
# V0 probe: GPU memory and throughput of fully fine-tuning starVLA (QwenPI + Qwen3-VL-2B) on LIBERO,
# on ONE A6000. It runs the unmodified starVLA trainer for a few steps per batch size and records the
# peak memory of this job's processes and the step time.
#
# Launch it ONLY through the GPU queue (the runner sets CUDA_VISIBLE_DEVICES); see submit_queue_job.sh.
# Output: $OUT/summary.tsv plus one log per batch size.
set -uo pipefail

W="${W:-/mnt/nas/xinyizheng/affordance_vla}"
ENV_PREFIX="${ENV_PREFIX:-/data/xinyizheng/envs/starvla}"
STARVLA="${STARVLA:-$W/repos/starVLA}"
FRAMEWORK="${FRAMEWORK:-QwenPI}"
VLM="${VLM:-$W/models/Qwen3-VL-2B-Instruct}"
STEPS="${STEPS:-60}"
BATCHES="${BATCHES:-16 8 4}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"   # logged steps up to this one are left out of the timing median
OUT="${OUT:-$W/runs/probe_$(date +%Y%m%d_%H%M%S)}"
: "${CUDA_VISIBLE_DEVICES:?must be set by the queue runner; do not start this script by hand}"

# shellcheck disable=SC1090
source ~/anaconda3/etc/profile.d/conda.sh && conda activate "$ENV_PREFIX"
cd "$STARVLA" || exit 1
export WANDB_MODE=disabled HF_HOME="$W/cache/hf" TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset NCCL_SOCKET_IFNAME NCCL_IB_HCA   # starVLA's launcher hard-codes values for the authors' cluster
mkdir -p "$OUT"

SESSION="$(ps -o sid= -p $$ | tr -d ' ')"
my_gpu_mib() {  # MiB currently held on any GPU by processes of this job's session
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', *' -v pids="$(ps -s "$SESSION" -o pid= | tr '\n' ' ')" \
        'BEGIN { n = split(pids, a, " "); for (i = 1; i <= n; i++) mine[a[i]] = 1 } ($1 in mine) { s += $2 } END { print s + 0 }'
}
poll_peak() {  # keep the largest value seen in file $1
  local peak=0 now
  while :; do
    now="$(my_gpu_mib)"
    if [ "$now" -gt "$peak" ]; then peak="$now"; echo "$peak" > "$1"; fi
    sleep 2
  done
}

{
  echo "card: $CUDA_VISIBLE_DEVICES (index ${GPU_INDEX:-?}) · framework $FRAMEWORK · vlm $(basename "$VLM") · steps $STEPS"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader -i "$CUDA_VISIBLE_DEVICES"
  echo "other tenants on this card at start (pid, MiB):"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$CUDA_VISIBLE_DEVICES"
  git -C "$STARVLA" log --oneline -1
} > "$OUT/context.txt" 2>&1

printf 'batch\trc\tpeak_mib\tparams_total_m\tparams_trainable_m\tgrad_accum\tmodel_s\tdata_s\tsamples_per_s\toom\n' > "$OUT/summary.tsv"
for bs in $BATCHES; do
  log="$OUT/bs${bs}.log"
  echo 0 > "$OUT/bs${bs}.peak_mib"
  poll_peak "$OUT/bs${bs}.peak_mib" & poller=$!
  accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml --num_processes 1 \
    starVLA/training/train_starvla.py \
    --config_yaml examples/simBenchmarks/LIBERO/train_files/starvla_cotrain_libero.yaml \
    --framework.name "$FRAMEWORK" --framework.qwenvl.base_vlm "$VLM" \
    --datasets.vla_data.data_root_dir "$W/data/libero_lerobot" --datasets.vla_data.data_mix libero_all \
    --datasets.vla_data.per_device_batch_size "$bs" --trainer.vla_data.video_backend torchvision_av \
    --trainer.freeze_modules --trainer.max_train_steps "$STEPS" --trainer.save_interval 100000000 \
    --trainer.logging_frequency 5 --trainer.eval_interval 100000000 \
    --run_root_dir "$OUT" --run_id "bs${bs}" > "$log" 2>&1
  rc=$?
  kill "$poller" 2>/dev/null; wait "$poller" 2>/dev/null
  rm -rf "$OUT/bs${bs}/final_model" "$OUT/bs${bs}/checkpoints"   # weights are not needed, only the numbers
  python - "$log" "$bs" "$rc" "$(cat "$OUT/bs${bs}.peak_mib")" "$WARMUP_STEPS" >> "$OUT/summary.tsv" <<'EOF'
import re, statistics, sys
log, bs, rc, peak, warmup = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5])
text = open(log, errors="replace").read()
params = re.search(r"Parameters \(in millions\): ([\d.]+) Total, ([\d.]+) Trainable", text)
accum = re.search(r"Gradient accumulation steps = (\d+)", text)
steps = [(int(s), float(m), float(d)) for s, d, m in
         re.findall(r"Step (\d+), Loss: \{.*?'timing/data': ([\d.eE+-]+).*?'timing/model': ([\d.eE+-]+)", text)]
kept = [(m, d) for s, m, d in steps if s > warmup]
model_s = statistics.median(m for m, _ in kept) if kept else float("nan")
data_s = statistics.median(d for _, d in kept) if kept else float("nan")
rate = bs / (model_s + data_s) if kept else float("nan")
oom = "yes" if re.search(r"out of memory|OutOfMemoryError", text, re.I) else "no"
print("\t".join(str(x) for x in (bs, rc, peak, params.group(1) if params else "?", params.group(2) if params else "?",
                                 accum.group(1) if accum else "?", f"{model_s:.3f}", f"{data_s:.3f}", f"{rate:.2f}", oom)))
EOF
done
echo "PROBE-FINISHED: $OUT/summary.tsv"
cat "$OUT/summary.tsv"
