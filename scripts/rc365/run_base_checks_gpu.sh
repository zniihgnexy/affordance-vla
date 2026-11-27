#!/usr/bin/env bash
# AFFO-19 base-policy checks as ONE job each for the shared GPU queue (A6000 server). Calls run_eval_base_gpu.sh.
#   PHASE=steps      the 16000- and 32000-step checkpoints, condition A (configs/rc365/base_checks_steps.yaml), in turn
#   PHASE=stuck      TurnOffStove and TurnOnSinkFaucet with the part probe, n_act=8 and n_act=4 AT THE SAME TIME (two
#                    policy servers on the one card, 14 simulator clients each; configs/rc365/base_checks_stuck.yaml)
#   PHASE=instances  seen (A) and unseen (B) instances, episodes 20-79 of the four pick tasks (base_checks_instances.yaml)
# The checkpoints are MERGED ones ($W/runs/robocasa365_starvla/affo16_base_oft_lora_merged_<N>, scripts/rc365/merge_lora_checkpoint.py).
#   EXTRA  config overrides added in the instances phase, e.g. "episodes=60" to cut the episodes per cell
# Outputs: $W/runs/robocasa365_eval/affo19_<phase run names>/ (an existing folder is continued, never overwritten).
# Submit (never run directly), from the repo root:
#   bash scripts/server/submit_queue_job.sh 164 rc365_affo19_stuck 25000 1 "cards=0 1 2 3 5 6 7" -- \
#       env PHASE=stuck bash $PWD/scripts/rc365/run_base_checks_gpu.sh
set -uo pipefail
W="${W:-/mnt/nas/xinyizheng/affordance_vla}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
RUNS="$W/runs/robocasa365_starvla"
merged() { echo "$RUNS/affo16_base_oft_lora_merged_$1/checkpoints/steps_$1_pytorch_model.pt"; }
eval_run() {  # eval_run <steps> <run name> <config> <overrides>
  env CKPT="$(merged "$1")" RUN="$2" CONFIG="$3" OVERRIDES="$4" bash "$REPO/scripts/rc365/run_eval_base_gpu.sh"
}
case "${PHASE:?PHASE is steps, stuck or instances}" in
  steps)
    for n in 16000 32000; do eval_run "$n" "affo19_steps${n}_A" configs/rc365/base_checks_steps.yaml ""; done ;;
  stuck)
    eval_run 48000 affo19_stuck_nact8 configs/rc365/base_checks_stuck.yaml "n_act=8" &
    eval_run 48000 affo19_stuck_nact4 configs/rc365/base_checks_stuck.yaml "n_act=4" &
    wait ;;
  instances)
    eval_run 48000 affo19_instances_AB_20_79 configs/rc365/base_checks_instances.yaml "${EXTRA:-}" ;;
  *) echo "PHASE must be steps, stuck or instances" >&2; exit 2 ;;
esac
