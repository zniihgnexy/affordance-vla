#!/usr/bin/env bash
# Put one job into the shared GPU queue on the A6000 server (estar).
#
# The queue belongs to the vid_tokenizer project (runner: scripts/queue_runner.py there). It launches a
# pending job on ONE card as soon as that card has MINMB MiB free and runs fewer than SHARE queue jobs;
# the job gets CUDA_VISIBLE_DEVICES and GPU_INDEX. Without cards= the runner uses its default list
# (0 1 4 5 6 7). Since 2026-11-11 every empty card may be used, 0-3 preferred, so cards 2 and 3 may be
# named here; the runner still skips cards listed in its queue/deny_cards.txt. Pending jobs are taken in
# lexicographic order of the file name, so the 3-digit priority decides who goes first.
# Never start GPU work outside this queue.
#
# Usage:  submit_queue_job.sh <priority 000-999> <name> <minmb> <share> [cards=<"1 5">] -- <command...>
# Result: logs/queue/<priority>_<name>.log under the vid_tokenizer repo; status in queue/status.tsv.
set -euo pipefail

QUEUE="${QUEUE:-/home/xinyizheng/vid_tokenizer/queue}"
usage() { sed -n 2,13p "$0" >&2; exit 2; }

[ $# -ge 6 ] || usage
priority="$1"; name="$2"; minmb="$3"; share="$4"; shift 4
cards=""
if [[ "$1" == cards=* ]]; then cards="${1#cards=}"; shift; fi
[ "$1" = "--" ] || usage
shift
[[ "$priority" =~ ^[0-9]{3}$ ]] || { echo "priority must be three digits" >&2; exit 2; }
[[ "$name" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "name may contain only letters, digits, _ and -" >&2; exit 2; }
[[ "$minmb" =~ ^[0-9]+$ && "$share" =~ ^[1-9][0-9]*$ ]] || { echo "minmb and share must be integers" >&2; exit 2; }
for card in $cards; do
  [[ "$card" =~ ^[0-7]$ ]] || { echo "card $card does not exist (cards are 0-7)" >&2; exit 2; }
done
[ -d "$QUEUE/pending" ] || { echo "queue not found: $QUEUE" >&2; exit 1; }

job="$QUEUE/pending/${priority}_${name}.sh"
for state in pending running; do
  [ ! -e "$QUEUE/$state/${priority}_${name}.sh" ] || { echo "job already $state: ${priority}_${name}.sh" >&2; exit 1; }
done

# Write outside pending/ first and move it in: the runner must never see a half-written job.
tmp="$(mktemp "$QUEUE/.submit_${name}.XXXXXX")"
{
  echo "#Q minmb=${minmb} share=${share}${cards:+ cards=${cards}}"
  echo "set -uo pipefail"
  printf '%q ' "$@"; echo
} > "$tmp"
chmod 644 "$tmp"
mv "$tmp" "$job"
echo "submitted: $job"
