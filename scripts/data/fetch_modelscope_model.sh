#!/usr/bin/env bash
# Download one ModelScope model repository with curl (HTTP/2): resumable, every file checked by size
# and, when the listing has one, by sha256. No GPU is used.
#
# Why not the hub CLIs: on the A6000 server HTTP/1.1 is throttled to ~0.8 MB/s and Hugging Face drops
# connections, while curl over HTTP/2 to modelscope.cn reaches ~25 MB/s (measured 2026-11-10).
#
# Usage:  fetch_modelscope_model.sh <owner>/<name> <destination-dir>
# Exit status is non-zero unless every listed file is present and correct. Re-running resumes; a file
# that fails its checksum is removed, so the next run downloads it again.
set -uo pipefail

[ $# -eq 2 ] || { sed -n 2,9p "$0" >&2; exit 2; }
MODEL="$1"
DEST="$2"
PY="${PY:-$HOME/anaconda3/bin/python}"
MAX_TRIES="${MAX_TRIES:-12}"
LISTING="$DEST/.modelscope_files.tsv"
mkdir -p "$DEST"

curl -fsS -m 60 "https://modelscope.cn/api/v1/models/$MODEL/repo/files?Revision=master&Recursive=true" \
  | "$PY" -c 'import json, sys
for f in json.load(sys.stdin)["Data"]["Files"]:
    if f.get("Type") != "tree":
        print(f["Size"], f.get("Sha256") or "-", f["Path"], sep="\t")' > "$LISTING.tmp"
[ -s "$LISTING.tmp" ] || { echo "FAILED: could not list $MODEL on ModelScope" >&2; exit 1; }
mv "$LISTING.tmp" "$LISTING"

has_size() { [ -f "$1" ] && [ "$(stat -c %s "$1")" = "$2" ]; }
is_larger() { [ -f "$1" ] && [ "$(stat -c %s "$1")" -gt "$2" ]; }

failed=0
while IFS=$'\t' read -r size sha path; do
  case "$path" in
    /*|*..*) echo "FAILED: refusing a path that could leave the destination: $path"; failed=1; continue ;;
  esac
  target="$DEST/$path"
  mkdir -p "$(dirname "$target")"
  for _ in $(seq "$MAX_TRIES"); do
    has_size "$target" "$size" && break
    is_larger "$target" "$size" && rm -f "$target"   # cannot be resumed (the server would answer 416)
    curl -fsSL --retry 3 -C - -o "$target" "https://modelscope.cn/models/$MODEL/resolve/master/$path" || sleep 5
  done
  if ! has_size "$target" "$size"; then
    echo "FAILED: $path (size differs from the listing)"; failed=1; continue
  fi
  if [ "$sha" != "-" ] && ! echo "$sha  $target" | sha256sum -c --status; then
    rm -f "$target"
    echo "FAILED: $path (sha256 differs from the listing; removed, run again to fetch it)"; failed=1; continue
  fi
  echo "ok: $path ($size B)"
done < "$LISTING"

[ "$failed" -eq 0 ] && echo "MODEL-READY: $DEST" || { echo "MODEL-INCOMPLETE: $DEST" >&2; exit 1; }
