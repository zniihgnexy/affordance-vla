#!/usr/bin/env bash
# One LIBERO-Object task file (50 demos, ~664 MB) and its BDDL scene description, used by
# report/2026-11-10_vla_datasets_sim_report.ipynb. It goes to <repo>/data/libero/ (on the server `data`
# is a symlink into the NAS work dir) and is not versioned. Resumable; the file's sha256 is checked.
#
# Sources: SOURCE=modelscope (default; a byte-identical re-upload, fast from the A6000 server) or
# SOURCE=huggingface (the original release, yifengzhu-hf/LIBERO-datasets).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
MAIN="$(cd "$(git -C "$REPO" rev-parse --git-common-dir)/.." && pwd)"
OUT="$MAIN/data/libero/libero_object"
TASK=pick_up_the_salad_dressing_and_place_it_in_the_basket
SHA256=45960606a85b9c68980eb8130e0fe16bf3621539cf5d4147f35aa8139573c5ac
SOURCE="${SOURCE:-modelscope}"
mkdir -p "$OUT"

case "$SOURCE" in
  modelscope)  url="https://modelscope.cn/datasets/szvdzfv/LIBERO-datasets/resolve/master/libero_object/${TASK}_demo.hdf5" ;;
  huggingface) url="https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/main/libero_object/${TASK}_demo.hdf5" ;;
  *) echo "SOURCE must be modelscope or huggingface" >&2; exit 2 ;;
esac
sha256_of() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }

target="$OUT/${TASK}_demo.hdf5"
if [ ! -f "$target" ] || [ "$(sha256_of "$target")" != "$SHA256" ]; then
  curl -fL --retry 5 -C - -o "$target" "$url"
  [ "$(sha256_of "$target")" = "$SHA256" ] || { echo "sha256 mismatch: $target" >&2; exit 1; }
fi
[ -s "$OUT/${TASK}.bddl" ] || curl -fsL --retry 3 -o "$OUT/${TASK}.bddl" \
  "https://raw.githubusercontent.com/Lifelong-Robot-Learning/LIBERO/master/libero/libero/bddl_files/libero_object/${TASK}.bddl"
echo "done: $OUT"
