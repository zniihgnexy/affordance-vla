"""Download one ModelScope dataset repository with curl (HTTP/2), in parallel, with sha256 checks.

Why not the hub CLIs: on the A6000 server HTTP/1.1 is throttled and connections to Hugging Face drop
(`hf download` then reports success with files missing). This script lists the repository through the
ModelScope API, fetches every file with curl, verifies its sha256 against the listing, and exits with
a non-zero status unless every listed file is present and correct. Re-running it resumes.

Usage:  python fetch_modelscope_dataset.py <owner>/<name> <destination-dir> [--workers 16]
Only the standard library is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API = "https://modelscope.cn/api/v1/datasets/{repo}/repo/tree?Revision=master&Root=/&Recursive=True&PageSize=100000"
FILE_URL = "https://modelscope.cn/datasets/{repo}/resolve/master/{path}"
LIST_TRIES = 5
FILE_TRIES = 4


def list_files(repo: str) -> list[tuple[str, int, str]]:
    """(path, size, sha256) of every file in the repository."""
    error: Exception | None = None
    for _ in range(LIST_TRIES):
        try:
            with urllib.request.urlopen(API.format(repo=repo), timeout=120) as response:
                payload = json.load(response)
            if payload.get("Code") != 200:
                raise RuntimeError(f"ModelScope API: {payload.get('Code')} {payload.get('Message')}")
            entries = payload["Data"]["Files"]
            if len(entries) != payload.get("TotalCount", len(entries)):
                raise RuntimeError(f"listing truncated: {len(entries)} of {payload['TotalCount']} entries")
            return [(e["Path"], int(e["Size"]), e["Sha256"]) for e in entries if e["Type"] == "blob"]
        except Exception as caught:  # network errors are retried; the last one is re-raised below
            error = caught
    raise RuntimeError(f"could not list {repo}: {error}")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(repo: str, destination: Path, entry: tuple[str, int, str]) -> str | None:
    """Download one file unless a correct copy exists. Returns an error message, or None on success."""
    path, size, sha = entry
    target = destination / path
    if target.is_file() and target.stat().st_size == size and sha256_of(target) == sha:
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    url = FILE_URL.format(repo=repo, path=urllib.parse.quote(path))
    problem = "not attempted"
    for _ in range(FILE_TRIES):
        result = subprocess.run(["curl", "-fsSL", "--http2", "--retry", "3", "-m", "600", "-o", str(part), url],
                                capture_output=True, text=True)
        if result.returncode != 0:
            problem = f"curl exit {result.returncode}: {result.stderr.strip()[:120]}"
            continue
        if part.stat().st_size != size or sha256_of(part) != sha:
            problem = "size or sha256 mismatch"
            continue
        part.replace(target)
        return None
    part.unlink(missing_ok=True)
    return f"{path}: {problem}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo", help="ModelScope dataset id, <owner>/<name>")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    entries = list_files(args.repo)
    total_mb = sum(size for _, size, _ in entries) / 1e6
    print(f"{args.repo}: {len(entries)} files, {total_mb:,.0f} MB -> {args.destination}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        errors = [e for e in pool.map(lambda entry: fetch(args.repo, args.destination, entry), entries) if e]
    for error in errors[:20]:
        print("FAILED:", error)
    print(f"{args.repo}: {len(entries) - len(errors)} of {len(entries)} files verified"
          + ("" if not errors else f", {len(errors)} FAILED"), flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
