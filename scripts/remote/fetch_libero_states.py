"""Fetch only the simulator states and actions of the raw LIBERO demonstrations, by HTTP byte ranges.

A raw task file (yifengzhu-hf/LIBERO-datasets, about 650 MB) is mostly images. State replay
(rollout_libero_logged.py --replay-states) renders its own images and needs only `states` and
`actions` of every demonstration, well under 1 % of the file. From the A6000 server the connection
to Hugging Face gives 0.2-0.9 MB/s in total (2026-11-11), so whole files take hours; this script
opens each remote file through range requests, reads those two datasets, and writes a small local
HDF5 file with the same layout (data/demo_<k>/{states, actions}) and the same name.

Usage (LIBERO env, it has h5py):
    $LIBERO_PYTHON fetch_libero_states.py libero_object $W/data/libero_states/libero_object [--only WORD ...]
A file that already exists locally is skipped; an interrupted run continues where it stopped.
`--check FULL.hdf5` compares a fetched file with a complete local copy (states and actions must be
identical). No GPU.
"""
from __future__ import annotations

import argparse
import http.client
import io
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "yifengzhu-hf/LIBERO-datasets"
HOST = "huggingface.co"  # hf-mirror.com answers range requests of the standard library with HTTP 308
BLOCK = 1 << 15  # 32 KiB: HDF5 metadata is scattered in small pieces
TRIES = 12
TIMEOUT_S = 60
FIELDS = ("states", "actions")


class RangeFile(io.RawIOBase):
    """A read-only, seekable view of a remote file: bytes are fetched on demand over one kept-alive
    connection and cached in fixed-size blocks; a read that misses several neighbouring blocks is
    one request."""

    def __init__(self, url: str, size: int) -> None:
        super().__init__()
        self.url, self.size, self.position = url, size, 0
        self.blocks: dict[int, bytes] = {}
        self.connection: http.client.HTTPSConnection | None = None
        self.path = ""

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = max(0, min(self.size, base + offset))
        return self.position

    def _connect(self) -> None:
        """Follow the repository's redirect once and keep a connection to the storage host it names."""
        request = urllib.request.Request(self.url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as reply:
            parts = urllib.parse.urlsplit(reply.geturl())
            reply.read()
        self.path = parts.path + (f"?{parts.query}" if parts.query else "")
        self.connection = http.client.HTTPSConnection(parts.netloc, timeout=TIMEOUT_S)

    def _fetch(self, first: int, last: int) -> bytes:
        error: Exception | None = None
        for attempt in range(TRIES):
            try:
                if self.connection is None:
                    self._connect()
                self.connection.request("GET", self.path, headers={"Range": f"bytes={first}-{last}"})
                reply = self.connection.getresponse()
                data = reply.read()
                if reply.status == 206 and len(data) == last - first + 1:
                    return data
                error = RuntimeError(f"HTTP {reply.status} with {len(data)} bytes for a range of {last - first + 1}")
            except (OSError, http.client.HTTPException) as caught:  # dropped connections are common on this link
                error = caught
            self.connection = None  # reconnect and resolve the (signed, expiring) storage address again
            time.sleep(min(1 + attempt, 5))
        raise RuntimeError(f"range {first}-{last} failed after {TRIES} tries: {error}")

    def _ensure(self, first_block: int, last_block: int) -> None:
        """Fetch the missing blocks of first_block..last_block, each run of neighbours in one request."""
        index = first_block
        while index <= last_block:
            if index in self.blocks:
                index += 1
                continue
            run_end = index
            while run_end < last_block and run_end + 1 not in self.blocks:
                run_end += 1
            data = self._fetch(index * BLOCK, min((run_end + 1) * BLOCK, self.size) - 1)
            for k in range(index, run_end + 1):
                self.blocks[k] = data[(k - index) * BLOCK:(k - index + 1) * BLOCK]
            index = run_end + 1

    def readinto(self, buffer) -> int:
        count = min(len(buffer), self.size - self.position)
        if count <= 0:
            return 0
        self._ensure(self.position // BLOCK, (self.position + count - 1) // BLOCK)
        done = 0
        while done < count:
            index, inside = divmod(self.position, BLOCK)
            piece = self.blocks[index][inside:inside + count - done]
            buffer[done:done + len(piece)] = piece
            done += len(piece)
            self.position += len(piece)
        return done


def list_files(folder: str) -> list[tuple[str, int]]:
    """(file name, size) of the HDF5 files of one suite."""
    error: Exception | None = None
    for attempt in range(TRIES):
        try:
            with urllib.request.urlopen(f"https://{HOST}/api/datasets/{REPO}/tree/main/{folder}",
                                        timeout=TIMEOUT_S) as reply:
                return [(entry["path"].rsplit("/", 1)[-1], int(entry["size"])) for entry in json.load(reply)
                        if entry["type"] == "file" and entry["path"].endswith(".hdf5")]
        except Exception as caught:  # the link drops connections; try again
            error = caught
            time.sleep(min(1 + attempt, 5))
    raise RuntimeError(f"could not list {REPO}/{folder}: {error}")


def demo_key(name: str) -> int:
    return int(re.sub(r"\D", "", name) or 0)


def fetch_states(folder: str, name: str, size: int, destination: Path) -> str:
    """Write the states and actions of every demonstration of one remote file; returns a one-line report.

    Demonstrations already in the partial file of an earlier, interrupted run are not fetched again.
    """
    import h5py

    target = destination / name
    if target.exists():
        return f"skip (exists): {name}"
    remote = RangeFile(f"https://{HOST}/datasets/{REPO}/resolve/main/{folder}/{name}", size)
    partial = target.with_name(f".{name}.partial")
    with h5py.File(remote, "r") as source, h5py.File(partial, "a") as out:
        data = out.require_group("data")
        for key, value in source["data"].attrs.items():
            data.attrs[key] = value
        demos = sorted(source["data"], key=demo_key)
        for count, demo in enumerate(demos, start=1):
            if demo in data and data[demo].attrs.get("complete", False):
                continue
            if demo in data:
                del data[demo]
            group = data.create_group(demo)
            for field in FIELDS:
                group.create_dataset(field, data=source["data"][demo][field][:])
            group.attrs["complete"] = True
            out.flush()
            if count % 10 == 0:
                print(f"  {name}: {count}/{len(demos)} demonstrations", flush=True)
    partial.rename(target)
    fetched = len(remote.blocks) * BLOCK / 1e6
    return f"ok: {name}: {len(demos)} demonstrations, {fetched:.1f} MB fetched of {size / 1e6:.0f}"


def check(fetched: Path, full: Path) -> None:
    """Fail unless the states and actions of both files are identical."""
    import h5py
    import numpy as np

    with h5py.File(fetched, "r") as a, h5py.File(full, "r") as b:
        if sorted(a["data"]) != sorted(b["data"]):
            raise SystemExit(f"CHECK-FAILED: {fetched.name} and {full.name} list different demonstrations")
        for demo in a["data"]:
            for field in FIELDS:
                if not np.array_equal(a["data"][demo][field][:], b["data"][demo][field][:]):
                    raise SystemExit(f"CHECK-FAILED: {demo}/{field} differs")
        print(f"CHECK-OK: {len(a['data'])} demonstrations identical in {fetched.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="suite folder of the repository, e.g. libero_object")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--only", nargs="*", default=[], help="keep files whose name contains one of these")
    parser.add_argument("--check", type=Path, help="a complete local copy of one fetched file to compare with")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_]+", args.folder):
        parser.error("folder may contain only letters, digits and _")
    files = [(name, size) for name, size in list_files(args.folder)
             if re.fullmatch(r"[A-Za-z0-9_.-]+", name) and (not args.only or any(word in name for word in args.only))]
    if not files:
        print("no file matches", file=sys.stderr)
        return 1
    args.destination.mkdir(parents=True, exist_ok=True)
    failed = 0
    with ThreadPoolExecutor(len(files)) as pool:
        jobs = [(name, pool.submit(fetch_states, args.folder, name, size, args.destination)) for name, size in files]
        for name, job in jobs:
            try:
                print(job.result(), flush=True)
            except Exception as error:  # report every file; one failure must not hide the others
                print(f"FAILED: {name}: {error}", flush=True)
                failed += 1
    if args.check:
        check(args.destination / args.check.name, args.check)
    print(f"FETCH-FINISHED failed={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
