"""Run a Python script as __main__ and print the CUDA peak memory of this process when it ends (any way it ends).

Usage: python peak_memory_launcher.py <script.py> [args...]   (also under `accelerate launch`)
"""

import atexit
import runpy
import sys

import torch


def _report() -> None:
    if torch.cuda.is_available():
        gib = 2**30
        print(
            f"PEAK_MEMORY allocated={torch.cuda.max_memory_allocated() / gib:.2f}GiB "
            f"reserved={torch.cuda.max_memory_reserved() / gib:.2f}GiB device={torch.cuda.current_device()}",
            flush=True,
        )


if __name__ == "__main__":
    script, sys.argv = sys.argv[1], sys.argv[1:]
    atexit.register(_report)
    runpy.run_path(script, run_name="__main__")
