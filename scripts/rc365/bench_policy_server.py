"""How many evaluation clients can one policy server carry? N threads send queries (the three-camera example of an
evaluation step) for a fixed time; prints queries per second and the latency a client sees (robocasa365 environment).

  PYTHONPATH=<starVLA snapshot>:. "$RC365_PYTHON" scripts/rc365/bench_policy_server.py --port P --clients 1 4 8 16 30
A real client asks once per `n_act` environment steps, about every 3.3 s (0.41 s per step measured in AFFO-10), so a
server answering q queries per second carries about q * 3.3 clients before they start to wait.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from affordance_vla.sim import rc365_policy_io as io  # noqa: E402
from affordance_vla.sim.rc365_eval import ServerPolicy  # noqa: E402


def example() -> dict:
    rng = np.random.default_rng(0)
    obs = {key: rng.integers(0, 256, (256, 256, 3), dtype=np.uint8) for key in io.VIDEO_KEYS}
    obs[io.LANGUAGE_KEY] = "Open the left drawer."
    return io.build_example(obs)


def run(port: int, clients: int, seconds: float) -> tuple[float, float, float]:
    latencies: list[float] = []
    lock = threading.Lock()
    stop = time.time() + seconds

    def worker() -> None:
        policy = ServerPolicy("127.0.0.1", port)
        ex = example()
        mine = []
        while time.time() < stop:
            started = time.perf_counter()
            policy(ex)
            mine.append(time.perf_counter() - started)
        policy.close()
        with lock:
            latencies.extend(mine)

    threads = [threading.Thread(target=worker) for _ in range(clients)]
    begun = time.time()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.time() - begun
    return len(latencies) / elapsed, statistics.median(latencies), float(np.percentile(latencies, 95))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--clients", type=int, nargs="+", default=[1, 4, 8, 16, 30])
    parser.add_argument("--seconds", type=float, default=40)
    args = parser.parse_args()
    for clients in args.clients:
        rate, median, p95 = run(args.port, clients, args.seconds)
        print(f"clients={clients}: {rate:.2f} queries/s, latency median {median * 1000:.0f} ms, "
              f"p95 {p95 * 1000:.0f} ms", flush=True)


if __name__ == "__main__":
    main()
