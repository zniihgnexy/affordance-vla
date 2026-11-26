"""Read the loss and step time out of a starVLA training log (it prints a metrics dict every `logging_frequency` steps)."""
from __future__ import annotations

import re
from pathlib import Path

_NUMBER = r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"


def read_series(log: Path | str, key: str) -> list[float]:
    """Every value printed for `key` (for example 'action_dit_loss' or 'timing/model'), in order of appearance."""
    text = Path(log).read_text(errors="ignore")
    return [float(v) for v in re.findall(rf"'{re.escape(key)}':\s+{_NUMBER}", text)]


def smoothed(values: list[float], window: int) -> list[float]:
    """Trailing mean over `window` values."""
    out = []
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1): i + 1]
        out.append(sum(chunk) / len(chunk))
    return out
