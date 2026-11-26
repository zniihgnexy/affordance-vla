"""Which episodes of a RoboCasa365 bundle hold which objects, read from `ep_meta.json` only (no simulator)."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


def instance_key(mjcf_path: str) -> str:
    """'objaverse/mug/mug_11' from an object's model.xml path (the registry key, whatever the checkout)."""
    found = re.search(r"objects/(.+)/model\.xml$", str(mjcf_path))
    if not found:
        raise ValueError(f"not an object model path: {mjcf_path}")
    return found.group(1)


def read_meta(ds: Path, ep: int) -> dict:
    return json.loads((ds / f"extras/episode_{ep:06d}/ep_meta.json").read_text())


def episode_count(ds: Path) -> int:
    return len(list(ds.glob("extras/episode_*")))


def episode_objects(meta: dict, categories: set[str] | None = None, names: set[str] | None = None) -> dict[str, dict]:
    """{env object name: {category, instance}} of the objects of an episode (of the given categories and names)."""
    found = {}
    for cfg in meta.get("object_cfgs", []):
        info = cfg.get("info", {})
        if (
            info.get("cat")
            and (categories is None or info["cat"] in categories)
            and (names is None or cfg["name"] in names)
        ):
            found[cfg["name"]] = {"category": info["cat"], "instance": instance_key(info["mjcf_path"])}
    return found


def manipulated_category(meta: dict) -> str | None:
    """The category of the object a pick-and-place episode moves: the config named `obj`."""
    for cfg in meta.get("object_cfgs", []):
        if cfg["name"] == "obj":
            return cfg["info"]["cat"]
    return None


def list_episodes(
    ds: Path, categories: Iterable[str] | None = None, limit: int | None = None, names: Iterable[str] | None = None
) -> list[int]:
    """Episode numbers to replay: all, or, with `categories`, those that hold such an object (of the given names)."""
    wanted, only = (set(categories) if categories else None), (set(names) if names else None)
    keep = [ep for ep in range(episode_count(ds)) if wanted is None or episode_objects(read_meta(ds, ep), wanted, only)]
    return keep[:limit] if limit else keep


def manipulated_counts(ds: Path) -> Counter:
    """Category of the manipulated object ('obj') over all episodes of a pick-and-place bundle."""
    return Counter(manipulated_category(read_meta(ds, ep)) for ep in range(episode_count(ds)))
