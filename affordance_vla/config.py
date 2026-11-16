"""Configuration files: YAML, an optional `base:` file they extend, and `key=value` overrides.

A config may name another config as `base:` (relative to its own folder); its keys are merged over the
base's, nested mappings key by key. Overrides from the command line use dotted keys (`viewer.realtime=0.5`)
and YAML values (`grasp.place=false`). `resolved` adds where the config came from and the git commit, so
a run can record exactly what it used.
"""
from __future__ import annotations

import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml


def merged(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """A new mapping: `update` over `base`, merging nested mappings key by key."""
    out = dict(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = merged(out[key], value)
        else:
            out[key] = value
    return out


def with_override(config: Mapping[str, Any], override: str) -> dict[str, Any]:
    """A new config with one `dotted.key=yaml_value` override applied."""
    key, sep, text = override.partition("=")
    if not sep or not key:
        raise ValueError(f"an override looks like key=value or section.key=value, got {override!r}")
    value: Any = yaml.safe_load(text) if text else None
    for part in reversed(key.split(".")):
        value = {part: value}
    return merged(config, value)


def load_config(path: Path | str, overrides: Iterable[str] = ()) -> dict[str, Any]:
    """The config at `path` merged over its `base:` chain, then the overrides in order."""
    path = Path(path)
    seen: list[Path] = []
    chain: list[dict[str, Any]] = []
    while path is not None:
        path = path.resolve()
        if path in seen:
            raise ValueError(f"config bases form a loop: {' -> '.join(map(str, seen + [path]))}")
        seen.append(path)
        data = yaml.safe_load(path.read_text()) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path} does not hold a mapping")
        base = data.pop("base", None)
        chain.append(data)
        path = path.parent / base if base else None
    config: dict[str, Any] = {}
    for data in reversed(chain):
        config = merged(config, data)
    for override in overrides:
        config = with_override(config, override)
    return config


def git_commit(repo: Path | str) -> str:
    """The checked-out commit of `repo`, with `+dirty` when tracked files have changes; 'unknown' outside git."""
    def run(*args: str) -> str:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout
    try:
        commit = run("rev-parse", "--short", "HEAD").strip()
        return commit + ("+dirty" if run("status", "--porcelain", "--untracked-files=no").strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def resolved(config: Mapping[str, Any], path: Path | str, overrides: Iterable[str], repo: Path | str) -> dict:
    """The config with a `_source` entry: the file, the overrides and the code version it was run with."""
    return {**config, "_source": {"config": str(path), "overrides": list(overrides), "commit": git_commit(repo)}}
