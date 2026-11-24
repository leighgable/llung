"""Shared fixtures for the trainer's property tests.

Every test runs the driver in-process on CPU with TINY configs: fast,
GPU-independent, and deterministic under a fixed seed.
"""

import csv
import sys
from pathlib import Path

import pytest
import yaml

# tests/ lives next to the modules under test.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train  # noqa: E402

# Small enough for sub-second runs; big enough to exercise every branch.
TINY = {
    "seed": 0,
    "task": {"name": "parity", "seq_len": 6, "val_size": 32},
    "model": {"hidden": 8},
    "train": {
        "pop_size": 6,
        "generations": 4,
        "sigma": 0.2,
        "checkpoint_every": 2,
        "epochs": 2,
        "batches_per_epoch": 2,
        "batch_size": 8,
        "steps": 3,
        "rollouts": 8,
    },
}


def write_config(run_dir: Path, overrides: dict) -> Path:
    """Merge overrides over TINY, write runs/<id>/config.yaml."""
    cfg = {"seed": TINY["seed"], **overrides}
    cfg = {**TINY, **cfg, "task": {**TINY["task"], **overrides.get("task", {})},
           "model": {**TINY["model"], **overrides.get("model", {})},
           "train": {**TINY["train"], **overrides.get("train", {})}}
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def run_mode(mode: str, run_dir: Path, overrides: dict | None = None) -> list[dict]:
    """Run one mode in-process and return the logged rows."""
    config = write_config(run_dir, overrides or {})
    cfg = train.load_config(config)
    train.seed_everything(int(cfg["seed"]))
    r = train.Runner(mode, cfg, run_dir, "cpu")
    try:
        if mode == "ga":
            train.run_ga(r)
        elif mode == "rl":
            train.run_rl(r)
        else:
            train.run_pretrain(r)
        r.checkpoint()
    finally:
        r.close()
    with (run_dir / "metrics.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def strip_wall(rows: list[dict]) -> list[dict]:
    """The `wall` column is wall-clock time — exclude it from
    determinism assertions."""
    return [{k: v for k, v in row.items() if k != "wall"} for row in rows]
