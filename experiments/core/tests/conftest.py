"""Shared fixtures for the trainer's property tests.

Every test runs the engine in-process on CPU with TINY configs: fast,
GPU-independent, and deterministic under a fixed seed. The model is a
test-only plug-in (toy_models.py) -- the engine knows no architecture.
"""

import csv
import sys
from pathlib import Path

import yaml

TESTS = Path(__file__).resolve().parent
# tests/ lives next to the modules under test.
sys.path.insert(0, str(TESTS.parent))

import train  # noqa: E402

TOY_MODEL = f"{TESTS / 'toy_models.py'}:MLP"

# Small enough for sub-second runs; big enough to exercise every branch.
TINY = {
    "seed": 0,
    "device": "cpu",
    "task": {"factory": "parity", "seq_len": 6, "val_size": 32},
    "model": {"factory": TOY_MODEL, "seq_len": 6, "hidden": 8},
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
    """Merge overrides (one level deep) over TINY, write <run>/config.yaml."""
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in TINY.items()}
    for key, value in overrides.items():
        if isinstance(value, dict) and "factory" in value and value["factory"] != cfg.get(key, {}).get("factory"):
            cfg[key] = dict(value)  # another factory: TINY's kwargs don't apply
        elif isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def run_mode(mode: str, run_dir: Path, overrides: dict | None = None) -> list[dict]:
    """Run one mode in-process through the real entry point; return rows."""
    config = write_config(run_dir, overrides or {})
    code = train.execute(config, mode=mode)
    failed = run_dir / "FAILED"
    assert code == 0, failed.read_text() if failed.exists() else f"exit {code}"
    with (run_dir / "metrics.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def strip_wall(rows: list[dict]) -> list[dict]:
    """The `wall` column is wall-clock time — exclude it from
    determinism assertions."""
    return [{k: v for k, v in row.items() if k != "wall"} for row in rows]
