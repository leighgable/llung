"""Property tests for core/train.py (hypothesis) + the sentinel contract.

The properties that MUST hold for every seed:
  - GA elitism: gen_best never worsens across generations.
  - Seeding: same config + same seed -> same metrics (modulo `wall`).
  - The sentinel contract: clean exit -> DONE, any exception -> FAILED,
    and a rerun never inherits a stale verdict.
"""

import sys

import pytest
import yaml
from hypothesis import HealthCheck, given, settings, strategies as st

# tmp_path is shared between generated examples by design: every
# example writes to a seed-namespaced subdir, so nothing collides.
PROPS = {
    "deadline": None,
    "derandomize": True,
    "suppress_health_check": [HealthCheck.function_scoped_fixture],
}

from conftest import run_mode, strip_wall, write_config

import train


@pytest.mark.parametrize("mode", ["ga", "pretrain", "rl"])
def test_every_mode_logs_and_checkpoints(tmp_path, mode):
    rows = run_mode(mode, tmp_path / mode)
    assert rows, f"{mode} must log at least one row"
    # Header-first contract: columns match the mode's declared shape.
    expected = ["step"] + train.MODE_COLUMNS[mode] + ["wall"]
    assert list(rows[0].keys()) == expected
    # Numeric sanity: no NaN/empty cells in required columns.
    for row in rows:
        for col in expected:
            assert row[col] != "", f"empty {col} in {row}"
    assert (tmp_path / mode / "checkpoints" / "latest.pt").exists()
    assert list((tmp_path / mode).glob("events.out.tfevents.*")), "TensorBoard events"


@given(seed=st.integers(0, 10_000))
@settings(max_examples=6, **PROPS)
def test_ga_gen_best_is_non_increasing(seed, tmp_path):
    """Elitism guarantees the best genome survives each generation."""
    rows = run_mode("ga", tmp_path / f"ga{seed}", {"seed": seed})
    best = [float(r["gen_best"]) for r in rows]
    assert len(best) == 4, "one row per generation"
    for prev, cur in zip(best, best[1:]):
        assert cur <= prev + 1e-9, f"gen_best worsened: {best}"


@given(seed=st.integers(0, 10_000))
@settings(max_examples=4, **PROPS)
def test_same_seed_same_metrics(seed, tmp_path):
    a = strip_wall(run_mode("ga", tmp_path / f"a{seed}", {"seed": seed}))
    b = strip_wall(run_mode("ga", tmp_path / f"b{seed}", {"seed": seed}))
    assert a == b, "a rerun of the same config must reproduce bit-for-bit"


@given(seed=st.integers(0, 10_000))
@settings(max_examples=3, **PROPS)
def test_ga_best_never_exceeds_mean(seed, tmp_path):
    """MSE is non-negative, and the best genome beats the mean."""
    rows = run_mode("ga", tmp_path / f"c{seed}", {"seed": seed})
    for row in rows:
        assert 0.0 <= float(row["gen_best"]) <= float(row["gen_mean"]) + 1e-6


def test_failed_run_writes_failed_sentinel(monkeypatch, tmp_path):
    config = write_config(tmp_path / "bad", {"task": {"name": "nope-such-task"}})
    monkeypatch.setattr(
        sys, "argv", ["train.py", "--mode", "ga", "--config", str(config)]
    )
    assert train.main() == 1
    failed = (tmp_path / "bad" / "FAILED").read_text()
    assert "ValueError" in failed and "unknown task" in failed
    assert not (tmp_path / "bad" / "DONE").exists()


def test_clean_run_writes_done_and_rerun_clears_stale_sentinel(
    monkeypatch, tmp_path
):
    config = write_config(tmp_path / "ok", {})
    monkeypatch.setattr(
        sys, "argv", ["train.py", "--mode", "pretrain", "--config", str(config)]
    )
    assert train.main() == 0
    done = (tmp_path / "ok" / "DONE").read_text()
    assert done.startswith("exit=0"), done
    meta = __import__("json").loads((tmp_path / "ok" / "meta.json").read_text())
    assert meta["mode"] == "pretrain" and meta["device"] in ("cpu", "cuda")

    # A rerun of the same run dir must not inherit the old verdict:
    # DONE is cleared at start, then written again on success.
    (tmp_path / "ok" / "DONE").write_text("exit=0\nsteps=999\n")  # simulate stale
    monkeypatch.setattr(
        sys, "argv", ["train.py", "--mode", "pretrain", "--config", str(config)]
    )
    assert train.main() == 0
    assert "steps=2" in (tmp_path / "ok" / "DONE").read_text()


def test_config_loader_keeps_unknown_keys(tmp_path):
    """The agent adds its own knobs; the loader must not drop them."""
    path = write_config(tmp_path / "cfg", {})
    cfg = yaml.safe_load(path.read_text())
    cfg["my_custom_knob"] = {"lr_schedule": "cosine"}
    path.write_text(yaml.safe_dump(cfg))
    loaded = train.load_config(path)
    assert loaded["my_custom_knob"] == {"lr_schedule": "cosine"}
    # Defaults still fill in underneath.
    assert loaded["train"]["sigma"] > 0
