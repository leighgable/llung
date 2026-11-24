"""Property tests for core/monitor.py."""

import csv
import sys
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

# tmp_path is shared between generated examples by design: every
# example writes to a seed-namespaced subdir, so nothing collides.
PROPS = {
    "deadline": None,
    "derandomize": True,
    "suppress_health_check": [HealthCheck.function_scoped_fixture],
}

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import monitor  # noqa: E402


def write_run(run: Path, rows: list[dict], sentinel: str | None = None) -> Path:
    run.mkdir(parents=True, exist_ok=True)
    if rows:
        with (run / "metrics.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    if sentinel:
        (run / sentinel).write_text("exit=0\n" if sentinel == "DONE" else "ValueError: boom\n")
    return run


def flat_rows(n: int, val: float = 0.25, grad: float = 1.0) -> list[dict]:
    return [
        {"step": str(i), "epoch": str(i), "train_loss": "0.5", "val_loss": str(val),
         "grad_norm": str(grad), "lr": "0.001", "wall": str(i)}
        for i in range(n)
    ]


def test_plateau_alert_fires_on_constant_val_loss(tmp_path):
    run = write_run(tmp_path / "r", flat_rows(8), "DONE")
    out = monitor.summarize(run)
    assert "run finished: DONE" in out
    assert "ALERT: val_loss plateaued for 3 evals" in out


def test_grad_spike_alert_fires(tmp_path):
    rows = flat_rows(8, grad=1.0)
    rows[-1]["grad_norm"] = "50.0"
    run = write_run(tmp_path / "r", rows, "DONE")
    assert "grad_norm spike >10x EMA" in monitor.summarize(run)


def test_finished_states_are_reported(tmp_path):
    running = monitor.summarize(write_run(tmp_path / "a", flat_rows(2)))
    assert "status: running" in running
    failed = monitor.summarize(write_run(tmp_path / "b", [], "FAILED"))
    assert "run finished: FAILED" in failed and "boom" in failed


def test_missing_metrics_is_not_an_error(tmp_path):
    out = monitor.summarize(write_run(tmp_path / "empty", []))
    assert "no metrics.csv yet" in out


@given(
    n=st.integers(0, 30),
    vals=st.lists(st.floats(min_value=0, max_value=100), min_size=0, max_size=30),
    sentinel=st.sampled_from([None, "DONE", "FAILED"]),
)
@settings(max_examples=25, **PROPS)
def test_summarize_never_crashes_and_reports_status(tmp_path, n, vals, sentinel):
    """Text in, text out — for ANY csv shape the agent might point it at."""
    rows = flat_rows(max(n, 4))
    for row, v in zip(rows, vals):
        row["val_loss"] = repr(v)
    run = write_run(tmp_path / f"r{n}-{sentinel}", rows, sentinel)
    out = monitor.summarize(run, tail=5)
    assert isinstance(out, str) and out
    if sentinel:
        assert f"run finished: {sentinel}" in out
    else:
        assert "status: running" in out
    # The table shows at most `tail` body rows.
    assert "( earlier rows omitted)" not in out or n + 1 > 5


@given(tail=st.integers(1, 50))
@settings(max_examples=10, **PROPS)
def test_tail_argument_is_respected(tmp_path, tail):
    run = write_run(tmp_path / f"t{tail}", flat_rows(60), "DONE")
    out = monitor.summarize(run, tail=tail)
    body = out.splitlines()
    start = body.index(next(l for l in body if l.startswith("step")))
    data_rows = [l for l in body[start + 2 :] if l and not l.startswith("(")]
    assert len(data_rows) == min(tail, 60)
