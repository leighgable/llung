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


# ------------------------------------------------- configurable monitor


import math  # noqa: E402

import yaml  # noqa: E402


def with_monitor(run: Path, section) -> Path:
    (run / "config.yaml").write_text(yaml.safe_dump({"seed": 0, "monitor": section}))
    return run


def acc_rows(values, **extra):
    return [{"step": str(i + 1), "val_loss": "1.0", "val_acc": repr(v), "grad_norm": "1.0",
             **{k: repr(vs[i]) for k, vs in extra.items()}, "wall": str(i)} for i, v in enumerate(values)]


def test_watch_a_max_metric(tmp_path):
    run = write_run(tmp_path / "up", acc_rows([0.1, 0.2, 0.3, 0.4, 0.5]), "DONE")
    with_monitor(run, {"watch": [{"metric": "val_acc", "goal": "max", "patience": 3}]})
    out = monitor.summarize(run)
    assert "watch: val_acc (max) last 0.5, best 0.5 at step 5" in out
    assert "ALERT" not in out, "improving: no alert -- and the flat val_loss is no longer watched"
    stuck = write_run(tmp_path / "stuck", acc_rows([0.1, 0.5, 0.4, 0.45, 0.5]), "DONE")
    with_monitor(stuck, {"watch": [{"metric": "val_acc", "goal": "max", "patience": 3}]})
    assert "ALERT: val_acc plateaued for 3 evals (best so far 0.500000, goal max)" in monitor.summarize(stuck)


def test_min_delta_and_patience(tmp_path):
    rows = acc_rows([0.10, 0.20, 0.205, 0.21, 0.215])
    run = with_monitor(write_run(tmp_path / "d", rows), {"watch": [
        {"metric": "val_acc", "goal": "max", "patience": 3, "min_delta": 0.05}]})
    assert "plateaued" in monitor.summarize(run), "gains below min_delta don't count"
    run2 = with_monitor(write_run(tmp_path / "p", rows), {"watch": [
        {"metric": "val_acc", "goal": "max", "patience": 5, "min_delta": 0.05}]})
    assert "plateaued" not in monitor.summarize(run2), "not enough evals for patience 5"


def test_non_finite_and_missing_columns(tmp_path):
    run = write_run(tmp_path / "nan", acc_rows([0.1, 0.2, float("nan")]))
    with_monitor(run, {"watch": ["val_acc", {"metric": "val_acc_k4", "goal": "max"}]})
    out = monitor.summarize(run)
    assert "ALERT: val_acc is not finite (nan) at step 3" in out
    assert "NOTE: monitor watches 'val_acc_k4' but metrics.csv has no such column" in out


def test_configured_spikes(tmp_path):
    rows = acc_rows([0.1] * 6, train_answer=[1, 1, 1, 1, 1, 6])
    run = with_monitor(write_run(tmp_path / "s", rows), {"spikes": [{"metric": "train_answer", "ratio": 5}]})
    out = monitor.summarize(run)
    assert "ALERT: train_answer spike >5x EMA" in out
    with_monitor(run, {"spikes": []})
    assert "spike" not in monitor.summarize(run)


def test_show_narrows_the_table(tmp_path):
    run = with_monitor(write_run(tmp_path / "w", acc_rows([0.1, 0.2])), {"show": ["val_acc", "nope"]})
    out = monitor.summarize(run)
    header = next(line for line in out.splitlines() if line.startswith("step"))
    assert header.split() == ["step", "val_acc", "wall"]
    assert "(3 of 5 columns; monitor.show)" in out and "show lists columns metrics.csv does not have: nope" in out


def test_legacy_watches_without_a_monitor_section(tmp_path):
    run = write_run(tmp_path / "l", flat_rows(8), "DONE")
    with_monitor(run, {"watch": []})
    assert "ALERT: val_loss plateaued for 3 evals" in monitor.summarize(run)


JUNK = st.recursive(
    st.one_of(st.none(), st.booleans(), st.integers(-3, 5), st.floats(allow_nan=True), st.text(max_size=8)),
    lambda inner: st.one_of(st.lists(inner, max_size=3), st.dictionaries(
        st.sampled_from(["metric", "goal", "patience", "min_delta", "ratio", "watch", "spikes", "show", "x"]),
        inner, max_size=4)),
    max_leaves=12,
)


@given(section=JUNK)
@settings(max_examples=150, **PROPS)
def test_any_monitor_section_is_survivable(tmp_path, section):
    """A malformed monitor section is reported, never fatal."""
    run = write_run(tmp_path / "junk", acc_rows([0.1, 0.2, 0.3, 0.3, 0.3]), "DONE")
    (run / "config.yaml").write_text(yaml.safe_dump({"monitor": section}))
    out = monitor.summarize(run)
    assert out.startswith("run: junk") and "run finished: DONE" in out


def test_target_reached_replaces_the_plateau_alert(tmp_path):
    run = write_run(tmp_path / "t", acc_rows([0.5, 1.0, 1.0, 1.0, 1.0]), "DONE")
    with_monitor(run, {"watch": [{"metric": "val_acc", "goal": "max", "target": 0.99}]})
    out = monitor.summarize(run)
    assert "best 1 at step 2 — reached its target 0.99" in out and "plateaued" not in out
    with_monitor(run, {"watch": [{"metric": "val_acc", "goal": "max", "target": "high"}]})
    assert "monitor config: watch[0] val_acc: patience / min_delta / target must be numbers" in monitor.summarize(run)
    low = write_run(tmp_path / "low", acc_rows([0.5, 0.6, 0.6, 0.6, 0.6]), "DONE")
    with_monitor(low, {"watch": [{"metric": "val_acc", "goal": "max", "target": 0.99}]})
    out = monitor.summarize(low)
    assert ", target 0.99" in out and "ALERT: val_acc plateaued" in out
