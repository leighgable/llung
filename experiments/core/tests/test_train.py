"""Property tests for core/train.py (hypothesis), the sentinel contract,
and the plug-in contract (the engine knows no architecture).

The properties that MUST hold for every seed:
  - GA elitism: gen_best never worsens across generations.
  - Seeding: same config + same seed -> same metrics (modulo `wall`).
  - The sentinel contract: clean exit -> DONE, any exception -> FAILED,
    and a rerun never inherits a stale verdict.
"""

import json
import shutil
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

from conftest import TESTS, run_mode, strip_wall, write_config

import train


@pytest.mark.parametrize("mode", ["ga", "pretrain", "rl"])
def test_every_mode_logs_and_checkpoints(tmp_path, mode):
    rows = run_mode(mode, tmp_path / mode)
    assert rows, f"{mode} must log at least one row"
    # Header-first contract: the mode's columns, then the task's
    # metrics (parity defines val_acc).
    expected = ["step"] + train.MODE_COLUMNS[mode] + ["val_acc", "wall"]
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
    """The loss is non-negative, and the best genome beats the mean."""
    rows = run_mode("ga", tmp_path / f"c{seed}", {"seed": seed})
    for row in rows:
        assert 0.0 <= float(row["gen_best"]) <= float(row["gen_mean"]) + 1e-6


def test_failed_run_writes_failed_sentinel(tmp_path):
    config = write_config(tmp_path / "bad", {"task": {"factory": "nope-such-task"}})
    assert train.execute(config, mode="ga") == 1
    failed = (tmp_path / "bad" / "FAILED").read_text()
    assert "ValueError" in failed and "unknown factory" in failed
    assert not (tmp_path / "bad" / "DONE").exists()


@pytest.mark.parametrize(
    "overrides, needle",
    [
        ({"model": {"factory": None}}, "model.factory is not set"),
        ({"model": {"factory": "no_such_file.py:Net"}}, "not found"),
        ({"model": {"factory": f"{TESTS / 'toy_models.py'}:Nope"}}, "'Nope' not found"),
        ({"model": {"factory": f"{TESTS / 'toy_models.py'}:broken"}}, "not a torch.nn.Module"),
        # model and task don't fit together: caught by the up-front val forward
        ({"model": {"seq_len": 3}}, "RuntimeError"),
        ({"mode": "nope"}, "unknown mode"),
    ],
)
def test_plugin_errors_land_in_failed(tmp_path, overrides, needle):
    config = write_config(tmp_path / "bad", overrides)
    assert train.execute(config) == 1
    assert needle in (tmp_path / "bad" / "FAILED").read_text()


def test_clean_run_writes_done_and_rerun_clears_stale_sentinel(tmp_path):
    config = write_config(tmp_path / "ok", {})
    assert train.execute(config, mode="pretrain") == 0
    done = (tmp_path / "ok" / "DONE").read_text()
    assert done.startswith("exit=0"), done
    meta = json.loads((tmp_path / "ok" / "meta.json").read_text())
    assert meta["mode"] == "pretrain" and meta["device"] == "cpu"
    assert meta["n_params"] > 0
    # Provenance: the plug-in sources are snapshotted into the run dir.
    # (laid out relative to experiments/, so packages don't collide)
    assert (tmp_path / "ok" / "src" / "core" / "tests" / "toy_models.py").exists()
    assert (tmp_path / "ok" / "src" / "core" / "tasks.py").exists()

    # A rerun of the same run dir must not inherit the old verdict:
    # DONE is cleared at start, then written again on success.
    (tmp_path / "ok" / "DONE").write_text("exit=0\nsteps=999\n")  # simulate stale
    assert train.execute(config, mode="pretrain") == 0
    assert "steps=2" in (tmp_path / "ok" / "DONE").read_text()


def test_relative_factory_resolves_against_the_config_dir(tmp_path):
    run = tmp_path / "rel"
    run.mkdir()
    shutil.copy(TESTS / "toy_models.py", run / "mynet.py")
    rows = run_mode("pretrain", run, {"model": {"factory": "mynet.py:MLP"}})
    assert len(rows) == 2


def test_dotted_module_factory():
    obj, source = train.load_object("collections:OrderedDict", [])
    assert obj.__name__ == "OrderedDict" and source is not None


def test_float_targets_use_mse(tmp_path):
    """A task with float targets -> default loss is MSE, no metrics needed."""
    (tmp_path / "reg").mkdir()
    (tmp_path / "reg" / "regress.py").write_text(
        "import torch\n"
        "class Sum:\n"
        "    def __init__(self, seed, seq_len=6, val_size=32):\n"
        "        self.seq_len = seq_len\n"
        "        self.val = self.sample(val_size, -1)\n"
        "    def sample(self, n, step):\n"
        "        x = torch.rand(n, self.seq_len, 1, generator=torch.Generator().manual_seed(step + 7))\n"
        "        return x, x.sum(dim=(1, 2)).unsqueeze(-1)\n"
    )
    rows = run_mode("pretrain", tmp_path / "reg",
                    {"task": {"factory": "regress.py:Sum"},
                     "model": {"n_classes": 1}})
    assert "val_acc" not in rows[0]
    assert float(rows[-1]["val_loss"]) >= 0


def test_distill_rebuilds_teacher_from_its_snapshot(tmp_path):
    run_mode("pretrain", tmp_path / "teacher", {"model": {"hidden": 16}})
    rows = run_mode(
        "distill", tmp_path / "student",
        {"distill": {"teacher_checkpoint": str(tmp_path / "teacher/checkpoints/latest.pt"), "alpha": 0.5}},
    )
    assert all(float(r["distill_loss"]) >= 0 for r in rows)


def test_check_reports_shapes_and_writes_nothing(tmp_path):
    config = write_config(tmp_path / "chk", {})
    report = train.check(config)
    assert "float32[32, 6, 1]" in report and "int64[32]" in report
    assert "output:  float32[32, 2]" in report and report.endswith("ok")
    assert sorted(p.name for p in (tmp_path / "chk").iterdir()) == ["config.yaml"]


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
