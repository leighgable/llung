"""core/cli.py: the typer front end, incl. a real detached launch."""

import os
import time

import pytest
import yaml
from typer.testing import CliRunner

import cli
from conftest import TESTS, write_config

TOY = f"{TESTS / 'toy_models.py'}:MLP"
runner = CliRunner()


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setenv("LLUNG_RUNS_DIR", str(tmp_path))
    return tmp_path


def invoke(*args):
    return runner.invoke(cli.app, list(args))


def wait_for_sentinel(run, seconds=60):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if (run / "DONE").exists() or (run / "FAILED").exists():
            return
        time.sleep(0.2)
    raise AssertionError(f"no sentinel in {run}: {(run / 'train.log').read_text()}")


def test_new_writes_full_config_with_overrides(runs):
    res = invoke("new", "exp01", "--model", TOY, "--mode", "ga",
                 "--set", "model.hidden=16", "-s", "train.lr=3e-4", "-s", "device=cpu")
    assert res.exit_code == 0, res.output
    cfg = yaml.safe_load((runs / "exp01" / "config.yaml").read_text())
    assert cfg["mode"] == "ga" and cfg["model"] == {"factory": TOY, "hidden": 16}
    assert cfg["train"]["lr"] == 3e-4 and cfg["train"]["pop_size"] == 50  # defaults spelled out
    again = invoke("new", "exp01", "--model", TOY)
    assert again.exit_code == 1 and "exists" in again.output


def test_check_and_foreground_run_and_status(runs):
    write_config(runs / "exp02", {})
    res = invoke("check", "exp02")
    assert res.exit_code == 0 and "output:  float32[32, 2]" in res.output, res.output
    res = invoke("run", "exp02", "--mode", "ga")
    assert res.exit_code == 0, res.output
    res = invoke("status", "exp02", "--tail", "2")
    assert "run finished: DONE" in res.output and "gen_best" in res.output
    res = invoke("ls")
    assert "exp02" in res.output and "DONE" in res.output


def test_check_failure_prints_traceback(runs):
    write_config(runs / "bad", {"model": {"seq_len": 3}})
    res = invoke("check", "bad")
    assert res.exit_code == 1 and "check failed" in res.output and "Traceback" in res.output


def test_missing_config_is_a_clear_error(runs):
    res = invoke("launch", "nope")
    assert res.exit_code == 1 and "no config" in res.output


def test_launch_detaches_and_finishes(runs):
    write_config(runs / "exp03", {})
    res = invoke("launch", "exp03")
    assert res.exit_code == 0, res.output
    assert "launched exp03" in res.output
    wait_for_sentinel(runs / "exp03")
    assert (runs / "exp03" / "DONE").exists(), (runs / "exp03" / "FAILED").read_text()


def test_stop_writes_failed(runs):
    write_config(runs / "long", {"train": {"epochs": 100_000, "checkpoint_every": 100_000}})
    assert invoke("launch", "long").exit_code == 0
    deadline = time.time() + 60  # wait until the trainer is in its loop
    while not (runs / "long" / "metrics.csv").exists() and time.time() < deadline:
        time.sleep(0.2)
    assert cli.monitor.run_state(runs / "long") == "RUNNING"
    res = invoke("stop", "long")
    assert res.exit_code == 0, res.output
    failed = (runs / "long" / "FAILED").read_text()
    assert "Stopped" in failed or "killed" in failed
    assert invoke("stop", "long").exit_code == 1  # not running any more


def test_dead_run_is_detected(runs):
    run = runs / "ghost"
    write_config(run, {})
    (run / "metrics.csv").write_text("step,val_loss\n1,0.5\n")
    (run / "PID").write_text(f"999999999\n{cli.monitor._pid_namespace()}\n")
    assert cli.monitor.run_state(run) == "DEAD"
    assert "status: DEAD" in invoke("status", "ghost").output
    (run / "PID").write_text("999999999\npid:[1]\n")  # another sandbox session
    assert cli.monitor.run_state(run) == "RUNNING?"
