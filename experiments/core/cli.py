#!/usr/bin/env python3
"""core/cli.py -- the experiment harness CLI (typer).

One entry point for humans and agents:

    python core/cli.py new exp07 --model models/mynet.py:Net --set model.hidden=64
    python core/cli.py check exp07          # build + one forward/backward, no writes
    python core/cli.py launch exp07         # detached; returns immediately
    python core/cli.py status exp07         # last metrics rows + alerts
    python core/cli.py ls                   # every run and its state
    python core/cli.py stop exp07           # SIGTERM -> FAILED "stopped"

A RUN is a run id (resolved under the runs directory -- the same one
the app uses, see runs_dir()), a run directory, or a config.yaml path. The run
directory is always the config's parent. The engine (and the plug-in
contract for models and tasks) is core/train.py.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
import yaml

CORE = Path(__file__).resolve().parent
sys.path.insert(0, str(CORE))

import monitor  # noqa: E402  (torch-free; `train` is imported lazily)

app = typer.Typer(
    help="Experiment harness: scaffold, check, launch and monitor training runs.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)


class Mode(str, Enum):
    pretrain = "pretrain"
    distill = "distill"
    ga = "ga"
    rl = "rl"
    vpo = "vpo"


RunArg = Annotated[str, typer.Argument(help="Run id, run directory, or config.yaml path.")]
ModeOpt = Annotated[Mode | None, typer.Option(help="Override the config's mode.")]
DeviceOpt = Annotated[str | None, typer.Option(help="Override the config's device (auto | cpu | cuda[:N]).")]


def runs_dir() -> Path:
    """Same resolution as the app (crates/agent/src/places.rs), so host
    and agent see the same runs: $LLUNG_RUNS_DIR (always set inside the
    sandboxes) > places.json "runs" (saved by `/runs` in the tui) >
    $LLUNG_DATA_DIR/runs > $XDG_DATA_HOME/llung/runs > ~/.local/share/llung/runs."""
    env = os.environ.get("LLUNG_RUNS_DIR")
    if env:
        return Path(env).expanduser()
    config = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    try:
        saved = json.loads((config / "llung" / "places.json").read_text()).get("runs")
        if saved:
            return Path(saved).expanduser()
    except (OSError, ValueError):
        pass
    data = os.environ.get("LLUNG_DATA_DIR")
    if data:
        return Path(data).expanduser() / "runs"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share") / "llung" / "runs"


def run_path(run: str) -> Path:
    """Run id | run dir | config path -> run dir."""
    p = Path(run).expanduser()
    if p.suffix in (".yaml", ".yml"):
        return p.resolve().parent
    if p.is_absolute() or os.sep in run or p.exists():
        return p.resolve()
    return runs_dir() / run


def config_path(run: str) -> Path:
    p = Path(run).expanduser()
    cfg = p.resolve() if p.suffix in (".yaml", ".yml") else run_path(run) / "config.yaml"
    if not cfg.is_file():
        fail(f"no config at {cfg} (create one with: cli.py new {run} --model file.py:Factory)")
    return cfg


def fail(msg: str, code: int = 1):
    typer.echo(f"error: {msg}", err=True)
    raise typer.Exit(code)


def _absolutize(spec: str) -> str:
    """Store a factory file given relative to the cwd as an absolute
    path, so it still resolves from the run dir."""
    where, sep, attr = spec.rpartition(":")
    if sep and where.endswith(".py") and not Path(where).is_absolute() and Path(where).is_file():
        return f"{Path(where).resolve()}:{attr}"
    return spec


def _set(cfg: dict, assignment: str) -> None:
    key, eq, raw = assignment.partition("=")
    if not eq or not key:
        fail(f"--set expects key=value (e.g. train.lr=3e-4), got '{assignment}'")
    *parents, leaf = key.split(".")
    node = cfg
    for part in parents:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            fail(f"--set {key}: '{part}' is not a section")
    value = yaml.safe_load(raw)
    if isinstance(value, str):  # YAML 1.1 reads "3e-4" as a string
        try:
            value = float(value)
        except ValueError:
            pass
    node[leaf] = value


@app.command()
def new(
    run: RunArg,
    model: Annotated[str, typer.Option(help="Model factory: 'file.py:Name' or 'pkg.module:Name'.")],
    task: Annotated[str, typer.Option(help="Task factory: built-in (parity, lastbit, tokens) or 'file.py:Name'.")] = "parity",
    mode: Annotated[Mode, typer.Option(help="Training mode.")] = Mode.pretrain,
    set_: Annotated[
        list[str] | None,
        typer.Option("--set", "-s", help="Override a config value: section.key=value (YAML-parsed). Repeatable."),
    ] = None,
    force: Annotated[bool, typer.Option(help="Overwrite an existing config.")] = False,
):
    """Write RUN/config.yaml with every default spelled out (edit it after)."""
    import train

    run_dir = run_path(run)
    cfg_file = run_dir / "config.yaml"
    if monitor.run_state(run_dir) in ("RUNNING", "RUNNING?") and cfg_file.exists():
        fail(f"{run_dir.name} is running -- never edit a running run; pick a new run id")
    if cfg_file.exists() and not force:
        fail(f"{cfg_file} exists (use --force, or a new run id)")
    cfg = train.merge_config({"mode": mode.value, "model": {"factory": _absolutize(model)},
                              "task": {"factory": _absolutize(task)}})
    for assignment in set_ or []:
        _set(cfg, assignment)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_file.write_text(yaml.safe_dump(cfg, sort_keys=False))
    typer.echo(f"wrote {cfg_file}\nnext: cli.py check {run}   then: cli.py launch {run}")


@app.command()
def check(run: RunArg, mode: ModeOpt = None, device: DeviceOpt = None):
    """Build task + model and run one forward (and backward) pass. Writes nothing."""
    import traceback

    import train

    try:
        typer.echo(train.check(config_path(run), mode and mode.value, device))
    except typer.Exit:
        raise
    except Exception:
        typer.echo(traceback.format_exc().rstrip(), err=True)
        fail("check failed -- the model/task contract is in core/train.py's docstring")


@app.command()
def run(run: RunArg, mode: ModeOpt = None, device: DeviceOpt = None):
    """Run in the foreground (blocks until the run ends)."""
    import train

    raise typer.Exit(train.execute(config_path(run), mode and mode.value, device))


@app.command()
def launch(run: RunArg, mode: ModeOpt = None, device: DeviceOpt = None):
    """Start a run DETACHED (own process group, output -> train.log) and return at once."""
    cfg = config_path(run)
    run_dir = cfg.parent
    state = monitor.run_state(run_dir)
    if state in ("RUNNING", "RUNNING?"):
        fail(f"{run_dir.name} is already {state}; stop it first or use a new run id")
    argv = [sys.executable, str(Path(__file__).resolve()), "run", str(cfg)]
    if mode:
        argv += ["--mode", mode.value]
    if device:
        argv += ["--device", device]
    for stale in ("DONE", "FAILED"):  # don't let `status` race the child
        (run_dir / stale).unlink(missing_ok=True)
    with (run_dir / "train.log").open("w") as log:
        p = subprocess.Popen(argv, cwd=run_dir, stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
    (run_dir / "PID").write_text(f"{p.pid}\n{monitor._pid_namespace()}\n")
    typer.echo(f"launched {run_dir.name} (pid {p.pid}) -> {run_dir}\npoll with: cli.py status {run}")


@app.command()
def status(
    run: RunArg,
    tail: Annotated[int, typer.Option(help="Metrics rows to show.")] = 20,
):
    """Run state, alerts, and the last metrics rows."""
    run_dir = run_path(run)
    if not run_dir.is_dir():
        fail(f"no run at {run_dir}")
    typer.echo(monitor.summarize(run_dir, tail))


@app.command("ls")
def list_runs(
    directory: Annotated[Path | None, typer.Argument(help="Runs directory (default: the app's runs directory).")] = None,
):
    """List runs with their state, mode, progress and last write."""
    root = directory or runs_dir()
    runs = sorted(
        (d for d in root.glob("*") if (d / "config.yaml").exists() or (d / "metrics.csv").exists()),
        key=lambda d: (d / "config.yaml").stat().st_mtime if (d / "config.yaml").exists() else 0,
    ) if root.is_dir() else []
    if not runs:
        typer.echo(f"no runs in {root}")
        return
    table = [("run", "state", "mode", "model", "rows", "last write")]
    for d in runs:
        try:
            cfg = yaml.safe_load((d / "config.yaml").read_text()) or {}
        except (OSError, yaml.YAMLError):
            cfg = {}
        model = str((cfg.get("model") or {}).get("factory") or "-").rsplit("/", 1)[-1]
        rows = len(monitor._rows(d))
        table.append((d.name, monitor.run_state(d), str(cfg.get("mode", "-")), model,
                      str(rows), monitor.ago(monitor.last_write(d))))
    widths = [max(len(r[i]) for r in table) for i in range(len(table[0]))]
    typer.echo(f"runs: {root}")
    for row in table:
        typer.echo("  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip())


@app.command()
def stop(
    run: RunArg,
    timeout: Annotated[float, typer.Option(help="Seconds to wait before SIGKILL.")] = 10.0,
):
    """SIGTERM a running run (it writes FAILED 'stopped'); SIGKILL after --timeout."""
    run_dir = run_path(run)
    state = monitor.run_state(run_dir)
    pid, _ = monitor.read_pid(run_dir)
    if state == "RUNNING?":
        fail(f"{run_dir.name}'s pid lives in another sandbox session -- stop it from that session")
    if state != "RUNNING" or pid is None:
        fail(f"{run_dir.name} is not running (state {state})")

    def send(sig):
        try:
            if os.getpgid(pid) == pid:
                os.killpg(pid, sig)
            else:
                os.kill(pid, sig)
        except ProcessLookupError:
            pass

    send(signal.SIGTERM)
    deadline = time.time() + timeout
    while time.time() < deadline and monitor.pid_alive(pid):
        time.sleep(0.1)
    if monitor.pid_alive(pid):
        send(signal.SIGKILL)
        time.sleep(0.2)
    if not (run_dir / "DONE").exists() and not (run_dir / "FAILED").exists():
        (run_dir / "FAILED").write_text(f"killed by cli.py stop (SIGKILL after {timeout:g}s)\n")
    typer.echo(f"stopped {run_dir.name}: {monitor.run_state(run_dir)}")


if __name__ == "__main__":
    app()
