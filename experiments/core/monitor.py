#!/usr/bin/env python3
"""core/monitor.py -- the agent's observability (plan 1.2).

Text in, text out, no torch import (fast). Usually reached through the
CLI -- `python core/cli.py status <run>` / `ls` -- but importable too:

    import sys; sys.path.insert(0, "<workspace>/experiments/core"); import monitor
    print(monitor.summarize("<runs dir>/exp07"))

TensorBoard is the human's browser tool; the agent never talks to it.

Run states (run_state): DONE, FAILED, RUNNING (pid alive), DEAD (pid gone
without a sentinel -- SIGKILL/OOM, or its sandbox session closed),
RUNNING? (launched from another sandbox session: its pid is in another
pid namespace, so liveness is unknown -- judge by "last write"), NEW
(config only).

Alerts (heuristics, not oracles), configured per run in config.yaml --
objectives can be weird (MUX: a weighted sum of moving parts), so each
experiment says what progress means:

    monitor:
      watch:                       # plateau + non-finite alerts, "last vs best" lines
        - {metric: val_acc_k2, goal: max, patience: 3, min_delta: 0.01}
        - {metric: val_ref, goal: min}          # patience 3, min_delta 0 by default
        - {metric: val_trace_acc, goal: max, target: 0.99}  # reached -> no plateau alert
      spikes: [grad_norm]          # or {metric: grad_norm, ratio: 10}: last > ratio x EMA
      show: [val_acc_k2, val_ref]  # optional: only these columns in the table

    - "run finished: DONE <exit code>" / "run finished: FAILED <last line>"
    - "<metric> plateaued for <patience> evals"  (no new best by min_delta,
                                                  unless its target was reached)
    - "<metric> reached its target"              (in the watch line, not an alert)
    - "<metric> is not finite"                   (nan / inf in the last row)
    - "<spike metric> spike ><ratio>x EMA"
    - NOTE when a watched column does not exist (a typo silences an alert)

Without monitor.watch the legacy watches apply, silently skipped when
the column is absent: val_loss (min, patience 3, min_delta 1e-4) and
gen_best (GA; min, patience 5). A malformed monitor section never breaks
summarize(): it is reported as a "monitor config:" line.
"""

from __future__ import annotations

import csv
import math
import os
import sys
import time
from pathlib import Path

PLATEAU_EVALS = 3
SPIKE_RATIO = 10.0
STAGNANT_GENS = 5

# What `cli.py new` writes (via train.DEFAULTS["monitor"]): an empty watch
# list = the legacy watches below.
DEFAULT_MONITOR = {"watch": [], "spikes": ["grad_norm"], "show": []}
LEGACY_WATCH = [
    {"metric": "val_loss", "goal": "min", "patience": PLATEAU_EVALS, "min_delta": 1e-4, "optional": True},
    {"metric": "gen_best", "goal": "min", "patience": STAGNANT_GENS, "min_delta": 1e-9, "optional": True},
]


def _pid_namespace() -> str:
    try:
        return os.readlink("/proc/self/ns/pid")
    except OSError:
        return ""


def read_pid(run_dir) -> tuple[int | None, str]:
    """PID file -> (pid, pid namespace). Old single-line files have no ns."""
    try:
        lines = (Path(run_dir) / "PID").read_text().split()
        return int(lines[0]), (lines[1] if len(lines) > 1 else "")
    except (OSError, ValueError, IndexError):
        return None, ""


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # an unreaped zombie has finished
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat.rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return True


def run_state(run_dir) -> str:
    run_dir = Path(run_dir)
    if (run_dir / "FAILED").exists():
        return "FAILED"
    if (run_dir / "DONE").exists():
        return "DONE"
    pid, ns = read_pid(run_dir)
    if pid is not None:
        if ns and ns != _pid_namespace():
            return "RUNNING?"
        return "RUNNING" if pid_alive(pid) else "DEAD"
    return "RUNNING?" if (run_dir / "metrics.csv").exists() else "NEW"


def last_write(run_dir) -> float | None:
    """Seconds since the run last wrote metrics.csv or train.log."""
    times = [
        (Path(run_dir) / f).stat().st_mtime
        for f in ("metrics.csv", "train.log")
        if (Path(run_dir) / f).exists()
    ]
    return time.time() - max(times) if times else None


def ago(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= size:
            return f"{seconds / size:.0f}{unit} ago"
    return f"{seconds:.0f}s ago"


def _log_tail(run_dir: Path, n: int = 5) -> list[str]:
    try:
        return (run_dir / "train.log").read_text(errors="replace").strip().splitlines()[-n:]
    except OSError:
        return []


def _rows(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics.csv"
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _floats(rows: list[dict], col: str) -> list[float]:
    out = []
    for row in rows:
        try:
            out.append(float(row[col]))
        except (KeyError, TypeError, ValueError):
            pass
    return out


def parse_monitor(section) -> tuple[dict, list[str]]:
    """config.yaml `monitor:` section -> (normalized config, problems).
    Never raises: bad entries are dropped and reported."""
    errors: list[str] = []
    out = {"watch": [], "spikes": [], "show": []}
    if section is None:
        section = {}
    if not isinstance(section, dict):
        return {**out, "spikes": [{"metric": "grad_norm", "ratio": SPIKE_RATIO}]}, [
            f"monitor must be a mapping, got {type(section).__name__}"
        ]
    watch = section.get("watch") or []
    if not isinstance(watch, list):
        errors.append(f"watch: expected a list of metrics, got {watch!r}")
        watch = []
    for i, w in enumerate(watch):
        if isinstance(w, str):
            w = {"metric": w}
        if not isinstance(w, dict) or not isinstance(w.get("metric"), str):
            errors.append(f"watch[{i}]: needs a 'metric' name, got {w!r}")
            continue
        goal = str(w.get("goal", "min"))
        if goal not in ("min", "max"):
            errors.append(f"watch[{i}] {w['metric']}: goal must be min or max, got {goal!r}")
            continue
        try:
            patience, min_delta = int(w.get("patience", PLATEAU_EVALS)), float(w.get("min_delta", 0.0))
            target = None if w.get("target") is None else float(w["target"])
        except (TypeError, ValueError):
            errors.append(f"watch[{i}] {w['metric']}: patience / min_delta / target must be numbers")
            continue
        if patience < 1:
            errors.append(f"watch[{i}] {w['metric']}: patience must be >= 1")
            continue
        out["watch"].append({"metric": w["metric"], "goal": goal, "patience": patience, "min_delta": min_delta,
                             "target": target, "optional": bool(w.get("optional", False))})
    spikes = section.get("spikes", ["grad_norm"])
    for i, sp in enumerate(spikes if isinstance(spikes, list) else [spikes]):
        if isinstance(sp, str):
            sp = {"metric": sp}
        try:
            out["spikes"].append({"metric": str(sp["metric"]), "ratio": float(sp.get("ratio", SPIKE_RATIO))})
        except (TypeError, KeyError, ValueError, AttributeError):
            errors.append(f"spikes[{i}]: expected a metric name or {{metric, ratio}}, got {sp!r}")
    show = section.get("show") or []
    if isinstance(show, list) and all(isinstance(c, str) for c in show):
        out["show"] = show
    else:
        errors.append(f"show: expected a list of column names, got {show!r}")
    return out, errors


def load_monitor(run_dir) -> tuple[dict, list[str]]:
    """The monitor config of a run (its config.yaml), normalized."""
    path = Path(run_dir) / "config.yaml"
    try:
        import yaml

        raw = yaml.safe_load(path.read_text()) if path.exists() else {}
    except Exception as e:  # unreadable config: legacy watches, say why
        return parse_monitor(None)[0], [f"config.yaml unreadable ({type(e).__name__}: {e})"]
    return parse_monitor((raw or {}).get("monitor") if isinstance(raw, dict) else None)


def watched(cfg: dict) -> list[dict]:
    return cfg["watch"] or LEGACY_WATCH


def _best(values: list[float], goal: str) -> float:
    return max(values) if goal == "max" else min(values)


def _watch_report(rows: list[dict], cfg: dict) -> tuple[list[str], list[str], list[str]]:
    """(alerts, notes, summary lines) for the watched / spike metrics."""
    alerts, notes, summary = [], [], []
    columns = set(rows[0].keys()) if rows else set()
    for w in watched(cfg):
        m, goal, patience, delta = w["metric"], w["goal"], w["patience"], w["min_delta"]
        if m not in columns:
            if not w.get("optional"):
                notes.append(f"monitor watches '{m}' but metrics.csv has no such column")
            continue
        raw = [(r.get("step", "?"), r.get(m)) for r in rows]
        pts = [(step, float(v)) for step, v in raw if _is_float(v)]
        if not pts:
            continue
        last_step, last = pts[-1]
        if not math.isfinite(last):
            alerts.append(f"{m} is not finite ({last}) at step {last_step}")
            continue
        finite = [(s_, v) for s_, v in pts if math.isfinite(v)]
        values = [v for _, v in finite]
        best = _best(values, goal)
        best_step = next(s_ for s_, v in finite if v == best)
        target = w.get("target")
        reached = target is not None and (best >= target if goal == "max" else best <= target)
        if not w.get("optional"):
            note = f" — reached its target {target:g}" if reached else (f", target {target:g}" if target is not None else "")
            summary.append(f"watch: {m} ({goal}) last {last:.6g}, best {best:.6g} at step {best_step}{note}")
        if len(values) > patience and not reached:
            recent, earlier = values[-patience:], values[:-patience]
            stuck = (
                min(recent) >= min(earlier) - delta if goal == "min" else max(recent) <= max(earlier) + delta
            )
            if stuck:
                alerts.append(f"{m} plateaued for {patience} evals (best so far {best:.6f}, goal {goal})")
    for sp in cfg["spikes"]:
        vals = [v for v in _floats(rows, sp["metric"]) if math.isfinite(v)]
        if len(vals) >= 5:
            ema = vals[0]
            for v in vals[1:-1]:
                ema = 0.9 * ema + 0.1 * v
            if vals[-1] > sp["ratio"] * max(ema, 1e-12) and vals[-1] > 1e-6:
                alerts.append(
                    f"{sp['metric']} spike >{sp['ratio']:.0f}x EMA ({vals[-1]:.4g} vs EMA {ema:.4g})"
                )
    return alerts, notes, summary


def _is_float(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _alerts(rows: list[dict], cfg: dict | None = None) -> list[str]:
    """Alert strings for rows under a monitor config (default: legacy)."""
    return _watch_report(rows, cfg or parse_monitor(None)[0])[0]


def _table(rows: list[dict], fields: list[str], tail: int) -> list[str]:
    shown = rows[-tail:]
    widths = {f: max(len(f), *(len(r.get(f, "")) for r in shown)) for f in fields}
    line = "  ".join(f.ljust(widths[f]) for f in fields)
    sep = "  ".join("-" * widths[f] for f in fields)
    body = ["  ".join(r.get(f, "").ljust(widths[f]) for f in fields) for r in shown]
    note = [f"({len(rows) - tail} earlier rows omitted)"] if len(rows) > tail else []
    return [line, sep, *body, *note]


def summarize(run_dir, tail: int = 20) -> str:
    """Last-K metrics rows + alert strings for one run directory."""
    run_dir = Path(run_dir)
    lines = [f"run: {run_dir.name}"]
    state = run_state(run_dir)
    if state == "FAILED":
        text = (run_dir / "FAILED").read_text().strip().splitlines()
        last = text[-1] if text else "(empty FAILED file)"
        lines.append(f"run finished: FAILED — {last}")
    elif state == "DONE":
        text = (run_dir / "DONE").read_text().strip().splitlines()
        lines.append(f"run finished: DONE ({text[0] if text else 'exit=0'})")
    elif state == "DEAD":
        pid, _ = read_pid(run_dir)
        lines.append(
            f"status: DEAD — pid {pid} is gone but left no DONE/FAILED "
            "(SIGKILL? out of memory? its sandbox session closed?). train.log tail:"
        )
        lines.extend(f"  | {line}" for line in _log_tail(run_dir))
    elif state == "RUNNING":
        lines.append(f"status: running (pid {read_pid(run_dir)[0]}, last write {ago(last_write(run_dir))})")
    elif state == "RUNNING?":
        lines.append(
            "status: running? (no DONE/FAILED sentinel yet; liveness unknown — the pid "
            "is in another sandbox session, or there is no PID file — "
            f"last write {ago(last_write(run_dir))})"
        )
    else:
        lines.append("status: not started (config only)")

    cfg, errors = load_monitor(run_dir)
    lines.extend(f"monitor config: {e}" for e in errors)
    rows = _rows(run_dir)
    if not rows:
        lines.append("no metrics.csv yet — the run has not logged a step")
        return "\n".join(lines)
    alerts, notes, summary = _watch_report(rows, cfg)
    lines.extend(summary)
    lines.extend(f"ALERT: {a}" for a in alerts)
    lines.extend(f"NOTE: {n}" for n in notes)
    fields = list(rows[0].keys())
    if cfg["show"]:
        keep = [f for f in fields if f in ("step", "wall") or f in cfg["show"]]
        missing = [c for c in cfg["show"] if c not in fields]
        if missing:
            lines.append(f"NOTE: show lists columns metrics.csv does not have: {', '.join(missing)}")
        fields = keep if len(keep) > 2 else fields
    lines.append(f"metrics.csv — last {min(tail, len(rows))} of {len(rows)} rows"
                 + (f" ({len(fields)} of {len(rows[0])} columns; monitor.show)" if len(fields) < len(rows[0]) else "") + ":")
    lines.extend(_table(rows, fields, tail))
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit("monitor.py is a library; use the CLI:  python core/cli.py status <run>")
