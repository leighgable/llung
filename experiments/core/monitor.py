#!/usr/bin/env python3
"""core/monitor.py -- the agent's observability (plan 1.2).

Text in, text out: the agent calls summarize() from a torch repl
session and reads the string -- it never talks to TensorBoard (that is
the human's browser tool):

    import sys; sys.path.insert(0, "core"); import monitor
    print(monitor.summarize("runs/exp07"))

Alerts (heuristics, not oracles):
    - "run finished: DONE <exit code>" / "run finished: FAILED <last line>"
    - "val_loss plateaued for 3 evals"   (no new best in the last 3)
    - "grad_norm spike >10x EMA"         (last step vs its running EMA)
    - "gen_best stagnant for 5 gens"     (GA runs)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PLATEAU_EVALS = 3
SPIKE_RATIO = 10.0
STAGNANT_GENS = 5


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


def _alerts(rows: list[dict]) -> list[str]:
    alerts = []
    val = _floats(rows, "val_loss")
    if len(val) > PLATEAU_EVALS and min(val[-PLATEAU_EVALS:]) >= min(val[:-PLATEAU_EVALS]) - 1e-4:
        alerts.append(
            f"val_loss plateaued for {PLATEAU_EVALS} evals (best so far {min(val):.6f})"
        )
    grad = _floats(rows, "grad_norm")
    if len(grad) >= 5:
        ema = grad[0]
        for g in grad[1:-1]:
            ema = 0.9 * ema + 0.1 * g
        if grad[-1] > SPIKE_RATIO * max(ema, 1e-12) and grad[-1] > 1e-6:
            alerts.append(f"grad_norm spike >{SPIKE_RATIO:.0f}x EMA ({grad[-1]:.4g} vs EMA {ema:.4g})")
    best = _floats(rows, "gen_best")
    if len(best) > STAGNANT_GENS and min(best[-STAGNANT_GENS:]) >= min(best[:-STAGNANT_GENS]) - 1e-9:
        alerts.append(f"gen_best stagnant for {STAGNANT_GENS} gens (best {min(best):.6f})")
    return alerts


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
    failed, done = run_dir / "FAILED", run_dir / "DONE"
    if failed.exists():
        text = failed.read_text().strip().splitlines()
        last = text[-1] if text else "(empty FAILED file)"
        lines.append(f"run finished: FAILED — {last}")
    elif done.exists():
        first = done.read_text().strip().splitlines()[0]
        lines.append(f"run finished: DONE ({first})")
    else:
        lines.append("status: running (no DONE/FAILED sentinel yet)")

    rows = _rows(run_dir)
    if not rows:
        lines.append("no metrics.csv yet — the run has not logged a step")
        return "\n".join(lines)
    for alert in _alerts(rows):
        lines.append(f"ALERT: {alert}")
    fields = list(rows[0].keys())
    lines.append(f"metrics.csv — last {min(tail, len(rows))} of {len(rows)} rows:")
    lines.extend(_table(rows, fields, tail))
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    print(summarize(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20))
