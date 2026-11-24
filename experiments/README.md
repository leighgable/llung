# experiments/

ML research harness (RESEARCH-HARNESS-PLAN.md, Phase 1). Everything here
is Python and agent-writable; Rust knows processes, files, grants, and
deadlines — never "training runs".

- `core/train.py` — the ONE mode-agnostic driver
  (`--mode {ga,pretrain,distill,rl}`). Loop shape for every mode:
  init → loop { step → log → checkpoint } → finish. Patch it per
  experiment; keep `metrics.csv` + the sentinels intact — they are the
  contract with the agent.
- `core/monitor.py` — text-in/text-out observability for the agent
  (`summarize(run_dir, tail=20)`). TensorBoard is the human's tool;
  run it on the host against `runs/`.
- `runs/<run-id>/` — per-run data: `config.yaml` (written with the
  agent's `edit` tool), `metrics.csv`, `events.out.tfevents.*`,
  `checkpoints/`, `meta.json`, `train.log`, and on exit exactly one of
  `DONE` / `FAILED` (the IPC contract — poll, don't ask).

## Testing (pytest + hypothesis)

`core/tests/` holds the trainer's property suite — run it from a torch
session (pytest lives in the torch env; experiments/ is the only
writable mount, which hypothesis's example DB needs):

```python
import subprocess, sys
subprocess.run([sys.executable, "-m", "pytest", "core/tests", "-q"])
```

or on the host: `torch-python -m pytest core/tests -q` from
`experiments/`. The suite pins the properties that must hold for EVERY
seed: GA elitism (gen_best never worsens), run reproducibility (same
seed -> same metrics), the DONE/FAILED sentinel contract, and
monitor.py's text-in/text-out robustness. Add a property test whenever
you patch `train.py` — the agent will, too.

## Streaming external data (teacher traces)

Data too big for the workspace — e.g. TBs of precomputed teacher traces
on an external drive — rides in via an operator-level read-only bind:

```bash
export LLUNG_RO_BINDS=/mnt/teacher-traces:/mnt/another-shard
# ... then launch the TUI; every repl session sees the paths,
# READ-ONLY, at native bind-mount speed.
```

Rules:
- The paths must be OUTSIDE /tmp (the sandbox mounts a fresh tmpfs over
  /tmp, which shadows any bind underneath it). Mounted drives under
  /mnt or /media are fine, including FUSE filesystems (exFAT/NTFS-3g) —
  decrypt LUKS volumes host-side first.
- Read-only on purpose: the trainer streams shards but can never mutate
  the source. Derived/cached artifacts belong in the run dir
  (`experiments/runs/<id>/`, writable).
- The path inside the sandbox is the SAME absolute path, so run configs
  reference it verbatim (`teacher_traces_path: /mnt/teacher-traces/…`).

## Detached launch convention (from a `gpu:torch` REPL session)

The REPL call *launches* a background process and returns immediately;
sentinels are how the agent learns a run ended (polling a directory is
idempotent IPC). `sys.executable` is the torch interpreter itself, so
the detached child inherits the exact closure:

```python
import os, subprocess, sys
os.chdir("experiments")  # repl sessions start at the workspace root
run = "runs/exp07"
os.makedirs(run, exist_ok=True)
p = subprocess.Popen(
    [sys.executable, "core/train.py", "--mode", "ga", "--config", f"{run}/config.yaml"],
    cwd=".", stdout=open(f"{run}/train.log", "w"),
    stderr=subprocess.STDOUT, start_new_session=True)  # own process group — killable
open(f"{run}/PID", "w").write(str(p.pid))
```

Later calls in the same (or another) torch session poll and summarize:

```python
import sys; sys.path.insert(0, "core"); import monitor, os
print(monitor.summarize("runs/exp07"))        # "" if no sentinel yet
```

Rules: never edit a running run's config (make a new run id); a fresh
run id per experiment attempt. Publish a real plot by saving a PNG from
the torch session (matplotlib: `matplotlib.use("Agg")`,
`plt.savefig("runs/<id>/loss.png")`) and calling the `show_image`
tool with that path — it lands in chat as a media attachment. Use
`plot_device` on `metrics.csv` only for a quick in-text glance.
