#!/usr/bin/env python3
"""core/train.py -- the architecture-agnostic training engine.

RESEARCH-HARNESS-PLAN.md 1.1: Rust never learns what a "training run"
is; this file is the whole trainer. It knows NO model architecture --
the model (and optionally the task) is a plug-in named in config.yaml.
Drive it through the CLI (`python core/cli.py --help`); this module is
the importable engine behind it.

Every experiment is the same loop:

    init -> loop { step -> log -> checkpoint } -> finish

Modes (what one "step" means):
    pretrain supervised epochs (autograd + optimizer)
    distill  pretrain + a frozen teacher forward in the loss
    ga       one generation = evaluate N parameter vectors (no autograd)
    rl       contextual bandit: model logits are a categorical policy,
             task.reward scores the sampled actions (REINFORCE)

Plug-in contract
----------------
config.yaml names factories as "<file.py>:<Name>" or
"<package.module>:<Name>". Relative files resolve against the config's
directory, then experiments/ (so "models/mynet.py:Net" is
experiments/models/mynet.py). The rest of the section is passed as
keyword arguments:

    model:
      factory: models/mynet.py:Net    # -> Net(hidden=64) -> nn.Module
      hidden: 64
    task:
      factory: parity                 # built-in (core/tasks.py: parity, lastbit,
      seq_len: 8                      # tokens) or file.py:Name
                                      # -> Parity(seed=<seed>, seq_len=8)
    init:                             # optional warm start (any architecture)
      checkpoint: /data/backbone.bin  # .pt/.bin state_dict, engine checkpoint,
                                      # or .safetensors
      convert: models/x.py:convert    # optional: fn(state_dict) -> state_dict
                                      # (rename a foreign backbone's keys)
      strict: false                   # true: every key must match exactly

If the model defines `training_loss(x, y) -> (loss, parts)`, the engine
uses it instead of `task.loss(model(x), y)` -- for models whose objective
is not a function of one forward output (e.g. LatentLM's MUX loss: a
latent loop, a discrete pass, several terms). `parts` (floats) become
metrics.csv columns: val_<k> from the eval pass, train_<k> averaged over
each epoch. Modes pretrain and ga support it (distill / rl need outputs).

If the model defines `warm_start_from(state_dict)`, init calls it (the
architecture's own copy policy); otherwise matching keys with matching
shapes are loaded. Either way a load report lands in train.log and
meta.json, and `cli.py check` prints it.

The model is any torch.nn.Module; it is called as `model(x)` with the
task's inputs. The task is any object with:

    val                          fixed validation batch (x, y)
    sample(n, step) -> (x, y)    deterministic in step
    loss(out, y)                 optional; default: cross-entropy over the
                                 last dim for integer y, MSE for float y
    metrics(out, y) -> dict      optional; extra metrics.csv columns
    distill_loss(s_out, t_out, y) optional; default: KL (int y) / MSE
    reward(x, actions, y)        optional, rl; default: (actions == y)

x and y may be tensors or (nested) tuples/lists/dicts of tensors.

Run directory contract (the run dir is the config file's parent; the
agent polls it, sentinels are idempotent IPC):

    metrics.csv            mode columns + task metrics, header first
    events.out.tfevents.*  TensorBoard scalars
    checkpoints/step_N.pt  + latest.pt (state_dict + config + step)
    meta.json              provenance: argv, resolved config, versions
    src/                   snapshot of the plug-in source files
    PID                    "<pid>\\n<pid namespace>" of the trainer
    DONE                   clean exit (first line "exit=0")
    FAILED                 any exception or stop signal (traceback)

Everything is seeded from config["seed"]; a rerun of the same config in
the same run dir reproduces the run (sentinels from a previous attempt
are removed at start).
"""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import random
import shutil
import signal
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import monitor  # noqa: E402  (torch-free; owns the monitor: config section)

CORE = Path(__file__).resolve().parent
EXPERIMENTS = CORE.parent
BUILTIN_TASKS = {
    "parity": f"{CORE / 'tasks.py'}:Parity",
    "lastbit": f"{CORE / 'tasks.py'}:LastBit",
    "tokens": f"{CORE / 'tasks.py'}:TokenStream",
}

# ------------------------------------------------------------------ config

DEFAULTS = {
    "seed": 0,
    "mode": "pretrain",
    "device": "auto",  # auto | cpu | cuda | cuda:N
    "model": {"factory": None},
    "task": {"factory": "parity"},
    "init": {"checkpoint": None, "convert": None, "strict": False},
    "train": {
        # pretrain / distill / rl ("epoch" = one eval + metrics row)
        "epochs": 10,
        "batches_per_epoch": 20,
        "batch_size": 64,
        "grad_accum": 1,  # micro-batches per optimizer step
        "optimizer": "adam",  # adam | adamw | sgd
        "lr": 1e-3,
        "lr_schedule": "constant",  # constant | cosine (after warmup)
        "warmup_steps": 0,  # linear warmup, in optimizer steps
        "min_lr_frac": 0.1,  # cosine floor, as a fraction of lr
        "weight_decay": 0.0,
        "grad_clip": 10.0,
        "amp": "none",  # none | bf16 (autocast)
        "eval_batch_size": 64,  # val is evaluated in chunks of this size
        # ga
        "pop_size": 50,
        "generations": 20,
        "sigma": 0.2,
        "sigma_decay": 0.98,
        "elite_frac": 0.1,
        "tournament": 3,
        # rl
        "steps": 100,
        "rollouts": 64,
        "entropy_beta": 0.01,
        # shared
        "checkpoint_every": 5,
        "keep_checkpoints": 0,  # keep only the newest N step_*.pt (0 = all)
    },
    "distill": {
        "teacher_checkpoint": None,  # path to a .pt written by this engine
        "teacher_model": None,  # model section; default: the checkpoint's own
        "alpha": 0.5,
        "temperature": 1.0,
    },
    # What core/monitor.py alerts on (cli.py status). Empty watch = legacy
    # (val_loss / gen_best); see monitor.py's docstring.
    "monitor": copy.deepcopy(monitor.DEFAULT_MONITOR),
}

MODE_COLUMNS = {
    "pretrain": ["epoch", "train_loss", "val_loss", "grad_norm", "lr", "peak_mem_mb"],
    "distill": ["epoch", "train_loss", "val_loss", "distill_loss", "grad_norm", "lr", "peak_mem_mb"],
    "ga": ["generation", "gen_best", "gen_mean", "gen_std", "sigma"],
    "rl": ["rollouts", "reward_mean", "reward_max", "reward_min", "policy_loss", "entropy"],
}
MODES = tuple(MODE_COLUMNS)


def merge_config(raw: dict) -> dict:
    """Plain dict over DEFAULTS (one level deep); unknown keys are kept
    (the agent adds its own knobs and reads them back in patches)."""
    cfg = copy.deepcopy(DEFAULTS)
    for key, value in (raw or {}).items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    return cfg


def load_config(path, mode: str | None = None, device: str | None = None) -> dict:
    cfg = merge_config(yaml.safe_load(Path(path).read_text()) or {})
    if mode:
        cfg["mode"] = mode
    if device:
        cfg["device"] = device
    if cfg["mode"] not in MODES:
        raise ValueError(f"unknown mode '{cfg['mode']}' (expected one of: {', '.join(MODES)})")
    if not cfg["model"].get("factory"):
        raise ValueError(
            "model.factory is not set -- point it at your architecture, "
            "e.g. 'models/mynet.py:Net' (see core/train.py 'Plug-in contract')"
        )
    return cfg


def search_path(config_path) -> list[Path]:
    """Where relative plug-in files are looked up."""
    return [Path(config_path).resolve().parent, EXPERIMENTS]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name: str) -> str:
    if name in (None, "", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


# ----------------------------------------------------------------- plug-ins

def _resolve_file(name: str, search: Sequence[Path]) -> Path:
    p = Path(name).expanduser()
    if p.is_absolute():
        if p.is_file():
            return p
    else:
        for d in search:
            for cand in (d / p, d / p.name):  # d/p.name: snapshots in run_dir/src
                if cand.is_file():
                    return cand.resolve()
    where = ", ".join(str(d) for d in search)
    raise FileNotFoundError(f"plug-in file '{name}' not found (searched: {where})")


def _modules_in(directory: Path) -> list[str]:
    """Names of imported modules whose file lives directly in `directory`
    (string compare: resolving every module path is slow)."""
    want = os.path.abspath(directory)
    return [
        name for name, mod in list(sys.modules.items())
        if isinstance(getattr(mod, "__file__", None), str)
        and os.path.dirname(os.path.abspath(mod.__file__)) == want
    ]


def _shared_modules() -> list[str]:
    """Modules imported from experiments/ outside core/ -- shared
    libraries such as codetok (experiments/ is on every plug-in's path)."""
    root, core = os.path.join(os.path.abspath(EXPERIMENTS), ""), os.path.join(os.path.abspath(CORE), "")
    return [
        name for name, mod in list(sys.modules.items())
        if isinstance(getattr(mod, "__file__", None), str)
        and os.path.abspath(mod.__file__).startswith(root)
        and not os.path.abspath(mod.__file__).startswith(core)
    ]


def local_sources(path: Path) -> list[Path]:
    """The plug-in file, every module it pulled in from its own directory
    (e.g. an adapter importing the architecture file), every shared
    module imported from experiments/, and the extra files those
    modules list in `__snapshot__` (e.g. codetok's tokenizer.json)."""
    found = {path.resolve()}
    for name in _modules_in(path.resolve().parent) + _shared_modules():
        mod = sys.modules.get(name)
        if mod is None:
            continue
        found.add(Path(mod.__file__).resolve())
        found.update(Path(f).resolve() for f in getattr(mod, "__snapshot__", ()) if Path(f).is_file())
    return sorted(found)


def snapshot_path(source: Path) -> Path:
    """Where a source lands under run_dir/src: its path relative to
    experiments/ (src/models/x.py, src/codetok/__init__.py), else flat."""
    try:
        return source.resolve().relative_to(EXPERIMENTS)
    except ValueError:
        return Path(source.name)


def _load_file(path: Path):
    """Execute a plug-in file as a fresh module -- never a cached one, so
    an edited model file is picked up even inside a long-lived repl. Its
    sibling modules and shared experiments/ modules are re-imported fresh
    too (except core's own). experiments/ itself is importable, so any
    plug-in can `import codetok`."""
    tag = hashlib.sha1(str(path).encode()).hexdigest()[:8]
    name = f"llung_plugin_{path.stem}_{tag}"
    stale = _shared_modules()
    if path.parent.resolve() != CORE:
        stale += _modules_in(path.parent.resolve())
    for old in stale:
        sys.modules.pop(old, None)
    if str(EXPERIMENTS) not in sys.path:
        sys.path.append(str(EXPERIMENTS))
    if str(path.parent) not in sys.path:  # let it import its siblings
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def load_object(spec: str, search: Sequence[Path], builtins: dict | None = None):
    """'file.py:Name' | 'pkg.module:Name' | builtin name -> (object, source file)."""
    builtins = builtins or {}
    spec = builtins.get(spec, spec)
    where, sep, attr = str(spec).rpartition(":")
    if not (sep and where and attr):
        known = f"a built-in ({', '.join(builtins)}) or " if builtins else ""
        raise ValueError(f"unknown factory '{spec}': expected {known}'file.py:Name' or 'package.module:Name'")
    if where.endswith(".py"):
        source = _resolve_file(where, search)
        module = _load_file(source)
    else:
        module = importlib.import_module(where)
        source = Path(module.__file__) if getattr(module, "__file__", None) else None
    if not hasattr(module, attr):
        raise AttributeError(f"'{attr}' not found in {source or where}")
    return getattr(module, attr), source


def build(section: dict, search: Sequence[Path], builtins: dict | None = None, **extra):
    """Call a config section's factory with the rest of the section."""
    kwargs = dict(section)
    spec = kwargs.pop("factory", None)
    if not spec:
        raise ValueError(f"config section has no 'factory': {section}")
    factory, source = load_object(spec, search, builtins)
    return factory(**kwargs, **extra), source


def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):  # namedtuple
        return type(obj)(*(to_device(v, device) for v in obj))
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device) for v in obj)
    return obj


def describe(obj) -> str:
    """Shape summary for check(): 'float32[256, 8, 1]', nested."""
    if isinstance(obj, torch.Tensor):
        return f"{str(obj.dtype).removeprefix('torch.')}{list(obj.shape)}"
    if isinstance(obj, dict):
        return "{" + ", ".join(f"{k}: {describe(v)}" for k, v in obj.items()) + "}"
    if isinstance(obj, (list, tuple)):
        return "(" + ", ".join(describe(v) for v in obj) + ")"
    return type(obj).__name__


def batch_len(obj) -> int:
    if isinstance(obj, torch.Tensor):
        return obj.shape[0]
    if isinstance(obj, dict):
        return batch_len(next(iter(obj.values())))
    return batch_len(obj[0])


def batch_slice(obj, i: int, j: int):
    if isinstance(obj, torch.Tensor):
        return obj[i:j]
    if isinstance(obj, dict):
        return {k: batch_slice(v, i, j) for k, v in obj.items()}
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return type(obj)(*(batch_slice(v, i, j) for v in obj))
    if isinstance(obj, (list, tuple)):
        return type(obj)(batch_slice(v, i, j) for v in obj)
    return obj


def load_weights(path: Path) -> dict:
    """A state_dict from .safetensors, a raw .pt/.bin state_dict, or an
    engine checkpoint ({"model": state_dict, ...})."""
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    obj = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("model", "state_dict"):
        if isinstance(obj, dict) and isinstance(obj.get(key), dict):
            obj = obj[key]
    if not isinstance(obj, dict):
        raise TypeError(f"{path} holds a {type(obj).__name__}, not a state_dict")
    return obj


def apply_init(model: torch.nn.Module, init: dict, search: Sequence[Path], sources: list) -> str:
    """Warm start `model` from init.checkpoint; returns a load report."""
    path = Path(init["checkpoint"]).expanduser()
    path = path if path.is_absolute() else _resolve_file(str(path), search)
    if not path.is_file():
        raise FileNotFoundError(f"init.checkpoint {path} does not exist")
    sd = load_weights(path)
    if init.get("convert"):
        fn, src = load_object(init["convert"], search)
        sources.extend(local_sources(src) if src else [])
        sd = fn(sd)
    own = model.state_dict()
    matched = [k for k, v in sd.items() if k in own and own[k].shape == v.shape]
    mismatched = [f"{k} {list(sd[k].shape)}->{list(own[k].shape)}" for k in sd if k in own and own[k].shape != sd[k].shape]
    unexpected = [k for k in sd if k not in own]
    loaded_ptrs = {own[k].data_ptr() for k in matched}
    missing = [k for k in own if k not in sd and own[k].data_ptr() not in loaded_ptrs]  # tied = loaded
    if init.get("strict"):
        model.load_state_dict(sd, strict=True)
        how = "load_state_dict(strict)"
    elif hasattr(model, "warm_start_from"):
        model.warm_start_from(sd)
        how = "model.warm_start_from"
    else:
        model.load_state_dict({k: sd[k] for k in matched}, strict=False)
        how = "load_state_dict(matching keys)"
    n_loaded = sum(own[k].numel() for k in matched)
    n_total = sum(p.numel() for p in model.parameters())

    def few(keys):
        return f"{len(keys)}" + (f" ({', '.join(keys[:6])}{', ...' if len(keys) > 6 else ''})" if keys else "")

    return (
        f"init: {path} via {how}: {len(matched)} tensors / {n_loaded:,} values copied "
        f"(model has {n_total:,} params); missing {few(missing)}; "
        f"unexpected {few(unexpected)}; shape-mismatch {few(mismatched)}"
    )


def autocast(cfg: dict, device: str):
    amp = str(cfg["train"].get("amp", "none")).lower()
    if amp not in ("none", "bf16"):
        raise ValueError(f"unknown train.amp '{amp}' (none | bf16)")
    return torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=amp == "bf16")


def lr_at(step: int, total: int, tr: dict) -> float:
    """Learning rate for optimizer step `step` (0-based) of `total`."""
    base, warmup = float(tr["lr"]), int(tr.get("warmup_steps", 0))
    if step < warmup:
        return base * (step + 1) / warmup
    schedule = str(tr.get("lr_schedule", "constant"))
    if schedule == "constant":
        return base
    if schedule == "cosine":
        floor = float(tr.get("min_lr_frac", 0.1))
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return base * (floor + (1 - floor) * 0.5 * (1 + np.cos(np.pi * progress)))
    raise ValueError(f"unknown train.lr_schedule '{schedule}' (constant | cosine)")


def default_loss(out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if torch.is_floating_point(y):
        return F.mse_loss(out, y)
    return F.cross_entropy(out.reshape(-1, out.size(-1)), y.reshape(-1))


# --------------------------------------------------------------- experiment

class Experiment:
    """A built (task, model) pair plus the loss/metric hooks. No I/O."""

    def __init__(self, cfg: dict, search: Sequence[Path], device: str):
        self.cfg, self.device = cfg, device
        self.task, task_src = build(cfg["task"], search, BUILTIN_TASKS, seed=int(cfg["seed"]))
        model, model_src = build(cfg["model"], search)
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"model.factory returned {type(model).__name__}, not a torch.nn.Module")
        self.sources = []
        for src in (model_src, task_src):
            if src is not None:
                self.sources.extend(s for s in local_sources(src) if s not in self.sources)
        self.init_report = None
        if (cfg.get("init") or {}).get("checkpoint"):
            self.init_report = apply_init(model, cfg["init"], search, self.sources)
            print(self.init_report, flush=True)
        self.model = model.to(device)
        self.val = to_device(self.task.val, device)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def batch(self, n: int, step: int):
        return to_device(self.task.sample(n, step), self.device)

    def loss(self, out, y) -> torch.Tensor:
        fn = getattr(self.task, "loss", None)
        return fn(out, y) if fn else default_loss(out, y)

    @property
    def custom_loss(self) -> bool:
        """The model computes its own objective (training_loss hook)."""
        return callable(getattr(self.model, "training_loss", None))

    def compute_loss(self, x, y) -> tuple[torch.Tensor, dict]:
        """(loss, parts) for one batch: the model's training_loss hook if it
        has one, else task loss on model(x) (parts: {})."""
        if self.custom_loss:
            loss, parts = self.model.training_loss(x, y)
            return loss, {k: float(v) for k, v in (parts or {}).items()}
        return self.loss(self.model(x), y), {}

    def distill_loss(self, s_out, t_out, y) -> torch.Tensor:
        fn = getattr(self.task, "distill_loss", None)
        if fn:
            return fn(s_out, t_out, y)
        if torch.is_floating_point(y):
            return F.mse_loss(s_out, t_out)
        temp = float(self.cfg["distill"]["temperature"])
        c = s_out.size(-1)
        return F.kl_div(
            F.log_softmax(s_out / temp, -1).reshape(-1, c),
            F.softmax(t_out / temp, -1).reshape(-1, c),
            reduction="batchmean",
        ) * temp * temp

    def reward(self, x, actions, y) -> torch.Tensor:
        fn = getattr(self.task, "reward", None)
        return (fn(x, actions, y) if fn else (actions == y)).float()

    @torch.no_grad()
    def evaluate(self) -> tuple[float, dict]:
        """Validation loss + task metrics on the fixed val batch, in chunks
        of train.eval_batch_size (metrics are chunk-size-weighted means)."""
        was_training = self.model.training
        self.model.eval()
        x, y = self.val
        n = batch_len(x)
        chunk = max(1, int(self.cfg["train"].get("eval_batch_size", 64)))
        fn = getattr(self.task, "metrics", None)
        loss, sums = 0.0, {}
        for i in range(0, n, chunk):
            xs, ys = batch_slice(x, i, i + chunk), batch_slice(y, i, i + chunk)
            w = batch_len(xs) / n
            with autocast(self.cfg, self.device):
                if self.custom_loss:  # the model's objective; parts -> val_<k>
                    l, parts = self.compute_loss(xs, ys)
                    loss += float(l) * w
                    metrics = {f"val_{k}": v for k, v in parts.items()}
                else:
                    out = self.model(xs)
                    loss += float(self.loss(out, ys)) * w
                    metrics = fn(out, ys) if fn else {}
            for k, v in metrics.items():
                sums[k] = sums.get(k, 0.0) + float(v) * w
        self.model.train(was_training)
        return loss, {k: round(v, 6) for k, v in sums.items()}


def make_optimizer(params, tr: dict) -> torch.optim.Optimizer:
    name = str(tr["optimizer"]).lower()
    lr, wd = float(tr["lr"]), float(tr["weight_decay"])
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
    raise ValueError(f"unknown optimizer '{name}' (adam | adamw | sgd)")


# ------------------------------------------------------------------- runner

class Runner:
    """metrics.csv + TensorBoard + checkpoints, one row per step."""

    def __init__(self, mode: str, exp: Experiment, run_dir: Path):
        self.mode, self.exp, self.run_dir = mode, exp, run_dir
        self.cfg, self.model, self.device = exp.cfg, exp.model, exp.device
        self.step = 0
        self.t0 = time.time()
        # One val forward up front: it fixes the metric columns AND fails
        # fast (-> FAILED) when model and task don't fit together.
        _, metrics = exp.evaluate()
        self.fields = metric_fields(mode, exp, metrics, train_part_names(exp))
        self._f = (run_dir / "metrics.csv").open("w", newline="")
        self.csv = csv.DictWriter(self._f, fieldnames=self.fields, extrasaction="ignore")
        self.csv.writeheader()
        self.tb = SummaryWriter(log_dir=str(run_dir))
        self.ckpt_dir = run_dir / "checkpoints"
        self.ckpt_dir.mkdir(exist_ok=True)

    def log(self, row: dict) -> None:
        row = {"step": self.step, "wall": round(time.time() - self.t0, 3), **row}
        self.csv.writerow(row)
        self._f.flush()  # the agent tails this file; every row must land
        for key, value in row.items():
            if key not in ("step", "wall") and isinstance(value, (int, float)):
                self.tb.add_scalar(key, value, self.step)

    def checkpoint(self, extra: dict | None = None) -> None:
        state = {
            "mode": self.mode,
            "step": self.step,
            "config": self.cfg,
            "model": self.model.state_dict(),
            **(extra or {}),
        }
        torch.save(state, self.ckpt_dir / f"step_{self.step}.pt")
        torch.save(state, self.ckpt_dir / "latest.pt")
        keep = int(self.cfg["train"].get("keep_checkpoints", 0))
        if keep > 0:
            steps = sorted(self.ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
            for old in steps[:-keep]:
                old.unlink(missing_ok=True)

    def close(self) -> None:
        self.tb.close()
        self._f.close()


def metric_fields(mode: str, exp: "Experiment", metrics: dict, train_parts: list[str] | None = None) -> list[str]:
    """The metrics.csv header a run will have: mode columns, eval metrics,
    train_<k> for the parts training_loss returns IN TRAINING (eval-only
    parts such as accuracy sweeps get no empty train_ column), wall."""
    cols = MODE_COLUMNS[mode]
    if not exp.custom_loss:
        train_cols = []
    elif train_parts is not None:
        train_cols = [f"train_{k}" for k in train_parts]
    else:
        train_cols = [f"train_{k[4:]}" for k in metrics if k.startswith("val_")]
    return ["step", *cols, *(k for k in metrics if k not in cols), *train_cols, "wall"]


def train_part_names(exp: "Experiment") -> list[str] | None:
    """The parts a training_loss model reports in training mode: one dry
    call without grad on a tiny batch (deterministic; no optimizer step)."""
    if not exp.custom_loss:
        return None
    was = exp.model.training
    exp.model.train()
    try:
        x, y = exp.batch(min(int(exp.cfg["train"]["batch_size"]), 2), 10**9)
        with torch.no_grad(), autocast(exp.cfg, exp.device):
            _, parts = exp.compute_loss(x, y)
    finally:
        exp.model.train(was)
    return list(parts)


def _due(step: int, total: int, every: int) -> bool:
    return step % max(every, 1) == 0 or step == total


# --------------------------------------------------------------------- modes

def run_ga(r: Runner) -> None:
    """One generation = evaluate N parameter vectors on the val batch.
    Genomes start as gaussian perturbations of the model's OWN init, so
    whatever init scheme the architecture uses is respected."""
    tr, exp = r.cfg["train"], r.exp
    params = list(r.model.parameters())
    base = parameters_to_vector(params).detach().cpu()
    g = torch.Generator().manual_seed(int(r.cfg["seed"]) + 3)
    pop_size, sigma = int(tr["pop_size"]), float(tr["sigma"])
    pop = [base.clone()] + [base + sigma * torch.randn(base.shape, generator=g) for _ in range(pop_size - 1)]
    elite_n = max(1, int(pop_size * float(tr["elite_frac"])))
    x, y = exp.val
    generations = int(tr["generations"])
    r.model.eval()

    @torch.no_grad()
    def fitness(vec: torch.Tensor) -> float:  # lower is better
        vector_to_parameters(vec.to(r.device), params)
        return float(exp.compute_loss(x, y)[0])

    for generation in range(1, generations + 1):
        r.step = generation
        losses = torch.tensor([fitness(v) for v in pop])
        order = losses.argsort()
        elites = [pop[int(i)] for i in order[:elite_n]]
        vector_to_parameters(elites[0].to(r.device), params)
        _, metrics = exp.evaluate()
        r.log(
            {
                "generation": generation,
                "gen_best": round(float(losses[order[0]]), 6),
                "gen_mean": round(float(losses.mean()), 6),
                "gen_std": round(float(losses.std()) if len(pop) > 1 else 0.0, 6),
                "sigma": round(sigma, 6),
                **metrics,
            }
        )
        if _due(generation, generations, int(tr["checkpoint_every"])):
            r.checkpoint({"genome": elites[0].clone()})
        # tournament selection (on cached fitness) -> uniform crossover
        # -> gaussian mutation; elites survive unchanged.
        t = max(1, min(int(tr["tournament"]), len(pop)))

        def pick() -> torch.Tensor:
            idx = torch.randperm(len(pop), generator=g)[:t]
            return pop[int(idx[losses[idx].argmin()])]

        parents = [pick() for _ in range(pop_size - elite_n)]
        children = []
        for i in range(0, len(parents) - 1, 2):
            a, b = parents[i], parents[i + 1]
            mask = torch.rand(a.shape, generator=g) < 0.5
            children.append(torch.where(mask, a, b) + torch.randn(a.shape, generator=g) * sigma)
            children.append(torch.where(mask, b, a) + torch.randn(a.shape, generator=g) * sigma)
        if len(children) < len(parents):  # odd count: mutate the leftover parent
            children.append(parents[-1] + torch.randn(base.shape, generator=g) * sigma)
        pop = elites + children
        sigma *= float(tr["sigma_decay"])
    vector_to_parameters(elites[0].to(r.device), params)


def run_pretrain(r: Runner, teacher: torch.nn.Module | None = None) -> None:
    tr, exp = r.cfg["train"], r.exp
    alpha = float(r.cfg["distill"]["alpha"]) if teacher is not None else 0.0
    opt = make_optimizer(r.model.parameters(), tr)
    epochs, per_epoch, bs = int(tr["epochs"]), int(tr["batches_per_epoch"]), int(tr["batch_size"])
    accum = max(1, int(tr.get("grad_accum", 1)))
    total_steps, opt_step = epochs * per_epoch, 0
    if teacher is not None and exp.custom_loss:
        raise ValueError("distill needs model outputs; this model computes its own loss (training_loss)")
    cuda = torch.device(r.device).type == "cuda"
    for epoch in range(1, epochs + 1):
        r.step = epoch
        r.model.train()
        total = distill_total = grad_norm = 0.0
        part_sums: dict[str, float] = {}
        if cuda:
            torch.cuda.reset_peak_memory_stats(r.device)
        for _ in range(per_epoch):
            for group in opt.param_groups:
                group["lr"] = lr_at(opt_step, total_steps, tr)
            opt.zero_grad(set_to_none=True)
            for micro in range(accum):
                x, y = exp.batch(bs, opt_step * accum + micro)
                with autocast(r.cfg, r.device):
                    if exp.custom_loss:
                        loss, parts = exp.compute_loss(x, y)
                        for k, v in parts.items():
                            part_sums[k] = part_sums.get(k, 0.0) + v / (accum * per_epoch)
                    else:
                        out = r.model(x)
                        loss = exp.loss(out, y)
                    if teacher is not None:
                        with torch.no_grad():
                            t_out = teacher(x)
                        d_loss = exp.distill_loss(out, t_out, y)
                        distill_total += float(d_loss.detach()) / accum
                        loss = (1 - alpha) * loss + alpha * d_loss
                (loss / accum).backward()
                total += float(loss.detach()) / accum
            grad_norm = float(torch.nn.utils.clip_grad_norm_(r.model.parameters(), float(tr["grad_clip"])))
            opt.step()
            opt_step += 1
        val_loss, metrics = exp.evaluate()
        row = {
            "epoch": epoch,
            "train_loss": round(total / max(per_epoch, 1), 6),
            "val_loss": round(val_loss, 6),
            "grad_norm": round(grad_norm, 6),
            "lr": float(f"{opt.param_groups[0]['lr']:.6g}"),
            "peak_mem_mb": round(torch.cuda.max_memory_allocated(r.device) / 2**20, 1) if cuda else 0.0,
            **metrics,
            **{f"train_{k}": round(v, 6) for k, v in part_sums.items()},
        }
        if teacher is not None:
            row["distill_loss"] = round(distill_total / max(per_epoch, 1), 6)
        r.log(row)
        if _due(epoch, epochs, int(tr["checkpoint_every"])):
            r.checkpoint()


def run_rl(r: Runner) -> None:
    """Contextual bandit: the model's output logits (last dim) define a
    categorical policy per input; task.reward scores the sampled actions.
    REINFORCE with a running-mean baseline; one step = one update from
    `rollouts` inputs."""
    tr, exp = r.cfg["train"], r.exp
    if exp.custom_loss:
        raise ValueError("rl needs policy logits from model(x); this model computes its own loss (training_loss)")
    beta, steps = float(tr["entropy_beta"]), int(tr["steps"])
    opt = make_optimizer(r.model.parameters(), tr)
    baseline = 0.0
    r.model.train()
    for step in range(1, steps + 1):
        r.step = step
        x, y = exp.batch(int(tr["rollouts"]), 10_000_000 + step)
        dist = torch.distributions.Categorical(logits=r.model(x))
        actions = dist.sample()
        rewards = exp.reward(x, actions, y).detach()
        entropy = dist.entropy().mean()
        baseline = 0.9 * baseline + 0.1 * float(rewards.mean())
        loss = -(dist.log_prob(actions) * (rewards - baseline)).mean() - beta * entropy
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(r.model.parameters(), float(tr["grad_clip"]))
        opt.step()
        _, metrics = exp.evaluate()
        r.log(
            {
                "rollouts": int(tr["rollouts"]),
                "reward_mean": round(float(rewards.mean()), 6),
                "reward_max": round(float(rewards.max()), 6),
                "reward_min": round(float(rewards.min()), 6),
                "policy_loss": round(float(loss.detach()), 6),
                "entropy": round(float(entropy.detach()), 6),
                **metrics,
            }
        )
        if _due(step, steps, int(tr["checkpoint_every"])):
            r.checkpoint()


def load_teacher(cfg: dict, search: Sequence[Path], device: str) -> torch.nn.Module:
    """The teacher's architecture comes from its checkpoint's stored
    config, built from the source snapshot in the teacher's run dir
    (so later edits to the model file don't change the teacher)."""
    d = cfg["distill"]
    if not d.get("teacher_checkpoint"):
        raise ValueError("distill mode needs distill.teacher_checkpoint (a .pt from an earlier run)")
    path = Path(d["teacher_checkpoint"]).expanduser()
    if not path.is_absolute():
        path = search[0] / path
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model_cfg = d.get("teacher_model") or (ckpt.get("config", {}).get("model") if isinstance(ckpt, dict) else None)
    if not model_cfg:
        raise ValueError(f"{path} carries no config; set distill.teacher_model to its model section")
    teacher_run = path.resolve().parent.parent
    teacher, _ = build(model_cfg, [teacher_run / "src", teacher_run, *search])
    teacher.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


# ---------------------------------------------------------- entry points

class Stopped(Exception):
    """Raised by the SIGTERM handler so a stopped run still gets FAILED."""


def _on_sigterm(signum, _frame):
    raise Stopped(f"stopped by signal {signum} (cli.py stop?)")


def pid_namespace() -> str:
    try:
        return os.readlink("/proc/self/ns/pid")
    except OSError:
        return ""


def write_sentinel(run_dir: Path, name: str, text: str) -> None:
    (run_dir / name).write_text(text)


def provenance(run_dir: Path, config_path: Path, exp: Experiment) -> None:
    src = run_dir / "src"
    src.mkdir(exist_ok=True)
    for s in exp.sources:
        dest = src / snapshot_path(s)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, dest)
    meta = {
        "mode": exp.cfg["mode"],
        "argv": sys.argv,
        "config": str(config_path),
        "run_id": run_dir.name,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": exp.device,
        "n_params": exp.n_params,
        "init": exp.init_report,
        "sources": [str(s) for s in exp.sources],
        "resolved_config": exp.cfg,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def execute(config, mode: str | None = None, device: str | None = None) -> int:
    """Run one experiment in the foreground; returns the exit code.
    EVERYTHING after the stale sentinels are cleared is inside the try,
    so a failed (or stopped) run always leaves FAILED behind."""
    config_path = Path(config).resolve()
    run_dir = config_path.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("DONE", "FAILED"):  # a rerun must not inherit a verdict
        (run_dir / stale).unlink(missing_ok=True)
    (run_dir / "PID").write_text(f"{os.getpid()}\n{pid_namespace()}\n")

    old_handler = None
    if threading.current_thread() is threading.main_thread():
        old_handler = signal.signal(signal.SIGTERM, _on_sigterm)
    r = None
    try:
        cfg = load_config(config_path, mode, device)
        seed_everything(int(cfg["seed"]))
        exp = Experiment(cfg, search_path(config_path), pick_device(cfg["device"]))
        provenance(run_dir, config_path, exp)
        r = Runner(cfg["mode"], exp, run_dir)
        if cfg["mode"] == "ga":
            run_ga(r)
        elif cfg["mode"] == "rl":
            run_rl(r)
        else:
            teacher = None
            if cfg["mode"] == "distill":
                teacher = load_teacher(cfg, search_path(config_path), exp.device)
            run_pretrain(r, teacher)
        r.checkpoint()
        write_sentinel(
            run_dir, "DONE",
            f"exit=0\nmode={cfg['mode']}\nsteps={r.step}\ndevice={exp.device}\n",
        )
        return 0
    except BaseException:  # incl. Stopped / KeyboardInterrupt
        write_sentinel(run_dir, "FAILED", traceback.format_exc())
        traceback.print_exc()
        return 1
    finally:
        if r is not None:
            r.close()
        if old_handler is not None:
            signal.signal(signal.SIGTERM, old_handler)


def check(config, mode: str | None = None, device: str | None = None) -> str:
    """Dry run: build task + model, one val forward (+ one backward for
    gradient modes). Writes nothing. Raises on any contract violation;
    returns a short report on success."""
    config_path = Path(config).resolve()
    cfg = load_config(config_path, mode, device)
    seed_everything(int(cfg["seed"]))
    exp = Experiment(cfg, search_path(config_path), pick_device(cfg["device"]))
    x, y = exp.val
    t0 = time.perf_counter()
    val_loss, metrics = exp.evaluate()
    ms = (time.perf_counter() - t0) * 1000
    out = None
    if not exp.custom_loss:  # loss-hook models need not have a plain forward
        with torch.no_grad(), autocast(cfg, exp.device):
            out = exp.model.eval()(batch_slice(x, 0, int(cfg["train"]["eval_batch_size"])))
    trainable = sum(p.numel() for p in exp.model.parameters() if p.requires_grad)
    lines = [
        f"config:  {config_path}",
        f"mode:    {cfg['mode']}   device: {exp.device}",
        f"model:   {cfg['model']['factory']}  ({exp.n_params:,} params, {trainable:,} trainable)",
        f"task:    {cfg['task']['factory']}",
        *([exp.init_report] if exp.init_report else []),
        f"val x:   {describe(x)}",
        f"val y:   {describe(y)}",
        f"output:  {describe(out)}" if out is not None else "output:  (the model's training_loss hook computes the loss)",
        f"val_loss {val_loss:.6f}   full val eval {ms:.1f} ms"
        + "".join(f"   {k} {v}" for k, v in metrics.items()),
    ]
    warnings = []
    if not np.isfinite(val_loss):
        warnings.append("val_loss is not finite")
    if cfg["mode"] in ("pretrain", "distill", "rl"):
        exp.model.train()
        xb, yb = exp.batch(min(int(cfg["train"]["batch_size"]), 8), 0)
        exp.model.zero_grad()
        with autocast(cfg, exp.device):
            loss, parts = exp.compute_loss(xb, yb)
        loss.backward()
        if parts:
            lines.append("train step: " + "   ".join(f"{k} {v:.4g}" for k, v in parts.items()))
        dead = [n for n, p in exp.model.named_parameters() if p.requires_grad and p.grad is None]
        if dead:
            warnings.append(f"{len(dead)} parameter tensors get no gradient: {', '.join(dead[:5])}")
        exp.model.zero_grad()
    if cfg["mode"] == "ga" and exp.n_params > 100_000:
        warnings.append(f"ga over {exp.n_params:,} params will be slow/ineffective")
    if cfg["mode"] in ("rl", "distill") and torch.is_floating_point(y) and not (
        hasattr(exp.task, "reward" if cfg["mode"] == "rl" else "distill_loss")
    ):
        warnings.append(f"{cfg['mode']} default assumes integer class targets; task y is float")
    # monitor: every watched / shown metric must be a column this run logs
    mon, problems = monitor.parse_monitor(cfg.get("monitor"))
    warnings.extend(f"monitor config: {e}" for e in problems)
    fields = metric_fields(cfg["mode"], exp, metrics, train_part_names(exp))
    for w in mon["watch"]:
        if w["metric"] not in fields:
            warnings.append(f"monitor watches '{w['metric']}', which this run does not log "
                            f"(columns: {', '.join(f for f in fields if f not in ('step', 'wall'))})")
    for c in mon["show"]:
        if c not in fields:
            warnings.append(f"monitor.show lists '{c}', which this run does not log")
    lines.append("monitor: " + (", ".join(f"{w['metric']} ({w['goal']})" for w in mon["watch"])
                                or "legacy (val_loss / gen_best)"))
    meta = getattr(exp.task, "meta", None) or {}
    if meta.get("tokenizer"):
        sha = str(meta.get("tokenizer_sha256") or "?")[:12]
        lines.insert(4, f"data:    tokenizer {meta['tokenizer']} (sha256 {sha}), vocab {meta.get('vocab_size', '?')}")
    task_vocab = getattr(exp.task, "vocab_size", None)
    if task_vocab and isinstance(out, torch.Tensor) and out.dim() > 1 and out.shape[-1] < int(task_vocab):
        warnings.append(f"model outputs {out.shape[-1]} logits but the task's vocab is {task_vocab}: ids >= {out.shape[-1]} would crash the loss")
    lines += [f"WARNING: {w}" for w in warnings] or ["ok"]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit("train.py is the engine; use the CLI:  python core/cli.py --help")
