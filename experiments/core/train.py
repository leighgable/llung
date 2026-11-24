#!/usr/bin/env python3

"""core/train.py -- ONE mode-agnostic experiment driver.

RESEARCH-HARNESS-PLAN.md 1.1: Rust never learns what a "training run"
is; this file is the whole trainer. Every experiment is the same loop:

    init -> loop { step -> log -> checkpoint } -> finish

Modes (what one "step" means):
    ga       one epoch = evaluate N genomes (no autograd)
    pretrain supervised epochs (autograd + Adam)
    distill  pretrain + a frozen teacher forward in the loss
    rl       rollout collection + vector policy update (REINFORCE)

Usage (from experiments/, inside a torch repl session):

    python3 core/train.py --mode ga --config runs/exp07/config.yaml

The run directory is the config file's parent. Contract -- the agent
polls the directory, sentinels are idempotent IPC:

    metrics.csv            mode-dependent columns, header written first
    events.out.tfevents.*  TensorBoard scalars
    checkpoints/step_N.pt  + latest.pt (state_dict + config + row)
    meta.json              provenance: argv, resolved config, versions
    DONE                   clean exit (first line "exit=0")
    FAILED                 any exception (full traceback)

Everything is seeded from config["seed"]; a rerun of the same config in
the same run dir reproduces the run bit-for-bit (sentinels from a
previous attempt are removed at start).
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import platform
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter

# ------------------------------------------------------------------ config

DEFAULTS = {
    "seed": 0,
    "task": {"name": "parity", "seq_len": 8, "val_size": 256},
    "model": {"hidden": 32},
    "train": {
        # ga
        "pop_size": 50,
        "generations": 20,
        "sigma": 0.2,
        "sigma_decay": 0.98,
        "elite_frac": 0.1,
        "tournament": 3,
        # pretrain / distill
        "epochs": 10,
        "batches_per_epoch": 20,
        "batch_size": 64,
        "lr": 1e-3,
        "grad_clip": 10.0,
        # rl
        "steps": 100,
        "rollouts": 64,
        "entropy_beta": 0.01,
        # shared
        "checkpoint_every": 5,
    },
    "distill": {"alpha": 0.5, "teacher_hidden": 64},
}

MODE_COLUMNS = {
    "ga": ["generation", "gen_best", "gen_mean", "gen_std"],
    "pretrain": ["epoch", "train_loss", "val_loss", "grad_norm", "lr"],
    "distill": ["epoch", "train_loss", "val_loss", "distill_loss", "grad_norm", "lr"],
    "rl": ["rollouts", "reward_mean", "reward_max", "reward_min", "policy_loss", "entropy"],
}


def load_config(path: Path) -> dict:
    """Plain YAML over DEFAULTS; unknown keys are kept (the agent adds
    its own knobs and reads them back in patched versions)."""
    raw = yaml.safe_load(path.read_text()) or {}
    cfg = copy.deepcopy(DEFAULTS)
    for key, value in raw.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------- task
# A task is an infinite random-sequence generator with a FIXED validation
# set, so val_loss is comparable across steps and across runs.

class Task:
    """Random bit sequences -> one target bit.

    parity:  target = XOR of the sequence
    lastbit: target = final bit (sanity task: loss floor is 0)
    """

    def __init__(self, cfg: dict, seed: int):
        self.name = cfg["name"]
        self.seq_len = int(cfg["seq_len"])
        g = torch.Generator().manual_seed(seed + 1)
        self.val = self._batch(int(cfg["val_size"]), g)
        self._seed = seed + 2

    def _batch(self, n: int, g: torch.Generator):
        x = torch.randint(0, 2, (n, self.seq_len, 1), generator=g).float()
        if self.name == "parity":
            y = x.squeeze(-1).sum(dim=1).remainder(2)
        elif self.name == "lastbit":
            y = x[:, -1, 0]
        else:
            raise ValueError(f"unknown task '{self.name}'")
        return x, y

    def sample(self, n: int, step: int):
        g = torch.Generator().manual_seed(self._seed + step)
        return self._batch(n, g)


def task_loss(model, batch, device) -> torch.Tensor:
    x, y = batch[0].to(device), batch[1].to(device)
    return F.mse_loss(model(x), y)


# -------------------------------------------------------------------- model
# Small gated RNN (GRUCell + linear head): reads a bit sequence, emits
# one logit. Small enough that a GA genome is one flat vector.

class GatedRNN(torch.nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.gru = torch.nn.GRUCell(1, hidden)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, x):  # (batch, seq, 1) -> (batch,)
        h = torch.zeros(x.size(0), self.gru.hidden_size, device=x.device)
        for t in range(x.size(1)):
            h = self.gru(x[:, t], h)
        return self.head(h).squeeze(-1)


def flat_size(model) -> int:
    return sum(p.numel() for p in model.parameters())


def set_flat(model, vec: torch.Tensor) -> None:
    with torch.no_grad():
        offset = 0
        for p in model.parameters():
            n = p.numel()
            p.copy_(vec[offset : offset + n].view_as(p))
            offset += n


# ------------------------------------------------------------------- runner

class Runner:
    """metrics.csv + TensorBoard + checkpoints, one row per step."""

    def __init__(self, mode: str, cfg: dict, run_dir: Path, device: str):
        self.mode = mode
        self.cfg = cfg
        self.run_dir = run_dir
        self.device = device
        self.step = 0
        self.t0 = time.time()
        self.fields = ["step"] + MODE_COLUMNS[mode] + ["wall"]
        self._f = (run_dir / "metrics.csv").open("w", newline="")
        self.csv = csv.DictWriter(
            self._f, fieldnames=self.fields, extrasaction="ignore"
        )
        self.csv.writeheader()
        self.tb = SummaryWriter(log_dir=str(run_dir))
        self.ckpt_dir = run_dir / "checkpoints"
        self.ckpt_dir.mkdir(exist_ok=True)
        self.model = GatedRNN(int(cfg["model"]["hidden"])).to(device)
        self.task = Task(cfg["task"], int(cfg["seed"]))

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
        }
        if extra:
            state.update(extra)
        torch.save(state, self.ckpt_dir / f"step_{self.step}.pt")
        torch.save(state, self.ckpt_dir / "latest.pt")

    def close(self) -> None:
        self.tb.close()
        self._f.close()


# --------------------------------------------------------------------- modes

def run_ga(r: Runner) -> None:
    """One epoch = evaluate N genomes. No autograd anywhere."""
    tr = r.cfg["train"]
    n_params = flat_size(r.model)
    g = torch.Generator().manual_seed(int(r.cfg["seed"]) + 3)
    pop = [torch.randn(n_params, generator=g) * 0.5 for _ in range(int(tr["pop_size"]))]
    sigma = float(tr["sigma"])
    elite_n = max(1, int(int(tr["pop_size"]) * float(tr["elite_frac"])))
    val = (r.task.val[0].to(r.device), r.task.val[1].to(r.device))
    generations = int(tr["generations"])
    every = int(tr["checkpoint_every"])

    def loss_of(vec: torch.Tensor) -> float:
        set_flat(r.model, vec.to(r.device))
        with torch.no_grad():
            return F.mse_loss(r.model(val[0]), val[1]).item()

    for generation in range(1, generations + 1):
        r.step = generation
        losses = torch.tensor([loss_of(v) for v in pop])
        order = losses.argsort()
        elites = [pop[int(i)] for i in order[:elite_n]]
        r.log(
            {
                "generation": generation,
                "gen_best": round(float(losses[order[0]]), 6),
                "gen_mean": round(float(losses.mean()), 6),
                "gen_std": round(float(losses.std()), 6),
            }
        )
        if generation % every == 0 or generation == generations:
            r.checkpoint({"genome": elites[0].cpu()})
        # tournament selection -> uniform crossover -> gaussian mutation
        t = int(tr["tournament"])
        parents = []
        for _ in range(len(pop) - elite_n):
            cand = random.sample(pop, t)
            parents.append(min(cand, key=lambda v: loss_of(v)))
        children = []
        for i in range(0, len(parents) - 1, 2):
            a, b = parents[i], parents[i + 1]
            mask = torch.rand(a.shape, generator=g) < 0.5
            children.append(torch.where(mask, a, b) + torch.randn(a.shape, generator=g) * sigma)
            children.append(torch.where(mask, b, a) + torch.randn(a.shape, generator=g) * sigma)
        if len(children) < len(parents):  # odd population: mutate the leftover parent
            last = parents[-1]
            children.append(last + torch.randn(last.shape, generator=g) * sigma)
        pop = elites + children
        sigma *= float(tr["sigma_decay"])
    set_flat(r.model, elites[0].to(r.device))


def run_pretrain(r: Runner, teacher: GatedRNN | None = None) -> None:
    tr = r.cfg["train"]
    alpha = float(r.cfg["distill"]["alpha"]) if teacher is not None else 0.0
    opt = torch.optim.Adam(r.model.parameters(), lr=float(tr["lr"]))
    epochs, every = int(tr["epochs"]), int(tr["checkpoint_every"])
    for epoch in range(1, epochs + 1):
        r.step = epoch
        r.model.train()
        total, seen = 0.0, 0
        grad_norm = 0.0
        for b in range(int(tr["batches_per_epoch"])):
            batch = r.task.sample(int(tr["batch_size"]), epoch * 1000 + b)
            loss = task_loss(r.model, batch, r.device)
            distill_loss = 0.0
            if teacher is not None:
                x = batch[0].to(r.device)
                with torch.no_grad():
                    t_out = teacher(x)
                d_loss = F.mse_loss(r.model(x), t_out)
                distill_loss = float(d_loss)
                loss = (1 - alpha) * loss + alpha * d_loss
            opt.zero_grad()
            loss.backward()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(r.model.parameters(), float(tr["grad_clip"]))
            )
            opt.step()
            total += float(loss) * int(tr["batch_size"])
            seen += int(tr["batch_size"])
        r.model.eval()
        with torch.no_grad():
            val_loss = task_loss(r.model, r.task.val, r.device).item()
        row = {
            "epoch": epoch,
            "train_loss": round(total / max(seen, 1), 6),
            "val_loss": round(val_loss, 6),
            "grad_norm": round(grad_norm, 6),
            "lr": float(tr["lr"]),
        }
        if teacher is not None:
            row["distill_loss"] = round(distill_loss, 6)
        r.log(row)
        if epoch % every == 0 or epoch == epochs:
            r.checkpoint()


def run_rl(r: Runner) -> None:
    """Two-armed bandit tied to the task: arm 1 pays off when the bit
    sequence matches the task target, arm 2 pays off otherwise. The
    scalar logit is a Bernoulli preference -> REINFORCE with a running
    mean baseline. One step = one update from M rollouts."""
    tr = r.cfg["train"]
    beta, every = float(tr["entropy_beta"]), int(tr["checkpoint_every"])
    opt = torch.optim.Adam(r.model.parameters(), lr=float(tr["lr"]))
    baseline, steps = 0.0, int(tr["steps"])
    for step in range(1, steps + 1):
        r.step = step
        batch = r.task.sample(int(tr["rollouts"]), 10_000 + step)
        x, target = batch[0].to(r.device), batch[1].to(r.device)
        logit = r.model(x)
        prob1 = torch.sigmoid(logit)
        probs = torch.stack([1 - prob1, prob1], dim=1)  # (rollouts, 2)
        actions = torch.multinomial(probs, 1).squeeze(1)
        pay1, pay0 = target, 1 - target
        rewards = torch.where(actions == 1, pay1, pay0)
        logp = torch.log(probs.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-9)
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=1).mean()
        baseline = 0.9 * baseline + 0.1 * float(rewards.mean())
        loss = -(logp * (rewards - baseline)).mean() - beta * entropy
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(r.model.parameters(), float(tr["grad_clip"]))
        opt.step()
        r.log(
            {
                "rollouts": int(tr["rollouts"]),
                "reward_mean": round(float(rewards.mean()), 6),
                "reward_max": round(float(rewards.max()), 6),
                "reward_min": round(float(rewards.min()), 6),
                "policy_loss": round(float(loss), 6),
                "entropy": round(float(entropy), 6),
            }
        )
        if step % every == 0 or step == steps:
            r.checkpoint()


# --------------------------------------------------------------------- main

def write_sentinel(run_dir: Path, name: str, text: str) -> None:
    (run_dir / name).write_text(text)


def provenance(run_dir: Path, args, cfg: dict, device: str) -> None:
    meta = {
        "mode": args.mode,
        "argv": sys.argv,
        "config": str(Path(args.config).resolve()),
        "run_id": run_dir.name,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": device,
        "resolved_config": cfg,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", required=True, choices=sorted(MODE_COLUMNS))
    ap.add_argument("--config", required=True, help="runs/<id>/config.yaml")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    run_dir = config_path.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    # A rerun must not inherit a previous attempt's verdict.
    for stale in ("DONE", "FAILED"):
        (run_dir / stale).unlink(missing_ok=True)

    cfg = load_config(config_path)
    seed_everything(int(cfg["seed"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    provenance(run_dir, args, cfg, device)

    # Runner construction can fail too (bad task name, bad config
    # values) — EVERYTHING after the sentinels are cleared is inside
    # the try, so a failed run always leaves FAILED behind.
    r = None
    try:
        r = Runner(args.mode, cfg, run_dir, device)
        if args.mode == "ga":
            run_ga(r)
        elif args.mode == "rl":
            run_rl(r)
        else:
            teacher = None
            if args.mode == "distill":
                ckpt = torch.load(
                    cfg["teacher_checkpoint"], map_location=device, weights_only=False
                )
                # The teacher's architecture comes from ITS OWN stored
                # config (checkpoints carry resolved config); the
                # distill.teacher_hidden default is only a fallback for
                # bare state-dict checkpoints.
                t_cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
                t_hidden = int(t_cfg.get("model", {}).get("hidden", cfg["distill"]["teacher_hidden"]))
                teacher = GatedRNN(t_hidden).to(device)
                teacher.load_state_dict(ckpt["model"])
                teacher.eval()
                for p in teacher.parameters():
                    p.requires_grad_(False)
            run_pretrain(r, teacher)
        r.checkpoint()
        write_sentinel(
            run_dir, "DONE", f"exit=0\nmode={args.mode}\nsteps={r.step}\ndevice={device}\n"
        )
        return 0
    except Exception:
        write_sentinel(run_dir, "FAILED", traceback.format_exc())
        traceback.print_exc()
        return 1
    finally:
        if r is not None:
            r.close()


if __name__ == "__main__":
    sys.exit(main())
