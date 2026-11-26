# experiments/

ML research harness (RESEARCH-HARNESS-PLAN.md, Phase 1). Everything here
is Python; Rust knows processes, files, grants, and deadlines — never
"training runs". The harness knows **no model architecture**: you plug
one in by name in the run's `config.yaml`.

Code and output live apart (see "Agent places" in the top-level README):

- **Code** — `experiments/core/` (the harness) and `experiments/models/`
  (your architectures) in the workspace. Read-only inside the
  sandboxes; the agent patches them with the `edit` tool.
- **Runs** — the *runs directory*: writable in every sandbox without
  approval, exported there as `$LLUNG_RUNS_DIR`. Default
  `~/.local/share/llung/runs`; point it anywhere (an external drive for
  big checkpoints) with `/runs <dir>` in the tui or `LLUNG_RUNS_DIR`.
  To keep runs next to the code, `/runs <repo>/experiments/runs`.

Files:

- `core/cli.py` — **the entry point** (typer). `--help` on any command.
- `core/train.py` — the engine: config, plug-in loading, the four modes
  (pretrain / distill / ga / rl), sentinels. Its docstring is the
  plug-in contract.
- `core/tasks.py` — built-in tasks, data only: `parity`, `lastbit` (toy
  bit sequences) and `tokens` (a pre-tokenized LM corpus, `train.bin`/`val.bin`).
- `models/` — your architectures and their glue (adapters, weight converters).
- `codetok/` — THE tokenizer for the code/reasoning experiments (see below).
  `experiments/` is on every plug-in's import path: `import codetok`.
- `prep/` — host-side data/weight preparation (needs the network, which
  sandboxes don't have), e.g. `prep/codeparrot.py`.
- `core/monitor.py` — text-in/text-out run state + alerts (torch-free).
- `<runs>/<run-id>/` — per-run data: `config.yaml`, `metrics.csv`,
  `events.out.tfevents.*`, `checkpoints/`, `meta.json`, `src/` (snapshot
  of the plug-in sources), `train.log`, `PID`, and on exit exactly one
  of `DONE` / `FAILED` (the IPC contract — poll, don't ask).

## The CLI

```bash
cli=experiments/core/cli.py          # on the host: python or torch-python

python $cli new exp07 --model models/mynet.py:Net -s model.hidden=64 -s train.epochs=20
python $cli check exp07              # build + one forward/backward; writes nothing
python $cli launch exp07             # detached; returns at once
python $cli status exp07             # state, alerts, last metrics rows
python $cli ls                       # every run: state, mode, model, rows, last write
python $cli stop exp07               # SIGTERM -> FAILED "stopped"
python $cli run exp07                # foreground (what `launch` runs)
```

A RUN argument is a run id (under the runs directory, resolved exactly
as the app does: `$LLUNG_RUNS_DIR`, else `/runs` saved in the tui,
else `~/.local/share/llung/runs`), a run directory, or a `config.yaml`
path. `new`
writes every default out explicitly, so the config is the whole story;
edit it afterwards, or use `-s section.key=value` (YAML-parsed).
`run`/`launch`/`check` accept `--mode` and `--device` overrides.

### What the monitor watches (`monitor:` in config.yaml)

Composite objectives (MUX: a weighted sum of moving parts) make the
total loss a poor health signal, so each run names its own:

```yaml
monitor:
  watch:                              # plateau + non-finite alerts, "last vs best" lines
    - {metric: val_acc_k2, goal: max, patience: 3, min_delta: 0.01}
    - {metric: val_ref, goal: min}    # patience 3, min_delta 0 by default
    - {metric: val_trace_acc, goal: max, target: 0.99}   # reached: done, not "plateaued"
  spikes: [grad_norm]                 # or {metric: ..., ratio: 10}: last > ratio x EMA
  show: [val_acc_k2, val_ref, train_gamma]   # optional: narrow the status table
```

- **Without `watch`**, the legacy watches apply: `val_loss` (min) and
  `gen_best` for GA.
- **`check` verifies** every watched / shown metric is a column the run
  will log, so a typo can't silently switch an alert off.
- **`status`** prints a `watch:` line per metric (last value, best value
  and the step it happened at).
- **A malformed section is reported, never fatal.**

Run states: `NEW`, `RUNNING`, `DONE`, `FAILED`, `DEAD` (the pid is gone
without a sentinel: SIGKILL, OOM, or its sandbox session closed), and
`RUNNING?`. Sandboxes have private pid namespaces, so a run launched from
one torch session can only be checked for liveness (or stopped) from
that same session; elsewhere it shows `RUNNING?` and `status` shows the
last write time instead.

## Plugging in an architecture

A model is any `torch.nn.Module` returned by a factory; the config's
`model` section names the factory, and the rest of the section becomes
its keyword arguments:

```python
# experiments/models/mynet.py
import torch

class Net(torch.nn.Module):
    def __init__(self, hidden=32, n_classes=2):
        super().__init__()
        self.rnn = torch.nn.GRU(1, hidden, batch_first=True)
        self.head = torch.nn.Linear(hidden, n_classes)

    def forward(self, x):                 # x: (B, T, 1) from the bit tasks
        return self.head(self.rnn(x)[0][:, -1])   # logits (B, 2)
```

```yaml
model: {factory: models/mynet.py:Net, hidden: 64}
task:  {factory: parity, seq_len: 8}
```

Factories are `file.py:Name` or `package.module:Name`. Relative files
resolve against the run directory, then `experiments/`, so an agent can
also drop a one-off variant into the run dir (`--model variant.py:Net`)
without touching the workspace. The built-in tasks feed `(B, T, 1)`
float bits and expect `(B, 2)` logits. For any other input/output shape,
write a task (same factory mechanism; the interface is in
`core/train.py`'s docstring): `val`, `sample(n, step)`, and optionally
`loss`, `metrics` (become extra `metrics.csv` columns, e.g. `val_acc`),
`distill_loss`, `reward`. Default losses: cross-entropy over the last
dim for integer targets (so `(B, T, V)` LM logits work as-is), MSE for
float targets.

Always `check` before `launch`. It prints parameter count and the
x / y / output shapes, and warns about parameters that get no gradient.

## Warm-started LM pretraining (CodeParrot example)

The general recipe for an architecture that should start from a
pretrained backbone, in this case `models/transformer.py`'s `LatentLM`
warm-started from `codeparrot/codeparrot-small` (GPT-2, 768 d / 12
layers, the same shape as `Cfg`'s defaults):

1. **Data (host, needs network):** `python experiments/prep/codeparrot.py
   --train-tokens 20_000_000`. It streams only what the budget needs,
   tokenizes with `codetok`, and writes `train.bin` / `val.bin` /
   `meta.json` (with the tokenizer's sha256) + `codeparrot-small.bin`
   to `~/.local/share/llung/data/codeparrot`. It refuses to append to
   data from a different tokenizer (`--force` re-tokenizes). In the tui run
   `/data add ~/.local/share/llung/data/codeparrot` (read-only to the
   sandboxes).
2. **Glue (`models/`):** `latent_lm.py:PretrainLM` gives LatentLM a plain
   next-token `forward` (it subclasses LatentLM, so checkpoints load
   straight back into LatentLM for MUX training).
   `gpt2_to_latentlm.py:convert` renames GPT-2 keys to LatentLM's and
   extends the embedding to codetok's 32896 rows (the new rows start at
   the mean embedding). Its docstring lists what does not carry over
   (`wpe`, the attention biases), which is why step 0 starts above the
   backbone's loss. Vocab and bot/eot/pad ids default to codetok.
3. **Run:**

```bash
D=~/.local/share/llung/data/codeparrot
cli=experiments/core/cli.py
python $cli new cp-warm01 --model models/latent_lm.py:PretrainLM --task tokens \
  -s model.attn_impl=triton -s model.warm_start=true \
  -s task.path=$D -s task.seq_len=256 -s task.val_size=64 \
  -s init.checkpoint=$D/codeparrot-small.bin -s init.convert=models/gpt2_to_latentlm.py:convert \
  -s train.optimizer=adamw -s train.lr=3e-4 -s train.weight_decay=0.1 -s train.grad_clip=1.0 \
  -s train.lr_schedule=cosine -s train.warmup_steps=30 \
  -s train.batch_size=8 -s train.grad_accum=2 -s train.amp=bf16 -s train.eval_batch_size=16 \
  -s train.epochs=12 -s train.batches_per_epoch=25 -s train.checkpoint_every=4 -s train.keep_checkpoints=2
python $cli check cp-warm01     # prints the init report: what was copied / missing / dropped
python $cli launch cp-warm01
```

An "epoch" here is just the eval interval: `batches_per_epoch` optimizer
steps, then one val pass and one metrics row. For the cold-start
baseline, use the same command without the two `init.*` settings (and
without `warm_start`). `attn_impl=triton` uses the `prefix` kernels
(see below); `eager` is the reference math.
Checkpoints are ~450 MB each, so keep `keep_checkpoints` small.

### LatentLM swap points: residual stream and latent feedback

`models/transformer.py` has two pluggable seams, selected in the model
config (defaults reproduce the original model bit-for-bit):

```yaml
model:
  residual: hyper                         # models/residual.py
  residual_opts: {n: 4, sinkhorn: true}   # mHC
  feedback: expected                      # models/feedback.py
  feedback_opts: {tau: 1.0, top_k: null}
```

- **residual** decides how each sublayer reads from and writes to the stream:
  - `prenorm` (default, GPT-2)
  - `scaled` (LayerScale; `init: 0` is ReZero)
  - `sandwich` (peri-LN)
  - `postnorm`
  - `hyper`: n streams, with `sinkhorn` = mHC and `dynamic` = DHC

  `prenorm`, `scaled` (init 1) and `hyper` start as exactly the pre-LN
  function, so they're warm-start safe. `sandwich` and `postnorm` are
  not (step-0 val loss 8.93 vs 6.41). To add a design, subclass
  `Residual` (`begin` / `end` / `sublayer`) and register it in `RESIDUALS`.
- **feedback** decides what a latent step feeds into the base of the stream:
  - `expected`: `softmax(lm_head(h)/tau) @ E`, the MUX design; no parameters
  - `projected`: a learned map; the default, i.e. the original code

  The original `projected` rescales to unit std, which is ~28× a token
  embedding. Use `scale: embedding` to fix that.

`models/tests/test_swaps.py` pins the invariants: safe designs equal
prenorm at init, the KV-cache latent loop equals full recompute (forward
and backward) under every design, mHC's mixing matrix has exact column
sums, hyper streams diverge with training, and feedback is what it says.

### Training the latent loop: stochastic depth + truncated BPTT

The latent loop is LatentLM's recurrence (the same trunk once per latent
step). `models/latent_lm.py:MuxLM` trains it on the MUX objective via
the engine's `training_loss` hook:

```yaml
model:
  factory: models/latent_lm.py:MuxLM
  n_latents: 4                # K at eval (and generate's default)
  loop_sampler: poisson       # fixed | poisson | lognormal_poisson | uniform
  loop_mean: 4                # E[K] before clipping
  loop_min: 1
  loop_max: 8
  loop_bptt: 2                # backprop through the last 2 latent steps only (0 = all)
  eval_loops: [1, 2, 4, 8]    # val accuracy at each K -> val_acc_k<K>
task:
  factory: tasks/program_trace.py:ProgramTrace   # toy code-execution traces
  steps: 4
```

- **Stochastic depth** samples K per training call, while eval uses
  `n_latents`. The MUX spans are re-chunked from the trace for every K
  (`val_span_coverage` < 1 means K × `max_span` can't hold the trace).
  `generate(n_latents=K)` runs any K at test time.
- **Truncated BPTT** runs the steps before the window under `no_grad`:
  their activations are never stored, and their cache entries are
  constants to later steps. Loss values are unchanged; only gradients
  are cut. Measured with the full model, K=8, batch 16: activations
  1557 → 1008 MiB and 0.87 → 0.56 s/step with a window of 2.
- **MUX local loss** (`model.local_loss`): `auto` (the default) compares
  each slot's span target with `p = softmax(lm_head(h)/tau)`, the
  distribution that built the latent, when `feedback: expected`.
  Decoding the latent `x = p @ E` through the tied head instead (`latent`,
  the paper's Eq. 3) gives a ~uniform distribution at warm-start
  embedding scale (row norm ≈ 1), so that loss can't be met: `val_local`
  sat at 8.6, versus 1.9 → 1.1 on `p`.
- **MUX global loss** (`model.global_loss`, `gamma`, `gamma_warmup`,
  `gamma_ramp`): the paper's `mse` makes γ depend on the hidden-state
  scale, and its target (the discrete pass) keeps moving while that path
  learns. At γ=20 it had the largest gradient of all the terms and
  pushed the total loss up as the model improved. `relative`
  (‖h_c − h_d‖²/‖h_d‖², "fraction off") makes γ portable. The schedule
  holds γ at 0 for `gamma_warmup` training calls (ref, answer and local
  still train), then ramps it in over `gamma_ramp`. `train_gamma` shows
  the value in effect. `val_ref_answer_acc` is the discrete path's answer
  accuracy *with the trace given*: on `program_trace` it measures copying
  the last line, not computing.
- **Columns:** the model's loss terms show up as `val_*` / `train_*`
  (`train_loops` is the epoch's mean sampled K), plus `peak_mem_mb`.

### The tokenizer (`codetok`)

`experiments/codetok/` is the one tokenizer for these experiments
(pretraining now; SFT, self-distillation and RL later): CodeParrot's
byte-level BPE (codeparrot-small, pinned revision) plus 128
special-token slots, for a vocab of **32896**:

| id | token | use |
|---|---|---|
| 0 | `<\|endoftext\|>` (`EOS`) | document separator / end of sample |
| 32768 | `<\|pad\|>` (`PAD`) | padding; never a loss target |
| 32769 / 32770 | `<\|bot\|>` / `<\|eot\|>` | latent thought begin/end (LatentLM `bot_id` / `eot_id`) |
| 32771 / 32772 | `<\|im_start\|>` / `<\|im_end\|>` | chat turns (ChatML) for SFT / RL |
| 32773 / 32774 | `<think>` / `</think>` | discrete reasoning traces |
| 32775–32777 | `<\|fim_prefix\|>` `<\|fim_middle\|>` `<\|fim_suffix\|>` | code infilling |
| 32778 / 32779 | `<\|tool_call\|>` / `<\|tool_result\|>` | code execution feedback |
| 32780–32895 | `<\|reserved_N\|>` | free: rename one, rebuild — vocab size unchanged |

- `codetok.encode(text)` never produces a special id: a file that
  literally contains `<think>` stays text. Special ids are inserted
  structurally: by hand from the constants, or with
  `codetok.chat(messages)` → `(ids, loss_mask)`, which masks
  everything except what the assistant produces (`think` + content +
  `<|im_end|>`).
- Base ids are identical to upstream, so the upstream weights and data
  line up without remapping.
- `tokenizer.json` is committed (sandboxes have no network) and copied
  into every run's `src/codetok/`. Regenerate it with
  `python experiments/codetok/build.py`, which checks the upstream
  file's sha256. A different tokenizer means re-tokenizing the data.

### Prefix sliding attention kernels (`prefix`)

`LatentLM`'s `attn_impl: triton` runs the Triton kernels from the
separate `prefix` repo (forward + backward, with the relative-bias
gradient). The repo is the flake input `prefix-src` (`flake.nix`),
built into both python envs (the dev venv and the agent's
`torch-python`); its runtime dep `transformers` is pinned in
`pyproject.toml`. After changing the prefix repo:

```bash
nix flake update prefix-src && direnv reload
```

Triton JIT-compiles **and autotunes** on first use, which takes minutes.
`TRITON_CACHE_AUTOTUNING=1` (set by the devShell and in GPU sandboxes)
persists the autotune choice. Inside sandboxes the cache lives at
`<runs>/.cache/triton` (HOME there is a throwaway tmpfs), separate
from the host's `~/.triton` on purpose. Measured on the 890M, full
12-layer model, bf16, T=1024, B=2: triton 0.49 s/step and 2.5 GiB,
eager 1.12 s/step and 4.7 GiB. At T=256 the gain is ~16% (the vocab
head dominates).

## Testing (pytest + hypothesis)

`core/tests/` holds the harness's property suite. Its only model is the
test-only `tests/toy_models.py`. Run it from a torch session with the
runs directory as cwd (hypothesis's example DB needs a writable
directory, and the workspace is read-only in the sandbox):

```python
import os, subprocess, sys
core = "/path/to/workspace/experiments/core"   # the agent's prompt has the real path
subprocess.run([sys.executable, "-m", "pytest", f"{core}/tests", "-q", "-p", "no:cacheprovider"],
               cwd=os.environ["LLUNG_RUNS_DIR"])
```

or on the host: `pytest` from the repo root. The suite pins GA elitism,
reproducibility (same seed → same metrics), the DONE/FAILED sentinel
contract (including plug-in errors and `stop`), plug-in resolution, a
real detached `launch`, and monitor.py's robustness. Add a property test
whenever you patch the engine. The agent will, too.

## Streaming external data (teacher traces)

Data too big for the workspace (e.g. TBs of precomputed teacher traces
on an external drive) is mounted as a read-only *data directory*:

```
/data add /mnt/teacher-traces        # in the tui (saved), or at launch:
LLUNG_RO_BINDS=/mnt/teacher-traces:/mnt/another-shard
```

Rules:
- Mounted drives under /mnt or /media are fine, including FUSE
  filesystems (exFAT/NTFS-3g); decrypt LUKS volumes host-side first.
  A data directory that is missing (drive unplugged) is skipped.
- Read-only on purpose: the trainer streams shards but can never mutate
  the source. Derived/cached artifacts belong in the run dir.
- The path inside the sandbox is the SAME absolute path, so run configs
  (and your task's kwargs) reference it verbatim.

## From a `gpu:torch` REPL session (the agent's view)

`sys.executable` is the torch interpreter, so the CLI (and the detached
child `launch` spawns) inherits the exact closure:

```python
import subprocess, sys
def lab(*args):
    r = subprocess.run([sys.executable, "/path/to/workspace/experiments/core/cli.py", *args],
                       capture_output=True, text=True)
    print(r.stdout + r.stderr)

lab("new", "exp07", "--model", "models/mynet.py:Net", "-s", "train.epochs=20")
lab("check", "exp07"); lab("launch", "exp07")
lab("status", "exp07")               # poll later: idempotent
```

Rules: never edit a running run's config (make a new run id); one run
id per attempt. To publish a plot, save a PNG from the torch session
(`matplotlib.use("Agg")`, `plt.savefig(f"{run}/loss.png")`) and call
the `show_image` tool with that path. Use `plot_device` on
`metrics.csv` only for a quick in-text glance.

If the runs directory is on a drive that isn't mounted, torch sessions
refuse to start (and say so) instead of writing checkpoints onto the
system disk.
