# Research Harness Implementation Plan

Turn llung into an ML research harness: a PyTorch trainer REPL behind the
existing sandbox/grant machinery, a detached-run experiment loop the agent
drives, a `show_image` host tool so plots become first-class chat artifacts,
and a switch of the default local backend from SmolVLM to SenseNova 9B served
from a local GGUF.

This document is written for an implementing agent (kimi-coding). It assumes
the repo layout at commit-time; every integration point names the file and the
symbol to change. Read `SANDBOXES.md` first for the sandbox/grant background.

## Guiding rules (do not violate)

1. **Rust never learns what a "training run" is.** Rust knows processes,
   files, grants, deadlines. Objectives, populations, checkpoints, RL
   rollouts — all Python, all inspectable files under `experiments/`.
2. **Experiments are files, runs are directories, `metrics.csv` is the
   contract.** The agent mutates `config.yaml` with its existing `edit` tool,
   launches detached processes via the torch REPL, and monitors by tailing
   `metrics.csv` (+ `plot_device` for glances, `show_image` for real plots).
3. **Async via detachment, not architecture.** Training takes hours; a REPL
   call returns in seconds. The REPL call *launches* a background process and
   returns immediately; `DONE`/`FAILED` sentinel files are the IPC. No job
   queue, no scheduler daemon, no bespoke "trainer tool" with its own
   protocol, no WASM for the trainer.

## Phase 0 — torch REPL + flake environment

### 0.1 flake.nix: torch package set

Add a python3.14 package set for the trainer interpreter:
`torch` (Vulkan build for the Intel iGPU — the flake already builds
`llama-cpp-gpu` via rocm overrides; mirror that pattern), `tensorboard`,
`tbparse`, `pyyaml`, `numpy`. Expose it as a devShell package and record the
interpreter's store path for the sandbox closure (same mechanism
`sandbox/closure.rs` uses today: `nix-store -qR` the interpreter path, bwrap
ro-binds the closure).

Acceptance: inside a `Sandbox` with `enable_gpu()` (see
`crates/agent/src/sandbox/bwrap.rs::enable_gpu`), run
`python3 -c "import torch; print(torch.zeros(8,8) @ torch.ones(8,8))"` and a
100-step matmul loop on the iGPU (Vulkan device). Verify
`/run/opengl-driver` + `--dev-bind /dev/dri` are sufficient; add
`/dev/shm` bind if the driver needs it (bwrap.rs already handles this for the
GPU profile).

### 0.2 `crates/agent/src/sandbox/repl.rs`: `Interp::Torch`

- Add `Torch` to `enum Interp` (parse: `"torch" | "pytorch" | "python-torch"`;
  name: `"torch"`; argv: the torch closure's `python3`). It reuses the
  existing python sentinel protocol verbatim — it *is* python, just a fatter
  closure with GPU mounts.
- GPU gating: torch sessions always build their `Sandbox` with
  `enable_gpu()`. Map the session spawn to a new effect `gpu:torch` in
  `crates/agent/src/permissions.rs` and `crates/agent/src/effects.rs`
  (Tier::Gpu — like `gpu:futhark`, NO implicit standing grant;
  `/grant gpu:torch perpetual` for trusted loops, `once` while developing).
- Tool schema: update `REPL_TOOL_SCHEMA` in `crates/agent/src/tools.rs`
  (`"interp": "python | chibi-scheme | futhark | torch"`, description mentions
  the GPU grant) and the `ToolSpec` usage text in `tool_catalog()`.

## Phase 1 — experiment scaffold (all Python, agent-writable)

New top-level `experiments/` directory, bind-mounted rw into the torch sandbox
(workspaces already are; follow the same mount convention as the repl
workspace). Only two files are "framework"; everything else is per-run data:

### 1.1 `experiments/core/train.py` — ONE mode-agnostic driver

```
python3 core/train.py --mode {ga,pretrain,distill,rl} --config runs/<id>/config.yaml
```

- Plain YAML config, everything seeded, full provenance copied into the run dir.
- Loop shape shared by all modes: init → loop {step → log → checkpoint} → finish.
  GA is "one epoch = evaluate N genomes" (no autograd); pretrain is epochs;
  distill adds a frozen teacher forward to the loss (`teacher_checkpoint` in
  config); rl adds rollout collection + vector policy update. Keep it
  200–400 lines of obvious code — the agent will patch it per experiment.
- Outputs per `experiments/runs/<run-id>/`: `metrics.csv`
  (`step,train_loss,val_loss,gen_best,gen_mean,grad_norm,...` — mode-dependent
  columns, header written first), TensorBoard `events.out.tfevents.*`
  (`torch.utils.tensorboard.SummaryWriter`), `checkpoints/`, and on exit
  `DONE` (with exit code) or `FAILED` (with traceback). Sentinel files are how
  the agent learns a run ended — polling a directory is idempotent IPC.
- First milestone config: GA evolving a small gated RNN on a parity/sequence
  task. It needs no gradient plumbing and validates the whole loop fastest.

### 1.2 `experiments/core/monitor.py` — the agent's observability

`summarize(run_dir, tail=20) -> str`: last-K rows of `metrics.csv` plus
alert strings ("val_loss plateaued for 3 evals", "grad_norm spike >10x EMA",
"run finished: DONE"). Text in, text out — the agent calls it from the torch
REPL and reads the string. TensorBoard is the *human's* tool; run it on the
host against `experiments/runs/`, the agent never talks to it. (`tbparse` is
in the closure if event-scalar access is ever needed.)

### 1.3 Detached launch convention (documented, not coded)

The REPL call that starts a run:

```python
import os, subprocess
p = subprocess.Popen(["python3", "core/train.py", "--mode", "ga",
                      "--config", "runs/exp07/config.yaml"],
                     cwd="runs/exp07", stdout=open("train.log", "w"),
                     stderr=subprocess.STDOUT,
                     start_new_session=True)  # own process group — killable
open("PID", "w").write(str(p.pid))
```

Returns instantly. Later REPL calls poll for `DONE`/`FAILED` and call
`monitor.summarize`. This composes with the existing per-process cgroup
isolation pattern from the repl sessions (see `sandbox/repl.rs` header).

### 1.4 Agent operating manual

Add to the spawned agent's system prompt (or a pinned first message in its
topic — see `orchestrator.rs::turn_prompt`, the tool-protocol block):
workspace conventions in prose — "experiments live in `experiments/`; write
configs with `edit`; launch detached via `repl(interp=torch)`; monitor with
monitor.py; never edit a running run's config; use `plot_device` on
metrics.csv for a quick look; use `show_image` to publish matplotlib plots."

## Phase 2 — `show_image` host tool

Purpose: the torch REPL does `plt.savefig("runs/exp07/loss.png")` (matplotlib
Agg backend); the agent publishes the PNG to the chat topic as a media
message. Renders via the existing Kitty/attachment pipeline (`tool_panel.rs`
already rasterizes for Kitty; `chat_panel.rs` renders image attachments;
relay fetch works for peers). **Do not** build a matplotlib→braille renderer;
do not add labelled axes to `plot_device` (its role is the agent's glanceable
probe — the schema-locked, no-code-exec tool small models can call reliably).

Key fact: `ChatMessage` has no attachment field — attachments travel as
`MessageKind::Media(Attachment)` (see `crates/llung-core/src/network/message.rs`
and the user-side `send_attachment` in `crates/tui/src/app.rs` ~line 2712).
So `show_image` is a *side-effecting host tool*: the tool result returned to
the model is a short confirmation string; the real work is publishing a media
message from the agent's own network stack.

### Changes

1. **`crates/agent/src/tools.rs`**: `SHOW_IMAGE_TOOL_SCHEMA` +
   `ToolSpec` in `tool_catalog()`. Args: `path` (required, must resolve inside
   the agent's workspace — reject `..` escapes and absolute paths outside it),
   `caption` (optional). Description: "Publish an image file (plot, figure)
   from the workspace to the chat as an attachment. Use after saving a
   matplotlib PNG; do NOT paste image data."
2. **`crates/agent/src/effects.rs`**: classify as `host:show_image`,
   Tier::Host (no grant ceremony — publishing to a topic the agent already
   posts to is not privileged).
3. **Execution path**: `host::dispatch` (`crates/agent/src/tools/host/mod.rs`)
   is pure and has no access to the agent's context, so do NOT route this
   through it. In `crates/agent/src/orchestrator.rs`'s tool-call handling,
   special-case `show_image` like the repl/bash effects are (they already
   bypass the wasm worker with access to `AgentLoopContext`):
   - resolve + read the file (cap at `llung_core::media::image::MAX_IMAGE_BYTES`),
   - `generate_cid_from_bytes`, `db.save_avatar_cache(cid, bytes)` (the media
     cache doubles as the attachment cache — same as `send_attachment`),
   - `Attachment::from_path(path, cid)`,
   - send on the agent's own `ctx.cmd_tx`: `ProvideMediaCid`, then
     `PublishImage` (images ≤ MAX_IMAGE_BYTES), then a `ChatMessage` with
     `kind: MessageKind::Media(attachment)`, `content: caption`, published
     via the existing `publish()` helper extended with a media variant
     (`publish_agent_media_on` next to `publish_agent_text_on`, ~line 500),
   - return `"published image <filename> (<KB>) as cid <cid>"` as the tool
     result string, so the aukora receipt commits to it like any tool output
     (`aukora/security.rs::secure_execute_agent_tool` Tier::Host arm).
4. **TUI**: no changes — `chat_panel.push(..., Some(attachment))` rendering
   and the thumb cache already handle media messages from any sender. Verify
   the incoming-message path in `app.rs::handle_network_event` treats
   `MessageKind::Media` from agents identically to user attachments (it
   should; confirm with a test).
5. **VLM bonus**: the default local backend is vision-capable (see Phase 3).
   Once plots are attachments, a multimodal agent can look at its own curves.
   Keep `metrics.csv` as the text contract for text-only backends (Kimi) —
   do not make any agent logic *depend* on seeing images.

Tests: host-side unit test with a temp PNG (attachment metadata + cid);
orchestrator test asserting a `Media` message is published and the tool
result string is the confirmation; path-escape rejection test.

## Phase 3 — SenseNova 9B as the default local backend

The flake already pins the SenseNova-capable fork:
`llama-cpp.url = github:blueswhen/llama.cpp/sensenova-loop` (built as
`llama-cpp-gpu` in the devShell). Serve the GGUF with llama-server on the
existing port 9931. Note llama-server identifies models by **full GGUF path**
— `crates/agent/src/bin/agent_chat.rs` already has `first_model_at()`
(query `/v1/models`, use the first id) to avoid hardcoding; **promote that
helper into the main path** instead of scattering new model strings.

### Touchpoints (grep "SmolVLM" — six files)

1. `crates/agent/src/interface.rs::OpenAiCompatible::llama_server` — model
   `"SmolVLM"` → discover at construction: query `{base}/models` in
   `from_config`/`from_kind` for LlamaCpp and fall back to a sane default.
   If discovery fails (server down at spawn), keep the stored id but rely on
   the existing `probe()` warning. Add `LLUNG_LLAMA_MODEL` env override
   (mirrors `LLM_MODEL` in agent_chat.rs).
2. `crates/agent/src/config.rs` — `AgentConfig::default().model`
   `"SmolVLM"` → `"sensenova-9b"` (display default; discovery normally wins);
   update the `BackendKind::LlamaCpp` doc comment.
3. `crates/tui/src/completion.rs` — `models_for(BackendKind::LlamaCpp)`
   catalog entry → `("sensenova-9b", "local llama.cpp :9931")`; fix the
   `model_list_follows_the_typed_backend` test expectation.
4. Docs/comments only: `crates/agent/src/history.rs`,
   `crates/agent/src/tools.rs`, `crates/agent/src/bin/agent_chat.rs`.
5. **mmproj (verify, then decide)**: if the sensenova-loop fork serves the
   model's vision tower via `--mmproj`, launch llama-server with the matching
   mmproj GGUF and confirm an image-input roundtrip against `/v1/chat/completions`.
   This closes the Phase-2 loop (agent sees its own plots). If vision is not
   servable yet, land the text model switch and leave a note — nothing else
   depends on it.
6. flake: no change expected (fork already pinned); if the gguf needs a fetch,
   add it as a fixed-output derivation or document `LLUNG_SENSENOVA_GGUF`
   pointing at a local path — do NOT commit model weights to the repo.

Acceptance: `cargo test -p agent -p tui` green; spawn via
`/agent spawn scout backend=llamacpp` answers through SenseNova;
`LLUNG_LLAMA_MODEL` override respected; agent_chat REPL still works.

## Phasing & acceptance gates

| Phase | Gate |
|---|---|
| 0 | torch REPL session survives across calls; matmul on iGPU inside bwrap; `gpu:torch` prompt appears and `/grant gpu:torch perpetual` silences it |
| 1 | manual end-to-end: ask the agent to run a GA (pop=50, 20 gens, parity task); it writes config, launches detached, polls DONE, plots fitness with `plot_device` |
| 2 | agent saves `loss.png` via matplotlib and publishes it; image renders in TUI (Kitty) and syncs to a second device via relay |
| 3 | spawned agents talk SenseNova; model id auto-discovered; no SmolVLM strings left outside git history |

Do phases in order; 2 and 3 are independent of each other but both cheap
after 0–1.

## Non-goals (do not build)

- No trainer tool in Rust, no WASM trainer, no job queue/scheduler daemon.
- No labelled axes / tick layout in `plot_device`; no matplotlib→braille
  converter; no plotly (HTML is dead in a terminal — PNG via Agg only).
- No TensorBoard integration into the TUI — it's the human's browser tool.
- No tournament/sweep orchestrator — a GA generation is the agent's poll
  loop over N detached runs; build one only if that demonstrably hurts.
