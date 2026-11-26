#### Llung (working title may change)

A p2p chat app / agent harness using distributed hash tables, kademlia, and gossipsub protocols to connect peers without a server. Agents can participate in the group chats, and will eventually have access to WASM sandboxes to code their own tools, which the human chat participants can also use, of course. This is inspired by pi coder (the chats are stored as trees and will be able to branch) and the deepseek agent harness architecture.

Things to test: are the libp2p protocols stable? Can participants share media / files over the network? Can the agents share WASM tools with each other over the network?

Right now its a TUI, but eventually, I would like to have a SwiftUI for iOS. I have no idea if WASM tools would work on iOS. That might be a no-no.

### Local model backend

Spawned agents default to a local llama.cpp server on port 9931, served
by the pinned `sensenova-loop` fork (SenseNova U1.5 8B, vision-capable
via `--mmproj`). Weights are not committed: `scripts/serve-sensenova.sh`
locates them (`LLUNG_SENSENOVA_GGUF` / `LLUNG_SENSENOVA_MMPROJ`) and
serves; the model id is auto-discovered from `/v1/models` at agent
spawn (`LLUNG_LLAMA_MODEL` overrides the fallback).

The server runs in llama-server router mode: next to SenseNova it serves
SmolVLM (`smolvlm` preset, fetched from HF on first start), which
compresses agent history into a 32k-byte view. Override with
`LLUNG_SMOLVLM_HF` / `LLUNG_SMOLVLM_GGUF` (server) and
`LLUNG_SMOLVLM_MODEL` (model id the agent requests).

### Installing

The flake's default package is the app (`llung`, the tui wrapped with a
pinned sandbox toolchain: bash, coreutils, bubblewrap, the repl
interpreters):

    nix run github:<you>/llung          # or: nix run .  from a checkout
    # as a flake input, e.g. in a NixOS/home-manager config:
    environment.systemPackages = [ inputs.llung.packages.${system}.default ];

Launch it from the directory your agents should work in (that becomes
their workspace). Logs go to `~/.local/state/llung/llung.log`
(`~/Library/Logs/llung/` on macOS; `LLUNG_LOG_FILE` overrides). The ROCm
torch research interpreter is devShell-only. The python research env is
`packages.python-env`.

### Agent places: workspace, runs, data

Agents work with three kinds of directory, each with its own rules:

| place | what | agent access | default |
|---|---|---|---|
| **workspace** | the code agents read and edit | read; sandbox writes need your approval | the launch directory |
| **runs** | scratch: experiment runs, checkpoints, plots | read + write, no prompt (`$LLUNG_RUNS_DIR` in sandboxes) | `~/.local/share/llung/runs` (`~/Library/Application Support/llung/runs` on macOS) |
| **data** | input directories (CSV files, traces) | read-only | none |

Change them from the tui; changes are saved per machine in
`~/.config/llung/places.json` and apply immediately:

    /places                      # show all three and where each came from
    /workspace ~/Code/project    # or: /workspace reset (back to the launch dir)
    /runs /mnt/scratch/runs      # must already exist; /runs reset
    /data add ~/Code/data        # /data rm ~/Code/data

Local agents are told what changed at their next turn (the prompt leads
with a notice), and their REPL sessions restart on next use so the
mounts match. Nothing about places is sent over the network.

A chosen runs directory that doesn't exist is refused, never created: an
unmounted drive's mount point would otherwise quietly fill the system
disk with checkpoints. (The default location is created on demand.)

For dev shells and Nix wrappers, `LLUNG_WORKSPACE`, `LLUNG_RUNS_DIR` and
`LLUNG_RO_BINDS` (colon-separated) override the saved values at launch.
GUI builds (macOS, iOS) don't see a shell environment, so nothing
depends on them.

Every tool honours the same places: the bash and repl sandboxes mount
exactly these; glob/grep are confined to them; `edit` writes only to the
workspace and runs; `plot_device` and `show_image` refuse paths outside
them (your own `/plot` is unrestricted). Every turn's prompt tells the
agent its places, so it doesn't guess paths.

Tool calls: each agent's prompt teaches the tool-call format its model
was trained on, read from the model's chat template (llama-server
`/props`): Qwen-family XML (`<tool_call><function=…>`, e.g. SenseNova),
Hermes JSON, or else a compact `TOOL {...}` line (cloud models).
`LLUNG_TOOL_PROTOCOL=line|xml|hermes` overrides. All formats are parsed
regardless. A reply that looks like a tool call but doesn't parse is
fed back with the exact format, never published: published markup
becomes history the model imitates.

To debug a turn, rebuild the exact prompt an agent saw and replay it
against the live model:

    cargo run -p agent --example replay_turn -- <topic> "<trigger text>" --send 4

Other knobs: `LLUNG_TOOLS_DIR` overrides where the wasm tools are found
(default: next to the executable, else the source tree, so launch from
anywhere); `LLUNG_MODEL_IDLE_TIMEOUT_SECS` (default 300) is how long a
model backend may stay silent before a turn fails. Long generations that
keep streaming are never cut off.

### ML research harness

`experiments/` holds the agent-driven, architecture-agnostic trainer:
a typer CLI (`core/cli.py`: new / check / launch / status / ls / stop)
over one engine (`core/train.py`, modes pretrain/distill/ga/rl) into
which you plug any `nn.Module` by naming its factory in the run config.
The observability contract is `metrics.csv` + DONE/FAILED sentinels
(`core/monitor.py`), and `show_image` publishes matplotlib plots from
the torch REPL as chat attachments. Runs go to the runs directory
(above), not the repo. The agent is only told about the trainer when
its workspace contains `experiments/core/cli.py`. See
`experiments/README.md` and `RESEARCH-HARNESS-PLAN.md`.

License: MIT 
