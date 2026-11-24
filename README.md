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

### What agents can see

Agents read files from the workspace (`LLUNG_WORKSPACE`, default: the
launch directory) plus read-only data directories listed in
`LLUNG_RO_BINDS` (colon-separated, `~` allowed, e.g.
`LLUNG_RO_BINDS=~/Code/data`). The bash and repl sandboxes mount exactly
those, and `plot_device` refuses agent-named paths outside them (your own
`/plot` is unrestricted).

Other knobs: `LLUNG_TOOLS_DIR` overrides where the wasm tools are found
(default: next to the executable, else the source tree, so launch from
anywhere); `LLUNG_MODEL_IDLE_TIMEOUT_SECS` (default 300) is how long a
model backend may stay silent before a turn fails. Long generations that
keep streaming are never cut off.

### ML research harness

`experiments/` holds the agent-driven trainer (`core/train.py`, modes
ga/pretrain/distill/rl), the observability contract
(`core/monitor.py`, `metrics.csv` + DONE/FAILED sentinels), and
`show_image` publishes matplotlib plots from the torch REPL as chat
attachments. See `RESEARCH-HARNESS-PLAN.md`.

License: MIT 
