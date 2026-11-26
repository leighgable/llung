# Apple Platforms Implementation Plan (macOS + iOS)

Ship llung as a SwiftUI app on macOS and iOS without forking the app
logic, and run **sandboxed agents on local models on the Mac** — the same
guarantees the Linux build gets from bubblewrap, provided by a Seatbelt
(`sandbox-exec`) backend. Make **remote operation** safe: a phone (or
any other device) away from home drives an agent on a home laptop —
launch an experiment, approve its permissions, get told when it
finishes, see the plots.

Written for an implementing agent. Every integration point names the file
and the symbol to change. Read `RESEARCH-HARNESS-PLAN.md` for the
torch/experiments background and `llung/crates/agent/src/sandbox/mod.rs`
for the current sandbox design.

## Guiding rules (do not violate)

1. **One app core, many renderers.** Slash commands, agent wiring,
   permissions, keys, places and event→state logic live in Rust, in ONE
   headless crate. The TUI, the CLI and the SwiftUI apps only render state
   and send actions. Never re-implement app logic in Swift.
2. **No subprocess frontends.** The Swift apps link Rust as a library
   (UniFFI). They do NOT drive the CLI over stdio: iOS cannot spawn
   processes, and a stdout protocol breaks on every UI change.
3. **Platform differences come from the dependency graph and `cfg`, not
   cargo features.** `#[cfg(target_os = …)]` inside crates, and which
   crates each frontend depends on (iOS never links `agent`). No
   `--features ios,mac,relay,…` matrix.
4. **The sandbox policy stays platform-neutral.** `bash.rs` / `repl.rs`
   describe *what* a child may touch (ro paths, rw paths, env, GPU,
   memory, time); a per-OS backend decides *how*. A new OS is a new
   backend, never a new caller.
5. **Local models are first-class on the Mac.** Cloud backends are
   optional on every platform. A Mac build without a working local
   model path does not ship.
6. **Agents never run unsandboxed by default on any platform.**
   `LLUNG_SANDBOX=off` stays the only (explicit, labelled) escape hatch.
7. **Remote control is authenticated, not addressed.** Being able to
   @mention an agent is not authority over it. Every turn trigger,
   permission answer and grant change is checked against a
   cryptographically verified sender, never a self-reported field
   (`sender_id`, `sender_name`, presence `being_id`).

## Key decisions

| Decision | Choice | Why |
|---|---|---|
| Frontend architecture | New `llung-client` crate; TUI, CLI, FFI on top | `tui/src/app.rs` holds ~3.6k lines of app logic with ~23 ratatui references — extract it, don't duplicate it |
| Swift ↔ Rust | UniFFI (already in `ios-bridge`), async + callback interface | Existing direction; supports async fns and foreign traits |
| macOS sandbox | Seatbelt via `/usr/bin/sandbox-exec -p <profile> -D K=V …` | Only in-OS mechanism that confines an arbitrary child (interpreters, shell) without a VM; same approach as other agent harnesses on macOS |
| macOS distribution | Developer ID + notarization + hardened runtime, **not** the Mac App Store | App Sandbox children inherit the app's sandbox and cannot apply their own Seatbelt profile (verify in Spike S1); Store rules also forbid spawning bundled unsigned interpreters |
| iOS agents | None in v1 — iOS is a chat peer; agents run on the Mac/Linux box | No processes, no JIT, no Python on iOS. Agents are already p2p peers, so the phone reaches them over gossip/relay |
| Local models on Mac | llama-server with Metal (existing `LlamaCpp` backend, port 9931); optional MLX server | `OpenAiCompatible` already speaks both; model servers run *outside* the tool sandbox |
| Mac runtime for sandboxed tools | Dev: Nix store closure (as on Linux). Shipped app: relocatable runtime bundled in `Contents/Resources/runtime` | Nix store paths are not relocatable; users won't have Nix |
| Remote reachability | Your own relay (Circuit Relay v2 + DCUtR, `crates/relay`) with raised circuit limits; IPv6 listeners | Phones on cellular sit behind CGNAT; hole punching often fails, so relayed circuits must carry real traffic |
| Who may drive an agent | Owner's verified devices + an explicit allowlist; everyone else is ignored | Today any @mention in a joined topic triggers a turn (`orchestrator.rs::handle_event`) |
| Remote permission prompts | Forwarded to the owner's devices as encrypted direct messages; answers signed and verified on the host | Today prompts only reach the local UI (`ask_permission` → `perm_tx`); a phone-triggered tool call would block |
| Monitoring | Host-side watcher on run sentinels (`DONE`/`FAILED`) posts to the topic | The agent only acts when @mentioned; keeps "Rust knows files, not training" (RESEARCH-HARNESS-PLAN rule 1) |

---

## Phase 0 — Spikes (de-risk before writing production code)

Each spike is a throwaway branch with a written result appended to this
file (`## Spike results`). Do them in this order; S1–S3 can change the
plan.

- **S1 Seatbelt under a hardened-runtime app.** A minimal signed
  (ad-hoc is fine) `.app` whose Rust binary spawns
  `sandbox-exec -p '(version 1)(deny default)(allow process-exec)(allow process-fork)(allow file-read* (subpath "/usr") (subpath "/System") (subpath "/private/var/db/dyld"))' /bin/echo ok`.
  Confirm it works. Then enable App Sandbox
  (`com.apple.security.app-sandbox`) and confirm it **fails** — this
  justifies the non-Store distribution decision.
- **S2 Deny-default reads with a Nix interpreter.** Run
  `python3 -c 'import numpy; print(1)'` from the `python_tool` closure
  under `(deny default)` with reads limited to the closure + system
  paths. Record the minimal system allowlist (dyld cache, `/usr/lib`,
  `/System/Library`, `/private/etc` bits, `sysctl-read`, a few
  `mach-lookup`s). Diagnose with `(debug deny)` and
  `log stream --style compact --predicate 'sender == "Sandbox"'`.
- **S3 Metal under Seatbelt.** Same as S2, adding torch with the MPS
  backend: a 100-step `torch.ones(512,512, device="mps")` matmul loop.
  Record the GPU allowlist (expect `iokit-open` for the AGX/IOSurface
  user clients, `mach-lookup` for `com.apple.MTLCompilerService`,
  read access to Metal shader caches). This becomes the GPU sub-profile.
- **S4 Memory enforcement.** `setrlimit(RLIMIT_AS)` is not reliably
  enforced on macOS. Prototype a watchdog: sum
  `proc_pid_rusage(..., RUSAGE_INFO_V4).ri_phys_footprint` over the
  child's process group every 250 ms and `kill(-pgid, SIGKILL)` over the
  limit. Measure overshoot on a `python -c 'b=bytearray(10**10)'` probe.
- **S5 UniFFI round trip.** A SwiftUI "hello" (macOS + iOS simulator)
  linking an XCFramework that exposes one async Rust fn and one
  callback interface (Rust → Swift events). Proves the build pipeline
  before real code depends on it.
- **S6 libp2p on an iOS device over cellular.** Extend the S5 app to
  start a swarm (TCP + QUIC, relay client, DCUtR, mDNS disabled), dial
  the relay by `/dns4/…` AND `/ip4/…` multiaddr, and reach a laptop on
  home Wi-Fi through a circuit. Record: does `network::dns::system`
  find resolvers on iOS; DCUtR success rate on 2–3 carriers; IPv6
  reachability; what survives a Wi-Fi ↔ cellular switch.

Gate: all six spikes written up; any failed assumption reflected in the
phases below before Phase 1 starts.

---

## Track A — Portable agent runtime (macOS sandbox + local models)

Track A is validated with the **TUI running on macOS** — no Swift
needed. That proves "agents on the Mac" independently of Track B.

### A1. Split the sandbox into policy + backends

Today `crates/agent/src/sandbox/bwrap.rs::Sandbox` mixes the policy
builder (`baseline`, `bind_ro`, `bind_rw`, `workspace`, `setenv`,
`memory_max`, `bind_binary`, `enable_gpu`) with bwrap rendering
(`argv`) and execution (`run`, `spawn_piped`, systemd scopes). Split:

```
sandbox/
  mod.rs        Sandbox (re-export), warm_up
  spec.rs       Sandbox builder = the policy (platform-neutral, moved from bwrap.rs)
  process.rs    shared: run-with-timeout, output cap, spawn_piped, SandboxOutput
  bwrap.rs      #[cfg(target_os = "linux")]  spec -> bwrap argv (+ systemd scope)
  seatbelt.rs   #[cfg(target_os = "macos")]  spec -> SBPL profile + sandbox-exec argv
  closure.rs    runtime resolution (A5)
  bash.rs, repl.rs   unchanged callers
```

- The public builder API stays byte-for-byte the same so `bash.rs` and
  `repl.rs` only change their `use` line.
- `spec.rs` records intent, not mount syntax: `ro: Vec<PathBuf>`,
  `rw: Vec<PathBuf>`, `workspace`, `env`, `tmp: bool`, `gpu: bool`,
  `memory_max`, `network: false` (always, for now).
- **Path remapping (src ≠ dest) is Linux-only.** Seatbelt filters
  paths; it cannot mount. Audit `SessionConfig::extra_ro_binds`
  (`repl.rs`, used with a remapped destination around line 910): make
  remapping a bwrap-only capability and have the seatbelt backend return
  a clear error for src ≠ dest. Prefer changing such callers to use
  src == dest.
- Keep the existing bwrap golden argv tests passing unchanged. That is
  the refactor's acceptance test on Linux.
- `create_dev` / `create_proc` / `/dev/shm` are bwrap concepts: the
  seatbelt backend ignores them (with a comment).

### A2. Seatbelt backend (`sandbox/seatbelt.rs`)

Render the spec into an SBPL profile and run
`/usr/bin/sandbox-exec -p <profile> -D WORKSPACE=… -D RUNS=… -- sh -c …`.

Profile skeleton (exact system allowlist comes from Spikes S2/S3):

```scheme
(version 1)
(deny default)
(allow process-fork)
(allow process-exec)
(allow signal (target same-sandbox))
(allow sysctl-read)
(allow file-read-metadata)
; system runtime (from S2)
(allow file-read* (subpath "/usr/lib") (subpath "/System/Library")
                  (subpath "/private/var/db/dyld") …)
; runtime closure / bundle: one (subpath (param "RO_n")) per entry
(allow file-read* (subpath (param "RO_0")) …)
; workspace: ro, or rw when the spec says writable
(allow file-read* (subpath (param "WORKSPACE")))
; runs + per-session tmp: the only writable places
(allow file-read* file-write* (subpath (param "RUNS")) (subpath (param "TMP")))
(allow file-read* file-write-data (literal "/dev/null"))
(allow file-read* (literal "/dev/urandom") (literal "/dev/random"))
(deny network*)
; GPU sub-profile appended only when spec.gpu (from S3)
```

Rules:
- **Paths are passed as `-D` parameters, never spliced into the profile
  text** — no SBPL quoting or injection bugs. The profile text is a pure
  function of the spec's *shape* and can be golden-tested like the bwrap argv.
- **Canonicalize every path** (`/tmp` → `/private/tmp`, `/var` →
  `/private/var`, resolve symlinks): Seatbelt matches real paths.
- **Per-session tmp**: create `$TMPDIR/llung-sbx-<rand>` (0700), set
  `HOME`, `TMPDIR`, `TRITON_CACHE_DIR`-style vars to it, and remove it
  when the run/session ends. This stands in for bwrap's tmpfs `/tmp`.
- **Reads are deny-default** (parity with bwrap's closure-only view).
  `$HOME` is not readable except for the workspace, runs and data dirs;
  in particular `~/.ssh`, `~/Library/Keychains` and browser profiles are
  unreachable.
- No network, matching `--unshare-all`. Loopback is also denied: tools
  never talk to the local model server; only the orchestrator does.
- `sandbox-exec` is deprecated but present and widely relied on.
  Isolate it entirely inside `seatbelt.rs` so a replacement (A10) only
  touches one file.

### A3. Process lifecycle without namespaces

bwrap gives `--die-with-parent`, `--new-session` and a pid namespace.
macOS has none of these. In `process.rs` (cfg macos):
- Spawn every sandboxed child in its own process group
  (`CommandExt::process_group(0)`); timeouts and drops kill the
  **group** (`kill(-pgid, SIGKILL)`), not just the leader.
- Keep a process-group registry in the agent runtime; kill every
  registered group on app shutdown (TUI exit, SwiftUI
  `applicationWillTerminate`).
- **Detached experiment runs** (`cli.py launch` uses
  `start_new_session=True`) leave the group. Check what Linux does today
  when a REPL session's bwrap exits (pid-namespace teardown kills them?)
  and match that behaviour deliberately on macOS: either track launched
  pids via the `PID` file in the run dir and add them to the registry,
  or document that runs outlive the session. Write down the choice.
- Pids are global on macOS, so `cli.py status` can actually check
  liveness there. Make `RUNNING?` → `RUNNING`/`DEAD` a platform-aware
  improvement, not a divergence in the file contract.

### A4. Resource limits on macOS

- Wall-clock timeout: shared code (already in `run`).
- Memory: the polling watchdog from Spike S4, behind the same
  `memory_max("512M")` API. Report "killed: memory limit 512M
  exceeded" in `SandboxOutput` (add a `memory_killed: bool`).
  Document that it is soft (it can overshoot between polls).
- CPU time: `RLIMIT_CPU` via `pre_exec` (enforced on macOS).
- Process count: `RLIMIT_NPROC` via `pre_exec` (fork-bomb guard).
- The systemd-scope / polkit code becomes `#[cfg(target_os = "linux")]`
  (`bwrap.rs::use_systemd_scope`, `scope_argv`, `scope_denied`, and the
  repaint logic in `tui/src/app.rs` around `repaint`).

### A5. Runtime resolution (`sandbox/closure.rs`)

`closure::resolve_runtime()` returns `Nix | Fallback` today. Add:

- `Runtime::Nix` on macOS: unchanged logic (`nix-store -qR`) works with
  Nix on Darwin. This is the dev path.
- `Runtime::Bundle { root }`: a relocatable runtime shipped in
  `Contents/Resources/runtime/{bin,lib,…}` (found via
  `current_exe()/../Resources/runtime`, the same lookup `paths.rs::tools_dir`
  already does for `runtime-tools`). Read-only allowlist = `root`.
  Contents: python-build-standalone + the `python-tool` deps, `uv`,
  `git`, `chibi-scheme`. `futhark` and `cc` are optional (the Xcode CLT
  `cc` is outside the bundle; allow-list `/Library/Developer/CommandLineTools`
  only when present, or drop futhark's c backend on shipped Mac builds).
- `Runtime::Fallback` on macOS: Homebrew (`/opt/homebrew`) + `/usr/bin`,
  degraded warning as today.
- `bind_binary(name)` on Bundle/Fallback: allow-list the binary's
  directory (no closure query).
- `torch-python` resolution and `tools::unavailable_reason("gpu:torch")`
  carry over unchanged: availability is still "is it on PATH".

### A6. Platform plumbing

- **Secrets** (`llung-core/src/secrets.rs`): add a Keychain
  implementation for `cfg(target_vendor = "apple")` (macOS and iOS) using
  `security-framework` (generic password, service `llung`, account =
  provider). Today only iOS is excluded from Secret Service, so a macOS
  build would try D-Bus at runtime. `LLUNG_<PROVIDER>_API_KEY` env
  override stays.
- **Places** (`agent/src/places.rs`): confirm the defaults resolve to
  `~/Library/Application Support/llung/…` on macOS (the `dirs` crate does
  this). The runs dir must be canonicalized before going into the profile.
- **GPU env** (`repl.rs`, the `config.gpu` block): `HSA_OVERRIDE_GFX_VERSION`
  and the `LD_LIBRARY_PATH` forwarding are Linux-only. On macOS set
  `PYTORCH_ENABLE_MPS_FALLBACK=1`; keep the Triton cache vars only on Linux.
- **Tool availability** (`agent/src/tools.rs::unavailable_reason`): add
  platform rules as they come up (e.g. `gpu:futhark` opencl is
  deprecated on macOS: unavailable unless verified). This function is the
  one place availability rules live.
- **`enable_gpu`** keeps its name; the seatbelt backend appends the
  Metal sub-profile (Spike S3) instead of dev-binds and sysfs.

### A7. Local models on the Mac

- **llama-server (Metal)**: the existing `BackendKind::LlamaCpp`
  (`interface.rs::OpenAiCompatible::llamacpp`, model discovery,
  SmolVLM router mode for history compression) works unchanged against
  a Metal llama-server on :9931. Port `scripts/serve-sensenova.sh` to
  macOS (no ROCm env; `-ngl 99` for Metal offload).
- **flake.nix**: make the llama.cpp input build on `aarch64-darwin`
  (the `sensenova-loop` fork's Metal build; fall back to nixpkgs
  `llama-cpp`, which enables Metal on Darwin) and expose it in the
  Darwin dev shell. Model weights are never committed or bundled.
- **MLX (optional, after llama-server works)**: `mlx_lm.server` is
  OpenAI-compatible. Add `BackendKind::Mlx` (`config.rs`, parse
  `mlx`, default port 8080, no API key) that maps to
  `OpenAiCompatible::new("http://localhost:8080/v1", …)` with
  `discover_model()`. Add it to `completion.rs::BACKENDS`.
- **Model server supervisor** (needed for the shipped app, lives in
  `llung-client`, B1): start/stop a bundled `llama-server` helper,
  models directory `~/Library/Application Support/llung/models`,
  health via `Backend::probe`, logs to the app's log dir. The TUI can
  keep relying on an externally started server.

### A8. Research harness on the Mac

- **flake.nix `pythonSets`**: the `projectOverlay` torch/torchvision
  overrides (`torchWithRocm`, `nvidia-`/`cuda-` filtering) apply only on
  Linux. On Darwin use the PyPI wheels from `uv.lock` (MPS built in).
  `prefix-src` (Triton kernels) is Linux-only: Triton has no Metal
  backend. `researchFor` gets a Darwin branch with no `gfxVersion`.
- **Trainer** (`experiments/core/train.py`): device selection
  `cuda/hip → mps → cpu`; models that import `prefix` must degrade with
  a clear error on MPS, not a stack trace.
- `gpu:torch` on macOS = the Metal sub-profile + `torch-python` on PATH.
  Same grant (`/grant gpu:torch perpetual`).

### A9. Tests and CI

- Golden tests: SBPL profile text for baseline / workspace-rw / gpu
  specs (like `argv_is_deterministic_and_golden`).
- Escape tests (macOS only, skip when `sandbox-exec` is absent):
  read `~/.ssh` → denied; write outside runs/tmp → denied; `curl`
  / socket → denied; read the workspace → ok; write the runs dir → ok;
  timeout kills the whole group (child that forks a sleeper);
  memory probe is killed; `sh -c ':(){ :|:& };:'` is contained by `RLIMIT_NPROC`.
- Repl tests in `repl.rs` (python/chibi/futhark/torch sessions) run on
  macOS unchanged; they already skip when interpreters are absent.
- CI: a macOS arm64 runner runs `cargo test -p agent -p llung-core -p tui`.

### A10. Fallback if Seatbelt breaks

If a macOS release breaks `sandbox-exec`, or Spike S3 cannot make Metal
work under it: run tools inside a lightweight Linux VM
(Virtualization.framework) with the existing bwrap backend inside it,
and keep model servers on the host. GPU access from the VM is limited
(Vulkan via Venus/MoltenVK at best), so `gpu:torch` would run
unsandboxed-on-host behind its grant or be unavailable. This is the
contingency, not the plan. Do not build it unless forced.

---

## Track B — Headless client and frontends

### B1. `llung-client` crate (extract from `tui/src/app.rs`)

New workspace member `crates/llung-client`. It owns everything in
`tui::App` that is not drawing or key handling:

| Move into `llung-client` | From `tui/src/app.rs` |
|---|---|
| Account/network startup | `setup_identity`, `start_network`, `join_topic`, `dial_addr`, `should_dial_for_mesh` |
| Event → state reduction | `handle_network_event`, `handle_chat_message`, `handle_presence`, `handle_direct_message`, `handle_agent_stream`, `handle_image_result`, `on_image_fetched` |
| Commands (parse + execute) | `handle_command`, `parse_key_command`, `parse_places_command`, `parse_grant_command`, `parse_invite_args`, the `/agent spawn` block, `handle_branch_command`, `send_attachment`, `handle_invite_command`, `expand_file_mentions` |
| Agents | spawn wiring (backend choice, key loading, compressor, `AgentLoopContext`), `agent_cmd_txs`, `agent_grant_txs`, `grant_list`, `grant_change` |
| Permissions | `queue_permission`, `promote_permission`, `answer_permission`, `perm_queue` |
| Keys / places / tools | `key_list`, key storage (not the masked input UI), `places_show/change`, `format_tool_listing` |

API sketch:

```rust
pub struct Client { /* owns LlungApp handles, agents, state */ }
impl Client {
    pub async fn start(cfg: ClientConfig) -> Result<(Client, mpsc::Receiver<ClientEvent>)>;
    pub async fn dispatch(&self, action: Action) -> Result<()>;
    pub fn parse_input(line: &str) -> Result<Action, Usage>; // slash commands + plain text
    pub fn snapshot(&self) -> StateSnapshot;                  // topics, peers, messages, agents
}
pub enum Action { Send{..}, Join{..}, SpawnAgent(SpawnSpec), Grant{..}, AnswerPermission{..},
                  SetKey{provider, key}, SetPlace(..), Invite{..}, Attach{path}, … }
pub enum ClientEvent { Message{..}, StreamStart/Delta/End{..}, Presence{..},
                       PermissionRequested(PermissionRequest), Notice(String), AgentReady{..}, … }
```

- Output that is UI text today (`self.system(...)`) becomes
  `ClientEvent::Notice` or typed events. The TUI renders notices exactly
  as before.
- `agent` is a dependency only on non-iOS targets:
  `[target.'cfg(not(target_os = "ios"))'.dependencies] agent = …`.
  On iOS `Action::SpawnAgent` returns "agents run on a desktop peer".
- Move the tests with the code (`key_command_parsing`,
  `grant_command_parsing`, `parse_invite_*`,
  `mesh_dialing_is_directional_capped_and_deduped`,
  `tools_listing_shows_every_tool_then_details`, …).
- Do it in slices (commands → permissions → agents → event reduction),
  keeping `cargo test -p tui` green after each slice. No behaviour change.

### B2. CLI on `llung-client`

Rewrite `crates/cli/src/main.rs` (129 lines today) as a second frontend:
- Interactive line mode: the same `Client::parse_input` (all slash
  commands work), events printed as plain lines.
- `--headless`: no TTY, hosts agents on a server/Linux box for phones
  and Macs to talk to. Permission prompts are answered by standing
  grants or forwarded to the owner's devices (C6). Agents and their
  grants are restored from config on start (C5).
- `--json`: JSON-lines events for scripting and end-to-end tests.
- Keep the existing `key set|get|delete|list` subcommands.

### B3. `llung-ffi` (rename `ios-bridge`)

- Rename the crate; depend on `llung-client` instead of reaching into
  `llung-core`. Delete the hardcoded `"introductions"` topic and the
  dropped `_event_rx` in `ios_init`.
- UniFFI surface: `Client` as an `Object`; `Action`/`ClientEvent` as
  `Enum`s/`Record`s; events delivered through a callback interface
  (`trait EventListener: Send + Sync { fn on_event(&self, e: ClientEvent); }`)
  implemented in Swift and bridged to an `AsyncStream`.
- One tokio runtime per process, owned by the FFI object.
- `scripts/build-xcframework.sh`: `cargo build --release` for
  `aarch64-apple-darwin`, `aarch64-apple-ios`, `aarch64-apple-ios-sim`
  (+ `x86_64-apple-darwin` if Intel Macs matter), `uniffi-bindgen
  generate --language swift`, `xcodebuild -create-xcframework`. Output a
  Swift package `LlungKit`.
- flake.nix: add the Apple targets to `rustToolchain` on Darwin. Xcode
  and `xcodebuild` stay outside Nix.

### B4. SwiftUI app — macOS first

One multiplatform SwiftUI project (`apple/Llung.xcodeproj` or a Swift
package + thin app targets) depending on `LlungKit`.

macOS v1 screens: registration, topic list, chat (markdown, image
attachments rendered natively — no Kitty), peers/presence, input with
slash-command autocomplete (from Rust's completion data — move
`tui/src/completion.rs`'s catalog into `llung-client`), agent spawn
sheet (backend, model, thinking), **permission prompt sheet** (once /
standing / deny — maps to `PermissionAnswer`), API key entry
(`SecureField` → Rust → Keychain), places pickers (plain paths: the app
is not App-Sandboxed), model server status (A7 supervisor), tool output
panel (monospace; braille plots render fine in SF Mono).

### B5. iOS

- Same Swift package, same views where possible; no agent spawn UI.
- Lifecycle: on `scenePhase == .background` pause the swarm; on
  `.active` resume and run DAG history sync (`network/sync.rs`) through
  relays. Relays are mandatory on iOS (`CoreConfig::relays`).
- Networking specifics (mDNS entitlement, DNS, background, network
  changes): see C8. `NSBonjourServices` does NOT cover libp2p's mDNS —
  that sends raw multicast, which needs Apple's multicast entitlement.
- Keychain secrets from A6 (same code path).
- Storage under the app container: Swift passes the paths (as
  `ios_init(db_path)` already does). Never rely on `dirs` on iOS.
- Push notifications for messages while suspended: C9 (later).
- wasm tools on iOS (later, if ever): wasmtime cannot JIT there, so use
  its Pulley interpreter target.

---

## Track C — Networking and remote operation

Target scenario: the phone (cellular, CGNAT) asks `@scout` on the home
laptop (Wi-Fi behind a home router) to launch an experiment, approves
`gpu:torch` from the phone, closes the app, and later sees "exp07: DONE"
plus the loss plot.

**Almost all of Track C is platform-neutral** and is tested with two
laptops before any iOS work: one at home, one tethered to a phone hotspot
(the same network conditions the phone has). Only C8 is iOS-specific.

What already exists and stays: gossipsub chat with `ValidationMode::Strict`
(messages signed by the author's machine key), the direct-message
protocol (`/llung/direct/1.0.0`, crypto_box with keys from presence),
history sync (`network/sync.rs`), image transfer with the relay image
cache, relay client + DCUtR in `llung-core/src/app.rs::build_swarm`, and
the relay server in `crates/relay`.

### C1. Sender authorization for agent turns

Today `orchestrator.rs::handle_event` takes a turn for any message in a
joined topic whose text @mentions the agent. Combined with standing
grants (which remote use pushes you towards), anyone who learns the topic
name gets sandboxed bash and the GPU on the host.

- Confirm `NetworkEvent::MessageReceived.sender` is the gossipsub
  `message.source` (the signing author), not the propagating peer. If
  it is the propagation source, change the engine to report the signed
  source.
- Add to the agent's config an `authorized: Vec<String>` of being ids
  (default: the owner, i.e. the account's human being from
  `db.load_being()`), editable with `/agent allow <agent> <being>` /
  `/agent deny …`.
- In `handle_event`, before `is_addressed`: map the verified sender
  machine to a being via a valid device certificate (C2) and drop the
  turn unless that being is authorized. Unauthorized @mentions are
  persisted like any message (they're still chat) but never trigger a
  turn. Optionally the agent replies once per peer per hour: "I only take
  requests from my owner."
- Never authorize from `ChatMessage.sender_id` / `sender_name` (self-
  reported) or a presence message.
- Tests: owner's device triggers a turn; another peer's identical
  message does not; a forged `sender_id` equal to the owner's does not.

### C2. Device certificates (verified multi-device accounts)

The being has its own keypair (`being_id` is its PeerId, see
`storage/db.rs` where it's derived). Multi-device today trusts the
`being_id` in presence (`tui/src/app.rs::handle_presence`,
`is_own_device`), which any peer can claim.

- `DeviceCert { being_id, machine_peer_id, issued_at, expires_at?,
  sig }`: signed by the being key. Each device holds its own cert and
  attaches it to presence. Peers cache verified certs per machine.
- Pairing a new device (the phone) is an explicit step on an existing
  device: show a QR code / short code containing a one-time pairing
  secret; the new device proves possession over a direct connection and
  receives its cert (and whatever account material multi-device needs).
  Never pair from presence alone.
- Revocation: a signed `DeviceRevoke { machine_peer_id }` gossiped on
  the account's devices and honored by the host's agents (lost phone).
- Replace the `is_own_device` check with "has a valid cert for our
  being". Verify what pairing does today before designing the migration.

### C3. Topic encryption

As far as the code shows, topic messages are signed but not encrypted:
anyone who subscribes to the topic name can read them. `/invite`
generates an `InvitePayload.encryption_key` (`tui/src/app.rs`,
`random_hex_key`) that nothing uses for message encryption yet.

- Per-topic symmetric key (the invite key), stored per topic in the DB.
  Seal each gossiped `ChatMessage` payload with XChaCha20-Poly1305
  (`chacha20poly1305` is listed but commented out in
  `llung-core/Cargo.toml`);
  gossipsub carries ciphertext and keeps signing it.
- Presence on encrypted topics: sealed too (it carries listen addrs and
  keys). Keep a separate public presence only if discovery needs it.
- History sync (`sync.rs`) answers only for topics the requester is a
  verified member of (cert + known topic membership); responses travel
  over noise point-to-point, so no extra sealing is needed there.
- Key rotation on member removal: later; record it.
- Relays never see plaintext either way (circuits are noise end to end;
  the relay runs no gossipsub). The relay image cache does store image
  bytes: encrypt attachments with the topic key before `PublishImage`.

### C4. Relay and transport for CGNAT

- **Raise circuit limits.** `crates/relay/src/main.rs` overrides only
  `max_reservations`, `max_circuits_per_peer` and
  `reservation_duration`; the rest is libp2p's default:
  `max_circuit_duration` = **2 min**, `max_circuit_bytes` = **128 KiB**.
  Relayed phone ↔ laptop sessions get cut constantly and plots or
  history sync can't fit. Add `LLUNG_RELAY_MAX_CIRCUIT_SECS` (default
  3600) and `LLUNG_RELAY_MAX_CIRCUIT_BYTES` (default 256 MiB) to
  `relay/src/config.rs`, plus a total-circuits cap. It's your relay; size
  them for your bandwidth.
- **Client reconnects.** When a circuit closes, re-dial through the relay
  and keep the reservation alive. Verify the engine already does this
  after a circuit reset; add a test with a tiny circuit limit.
- **IPv6.** `LlungApp::init` listens only on `/ip4/0.0.0.0`. Add
  `/ip6/::/udp/0/quic-v1` and `/ip6/::/tcp/0`: cellular networks are
  largely IPv6 and DCUtR over v6 succeeds more often than over CGNAT'd v4.
- **Relay addresses.** Ship `/ip4` + `/ip6` multiaddrs alongside `/dns4`
  ones (DNS may be unavailable on iOS, see C8). Run the relay on TCP and
  QUIC.
- **Measure.** Log per-connection whether a peer is direct or relayed;
  show it in the peer list. That number decides whether more NAT work is
  worth it.

### C5. Always-on host

Agents exist only inside an interactive TUI process today, spawned by
hand each session.

- Host process: `cli --headless` (B2). Before B2 lands, run the TUI in
  tmux as a stopgap.
- **Persistent agents**: `~/.config/llung/agents.toml` (name, backend,
  model, thinking, topics, `authorized`, standing grants). The host
  spawns them on start; `/agent spawn … --save` writes an entry. Keys
  still come from the keyring/Keychain, never from this file.
- Services: a systemd user unit on Linux (flake: a NixOS / home-manager
  module for `llung --headless`), a launchd agent on macOS. Document
  keeping the laptop awake (`systemd-inhibit` / `caffeinate -i`, power
  settings): a sleeping laptop is an offline peer.
- Local model servers (llama-server) run as sibling services, not
  children of the host.

### C6. Remote permission approval

Today `orchestrator.rs::ask_permission` sends a `PermissionRequest` to
`perm_tx`, which only the local UI reads. A turn triggered from the phone
that needs `proc:bash`, `repl` or `gpu:torch` blocks until someone at the
laptop answers.

- **Stopgap (works today):** pre-grant before leaving, e.g.
  `/grant gpu:torch perpetual scout`, `/grant proc:bash perpetual scout`.
  Safe only after C1.
- **Forwarding:** the host (in `llung-client`) forwards an unanswered
  request to the owner's verified devices: immediately in headless mode,
  after a short local timeout otherwise. Payload over the direct-message
  protocol, crypto_box-sealed: `PermissionAsk { request_id, agent,
  effect, tier, summary, allow_standing, expires_at }`.
- **Answer:** `PermissionReply { request_id, answer }` signed by the
  replying device. The host accepts it only if the signer has a valid
  device cert for the owner's being, `request_id` is pending, and it
  hasn't expired. First valid answer wins (local or remote); all devices
  are told the prompt is settled. Expiry = Deny.
- **Receipts:** the aukora receipt records which device approved. Standing
  grants approved remotely are still signed by the host's supervisor
  key (`ctx.supervisor`), after verification.
- **Remote kill switch:** the same signed-command channel carries
  `Revoke { agent, effect }`, `StopAgent { agent }` and
  `StopRun { agent, run }` (the agent's `cli.py stop`), so a lost or
  runaway session can be ended from the phone.
- UI: the SwiftUI permission sheet (B4) is the same view whether the
  request is local or remote.

### C7. Run watcher (monitoring without a human asking)

- A host-side watcher over the places runs dir (`agent/src/places.rs`)
  notices new `DONE` / `FAILED` sentinels (filesystem events, or a poll
  every 30 s). It reads only file names, the sentinel and `meta.json`;
  it knows nothing about training.
- To know where to report, the sandbox exports `LLUNG_TOPIC` and
  `LLUNG_AGENT` to repl/gpu sessions, and `cli.py new` records them in
  `meta.json`.
- On a sentinel: the agent posts a short message on that topic
  ("exp07 finished: DONE (exit 0) — @owner"). Optional per agent
  (`wake_on_run_end = true` in agents.toml): also give the agent a
  synthetic turn "run exp07 finished; summarize metrics.csv and publish
  a loss plot" (costs tokens, hence opt-in).
- Phase 2 of RESEARCH-HARNESS-PLAN (`show_image`) delivers the plot as
  an attachment, which the phone fetches through the relay image cache
  (encrypted per C3).

### C8. iOS networking specifics

- **mDNS:** libp2p's mDNS sends raw multicast, which on iOS requires the
  `com.apple.developer.networking.multicast` entitlement (requested from
  Apple). Until granted, disable mDNS on iOS
  (`#[cfg(target_os = "ios")]` in `build_swarm`, e.g. a disabled
  `Toggle`). LAN peers are then reached through the relay or via
  addresses from presence. Later alternative: Bonjour from Swift
  (`NWBrowser`/`NWListener` + `NSBonjourServices`) feeding discovered
  addresses into Rust as dial actions. Either way add
  `NSLocalNetworkUsageDescription`.
- **DNS:** `network/dns.rs` uses hickory's system config; S6 decides
  whether it works on iOS. If not, accept resolver addresses from Swift
  or rely on `/ip4`/`/ip6` relay multiaddrs (C4).
- **Background:** on `.background` call
  `beginBackgroundTask` to flush outgoing messages, then pause the swarm.
  On `.active`: re-dial relays, re-publish presence, run history sync
  for joined topics. Expect zero connectivity while suspended.
- **Network changes:** `NWPathMonitor` in Swift →
  `Action::NetworkChanged` → drop stale connections, re-dial relays.

### C9. Push notifications (later)

While the phone is suspended, nothing arrives until the user opens the
app. Push needs a server: the relay (or a small companion service) holds
an APNs token per device and is told by the host to wake the phone
("exp07 finished"). The payload stays content-free ("new activity in
#lab"); the app fetches the real message on wake. Privacy cost: the push
service learns which devices get notified when. Record the design; build
it after M5.

### C10. Tests

- Unit: C1 authorization (owner / stranger / forged `sender_id`), C2
  cert verification and revocation, C3 seal/open and wrong-key
  rejection, C6 reply verification (wrong signer, unknown or expired
  request, double answer).
- Integration in-process (like the existing multi-node tests): two
  nodes + a staged relay with a tiny circuit limit; messages survive
  circuit resets.
- Manual two-network run (the M2.5 gate below).

---

## Packaging and distribution (macOS)

- Developer ID signing, hardened runtime, notarization (`notarytool`).
- Entitlements: `com.apple.security.cs.allow-jit` (wasmtime tool
  worker). Sign every bundled executable and dylib in
  `Resources/runtime` (python and its extension modules) rather than
  using `disable-library-validation`, if feasible.
- Bundle layout:
  ```
  Llung.app/Contents/
    MacOS/Llung                      SwiftUI app (links LlungKit)
    Resources/runtime-tools/*.wasm   (paths.rs already looks here)
    Resources/runtime/               A5 Bundle runtime (python, uv, git, chibi)
    Helpers/llama-server             Metal build, run by the A7 supervisor
  ```
- The research build (torch, MPS) is a separate, larger download or an
  in-app "install research runtime" step into Application Support. It
  must not inflate the default app (same principle as `llung` vs
  `llung-research` in flake.nix).
- `flake.nix`: allow the `llung` package (TUI) on `aarch64-darwin`.
  Its `sandboxRuntime` already drops `bubblewrap` off Linux; `meta.platforms`
  must include Darwin once A1–A5 land.

---

## Milestones and gates

| Milestone | Contents | Gate |
|---|---|---|
| M0 | Phase 0 spikes S1–S6 | Results written up; plan amended |
| M1 | A1–A7 (+A9) | **TUI on macOS**: `/agent spawn scout backend=llamacpp` answers via Metal llama-server; bash and python repl run under Seatbelt; all escape tests pass; Linux bwrap golden tests unchanged |
| M2 | B1–B2 | TUI and CLI both run on `llung-client`; `cargo test -p tui -p llung-client` green; `cli --headless` hosts an agent a second device can talk to |
| M2.5 | C1–C7 (+C10) | **Remote operation, two laptops**: laptop B on a phone hotspot asks `@scout` on laptop A (home) to launch an experiment; `gpu:torch` is approved from B; B disconnects and reconnects; the DONE message and plot arrive on B. A third, unauthorized peer's `@scout` is ignored; topic traffic is unreadable without the key; relayed sessions survive past 2 min / 128 KiB |
| M3 | B3–B4 | macOS SwiftUI app: chat, spawn a local-model agent, approve a permission prompt, see a tool result and a published plot |
| M4 | Packaging | Notarized `.app` on a clean Mac (no Nix, no Homebrew): bundled runtime, supervised llama-server, sandboxed tools work |
| M5 | B5 + C8 | The M2.5 scenario from an iOS device on cellular: pair the phone (C2), drive the home agent, approve a prompt from the phone, survive background/foreground and a Wi-Fi ↔ cellular switch |
| M6 | A8 | Research harness on the Mac: `gpu:torch` MPS session, `cli.py` new → check → launch → status on parity |

M1 and M2 are independent and can run in parallel. M2.5 needs M2 for
`cli --headless` and the forwarding hooks in `llung-client`, but C1, C3
and C4 can start immediately against the current TUI (do C1 first: it
is the prerequisite for safe standing grants). M6 can start any time
after M1.

## Risks

- **`sandbox-exec` deprecation.** Present and widely used, but
  unsupported. Mitigation: everything lives in `seatbelt.rs`; A10
  contingency; CI on every new macOS release.
- **Metal allowlist drift.** The GPU sub-profile is empirical (S3) and
  may change across macOS versions. Mitigation: opt-in (only `gpu:*`
  effects), escape tests + an MPS smoke test in CI.
- **Soft memory limits.** The polling watchdog can overshoot. Document it
  in the tool output; keep timeouts as the hard bound.
- **No die-with-parent.** Orphans are possible if the app crashes.
  Mitigation: the process-group registry plus a startup sweep of stale
  `llung-sbx-*` temp dirs and recorded pgids.
- **No Mac App Store build** with local agents. Accepted.
- **Triton has no Metal backend.** `prefix` kernels stay Linux-only.
- **CGNAT.** Hole punching from cellular often fails, so the relay
  carries real traffic. Mitigation: IPv6 listeners, raised circuit
  limits, measuring direct vs relayed (C4); size the VPS for it.
- **Relay as a single point of availability.** If it's down, remote
  peers can't meet. Support several relays (`LLUNG_RELAY` is already a
  list); the relay never needs to be trusted for confidentiality (C3).
- **Identity migration.** Moving multi-device from self-asserted
  presence to device certificates (C2) touches existing accounts. Plan
  a one-time re-pairing.
- **Remote approval UX.** Prompts that expire while the phone is
  suspended read as denials. Mitigation: pre-granted standing grants for
  routine effects, push later (C9).
- **Apple's multicast entitlement** may be slow to get or refused.
  Mitigation: relay-only LAN, Bonjour from Swift (C8).

## Non-goals

- Mac App Store distribution of the agent-capable build.
- Agents running on iOS (v1).
- Windows / Android.
- Per-platform cargo features.
- A Swift reimplementation of any `llung-client` logic.
- A VM-based sandbox, unless forced by A10.
- A llung-operated hosted service: every user runs (or picks) their own
  relay; push (C9) follows the same rule.
- Arbitrary remote shell or remote control beyond C6's signed
  approve / revoke / stop commands. Everything else goes through chat
  and the agent.

## Spike results

_(append S1–S6 findings here)_
