# Sandboxes / Tools — Implementation Plan

Derived from `SANDBOXES.md`. Audited current state: `crates/agent/src/aukora/`
(kernel, security, verification), `orchestrator.rs`, `tools.rs`, `agent-rt`,
TUI `app.rs` slash commands, `flake.nix`.

## Audit findings (what exists today)

- `aukora/kernel.rs` — Manifest/Grant/Receipt, `KernelState` (pinned_keys,
  executed_nonces, merkle root). `consume_manifest_use_core` = anti-replay +
  pin check + hash check + signature check. `verify_signature` is a **stub
  that always returns true**; self-grants carry empty signature + all-zeros key.
- `aukora/security.rs` — single gate `secure_execute_agent_tool`. Effect name
  is mapped 1:1 to `runtime-tools/{effect}.wasm` → only wasm tools can ever
  execute. No host-tool dispatch path.
- `orchestrator.rs` — `execute_secured_tool` auto-creates a self-grant for
  every tool call. **Zero human involvement**; there is no permissions UX.
- Kernel state is in-memory only → anti-replay resets on restart. Merkle root
  is not persisted either.
- `PeerStateVerifier` (verification.rs) is unused by anything; receipts never
  leave the process.
- TUI has slash-command dispatch (`handle_command` in `app.rs`), `/agent
  spawn` creates local agents. No `/grant`, no permission prompt UI.
- flake.nix devShell lacks: bubblewrap, chibi-scheme, futhark. Has python
  (via uv venv), ROCm/OpenCL env already wired (`OCL_ICD_VENDORS`,
  `/run/opengl-driver` pattern applicable to sandbox).
- Missing crates: `ignore`, `grep-searcher`/`grep-regex`, `similar`,
  `ed25519-dalek`.

## Design decisions (answers to the notes' open questions)

1. **bwrap direct, not nixwrap/nix-bwrap.** nixwrap targets nix-*build*
   isolation; llung already lives in a nix devshell with real store paths.
   Direct bwrap + `nix-store -qR <dep>` closure enumeration is the minimal
   fit. Add `pkgs.bubblewrap` to the flake.
2. **The harness asks for permissions, not the agent.** The kernel runs in
   the harness process; the agent is untrusted prompt-driven text. Flow:
   agent emits tool call → orchestrator builds Manifest → kernel finds no
   standing grant → emits `EffectRequest` → TUI prompts human → human grants
   (once/perpetual) or denies → receipt recorded either way. Denial is fed
   back to the model as tool feedback (same channel as zoom-error feedback).
   A human-feedback *tool* (agent asks human a question mid-turn) is a
   separate feature — defer to Phase 7 stretch.
3. **`perpetual` and `once` variants: yes.** Model as `GrantKind`:
   `Once` (current behavior: manifest-hash + nonce), `Standing` (matches
   (agent_id, effect-class), argument-agnostic, skips nonce consumption).
   `/grant <effect> perpetual` creates a Standing grant rule in the kernel.
4. **Rollback via overlayfs/git worktree: overkill for v1.** Defer. Instead,
   bash tool defaults to *workspace-readonly* + explicit `writable=true` arg
   behind a grant tier; edit tool does exact-match patches (reversible by
   construction). Revisit after Phase 4.
5. **Effect namespace.** Kernel moves from "effect name = wasm filename" to a
   tiered registry:
   - `wasm:<tool>` — prompt-free (pure compute, already sandboxed)
   - `fs:read` / `fs:glob` / `fs:grep` — prompt-free, workspace-only
   - `edit:file` — prompt-free once, standing grant common
   - `proc:bash` — prompt once/perpetual; default workspace readonly
   - `repl:<interp>` — prompt once, then standing for the session
   - `gpu:<backend>` — *never* standing-by-default; perpetual possible via
     explicit `/grant`

## Phases

### Phase 0 — Kernel real signatures + persistence (foundation)
- [ ] `ed25519-dalek` in `crates/agent`. Supervisor keypair generated at
      agent spawn, public key pinned in `KernelState.pinned_keys` (replace
      all-zeros placeholder). Implement `verify_signature` for real; keep
      empty-sig rejection explicit.
- [ ] Persist `KernelState` (pinned keys, consumed nonces, standing grants,
      merkle root) to SQLite (`llung-core` storage) — new table
      `kernel_state(agent_id, …)`. Load on `AgentLoopContext::new`.
- [ ] Replace random SysRng nonce with a persisted atomic counter (monotonic
      nonce survives restarts → real anti-replay).
- [ ] Unit tests: signature verify/forge, nonce replay across reload.

### Phase 1 — Effect registry + host tool dispatch
- [ ] New `crates/agent/src/effects.rs`: parse `target_effect` into
      `(Tier, name)`; classify per registry above. `secure_execute_agent_tool`
      refactored into dispatcher: wasm tier → existing `wasm_tx` path; other
      tiers → host executors (Phase 2/3). Receipt + merkle flow unchanged —
      every tier still goes through `consume_manifest_use_core`.
- [ ] Tool catalog (`tools.rs`) gains host tools (specs + usage lines) so the
      prompt advertises them.

### Phase 2 — Host tools: glob, grep, diff/edit (no external processes)
- [ ] Deps: `ignore`, `grep-searcher`, `grep-regex`, `similar`.
- [ ] `crates/agent/src/tools/host/glob.rs` — `ignore::WalkBuilder` from
      workspace root, gitignore-aware, pattern arg, cap results (e.g. 1000).
- [ ] `crates/agent/src/tools/host/grep.rs` — `grep_regex::RegexMatcher` +
      `grep_searcher`, file filtering via `ignore`, return matches with
      path:line:col, cap bytes.
- [ ] `crates/agent/src/tools/host/edit.rs` — args: path, one or more
      exact (old → new) pairs. Enforce exact unique match (reject 0 or ≥2
      matches); on success return `similar::TextDiff` unified diff of the
      change. No fuzzy.
- [ ] Tests: gitignore fixture, multi-match rejection, diff output shape.

### Phase 3 — bwrap sandbox executor + bash tool
- [ ] New module `crates/agent/src/sandbox/`. Flake: add `bubblewrap`.
- [ ] `closure.rs` — given a runtime spec (set of nix store paths: bash,
      coreutils, python, chibi, futhark, ROCm), run `nix-store -qR` to get
      full closure; cache per session. Non-nix fallback (PATH resolution,
      wider binds) behind `cfg`/runtime detection so plain `cargo build`
      keeps working — degrade with a warning, don't fail.
- [ ] `bwrap.rs` — argv builder:
      - `--unshare-all --die-with-parent --new-session`
      - ro-bind store closure at same paths; rw-bind (or ro by default)
        workspace
      - tmpfs /tmp, minimal /dev, `--proc /proc`, bind
        `/etc/resolv.conf`, `$HOME` only when granted
      - resource limits: prefer `systemd-run --scope -p MemoryMax= -p
        CPUQuota=` wrapper when systemd available; fallback `setrlimit` in
        pre-exec; always wall-clock timeout in harness
- [ ] GPU profile (for futhark opencl/cuda): ro-bind `/run/opengl-driver`,
      bind `/etc/OpenCL` or set `OCL_ICD_VENDORS`, `--dev-bind` for GPU
      char devices (`/dev/dri/*`, `/dev/kfd*`, nvidia nodes), tmpfs
      `/dev/shm`. Activated only by `gpu:*` grants.
- [ ] `bash` tool: args `{cmd, timeout_secs?, writable?}`. Streams output
      (cap bytes), exit code, timing into receipt commitment.
- [ ] Tests: argv construction golden tests; smoke test spawning
      `echo hello` in bwrap (skip when not linux/no bwrap, like the
      landscape wasm skip pattern).

### Phase 4 — Stateful REPL tool
- [ ] `repl.rs` — `ReplManager` keyed by session id; spawns long-running
      child inside the Phase 3 sandbox, pipes stdin/stdout across turns.
- [ ] Sentinels: per-interpreter injection after every input —
      python `print("__LLUNG_END__")`, chibi `(display "__LLUNG_END__")`,
      futhark: probe support (it has no direct print; fallback = idle-read:
      return after N ms of stdout silence, with timeout cap).
- [ ] One systemd scope (cgroup) per REPL process → clean kill of runaway
      python/futhark without touching siblings; fallback `killpg`.
- [ ] Interpreters in flake: `chibi-scheme`, `futhark` (pulls GHC — gate
      behind a flake flag `withFuthark` if closure size hurts).
- [ ] futhark defaults to c backend; `--backend=opencl|cuda` requires
      `gpu:futhark` grant → Phase 5 prompt.
- [ ] Tests: scripted fake interpreter verifying sentinel read, cross-turn
      state retention, kill on runaway loop.

### Phase 5 — Permissions UX: `/grant` + prompts
- [ ] `GrantKind::{Once, Standing}` + standing-grant rules in kernel;
      `consume_manifest_use_core` matches standing rules before nonce path.
- [ ] `mpsc` channel `EffectRequest { manifest, tier, tx }` from agent loop
      to TUI; `handle_event` flow in orchestrator awaits the answer when no
      grant matches (perpetual grants skip this entirely).
- [ ] TUI: pending-permission panel (like pending tool panel): show effect,
      args summary, agent name; keys: `y` once, `Y`/always (perpetual),
      `n` deny. Answer routes back on the channel.
- [ ] `/grant` slash command in `app.rs`:
      - `/grant list` — standing grants + recently denied
      - `/grant <effect> perpetual [agent]` — pre-authorize
      - `/grant revoke <effect> [agent]`
      - `/grant pending` — show queue
- [ ] Denied receipts get `ReceiptStatus::Denied` and are published as tool
      errors to the model (existing feedback path).
- [ ] Tests: standing-grant match, `/grant` parsing, denial feedback loop.

### Phase 6 — Non-NixOS build path
- [ ] Runtime detection: bwrap present? store paths resolvable? If not,
      `sandbox::bwrap` falls back to "no isolation" mode behind an explicit
      config flag (`AgentConfig.sandbox = off|warn-only|enforce`).
- [ ] CI-ish check: `cargo build` on a non-nix linux stays green (feature
      gate GPU profile to nix builds initially).

### Phase 7 — Stretch (post-MVP)
- [ ] Human-feedback tool (`ask_human`) — agent posts a question to the
      topic, blocks turn until a reply addressed to it arrives.
- [ ] Rollback: git worktree/overlayfs snapshot before `proc:bash` with
      `writable=true`; `/revert`.
- [ ] Wire `PeerStateVerifier`: publish receipts over gossipsub so remote
      peers can audit agent history roots.
- [ ] Receipt export: `/audit` dumps the merkle chain.

## Sequencing

```
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5 ──► Phase 6
 (kernel)   (dispatch)  (host)     (bwrap)     (repl)      (ux)
```

Phases 2 and 3 are independent after Phase 1 (can parallelize). Phase 5
depends only on Phase 1's kernel surface — its UI work can start while
Phase 3/4 land, using mock executors.

## Risk notes

- **Futhark closure size** (GHC) — gate behind flake flag; interpreter
  availability detection at runtime.
- **systemd scopes** unavailable in some containers/CI → always keep the
  `killpg` fallback path.
- **Sentinel fragility** — a model printing the sentinel string inside
  evaluated code breaks turn detection; use per-session random sentinel.
- **Kernel persistence** must be transactional with message persistence or
  anti-replay state can diverge from published history.
