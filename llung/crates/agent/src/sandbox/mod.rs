// crates/agent/src/sandbox/mod.rs
//
// bubblewrap-backed execution for effect tiers that need real
// processes: `proc:bash` (this phase) and later `repl:*` / `gpu:*`.
//
// Design (from SANDBOXES.md):
// - `closure` enumerates the minimal Nix store closure for the runtime
//   deps and ro-binds exactly those paths (hermetic, nothing else from
//   the host store is visible). Non-Nix hosts (plain `cargo build`)
//   degrade to binding /usr /bin /lib /etc read-only with a warning.
// - `bwrap` builds the sandbox argv (`--unshare-all --die-with-parent
//   --new-session`, minimal /dev, /dev/shm tmpfs, /proc, tmpfs /tmp)
//   and runs it with a wall-clock timeout plus an optional memory limit
//   (systemd scope when available, setrlimit fallback).
// - `bash` is the `proc:bash` tool the model calls.

pub mod bwrap;
pub mod closure;
pub mod bash;
pub mod repl;

/// Resolve the sandbox runtime ahead of the first tool call. The first
/// resolution is the slow part of a cold sandboxed call (~2 s on a Nix
/// host: one closure query per runtime binary); afterwards it's served
/// from the in-process cache and a sandboxed bash call costs ~30 ms.
/// Lookup only — no sandbox or interpreter is started, so this needs no
/// permission grant. Blocking: call it via `spawn_blocking`.
/// Platform-neutral: on non-Nix hosts `resolve_runtime` takes its
/// fallback path and this is near-instant.
pub fn warm_up() {
    if std::env::var("LLUNG_SANDBOX").as_deref() == Ok("off") {
        return; // tools run unsandboxed: nothing to resolve
    }
    let start = std::time::Instant::now();
    // Exactly what every sandboxed call builds first: the runtime
    // closure plus the optional companions (uv, git).
    let _ = bwrap::Sandbox::baseline("true", Vec::new());
    repl::warm_closures();
    tracing::debug!(elapsed = ?start.elapsed(), "sandbox warm-up done");
}
