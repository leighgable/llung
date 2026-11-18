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
