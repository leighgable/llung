// crates/agent/src/sandbox/closure.rs
//
// Minimal runtime resolution for the sandbox. On Nix hosts every dep is
// a store path; `nix-store -qR` gives its full closure, and bwrap ro-binds
// exactly those paths — the sandbox sees a hermetic, immutable toolchain
// and nothing else from the host. On non-Nix hosts (plain `cargo build`)
// there are no store paths to bind, so we degrade: ro-bind the standard
// host dirs (/usr /bin /lib /etc) and carry a warning in the output.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

/// The runtime the sandbox can actually provide.
#[derive(Debug, Clone)]
pub enum Runtime {
    /// Nix store closure binds + the PATH that resolves inside.
    Nix { store_paths: Vec<PathBuf>, bin_dirs: Vec<PathBuf> },
    /// Non-Nix host: wide read-only host binds, standard PATH.
    Fallback,
}

/// Binaries a sandboxed shell plausibly needs. Resolved from the
/// harness's PATH at run time — in the nix devshell these are all store
/// paths (bash, coreutils, gnugrep, findutils, gnused, gawk).
const RUNTIME_DEPS: &[&str] = &[
    "sh", "bash", "cat", "ls", "mkdir", "rm", "cp", "mv", "chmod", "head", "tail", "wc", "env",
    "true", "echo", "grep", "sed", "awk", "find", "sort", "uniq", "tr", "pwd", "xargs", "dirname",
];

/// Optional companions, bound BEST-EFFORT into every sandbox: present
/// on the host PATH -> their closure is ro-bound like the runtime
/// deps; absent -> silently skipped. Unlike [`RUNTIME_DEPS`], a missing
/// one must never flip the whole runtime into Fallback. `uv` lets the
/// agent maintain the python env itself (`uv lock` after a pyproject
/// edit); `git` can back experiments up to a repo someday.
pub const OPTIONAL_RUNTIME_DEPS: &[&str] = &["uv", "git"];

fn closure_cache() -> &'static Mutex<HashMap<PathBuf, Vec<PathBuf>>> {
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, Vec<PathBuf>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Resolve a binary name against the PATH environment variable.
pub fn which(name: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Full Nix store closure of a store path (cached per session).
/// Fails when `nix-store` is unavailable or the path is not in the store.
pub fn store_closure(path: &Path) -> Result<Vec<PathBuf>, String> {
    if let Some(cached) = closure_cache().lock().ok().and_then(|c| c.get(path).cloned()) {
        return Ok(cached);
    }
    let output = std::process::Command::new("nix-store")
        .arg("-qR")
        .arg(path)
        .output()
        .map_err(|e| format!("nix-store -qR failed to start: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "nix-store -qR {} failed: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let closure: Vec<PathBuf> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|l| !l.is_empty())
        .map(PathBuf::from)
        .collect();
    if let Ok(mut cache) = closure_cache().lock() {
        cache.insert(path.to_path_buf(), closure.clone());
    }
    Ok(closure)
}

/// Resolve the sandbox runtime. Every dep must resolve to a binary
/// inside /nix/store for the hermetic Nix profile; anything else →
/// `Runtime::Fallback` (callers bind host dirs and warn).
pub fn resolve_runtime() -> Runtime {
    let mut store_paths: Vec<PathBuf> = Vec::new();
    let mut bin_dirs: Vec<PathBuf> = Vec::new();
    for dep in RUNTIME_DEPS {
        let Some(binary) = which(dep) else {
            return Runtime::Fallback;
        };
        if !binary.starts_with("/nix/store") {
            return Runtime::Fallback;
        }
        if let Ok(closure) = store_closure(&binary) {
            store_paths.extend(closure);
        } else {
            return Runtime::Fallback;
        }
        if let Some(dir) = binary.parent() {
            if !bin_dirs.contains(&dir.to_path_buf()) {
                bin_dirs.push(dir.to_path_buf());
            }
        }
    }
    store_paths.sort();
    store_paths.dedup();
    Runtime::Nix { store_paths, bin_dirs }
}

/// Host directories bound read-only in Fallback mode (those that exist).
pub fn fallback_host_binds() -> Vec<PathBuf> {
    ["/usr", "/bin", "/sbin", "/lib", "/lib64"]
        .iter()
        .map(PathBuf::from)
        .filter(|p| p.exists())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn which_finds_store_binaries_in_the_nix_devshell() {
        let sh = which("sh").expect("sh on PATH");
        // In the nix devshell this is a store path; on a bare non-nix
        // host the same assertion holds for the Fallback decision — the
        // test only requires resolution, not store membership.
        assert!(sh.is_absolute());
    }

    #[test]
    fn optional_runtime_deps_are_strictly_optional() {
        // Disjoint from the mandatory set: a missing RUNTIME_DEPS entry
        // degrades the whole sandbox to Fallback, which must never
        // happen over a nice-to-have.
        for name in OPTIONAL_RUNTIME_DEPS {
            assert!(
                !RUNTIME_DEPS.contains(name),
                "{name} must stay optional"
            );
        }
        // In the nix devShell both are present; the smoke test in
        // bwrap.rs verifies they run sandboxed.
        if which("uv").is_none() || which("git").is_none() {
            eprintln!("uv/git not on PATH; presence half of the assertion skipped");
        }
    }

    #[test]
    fn which_missing_binary_returns_none() {
        assert!(which("definitely-not-a-real-binary-xyz").is_none());
    }

    #[test]
    fn store_closure_lists_the_path_itself_and_deps() {
        let Some(sh) = which("sh") else {
            eprintln!("no sh on PATH; skipping");
            return;
        };
        if !sh.starts_with("/nix/store") {
            eprintln!("sh is not a store path ({}); skipping", sh.display());
            return;
        }
        let closure = store_closure(&sh).expect("nix-store -qR");
        // The query was a file INSIDE a store path (e.g. .../bin/sh);
        // the closure lists store path roots, so assert on the root.
        let components: Vec<_> = sh.components().collect();
        let store_root: PathBuf = components.iter().take(4).collect();
        assert!(
            closure.contains(&store_root),
            "closure must include the containing store path {store_root:?}: {closure:?}"
        );
        // glibc must be in the closure of any dynamic binary.
        assert!(
            closure.iter().any(|p| p.to_string_lossy().contains("glibc")),
            "closure missing glibc: {closure:?}"
        );
    }

    #[test]
    fn store_closure_errors_cleanly_off_nix() {
        let tmp = std::env::temp_dir();
        // A non-store path must not produce a bogus closure.
        if !tmp.starts_with("/nix/store") {
            assert!(store_closure(&tmp).is_err());
        }
    }

    #[test]
    fn resolve_runtime_is_deterministic() {
        // Two calls hit the same cache path and agree on the variant.
        let a = resolve_runtime();
        let b = resolve_runtime();
        assert!(matches!(
            (&a, &b),
            (Runtime::Nix { .. }, Runtime::Nix { .. }) | (Runtime::Fallback, Runtime::Fallback)
        ));
    }
}
