// crates/agent/src/paths.rs
//
// Where things live, independent of the process's working directory:
//
// * `tools_dir()` — the wasm tool components (`<name>.wasm`).
// * `ro_binds()`  — extra read-only data roots (`LLUNG_RO_BINDS`).
// * `agent_readable()` — the single read rule for files an AGENT names:
//   the workspace plus the read-only binds. The same set the bash/repl
//   sandboxes mount, so every tool agrees on what the agent can see.

use std::path::{Path, PathBuf};

/// Directory holding the wasm tool components. First match wins:
///
/// 1. `LLUNG_TOOLS_DIR` (explicit override, used as-is).
/// 2. Next to the executable — how packaged builds ship:
///    `<exe dir>/runtime-tools`, `<exe dir>/../Resources/runtime-tools`
///    (macOS / iOS app bundles), `<exe dir>/../share/llung/runtime-tools`
///    (Unix installs).
/// 3. The source tree this binary was built from (dev builds:
///    `cargo run` from any directory).
/// 4. `runtime-tools` relative to the working directory (legacy).
///
/// Not cached: a few `stat`s per tool call, and tests change the env.
pub fn tools_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("LLUNG_TOOLS_DIR").filter(|d| !d.is_empty()) {
        return PathBuf::from(dir);
    }
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(exe_dir) = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf))
    {
        candidates.push(exe_dir.join("runtime-tools"));
        candidates.push(exe_dir.join("../Resources/runtime-tools"));
        candidates.push(exe_dir.join("../share/llung/runtime-tools"));
    }
    // crates/agent -> workspace root (llung/) -> runtime-tools
    candidates.push(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime-tools"));
    candidates
        .into_iter()
        .find(|dir| dir.is_dir())
        .unwrap_or_else(|| PathBuf::from("runtime-tools"))
}

/// The wasm component for tool `name`.
pub fn tool_wasm_path(name: &str) -> PathBuf {
    tools_dir().join(format!("{name}.wasm"))
}

/// Extra read-only data roots from `LLUNG_RO_BINDS`: colon-separated
/// absolute host paths (e.g. `~/Code/data:/mnt/traces`); nonexistent
/// entries are skipped. The sandboxes bind each at its own path; the
/// host-side readers accept files under them.
pub fn ro_binds() -> Vec<PathBuf> {
    parse_ro_binds(std::env::var("LLUNG_RO_BINDS").ok().as_deref())
}

/// Pure parser behind [`ro_binds`] (unit-testable). `~` is expanded so
/// `LLUNG_RO_BINDS=~/Code/data` works as people expect.
pub fn parse_ro_binds(value: Option<&str>) -> Vec<PathBuf> {
    let Some(value) = value else {
        return Vec::new();
    };
    value
        .split(':')
        .filter(|s| !s.is_empty())
        .map(|s| PathBuf::from(shellexpand::tilde(s).as_ref()))
        .filter(|p| p.exists())
        .collect()
}

/// Resolve a path an AGENT asked to read, enforcing the read rule: it
/// must lie inside `workspace` or one of [`ro_binds`]. Relative paths
/// resolve against the workspace; `~` is expanded. Both sides are
/// canonicalized, so `..` and symlinks can't escape. Returns the
/// canonical path to read.
pub fn agent_readable(path: &str, workspace: &Path) -> Result<PathBuf, String> {
    agent_readable_in(path, workspace, &ro_binds())
}

/// [`agent_readable`] with explicit roots (unit-testable).
pub fn agent_readable_in(
    path: &str,
    workspace: &Path,
    ro_binds: &[PathBuf],
) -> Result<PathBuf, String> {
    let expanded = PathBuf::from(shellexpand::tilde(path).as_ref());
    let full = if expanded.is_absolute() {
        expanded
    } else {
        workspace.join(expanded)
    };
    let canonical = full
        .canonicalize()
        .map_err(|e| format!("cannot read '{path}': {e}"))?;
    let roots = std::iter::once(workspace.to_path_buf()).chain(ro_binds.iter().cloned());
    for root in roots {
        if let Ok(root) = root.canonicalize() {
            if canonical.starts_with(&root) {
                return Ok(canonical);
            }
        }
    }
    Err(format!(
        "'{path}' is outside what this agent may read (the workspace {} and \
         LLUNG_RO_BINDS). Ask the operator to add its directory to \
         LLUNG_RO_BINDS, or to copy the file into the workspace.",
        workspace.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDir(PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            let p = std::env::temp_dir().join(format!(
                "llung-paths-{tag}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&p).unwrap();
            Self(p)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).ok();
        }
    }

    #[test]
    fn dev_builds_find_the_source_tree_tools_dir() {
        // Wherever the test runs from, the fallback chain lands on a
        // real directory (the source tree's runtime-tools/ in dev) —
        // never the cwd-relative legacy path when that doesn't exist.
        if std::env::var_os("LLUNG_TOOLS_DIR").is_some() {
            return; // explicit override wins; nothing to check
        }
        let src_tree = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime-tools");
        if src_tree.is_dir() {
            assert!(tools_dir().is_dir(), "{}", tools_dir().display());
        }
    }

    #[test]
    fn ro_binds_parse_skips_missing_and_expands_tilde() {
        let d = TempDir::new("binds");
        let value = format!("{}::/definitely/not/here", d.0.display());
        assert_eq!(parse_ro_binds(Some(&value)), vec![d.0.clone()]);
        assert!(parse_ro_binds(None).is_empty());
        if let Some(home) = std::env::var_os("HOME") {
            assert_eq!(parse_ro_binds(Some("~")), vec![PathBuf::from(home)]);
        }
    }

    #[test]
    fn agent_reads_are_confined_to_workspace_and_binds() {
        let ws = TempDir::new("ws");
        let data = TempDir::new("data");
        let secret = TempDir::new("secret");
        std::fs::write(ws.0.join("in.csv"), "a\n1\n").unwrap();
        std::fs::write(data.0.join("flights.csv"), "a\n1\n").unwrap();
        std::fs::write(secret.0.join("id_ed25519"), "KEY").unwrap();
        let binds = vec![data.0.clone()];

        // Workspace: relative and absolute.
        assert!(agent_readable_in("in.csv", &ws.0, &binds).is_ok());
        let abs = ws.0.join("in.csv");
        assert!(agent_readable_in(abs.to_str().unwrap(), &ws.0, &binds).is_ok());
        // A read-only bind.
        let f = data.0.join("flights.csv");
        assert!(agent_readable_in(f.to_str().unwrap(), &ws.0, &binds).is_ok());
        // Anything else is refused, with guidance.
        let s = secret.0.join("id_ed25519");
        let err = agent_readable_in(s.to_str().unwrap(), &ws.0, &binds).unwrap_err();
        assert!(err.contains("LLUNG_RO_BINDS"), "{err}");
        // `..` can't climb out of the workspace...
        let climb = format!("../{}/id_ed25519", secret.0.file_name().unwrap().to_str().unwrap());
        assert!(agent_readable_in(&climb, &ws.0, &binds).is_err());
        // ...and neither can a symlink inside it.
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&s, ws.0.join("link")).unwrap();
            assert!(agent_readable_in("link", &ws.0, &binds).is_err());
        }
    }
}
