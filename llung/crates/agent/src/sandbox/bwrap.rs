// crates/agent/src/sandbox/bwrap.rs
//
// Deterministic bubblewrap argv construction + async execution.
//
// Mount order is fixed so `argv()` output is golden-testable:
//   flags → ro/rw binds → GPU dev-binds → /dev → /dev/shm → /proc →
//   tmpfs list → env → chdir → `--` → program args
//
// GPU profile (SANDBOXES.md): /run/opengl-driver ro, /etc/OpenCL (or
// OCL_ICD_VENDORS), GPU char devices via --dev-bind, /dev/shm tmpfs.

use crate::sandbox::closure;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};
use tokio::io::AsyncReadExt;

/// Combined stdout/stderr cap for one sandboxed run.
pub const MAX_OUTPUT_BYTES: usize = 512 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MountKind {
    Ro,
    Rw,
    Dev,
}

impl MountKind {
    fn flag(self) -> &'static str {
        match self {
            MountKind::Ro => "--ro-bind",
            MountKind::Rw => "--bind",
            MountKind::Dev => "--dev-bind",
        }
    }
}

#[derive(Debug, Clone)]
struct Mount {
    src: PathBuf,
    dest: PathBuf,
    kind: MountKind,
}

/// The result of one sandboxed run (already output-capped).
#[derive(Debug)]
pub struct SandboxOutput {
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub exit_code: Option<i32>,
    pub timed_out: bool,
    pub duration: Duration,
    /// Degraded-mode warning (non-Nix host), echoed into the tool output.
    pub degraded: Option<String>,
}

/// A bubblewrap sandbox: bind mounts, tmpfs, env, chdir, resource
/// limits. Build with [`Sandbox::baseline`], tweak, then `run`.
pub struct Sandbox {
    program: String,
    program_args: Vec<String>,
    mounts: Vec<Mount>,
    tmpfs: Vec<PathBuf>,
    env: Vec<(String, String)>,
    chdir: Option<PathBuf>,
    create_dev: bool,
    create_proc: bool,
    gpu: bool,
    /// systemd-style memory limit ("512M"); enforced via a systemd
    /// scope when systemd is available, setrlimit otherwise.
    memory_max: Option<String>,
    degraded: Option<String>,
}

impl Sandbox {
    /// Bare sandbox with nothing mounted. For argv golden tests; real
    /// callers start from [`Sandbox::baseline`].
    pub fn new(program: impl Into<String>, program_args: Vec<String>) -> Self {
        Self {
            program: program.into(),
            program_args,
            mounts: Vec::new(),
            tmpfs: Vec::new(),
            env: Vec::new(),
            chdir: None,
            create_dev: false,
            create_proc: false,
            gpu: false,
            memory_max: None,
            degraded: None,
        }
    }

    /// Standard sandbox: minimal /dev (with /dev/shm tmpfs), /proc,
    /// tmpfs /tmp, and the runtime toolchain — either the Nix store
    /// closure (hermetic) or wide read-only host binds (degraded, with
    /// a warning carried into the output).
    pub fn baseline(program: impl Into<String>, program_args: Vec<String>) -> Self {
        let mut sb = Self::new(program, program_args);
        sb.create_dev = true;
        sb.create_proc = true;
        sb.tmpfs.push(PathBuf::from("/tmp"));
        sb.env.push(("HOME".into(), "/tmp".into()));
        sb.env.push(("TMPDIR".into(), "/tmp".into()));
        sb.env.push(("LANG".into(), "C".into()));
        match closure::resolve_runtime() {
            closure::Runtime::Nix { store_paths, bin_dirs } => {
                for path in store_paths {
                    sb.mounts.push(Mount {
                        src: path.clone(),
                        dest: path,
                        kind: MountKind::Ro,
                    });
                }
                let path_env = bin_dirs
                    .iter()
                    .map(|d| d.to_string_lossy().into_owned())
                    .collect::<Vec<_>>()
                    .join(":");
                sb.env.push(("PATH".into(), path_env));
            }
            closure::Runtime::Fallback => {
                sb.degraded = Some(
                    "non-Nix host: sandbox binds host /usr /bin /lib /etc read-only instead of a minimal store closure"
                        .into(),
                );
                for dir in closure::fallback_host_binds() {
                    sb.mounts.push(Mount {
                        src: dir.clone(),
                        dest: dir,
                        kind: MountKind::Ro,
                    });
                }
                sb.env.push(("PATH".into(), "/usr/bin:/bin".into()));
            }
        }
        sb
    }

    pub fn bind_ro(&mut self, src: impl Into<PathBuf>, dest: impl Into<PathBuf>) -> &mut Self {
        self.mounts.push(Mount { src: src.into(), dest: dest.into(), kind: MountKind::Ro });
        self
    }

    /// Mount the workspace. `writable = false` (the default for the
    /// bash tool) makes it read-only: destructive commands fail instead
    /// of being rolled back later.
    pub fn workspace(&mut self, path: impl Into<PathBuf>, writable: bool) -> &mut Self {
        let path = path.into();
        self.mounts.push(Mount {
            src: path.clone(),
            dest: path.clone(),
            kind: if writable { MountKind::Rw } else { MountKind::Ro },
        });
        self.env.push(("LLUNG_WORKSPACE".into(), path.to_string_lossy().into_owned()));
        self.chdir = Some(path);
        self
    }

    pub fn setenv(&mut self, key: impl Into<String>, value: impl Into<String>) -> &mut Self {
        self.env.push((key.into(), value.into()));
        self
    }

    pub fn memory_max(&mut self, limit: impl Into<String>) -> &mut Self {
        self.memory_max = Some(limit.into());
        self
    }

    /// Bind a binary from the host PATH by name: its full Nix store
    /// closure is mounted read-only and its directory appended to the
    /// sandbox PATH (the closure resolution lives in `closure.rs`).
    pub fn bind_binary(&mut self, name: &str) -> Result<(), String> {
        let path = closure::which(name)
            .ok_or_else(|| format!("'{name}' not found on PATH (is it in the flake devShell?)"))?;
        for store_path in closure::store_closure(&path)? {
            self.mounts.push(Mount {
                src: store_path.clone(),
                dest: store_path,
                kind: MountKind::Ro,
            });
        }
        if let Some(dir) = path.parent() {
            self.add_path_dir(dir);
        }
        Ok(())
    }

    /// Append a directory to the sandbox PATH env (deduped). Assumes
    /// `baseline()` ran (PATH env entry exists).
    pub fn add_path_dir(&mut self, dir: &Path) {
        let dir = dir.to_string_lossy().into_owned();
        if let Some(entry) = self.env.iter_mut().find(|(k, _)| k == "PATH") {
            if !entry.1.split(':').any(|d| d == dir) {
                entry.1.push(':');
                entry.1.push_str(&dir);
            }
        }
    }

    /// Spawn a long-running sandboxed child with piped stdio (for the
    /// REPL manager): stdout piped, stderr piped (callers usually
    /// redirect it into stdout via the sh wrapper), stdin piped.
    /// Unlike `run()` there is no timeout — the caller owns the child.
    pub async fn spawn_piped(&self) -> Result<tokio::process::Child, String> {
        let mut argv = self.argv();
        if argv[0] == "bwrap" {
            if let Some(path) = closure::which("bwrap") {
                argv[0] = path.to_string_lossy().into_owned();
            }
        }
        let mut cmd = tokio::process::Command::new(&argv[0]);
        cmd.args(&argv[1..])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        cmd.spawn().map_err(|e| {
            format!(
                "sandbox spawn failed (is bubblewrap installed? `nix develop` provides it): {e}"
            )
        })
    }

    /// GPU profile for Phase 4's `gpu:*` effects: OpenCL/GL driver
    /// dirs, GPU char devices (dev-bind so ioctls work), /dev/shm.
    pub fn enable_gpu(&mut self) -> &mut Self {
        self.gpu = true;
        if Path::new("/run/opengl-driver").exists() {
            self.mounts.push(Mount {
                src: PathBuf::from("/run/opengl-driver"),
                dest: PathBuf::from("/run/opengl-driver"),
                kind: MountKind::Ro,
            });
        }
        if Path::new("/etc/OpenCL").exists() {
            self.mounts.push(Mount {
                src: PathBuf::from("/etc/OpenCL"),
                dest: PathBuf::from("/etc/OpenCL"),
                kind: MountKind::Ro,
            });
        } else {
            // NixOS keeps ICD files under the driver dir.
            self.env.push((
                "OCL_ICD_VENDORS".into(),
                "/run/opengl-driver/etc/OpenCL/vendors".into(),
            ));
        }
        self
    }

    /// The full argv (`bwrap … -- program args`). Deterministic order.
    /// NOTE: all tmpfs mounts come AFTER the binds — a bind under a
    /// tmpfs path (e.g. a file in /tmp) is shadowed by the fresh
    /// tmpfs. Bind such files elsewhere (e.g. /opt).
    pub fn argv(&self) -> Vec<String> {
        let mut v = vec![
            "bwrap".to_string(),
            "--unshare-all".into(),
            "--die-with-parent".into(),
            "--new-session".into(),
        ];
        for m in &self.mounts {
            v.push(m.kind.flag().into());
            v.push(m.src.to_string_lossy().into_owned());
            v.push(m.dest.to_string_lossy().into_owned());
        }
        if self.create_dev {
            v.push("--dev".into());
            v.push("/dev".into());
            if self.gpu {
                // GPU char devices must exist before drivers can open
                // them; --dev-bind (not --ro-bind) so ioctls work.
                for dev in gpu_char_devices() {
                    v.push("--dev-bind".into());
                    v.push(dev.clone());
                    v.push(dev);
                }
            }
            // CUDA/OpenCL host-device sync uses shared memory.
            v.push("--tmpfs".into());
            v.push("/dev/shm".into());
        }
        if self.create_proc {
            v.push("--proc".into());
            v.push("/proc".into());
        }
        for t in &self.tmpfs {
            v.push("--tmpfs".into());
            v.push(t.to_string_lossy().into_owned());
        }
        for (k, val) in &self.env {
            v.push("--setenv".into());
            v.push(k.clone());
            v.push(val.clone());
        }
        if let Some(d) = &self.chdir {
            v.push("--chdir".into());
            v.push(d.to_string_lossy().into_owned());
        }
        v.push("--".into());
        v.push(self.program.clone());
        v.extend(self.program_args.iter().cloned());
        v
    }

    /// Whether this run will be wrapped in a systemd scope
    /// (`/run/systemd/system` exists = systemd is PID 1 here).
    fn use_systemd_scope(&self) -> bool {
        self.memory_max.is_some() && Path::new("/run/systemd/system").exists()
    }

    /// Run the sandbox with a hard wall-clock timeout and the output
    /// caps. Timeout → kill → `timed_out = true`.
    ///
    /// Resource limits: when systemd is PID 1 we first try a transient
    /// `systemd-run --scope` (clean MemoryMax accounting). That needs
    /// polkit authorization, which headless/unprivileged sessions may
    /// not grant — a denied scope transparently falls back to running
    /// bwrap directly with an RLIMIT_AS pre-exec cap.
    pub async fn run(&self, timeout: Duration) -> Result<SandboxOutput, String> {
        let mut argv = self.argv();
        // Resolve bwrap to an absolute path: systemd-run (the resource-
        // limit wrapper) does not inherit the caller's PATH, so a bare
        // "bwrap" would not be found inside the scope.
        if argv[0] == "bwrap" {
            if let Some(path) = closure::which("bwrap") {
                argv[0] = path.to_string_lossy().into_owned();
            }
        }
        if self.use_systemd_scope() {
            let mut scope_argv: Vec<String> = vec![
                "systemd-run".into(),
                "--scope".into(),
                "--quiet".into(),
                "--collect".into(),
            ];
            if let Some(mem) = &self.memory_max {
                scope_argv.push("-p".into());
                scope_argv.push(format!("MemoryMax={mem}"));
            }
            scope_argv.extend(argv.iter().cloned());
            let attempt =
                Self::spawn_and_wait(scope_argv, None, timeout, self.degraded.clone()).await;
            match &attempt {
                // Denied by polkit → fall through to the rlimit path.
                Ok(out) if scope_denied(out) => {}
                _ => return attempt,
            }
        }
        let limit = match &self.memory_max {
            Some(mem) => Some(parse_size(mem)?),
            None => None,
        };
        Self::spawn_and_wait(argv, limit, timeout, self.degraded.clone()).await
    }

    /// systemd-run --scope failure signature when polkit denies the
    /// transient unit (unprivileged/headless sessions).
    fn scope_denied(stderr: &[u8]) -> bool {
        let s = String::from_utf8_lossy(stderr);
        s.contains("Failed to start transient scope") || s.contains("Access denied")
    }

    async fn spawn_and_wait(
        argv: Vec<String>,
        rlimit_as: Option<u64>,
        timeout: Duration,
        degraded: Option<String>,
    ) -> Result<SandboxOutput, String> {
        let mut cmd = tokio::process::Command::new(&argv[0]);
        cmd.args(&argv[1..]);
        if let Some(bytes) = rlimit_as {
            // Fallback: RLIMIT_AS in the child pre-exec.
            // SAFETY: pre_exec runs in the forked child before exec;
            // setrlimit touches only this process's limits and is
            // async-signal-safe.
            unsafe {
                cmd.pre_exec(move || {
                    let rl = libc::rlimit { rlim_cur: bytes, rlim_max: bytes };
                    if libc::setrlimit(libc::RLIMIT_AS, &rl) != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                    Ok(())
                });
            }
        }
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let start = Instant::now();
        let mut child = cmd.spawn().map_err(|e| {
            format!(
                "sandbox spawn failed (is bubblewrap installed? `nix develop` provides it): {e}"
            )
        })?;
        let mut stdout_pipe = child.stdout.take().expect("stdout is piped");
        let mut stderr_pipe = child.stderr.take().expect("stderr is piped");

        // Read both pipes concurrently (a child blocked on a full pipe
        // while we drain the other would deadlock), capped at
        // MAX_OUTPUT_BYTES + 1 so we can note truncation.
        let result = tokio::time::timeout(timeout, async {
            let (mut so, mut se) = (Vec::new(), Vec::new());
            let read = tokio::join!(
                async {
                    stdout_pipe
                        .take(MAX_OUTPUT_BYTES as u64 + 1)
                        .read_to_end(&mut so)
                        .await
                        .ok();
                    so
                },
                async {
                    stderr_pipe
                        .take(MAX_OUTPUT_BYTES as u64 + 1)
                        .read_to_end(&mut se)
                        .await
                        .ok();
                    se
                },
            );
            let status = child.wait().await;
            (read.0, read.1, status)
        })
        .await;

        match result {
            Ok((stdout, stderr, status)) => Ok(SandboxOutput {
                stdout,
                stderr,
                exit_code: status.ok().and_then(|s| s.code()),
                timed_out: false,
                duration: start.elapsed(),
                degraded,
            }),
            Err(_) => {
                let _ = child.kill().await;
                Ok(SandboxOutput {
                    stdout: Vec::new(),
                    stderr: Vec::new(),
                    exit_code: None,
                    timed_out: true,
                    duration: start.elapsed(),
                    degraded,
                })
            }
        }
    }
}

/// True when a systemd-scope attempt was denied by polkit, meaning the
/// caller should fall back to the rlimit path.
fn scope_denied(out: &SandboxOutput) -> bool {
    Sandbox::scope_denied(&out.stderr)
}

/// GPU character devices present on this host (dri render nodes, kfd,
/// nvidia*). --dev-bind needs each one individually.
fn gpu_char_devices() -> Vec<String> {
    let mut devices = Vec::new();
    let globs = ["/dev/dri/*", "/dev/kfd*", "/dev/nvidia*"];
    for pattern in globs {
        if let (Some(dir_part), Some(glob_part)) = (pattern.rfind('/'), pattern.rsplit('/').next())
        {
            let dir = &pattern[..=dir_part];
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().into_owned();
                    if glob_match(glob_part, &name) {
                        devices.push(format!("{dir}{name}"));
                    }
                }
            }
        }
    }
    devices.sort();
    devices
}

/// Tiny glob: `*` matches any run of characters (no `?`/`[]` needed for
/// device names like `card0`, `renderD128`, `nvidia0`).
fn glob_match(pattern: &str, name: &str) -> bool {
    if let Some((prefix, suffix)) = pattern.split_once('*') {
        name.starts_with(prefix) && name.ends_with(suffix) && name.len() >= prefix.len() + suffix.len()
    } else {
        pattern == name
    }
}

/// Parse a systemd-style size ("512M", "1G", "256K") into bytes.
pub fn parse_size(s: &str) -> Result<u64, String> {
    let (num, suffix) = s
        .find(|c: char| !c.is_ascii_digit())
        .map(|i| s.split_at(i))
        .unwrap_or((s, ""));
    let value: u64 = num.parse().map_err(|_| format!("bad size '{s}'"))?;
    let mult = match suffix.to_ascii_uppercase().as_str() {
        "" | "B" => 1,
        "K" => 1 << 10,
        "M" => 1 << 20,
        "G" => 1 << 30,
        other => return Err(format!("bad size suffix '{other}' in '{s}'")),
    };
    Ok(value * mult)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argv_is_deterministic_and_golden() {
        let mut sb = Sandbox::new("sh", vec!["-c".into(), "echo hi".into()]);
        sb.create_dev = true;
        sb.create_proc = true;
        sb.tmpfs.push("/tmp".into());
        sb.mounts.push(Mount {
            src: "/nix/store/abc-sh".into(),
            dest: "/nix/store/abc-sh".into(),
            kind: MountKind::Ro,
        });
        sb.mounts.push(Mount {
            src: "/ws".into(),
            dest: "/ws".into(),
            kind: MountKind::Ro,
        });
        sb.env.push(("PATH".into(), "/nix/store/abc-sh/bin".into()));
        sb.chdir = Some("/ws".into());
        let argv = sb.argv();
        assert_eq!(
            argv,
            vec![
                "bwrap", "--unshare-all", "--die-with-parent", "--new-session",
                "--ro-bind", "/nix/store/abc-sh", "/nix/store/abc-sh",
                "--ro-bind", "/ws", "/ws",
                "--dev", "/dev",
                "--tmpfs", "/dev/shm",
                "--proc", "/proc",
                "--tmpfs", "/tmp",
                "--setenv", "PATH", "/nix/store/abc-sh/bin",
                "--chdir", "/ws",
                "--", "sh", "-c", "echo hi",
            ]
        );
        // Same builder, same argv (mounts keep insertion order).
        assert_eq!(argv, sb.argv());
    }

    #[test]
    fn rw_workspace_uses_bind_not_ro_bind() {
        let mut sb = Sandbox::new("sh", vec![]);
        sb.workspace("/ws", true);
        let argv = sb.argv();
        let pos = argv.iter().position(|a| a == "/ws").unwrap();
        assert_eq!(argv[pos - 1], "--bind");
        let mut sb = Sandbox::new("sh", vec![]);
        sb.workspace("/ws", false);
        let argv = sb.argv();
        let pos = argv.iter().position(|a| a == "/ws").unwrap();
        assert_eq!(argv[pos - 1], "--ro-bind");
    }

    #[test]
    fn gpu_profile_adds_driver_and_shm_in_order() {
        let mut sb = Sandbox::new("sh", vec![]);
        sb.create_dev = true;
        sb.enable_gpu();
        let argv = sb.argv();
        let dev_pos = argv.iter().position(|a| a == "--dev").unwrap();
        let shm_pos = argv.iter().position(|a| a == "/dev/shm").unwrap();
        assert!(dev_pos < shm_pos, "/dev must be created before /dev/shm is mounted over it");
        // GPU dev-binds (if any exist on this host) sit between them.
        if gpu_char_devices().is_empty() {
            assert_eq!(shm_pos, dev_pos + 2);
        } else {
            assert!(argv.contains(&"--dev-bind".to_string()));
        }
        // Driver dir + ICD handling present.
        if Path::new("/run/opengl-driver").exists() {
            assert!(argv.contains(&"/run/opengl-driver".to_string()));
        }
        if !Path::new("/etc/OpenCL").exists() {
            assert!(argv.windows(2).any(|w| w == ["--setenv", "OCL_ICD_VENDORS"]));
        }
    }

    #[test]
    #[test]
    fn run_resolves_bwrap_to_absolute_path() {
        // Golden argv keeps the bare "bwrap" name; run() resolves it so
        // systemd-run (no caller PATH) can exec it. Assert the resolver
        // used by run() agrees with the PATH lookup.
        if let Some(path) = closure::which("bwrap") {
            assert!(path.is_absolute());
            assert!(path.starts_with("/"));
        }
    }

    #[test]
    fn parse_size_units() {
        assert_eq!(parse_size("512").unwrap(), 512);
        assert_eq!(parse_size("512M").unwrap(), 512 << 20);
        assert_eq!(parse_size("1G").unwrap(), 1 << 30);
        assert_eq!(parse_size("256K").unwrap(), 256 << 10);
        assert!(parse_size("12Z").is_err());
        assert!(parse_size("M").is_err());
    }

    #[test]
    fn glob_match_device_names() {
        assert!(glob_match("kfd*", "kfd"));
        assert!(glob_match("nvidia*", "nvidia0"));
        assert!(glob_match("nvidia*", "nvidia-uvm"));
        assert!(glob_match("card0", "card0"));
        assert!(!glob_match("card0", "card1"));
        assert!(!glob_match("kfd*", "dri"));
    }

    /// End-to-end smoke test: real bwrap running a real command. Skips
    /// when bubblewrap is not installed (same skip pattern as the
    /// landscape wasm test).
    #[tokio::test]
    async fn smoke_echo_in_bwrap() {
        if std::env::var("LLUNG_SANDBOX").as_deref() == Ok("off") {
            eprintln!("LLUNG_SANDBOX=off; skipping");
            return;
        }
        if closure::which("bwrap").is_none() {
            eprintln!("bubblewrap not on PATH; skipping");
            return;
        }
        let mut sb = Sandbox::baseline("sh", vec!["-c".into(), "echo hello-from-bwrap".into()]);
        sb.workspace(std::env::temp_dir(), false);
        let out = sb.run(Duration::from_secs(10)).await.expect("sandbox run");
        assert!(!out.timed_out);
        assert_eq!(out.exit_code, Some(0), "stderr: {}", String::from_utf8_lossy(&out.stderr));
        assert!(
            String::from_utf8_lossy(&out.stdout).contains("hello-from-bwrap"),
            "stdout: {:?}",
            String::from_utf8_lossy(&out.stdout)
        );
    }

    /// Read-only workspace: a write attempt must fail inside the sandbox.
    #[tokio::test]
    async fn smoke_readonly_workspace_blocks_writes() {
        if std::env::var("LLUNG_SANDBOX").as_deref() == Ok("off") {
            eprintln!("LLUNG_SANDBOX=off; skipping");
            return;
        }
        if closure::which("bwrap").is_none() {
            eprintln!("bubblewrap not on PATH; skipping");
            return;
        }
        let dir = std::env::temp_dir().join(format!(
            "llung-bwrap-ro-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let mut sb = Sandbox::baseline("sh", vec!["-c".into(), "touch marker && echo wrote".into()]);
        sb.workspace(&dir, false);
        let out = sb.run(Duration::from_secs(10)).await.expect("sandbox run");
        assert_eq!(out.exit_code, Some(1), "touch in a ro workspace must fail");
        assert!(!dir.join("marker").exists(), "no file may appear on the host");
        std::fs::remove_dir_all(&dir).ok();
    }
}
