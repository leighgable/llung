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
//
// baseline() additionally binds OPTIONAL_RUNTIME_DEPS (uv, git)
// best-effort: present on the host PATH -> ro-bound with their
// closure; absent -> skipped. They must never trigger the degraded
// Fallback profile the way a missing RUNTIME_DEPS entry would.

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
        // Nice-to-have companions (uv, git — see closure.rs): bound
        // when present on the host PATH, skipped silently when not;
        // never a reason to degrade the runtime.
        for dep in closure::OPTIONAL_RUNTIME_DEPS {
            sb.bind_binary_optional(dep);
        }
        sb
    }

    pub fn bind_ro(&mut self, src: impl Into<PathBuf>, dest: impl Into<PathBuf>) -> &mut Self {
        self.mounts.push(Mount { src: src.into(), dest: dest.into(), kind: MountKind::Ro });
        self
    }

    /// Read-write bind (e.g. `experiments/` inside the otherwise ro
    /// workspace mount — a later bind of a subpath shadows the earlier
    /// ro parent, same as the tmpfs-over-bind note in `argv()`).
    pub fn bind_rw(&mut self, src: impl Into<PathBuf>, dest: impl Into<PathBuf>) -> &mut Self {
        self.mounts.push(Mount { src: src.into(), dest: dest.into(), kind: MountKind::Rw });
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
        self.bind_binary_at(&path)
    }

    /// [`Self::bind_binary`] for an explicit path (a Nix store binary
    /// named by an env override rather than found on PATH).
    pub fn bind_binary_at(&mut self, path: &Path) -> Result<(), String> {
        let path = path.to_path_buf();
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

    /// Best-effort variant of [`bind_binary`]: a missing binary (or a
    /// non-store one on a degraded host) is SKIPPED silently, so these
    /// are strictly nice-to-haves — see `closure::OPTIONAL_RUNTIME_DEPS`.
    pub fn bind_binary_optional(&mut self, name: &str) {
        let Some(path) = closure::which(name) else {
            return;
        };
        // Symlinked profile entries (/run/current-system/sw/bin/git,
        // ~/.nix-profile/bin/...) are not store paths themselves —
        // canonicalize to the real target before asking for its
        // closure. The resolved parent's bin dir goes on the PATH.
        let path = std::fs::canonicalize(&path).unwrap_or(path);
        let Ok(paths) = closure::store_closure(&path) else {
            return;
        };
        for p in paths {
            self.mounts.push(Mount {
                src: p.clone(),
                dest: p,
                kind: MountKind::Ro,
            });
        }
        if let Some(dir) = path.parent() {
            self.add_path_dir(dir);
        }
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
        // ROCm/HIP userspace (ROCr + libdrm amdgpu) enumerates devices
        // through sysfs: the kfd topology, PCI device attributes, and
        // the /sys/class/dri symlink mapping the render node to its PCI
        // address (amdgpu_get_auth walks it). Without these, GPU
        // discovery silently yields zero devices. Read-only: sysfs is
        // kernel-generated and the driver never writes it.
        for sys in ["/sys/devices", "/sys/bus/pci", "/sys/class", "/sys/dev"] {
            let path = PathBuf::from(sys);
            if path.exists() {
                self.mounts.push(Mount {
                    src: path.clone(),
                    dest: path,
                    kind: MountKind::Ro,
                });
            }
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
        // Filesystem layering: bwrap applies mounts in argv order and a
        // later mount covers whatever was at its destination. Emit binds
        // and tmpfs mounts parent-before-child (stable sort by
        // destination depth) so every one stays visible: the /tmp tmpfs
        // must precede a workspace or LLUNG_RO_BINDS dir under /tmp
        // (it used to come last and hide them), and an rw subdir bind
        // still follows its ro parent.
        enum Layer<'a> {
            Bind(&'a Mount),
            Tmpfs(&'a PathBuf),
        }
        let mut layers: Vec<Layer> = self
            .tmpfs
            .iter()
            .map(Layer::Tmpfs)
            .chain(self.mounts.iter().map(Layer::Bind))
            .collect();
        layers.sort_by_key(|layer| match layer {
            Layer::Bind(m) => m.dest.components().count(),
            Layer::Tmpfs(t) => t.components().count(),
        });
        for layer in layers {
            match layer {
                Layer::Bind(m) => {
                    v.push(m.kind.flag().into());
                    v.push(m.src.to_string_lossy().into_owned());
                    v.push(m.dest.to_string_lossy().into_owned());
                }
                Layer::Tmpfs(t) => {
                    v.push("--tmpfs".into());
                    v.push(t.to_string_lossy().into_owned());
                }
            }
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

    /// Whether this run will be wrapped in a systemd scope:
    /// `/run/systemd/system` exists (systemd is PID 1) and the caller
    /// has a user manager to talk to (`$XDG_RUNTIME_DIR/bus`).
    fn use_systemd_scope(&self) -> bool {
        self.memory_max.is_some()
            && Path::new("/run/systemd/system").exists()
            && std::env::var_os("XDG_RUNTIME_DIR")
                .is_some_and(|dir| Path::new(&dir).join("bus").exists())
    }

    /// The resource-limit wrapper: a transient scope on the USER
    /// manager. Never the system manager — that needs polkit, whose
    /// TTY agent writes a password prompt straight onto the controlling
    /// terminal (wrecking the TUI) and reads the password from the same
    /// TTY the UI reads keystrokes from. `--no-ask-password` makes any
    /// authorization need a fast failure instead of a prompt.
    fn scope_argv(&self, argv: &[String]) -> Vec<String> {
        let mut scope_argv: Vec<String> = vec![
            "systemd-run".into(),
            "--user".into(),
            "--scope".into(),
            "--no-ask-password".into(),
            "--quiet".into(),
            "--collect".into(),
        ];
        if let Some(mem) = &self.memory_max {
            scope_argv.push("-p".into());
            scope_argv.push(format!("MemoryMax={mem}"));
        }
        scope_argv.extend(argv.iter().cloned());
        scope_argv
    }

    /// Run the sandbox with a hard wall-clock timeout and the output
    /// caps. Timeout → kill → `timed_out = true`.
    ///
    /// Resource limits: when systemd is PID 1 we first try a transient
    /// `systemd-run --user --scope` (clean MemoryMax accounting via the
    /// user manager's delegated memory controller; no polkit). If the
    /// scope can't be created (no user manager, no systemd-run), we
    /// transparently fall back to running bwrap directly with an
    /// RLIMIT_AS pre-exec cap.
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
            let scope_argv = self.scope_argv(&argv);
            let attempt =
                Self::spawn_and_wait(scope_argv, None, timeout, self.degraded.clone()).await;
            match &attempt {
                // No scope (denied / no manager) or no systemd-run at
                // all: the command never ran → the rlimit path.
                Ok(out) if scope_denied(out) => {}
                Err(_) => {}
                _ => return attempt,
            }
        }
        let limit = match &self.memory_max {
            Some(mem) => Some(parse_size(mem)?),
            None => None,
        };
        Self::spawn_and_wait(argv, limit, timeout, self.degraded.clone()).await
    }

    /// systemd-run's own failure to create the scope (authorization
    /// denied, user manager unreachable). Matched at the START of
    /// stderr only: in that case systemd-run printed its error and the
    /// command never ran. A command that ran and merely printed
    /// "Access denied" must NOT match — the fallback would run it twice.
    fn scope_denied(stderr: &[u8]) -> bool {
        let s = String::from_utf8_lossy(stderr);
        let s = s.trim_start();
        s.starts_with("Failed to start transient scope")
            || s.starts_with("Failed to connect to")
            || s.starts_with("Failed to create bus connection")
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
    fn scope_uses_user_manager_and_never_prompts() {
        // The system manager needs polkit, whose TTY agent scribbles a
        // password prompt over the TUI: the scope must be a USER scope
        // and must never ask for a password.
        let mut sb = Sandbox::new("sh", vec!["-c".into(), "true".into()]);
        sb.memory_max("512M");
        let argv = sb.scope_argv(&["/bin/bwrap".into(), "--".into(), "sh".into()]);
        assert_eq!(
            &argv[..6],
            ["systemd-run", "--user", "--scope", "--no-ask-password", "--quiet", "--collect"]
        );
        assert!(argv.windows(2).any(|w| w == ["-p", "MemoryMax=512M"]));
        assert_eq!(&argv[argv.len() - 3..], ["/bin/bwrap", "--", "sh"]);
    }

    #[test]
    fn scope_denied_matches_only_systemd_run_failures() {
        // systemd-run's own errors (the command never ran) fall back...
        assert!(Sandbox::scope_denied(
            b"Failed to start transient scope unit: Access denied as the requested \
              operation requires interactive authentication.\n"
        ));
        assert!(Sandbox::scope_denied(
            b"Failed to connect to user scope bus via local transport: No such file or directory\n"
        ));
        // ...but a command that RAN and printed an error must not, or
        // the fallback would run it a second time.
        assert!(!Sandbox::scope_denied(b"cat: /root/x: Access denied\n"));
        assert!(!Sandbox::scope_denied(
            b"some output\nFailed to start transient scope unit: quoted\n"
        ));
    }

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
                // Parent-before-child layering (depth 1 before depth 3).
                "--tmpfs", "/tmp",
                "--ro-bind", "/ws", "/ws",
                "--ro-bind", "/nix/store/abc-sh", "/nix/store/abc-sh",
                "--dev", "/dev",
                "--tmpfs", "/dev/shm",
                "--proc", "/proc",
                "--setenv", "PATH", "/nix/store/abc-sh/bin",
                "--chdir", "/ws",
                "--", "sh", "-c", "echo hi",
            ]
        );
        // Same builder, same argv (stable order within a depth).
        assert_eq!(argv, sb.argv());
    }

    #[test]
    fn tmpfs_never_hides_a_bind_beneath_it() {
        // A workspace or LLUNG_RO_BINDS dir under /tmp must be mounted
        // AFTER the /tmp tmpfs, and an rw subdir after its ro parent.
        let mut sb = Sandbox::new("sh", vec![]);
        sb.tmpfs.push("/tmp".into());
        sb.workspace("/tmp/ws", false);
        sb.bind_rw("/tmp/ws/experiments", "/tmp/ws/experiments");
        let argv = sb.argv();
        let at = |needle: &str| argv.iter().position(|a| a == needle).unwrap();
        assert!(at("/tmp") < at("/tmp/ws"), "{argv:?}");
        assert!(at("/tmp/ws") < at("/tmp/ws/experiments"), "{argv:?}");
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
        // ROCm device discovery needs the sysfs view (kfd topology,
        // PCI attributes, the dri class symlink). Read-only binds.
        for sys in ["/sys/devices", "/sys/class"] {
            if Path::new(sys).exists() {
                assert!(argv.contains(&sys.to_string()), "sysfs bind missing: {sys}");
            }
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

    /// A rw subdir bind shadows the ro workspace mount: writes land in
    /// the subdir, the rest of the workspace stays read-only.
    #[tokio::test]
    async fn smoke_rw_subdir_writable_inside_ro_workspace() {
        if std::env::var("LLUNG_SANDBOX").as_deref() == Ok("off") {
            eprintln!("LLUNG_SANDBOX=off; skipping");
            return;
        }
        if closure::which("bwrap").is_none() {
            eprintln!("bubblewrap not on PATH; skipping");
            return;
        }
        // NOT under /tmp: baseline mounts a fresh tmpfs over /tmp,
        // which would shadow these binds (see the argv() note).
        let dir = std::env::current_dir()
            .unwrap()
            .join("target")
            .join(format!(
                "llung-bwrap-rwsub-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
        let sub = dir.join("experiments");
        std::fs::create_dir_all(&sub).unwrap();
        let mut sb = Sandbox::baseline(
            "sh",
            vec!["-c".into(), "touch experiments/marker && echo sub-ok".into()],
        );
        sb.workspace(&dir, false);
        sb.bind_rw(&sub, &sub);
        let out = sb.run(Duration::from_secs(10)).await.expect("sandbox run");
        assert_eq!(out.exit_code, Some(0), "stderr: {}", String::from_utf8_lossy(&out.stderr));
        assert!(sub.join("marker").exists(), "rw bind must write through to the host");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `bind_binary_optional` is strictly best-effort: a missing name
    /// adds no mounts and reports no error (unlike `bind_binary`).
    #[test]
    fn bind_binary_optional_is_silent_about_missing_names() {
        let mut sb = Sandbox::new("sh", vec![]);
        sb.env.push(("PATH".into(), "/nonexistent".into()));
        let before = sb.mounts.len();
        sb.bind_binary_optional("definitely-not-a-real-binary-xyz");
        assert_eq!(sb.mounts.len(), before, "nothing bound, nothing broken");
    }

    /// Optional companions (uv, git) actually run inside the sandbox —
    /// that is the agent's dependency-management and backup path.
    #[tokio::test]
    async fn smoke_uv_and_git_run_in_the_sandbox() {
        if std::env::var("LLUNG_SANDBOX").as_deref() == Ok("off") {
            eprintln!("LLUNG_SANDBOX=off; skipping");
            return;
        }
        if closure::which("bwrap").is_none()
            || closure::which("uv").is_none()
            || closure::which("git").is_none()
        {
            eprintln!("bwrap/uv/git not on PATH; skipping");
            return;
        }
        let dir = std::env::current_dir()
            .unwrap()
            .join("target")
            .join(format!("llung-bwrap-tools-{}-{}", std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut sb = Sandbox::baseline(
            "sh",
            vec!["-c".into(), "uv --version && git --version && echo tools-ok".into()],
        );
        sb.workspace(&dir, false);
        let out = sb.run(Duration::from_secs(30)).await.expect("sandbox run");
        assert_eq!(out.exit_code, Some(0), "stderr: {}", String::from_utf8_lossy(&out.stderr));
        assert!(
            String::from_utf8_lossy(&out.stdout).contains("tools-ok"),
            "uv+git must resolve on the sandbox PATH: {:?}",
            String::from_utf8_lossy(&out.stdout)
        );
        std::fs::remove_dir_all(&dir).ok();
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
