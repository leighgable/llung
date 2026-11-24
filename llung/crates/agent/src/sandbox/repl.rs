// crates/agent/src/sandbox/repl.rs
//
// Stateful REPL tool (SANDBOXES.md): long-running interpreters spawned
// INSIDE the Phase 3 bubblewrap sandbox, with stdin/stdout piped across
// turns. One session per (agent, session-name) keeps its variables and
// definitions between tool calls. Interpreters: python, chibi-scheme,
// futhark, and torch (the fat ROCm torch closure, GPU sandbox, only via
// the gpu:torch effect).
//
// Turn delimiters: interactive interpreters don't exit after a command,
// so the harness must know when output finished. After every input we
// append a unique per-session sentinel print (a model printing the
// token inside evaluated code can only fake ITS OWN turn's end, never
// another session's — the token is random per session) and read stdout
// until the token appears. Futhark's repl has no print primitive, so it
// uses the idle-read fallback: output is returned once stdout has been
// quiet for a grace period.
//
// Isolation: each session's interpreter is its own bwrap child, and
// `bwrap --new-session` makes it a session/group leader, so a runaway
// (infinite loop, huge allocation — capped separately by the sandbox
// memory limit) is killed via its process group without touching the
// other interpreters.

use rand::rand_core::Rng;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::{Mutex, Notify};

use crate::sandbox::bwrap::Sandbox;
use crate::sandbox::closure;

const DEFAULT_TIMEOUT_SECS: u64 = 30;
const MAX_TIMEOUT_SECS: u64 = 120;
/// Futhark idle-read: stdout silence that ends a turn.
const IDLE_QUIET: Duration = Duration::from_millis(500);
/// Futhark idle-read: if nothing arrived at all, call the turn done.
const FIRST_OUTPUT_GRACE: Duration = Duration::from_millis(700);
/// Poll interval while waiting for output.
const WAIT_TICK: Duration = Duration::from_millis(25);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    C,
    OpenCl,
    Cuda,
}

impl Backend {
    fn parse(s: &str) -> Result<Backend, String> {
        match s {
            "c" => Ok(Backend::C),
            "opencl" => Ok(Backend::OpenCl),
            "cuda" => Ok(Backend::Cuda),
            other => Err(format!("repl: unknown backend '{other}' (c | opencl | cuda)")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interp {
    Python,
    Chibi,
    Futhark,
    /// The torch trainer interpreter: python with the fat ROCm torch
    /// closure bound. Always spawned in the GPU-enabled sandbox and
    /// reachable only through the `gpu:torch` effect.
    Torch,
}

/// Pre-resolve every interpreter's toolchain closure (the same
/// `which` + `store_closure` lookups `ReplSession::spawn` does) so the
/// first session start skips them. Starts no process: this is lookup
/// only, so it needs no grant — not even for the `gpu:torch` closure.
/// Missing interpreters and non-Nix hosts are skipped silently.
pub(crate) fn warm_closures() {
    for interp in [Interp::Python, Interp::Chibi, Interp::Futhark, Interp::Torch] {
        for dep in interp.deps() {
            if let Some(path) = closure::which(dep) {
                let _ = closure::store_closure(&path);
            }
        }
    }
}

impl Interp {
    pub fn parse(s: &str) -> Result<Interp, String> {
        match s {
            "python" | "python3" => Ok(Interp::Python),
            "chibi" | "chibi-scheme" => Ok(Interp::Chibi),
            "futhark" => Ok(Interp::Futhark),
            "torch" | "pytorch" | "python-torch" => Ok(Interp::Torch),
            other => Err(format!(
                "repl: unknown interpreter '{other}' (python | chibi-scheme | futhark | torch)"
            )),
        }
    }

    fn default_session(&self) -> &'static str {
        match self {
            Interp::Python => "python",
            Interp::Chibi => "chibi",
            Interp::Futhark => "futhark",
            Interp::Torch => "torch",
        }
    }

    /// Host binaries whose store closure the sandbox needs. Futhark's
    /// c backend shells out to `cc` at run time; Triton compiles its
    /// runtime shims with `gcc` AT KERNEL-LAUNCH TIME (not a build dep
    /// of the python env, so it must be bound explicitly); torch
    /// resolves to the dedicated `torch-python` wrapper (flake:
    /// torch-interp) so the lean `python3` keeps meaning the plain
    /// python closure.
    fn deps(&self) -> &'static [&'static str] {
        match self {
            Interp::Python => &["python3"],
            Interp::Chibi => &["chibi-scheme"],
            Interp::Futhark => &["futhark", "cc"],
            Interp::Torch => &["torch-python", "gcc"],
        }
    }

    fn config(&self, backend: Backend) -> Result<SessionConfig, String> {
        let empty: Vec<(std::path::PathBuf, std::path::PathBuf)> = Vec::new();
        let base =
            |sentinel: SentinelStyle, prompt: PromptStyle, program: &str, args: Vec<String>, gpu: bool| {
                SessionConfig {
                    program: program.into(),
                    args,
                    sentinel,
                    prompt,
                    deps: self.deps(),
                    gpu,
                    extra_ro_binds: empty.clone(),
                    rw_subdirs: Vec::new(),
                }
            };
        match self {
            Interp::Python => Ok(base(
                SentinelStyle::Print,
                PromptStyle::None,
                "python3",
                vec!["-i".into(), "-q".into(), "-B".into()],
                false,
            )),
            Interp::Chibi => Ok(base(
                SentinelStyle::Display,
                PromptStyle::Chibi,
                "chibi-scheme",
                vec![],
                false,
            )),
            Interp::Futhark => {
                // NOTE: futhark repl has no --backend flag (verified
                // against futhark 0.25): the repl is what it is, and
                // opencl/cuda apply to *compiled* programs (futhark
                // run/test/bench) in a later phase. The backend choice
                // still controls the sandbox: non-c runs in the
                // GPU-enabled sandbox behind the gpu:futhark grant.
                Ok(base(
                    SentinelStyle::None,
                    PromptStyle::Futhark,
                    "futhark",
                    vec!["repl".into()],
                    backend != Backend::C,
                ))
            }
            Interp::Torch => {
                // Same sentinel protocol as python — it IS python,
                // just the fat torch closure with the GPU sandbox
                // always on. Reached via the gpu:torch effect; the
                // plain repl tool refuses this interpreter. The
                // trainer writes experiment runs under experiments/,
                // mounted read-write over the ro workspace.
                let _ = backend;
                let mut cfg = base(
                    SentinelStyle::Print,
                    PromptStyle::None,
                    "torch-python",
                    vec!["-i".into(), "-q".into(), "-B".into()],
                    true,
                );
                cfg.rw_subdirs = vec!["experiments".into()];
                Ok(cfg)
            }
        }
    }
}

/// LLUNG_RO_BINDS (see [`crate::paths::ro_binds`]): each existing
/// entry bound read-only under its own path.
fn extra_ro_binds_from_env() -> Vec<(std::path::PathBuf, std::path::PathBuf)> {
    crate::paths::ro_binds().into_iter().map(|p| (p.clone(), p)).collect()
}

/// How the sentinel expression is injected after each input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SentinelStyle {
    /// `print("<token>")` (python)
    Print,
    /// `(display "<token>")` (chibi-scheme)
    Display,
    /// No print primitive — idle-read fallback (futhark).
    None,
}

/// Prompt noise to strip from stdout. Several REPLs write prompts even
/// when stdin/stdout are pipes (they never got the readline memo):
/// chibi prefixes lines with "> ", futhark with "[N]> ".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptStyle {
    None,
    Chibi,
    Futhark,
}

impl PromptStyle {
    /// Strip prompt prefixes line-wise. Only LEADING prompts are
    /// removed, so user output like "10 > 5" mid-line survives.
    fn clean(&self, s: &str) -> String {
        match self {
            PromptStyle::None => s.trim_end().to_string(),
            PromptStyle::Chibi => s
                .lines()
                .map(|l| {
                    let mut l = l;
                    while let Some(rest) = l.strip_prefix("> ") {
                        l = rest;
                    }
                    if l == ">" { "" } else { l }
                })
                .collect::<Vec<_>>()
                .join("\n")
                .trim_end()
                .to_string(),
            PromptStyle::Futhark => s
                .lines()
                .map(|l| {
                    let mut l = l;
                    // Repeatedly strip a leading "[N]> " prompt.
                    loop {
                        if let Some(rest) = l.strip_prefix('[') {
                            if let Some(close) = rest.find(']') {
                                let digits = &rest[..close];
                                if !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()) {
                                    let after = &rest[close + 1..];
                                    if let Some(after) = after.strip_prefix("> ") {
                                        l = after;
                                        continue;
                                    }
                                    if after == ">" {
                                        l = "";
                                    }
                                }
                            }
                        }
                        break;
                    }
                    l
                })
                .collect::<Vec<_>>()
                .join("\n")
                .trim_end()
                .to_string(),
        }
    }
}

/// Everything needed to spawn one REPL session. Tests build custom
/// configs (e.g. a fake sh interpreter) through this type.
pub struct SessionConfig {
    pub program: String,
    pub args: Vec<String>,
    pub sentinel: SentinelStyle,
    pub prompt: PromptStyle,
    pub deps: &'static [&'static str],
    pub gpu: bool,
    /// Extra read-only binds, e.g. a test script living outside the
    /// workspace. Real interpreter configs leave this empty.
    pub extra_ro_binds: Vec<(std::path::PathBuf, std::path::PathBuf)>,
    /// Workspace subdirectories mounted READ-WRITE over the ro
    /// workspace mount (the torch trainer's `experiments/`). Created
    /// if missing at spawn time.
    pub rw_subdirs: Vec<std::path::PathBuf>,
}

impl SessionConfig {
    /// Interp-built configs only; tests use struct literal.
    fn for_interp(interp: Interp, backend: Backend) -> Result<Self, String> {
        interp.config(backend)
    }
}

#[derive(Debug, Deserialize)]
pub struct ReplArgs {
    /// python | chibi-scheme | futhark | torch (torch only via gpu:torch)
    pub interp: String,
    /// Code to evaluate this turn.
    pub input: String,
    /// Session name separating independent states (default: the interpreter name).
    #[serde(default)]
    pub session: Option<String>,
    /// Per-evaluation wall-clock seconds (default 30, max 120).
    #[serde(default)]
    pub timeout_secs: Option<u64>,
    /// Futhark only: c (default) | opencl | cuda. Non-c backends need the
    /// gpu:futhark grant (Phase 5); via the plain repl they are rejected.
    #[serde(default)]
    pub backend: Option<String>,
}

fn parse_args(arguments_json: &str) -> Result<(Interp, Backend, ReplArgs), String> {
    let args: ReplArgs = serde_json::from_str(arguments_json)
        .map_err(|e| format!("repl: arguments must be JSON with 'interp' and 'input': {e}"))?;
    let interp = Interp::parse(&args.interp)?;
    let backend = match args.backend.as_deref() {
        None => Backend::C,
        Some(b) => Backend::parse(b)?,
    };
    Ok((interp, backend, args))
}

/// Shared output buffers + wakeup for one session's reader tasks.
/// stdout and stderr stay SEPARATE: interactive interpreters emit
/// prompts (python's `>>>`) on stderr, which would corrupt the
/// sentinel protocol if merged. Tracebacks arrive in `err` and are
/// appended to the turn's output.
struct SessionIo {
    buf: std::sync::Mutex<String>,
    err: std::sync::Mutex<String>,
    notify: Notify,
}

impl SessionIo {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            buf: std::sync::Mutex::new(String::new()),
            err: std::sync::Mutex::new(String::new()),
            notify: Notify::new(),
        })
    }

    /// Drain the turn output: stdout content up to `content_end`, the
    /// buffer drained through `drain_end` (past the sentinel), plus any
    /// stderr the interpreter produced (REPLs are line-oriented, not
    /// turn-oriented, so all pending stderr is this turn's).
    fn take_through(&self, content_end: usize, drain_end: usize) -> String {
        let mut out = String::new();
        {
            let mut buf = self.buf.lock().unwrap();
            let content_end = content_end.min(buf.len());
            out.push_str(buf[..content_end].trim_end());
            let drain_end = drain_end.min(buf.len());
            buf.drain(..drain_end);
        }
        let err = std::mem::take(&mut *self.err.lock().unwrap());
        // Interactive prompts (python's `>>>`/`...`) are written to
        // stderr WITHOUT newlines, so they glue together into
        // ">>> ...>>> " strings. They are protocol noise, not output:
        // strip every occurrence before deciding stderr has content.
        let err = err.replace(">>> ", "").replace("... ", "").replace(">>>", "").replace("...", "");
        let err = err.trim();
        if !err.is_empty() {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str("[stderr]\n");
            out.push_str(err);
        }
        out
    }

    fn take_all_output(&self) -> String {
        let len = self.buf.lock().unwrap().len();
        self.take_through(len, len)
    }
}

/// One long-running sandboxed interpreter.
pub struct ReplSession {
    child: tokio::process::Child,
    stdin: tokio::process::ChildStdin,
    io: Arc<SessionIo>,
    token: String,
    style: SentinelStyle,
    prompt: PromptStyle,
    /// Set when the interpreter can no longer be talked to (write
    /// failure or a timed-out runaway); the manager then drops it.
    dead: bool,
}

impl ReplSession {
    /// Spawn the interpreter inside a fresh bubblewrap sandbox with the
    /// toolchain closure bound and stdout merged with stderr.
    async fn spawn(config: &SessionConfig) -> Result<ReplSession, String> {
        if std::env::var("LLUNG_SANDBOX").as_deref() == Ok("off") {
            return Err("repl: LLUNG_SANDBOX=off is not supported for repl sessions (they need persistent pipes)".into());
        }
        // No `exec 2>&1` merge: stderr must stay separate so REPL
        // prompts (which live on stderr) can't corrupt the sentinel
        // protocol on stdout.
        let mut sb = Sandbox::baseline(
            "sh",
            vec![
                "-c".into(),
                format!("exec {} {}", config.program, shell_join(&config.args)),
            ],
        );
        for dep in config.deps {
            sb.bind_binary(dep)?;
        }
        for (src, dest) in &config.extra_ro_binds {
            sb.bind_ro(src, dest);
        }
        let workspace = std::env::var("LLUNG_WORKSPACE")
            .map(std::path::PathBuf::from)
            .or_else(|_| std::env::current_dir())
            .map_err(|e| format!("repl: cannot resolve workspace: {e}"))?;
        sb.workspace(&workspace, false);
        // Bind-mount rw_subdirs AFTER the ro workspace mount: bwrap
        // applies mounts in order, so the later rw bind of the subdir
        // shadows the ro parent (same shadowing rule as tmpfs /tmp).
        for sub in &config.rw_subdirs {
            let path = workspace.join(sub);
            if let Err(e) = std::fs::create_dir_all(&path) {
                return Err(format!(
                    "repl: cannot create rw mount '{}' in the workspace: {e}",
                    path.display()
                ));
            }
            sb.bind_rw(&path, &path);
        }
        // Operator-level extra read-only binds (LLUNG_RO_BINDS): data
        // that is too big for the workspace — e.g. TBs of precomputed
        // teacher traces on an external drive. Same absolute path
        // inside the sandbox, read at native bind-mount speed, never
        // writable. Nonexistent entries are skipped.
        for (src, dest) in extra_ro_binds_from_env() {
            sb.bind_ro(src, dest);
        }
        if config.gpu {
            sb.enable_gpu();
            // GPU drivers live outside the store closure; propagate the
            // host's library path and ROCm overrides (set by the
            // devShell shellHook — e.g. HSA_OVERRIDE_GFX_VERSION for
            // the gfx1150 iGPU).
            if let Ok(ld) = std::env::var("LD_LIBRARY_PATH") {
                sb.setenv("LD_LIBRARY_PATH", ld);
            }
            if let Ok(hsa) = std::env::var("HSA_OVERRIDE_GFX_VERSION") {
                sb.setenv("HSA_OVERRIDE_GFX_VERSION", hsa);
            }
        }
        let mut child = sb.spawn_piped().await?;
        let stdin = child.stdin.take().expect("stdin is piped");
        let mut stdout = child.stdout.take().expect("stdout is piped");
        let mut stderr = child.stderr.take().expect("stderr is piped");

        let io = SessionIo::new();
        let reader = io.clone();
        tokio::spawn(async move {
            let mut chunk = [0u8; 8192];
            loop {
                match stdout.read(&mut chunk).await {
                    Ok(0) => break,
                    Ok(n) => {
                        reader
                            .buf
                            .lock()
                            .unwrap()
                            .push_str(&String::from_utf8_lossy(&chunk[..n]));
                        reader.notify.notify_waiters();
                    }
                    Err(_) => break,
                }
            }
        });
        let err_reader = io.clone();
        tokio::spawn(async move {
            let mut chunk = [0u8; 8192];
            loop {
                match stderr.read(&mut chunk).await {
                    Ok(0) => break,
                    Ok(n) => {
                        err_reader
                            .err
                            .lock()
                            .unwrap()
                            .push_str(&String::from_utf8_lossy(&chunk[..n]));
                        err_reader.notify.notify_waiters();
                    }
                    Err(_) => break,
                }
            }
        });

        let token = format!("__LLUNG_END_{:016x}__", {
            let mut rng = rand::rand_core::UnwrapErr(rand::rngs::SysRng);
            rng.next_u64()
        });
        Ok(ReplSession {
            child,
            stdin,
            io,
            token,
            style: config.sentinel,
            prompt: config.prompt,
            dead: false,
        })
    }

    /// Strip this interpreter's prompt noise from a turn's output.
    fn clean(&self, s: &str) -> String {
        self.prompt.clean(s)
    }

    /// Evaluate one input in this session and return its output.
    async fn evaluate(&mut self, input: &str, timeout: Duration) -> Result<String, String> {
        let mut payload = String::new();
        if !input.trim().is_empty() {
            payload.push_str(input);
            payload.push('\n');
        }
        match self.style {
            SentinelStyle::Print => payload.push_str(&format!("print({:?})\n", self.token)),
            SentinelStyle::Display => payload.push_str(&format!("(display {:?})\n", self.token)),
            SentinelStyle::None => {}
        }
        if let Err(e) = self.stdin.write_all(payload.as_bytes()).await {
            self.dead = true;
            return Err(format!("repl: writing to interpreter: {e} (session is dead)"));
        }
        let _ = self.stdin.flush().await;
        match self.style {
            SentinelStyle::None => self.await_idle(timeout).await,
            _ => self.await_sentinel(timeout).await,
        }
    }

    /// Sentinel read: drain everything up to and including the token.
    async fn await_sentinel(&mut self, timeout: Duration) -> Result<String, String> {
        let deadline = Instant::now() + timeout;
        loop {
            {
                let mut buf = self.io.buf.lock().unwrap();
                if let Some(pos) = buf.find(&self.token) {
                    let end = pos + self.token.len();
                    // '\n'/'\r' are ASCII: char count == byte count.
                    let trailing: usize = buf[end..]
                        .chars()
                        .take_while(|c| *c == '\n' || *c == '\r')
                        .count();
                    let upto = (end + trailing).min(buf.len());
                    let pos = pos.min(upto);
                    drop(buf);
                    let raw = self.io.take_through(pos, upto);
                    return Ok(self.clean(&raw));
                }
            }
            if Instant::now() >= deadline {
                let out = self.io.take_all_output();
                self.kill().await;
                self.dead = true;
                return Err(format!(
                    "repl: evaluation timed out after {}s — the session was killed; start a new session if it wedged [captured: {:?}]",
                    timeout.as_secs(),
                    out.chars().take(200).collect::<String>()
                ));
            }
            // `notified()` may miss a wakeup that raced our buffer check;
            // the tick branch bounds the loss to WAIT_TICK.
            tokio::select! {
                _ = self.io.notify.notified() => {}
                _ = tokio::time::sleep(WAIT_TICK) => {}
            }
        }
    }

    /// Idle read (futhark): no print primitive, so a turn ends when
    /// stdout has been quiet for IDLE_QUIET, or when nothing at all
    /// arrived within FIRST_OUTPUT_GRACE (definitions can be silent).
    async fn await_idle(&mut self, timeout: Duration) -> Result<String, String> {
        let deadline = Instant::now() + timeout;
        let start = Instant::now();
        let mut last_len = 0usize;
        let mut last_activity = Instant::now();
        loop {
            let len = self.io.buf.lock().unwrap().len();
            if len > last_len {
                last_len = len;
                last_activity = Instant::now();
            } else {
                let quiet_for = last_activity.elapsed();
                let done = if len == 0 {
                    start.elapsed() >= FIRST_OUTPUT_GRACE
                } else {
                    quiet_for >= IDLE_QUIET
                };
                if done {
                    let raw = self.io.take_all_output();
                    return Ok(self.clean(&raw));
                }
            }
            if Instant::now() >= deadline {
                self.kill().await;
                self.dead = true;
                return Err(format!(
                    "repl: evaluation timed out after {}s — the session was killed",
                    timeout.as_secs()
                ));
            }
            tokio::time::sleep(WAIT_TICK).await;
        }
    }

    /// Kill the whole interpreter process group. `bwrap --new-session`
    /// makes the sandboxed command a group leader (pgid == pid), so one
    /// signal takes down the interpreter AND anything it spawned.
    async fn kill(&mut self) {
        if let Some(pid) = self.child.id() {
            // SAFETY: kill(2) with a negative pid signals the process
            // group; the pid comes from a live child we own.
            unsafe {
                libc::kill(-(pid as i32), libc::SIGKILL);
            }
        }
        let _ = self.child.kill().await;
        let _ = self.child.wait().await;
    }
}

/// Join args for the `sh -c "exec 2>&1; exec prog args…"` wrapper.
fn shell_join(args: &[String]) -> String {
    args.iter()
        .map(|a| {
            if a.chars().all(|c| c.is_ascii_alphanumeric() || "-_=./+".contains(c)) {
                a.clone()
            } else {
                format!("'{}'", a.replace('\'', "'\\''"))
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Per-agent REPL session table. Global (like the wasm worker is per
/// context): sessions are keyed by "agent:session" so agents never
/// share interpreter state.
#[derive(Default)]
pub struct ReplManager {
    sessions: HashMap<String, ReplSession>,
}

static MANAGER: OnceLock<Mutex<ReplManager>> = OnceLock::new();

pub fn manager() -> &'static Mutex<ReplManager> {
    MANAGER.get_or_init(|| Mutex::new(ReplManager::default()))
}

impl ReplManager {
    /// Evaluate `args.input` in the session named by `args.session`
    /// (created on first use). Dead sessions are recreated on the next
    /// call — the model gets a clean interpreter instead of a wedged one.
    pub async fn evaluate(
        &mut self,
        agent: &str,
        config: &SessionConfig,
        args: &ReplArgs,
    ) -> Result<String, String> {
        let name = args
            .session
            .clone()
            .unwrap_or_else(|| config_default_session(config));
        let key = format!("{agent}:{name}");
        if self.sessions.get(&key).is_some_and(|s| s.dead) {
            self.sessions.remove(&key);
        }
        if !self.sessions.contains_key(&key) {
            let session = ReplSession::spawn(config).await?;
            self.sessions.insert(key.clone(), session);
        }
        let timeout = Duration::from_secs(
            args.timeout_secs
                .unwrap_or(DEFAULT_TIMEOUT_SECS)
                .clamp(1, MAX_TIMEOUT_SECS),
        );
        let session = self.sessions.get_mut(&key).expect("just inserted");
        let result = session.evaluate(&args.input, timeout).await;
        if session.dead {
            self.sessions.remove(&key);
        }
        result
    }

    /// Kill one session (exists so `/repl kill <name>`-style commands
    /// can land in a later phase; also used by tests).
    pub async fn kill_session(&mut self, key: &str) -> bool {
        if let Some(mut session) = self.sessions.remove(key) {
            session.kill().await;
            true
        } else {
            false
        }
    }

    pub fn session_keys(&self) -> Vec<String> {
        self.sessions.keys().cloned().collect()
    }
}

fn config_default_session(config: &SessionConfig) -> String {
    // Matches Interp::default_session for the built-in configs; custom
    // (test) configs fall back to the program name.
    match config.program.as_str() {
        "python3" => "python".into(),
        "chibi-scheme" => "chibi".into(),
        "futhark" => "futhark".into(),
        "torch-python" => "torch".into(),
        other => other.into(),
    }
}

/// Tier::Repl executor: the `repl` tool (python, chibi-scheme, futhark
/// c backend). Non-c futhark backends and the torch interpreter are
/// refused here — they require the `gpu:futhark` / `gpu:torch` effects.
pub async fn run(agent: &str, arguments_json: &str) -> Result<String, String> {
    let (interp, backend, args) = parse_args(arguments_json)?;
    if interp == Interp::Torch {
        return Err(
            "repl: interp 'torch' runs in the GPU sandbox and requires the gpu:torch effect — \
             call TOOL {\"name\":\"gpu:torch\",\"arguments\":{...}} instead"
                .into(),
        );
    }
    if backend != Backend::C {
        return Err(format!(
            "repl: futhark backend '{}' requires the gpu:futhark grant — Phase 5 wires the prompt",
            match backend {
                Backend::OpenCl => "opencl",
                Backend::Cuda => "cuda",
                Backend::C => unreachable!(),
            }
        ));
    }
    let config = SessionConfig::for_interp(interp, backend)?;
    manager()
        .lock()
        .await
        .evaluate(agent, &config, &args)
        .await
}

/// Tier::Gpu executor (`gpu:futhark`, `gpu:torch`): futhark repl with
/// the opencl backend, or the torch trainer interpreter — both inside
/// the GPU-enabled sandbox. The human grant prompt sits in the
/// orchestrator (Phase 5), on top of the kernel check.
pub async fn run_gpu(agent: &str, effect_name: &str, arguments_json: &str) -> Result<String, String> {
    match effect_name {
        "futhark" => {
            let (interp, _backend, mut args) = parse_args(arguments_json)?;
            if interp != Interp::Futhark {
                return Err("gpu:futhark expects args for interp='futhark'".into());
            }
            args.backend = Some("opencl".into());
            let config = SessionConfig::for_interp(Interp::Futhark, Backend::OpenCl)?;
            manager()
                .lock()
                .await
                .evaluate(agent, &config, &args)
                .await
        }
        "torch" => {
            let (interp, _backend, args) = parse_args(arguments_json)?;
            if interp != Interp::Torch {
                return Err("gpu:torch expects args for interp='torch'".into());
            }
            let config = SessionConfig::for_interp(Interp::Torch, Backend::C)?;
            manager()
                .lock()
                .await
                .evaluate(agent, &config, &args)
                .await
        }
        other => Err(format!(
            "gpu:{other} has no executor (gpu:futhark opencl, gpu:torch)"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn skip_without_bwrap() -> bool {
        if std::env::var("LLUNG_SANDBOX").as_deref() == Ok("off") {
            eprintln!("LLUNG_SANDBOX=off; skipping");
            return true;
        }
        if closure::which("bwrap").is_none() {
            eprintln!("bubblewrap not on PATH; skipping");
            return true;
        }
        false
    }

    /// A scripted fake interpreter (sh): counts turns in `n` (state
    /// persists across turns), echoes the random sentinel back on any
    /// line containing __LLUNG_END_, and hangs on lines containing HANG.
    fn fake_interp_script(dir: &Path) -> std::path::PathBuf {
        let path = dir.join("fake-repl.sh");
        std::fs::write(
            &path,
            r#"#!/bin/sh
n=0
while IFS= read -r line; do
  case "$line" in
    *HANG*) sleep 999 ;;
    *__LLUNG_END_*) n=$((n + 1)); echo "result $n"; echo "$line" | grep -o '__LLUNG_END_[0-9a-f]*__' ;;
  esac
done
"#,
        )
        .unwrap();
        path
    }

    fn fake_config(script: &Path) -> SessionConfig {
        // Bound at /opt, NOT /tmp: baseline mounts a fresh tmpfs over
        // /tmp, which would shadow any bind underneath it.
        let in_sandbox = std::path::PathBuf::from("/opt/fake-repl.sh");
        SessionConfig {
            program: "sh".into(),
            args: vec![in_sandbox.to_string_lossy().into_owned()],
            sentinel: SentinelStyle::Print,
            prompt: PromptStyle::None,
            deps: &[],
            gpu: false,
            extra_ro_binds: vec![(script.to_path_buf(), in_sandbox)],
            rw_subdirs: Vec::new(),
        }
    }

    fn fake_args(input: &str, session: &str, timeout_secs: u64) -> ReplArgs {
        ReplArgs {
            interp: "fake".into(),
            input: input.into(),
            session: Some(session.into()),
            timeout_secs: Some(timeout_secs),
            backend: None,
        }
    }

    #[tokio::test]
    async fn sentinel_delimits_turns_and_state_persists() {
        if skip_without_bwrap() {
            return;
        }
        let dir = std::env::temp_dir().join(format!(
            "llung-repl-fake-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let script = fake_interp_script(&dir);
        let config = fake_config(&script);

        let mut mgr = ReplManager::default();
        let out1 = mgr
            .evaluate("tester", &config, &fake_args("anything", "s1", 10))
            .await
            .expect("turn 1");
        assert_eq!(out1, "result 1", "got: {out1:?}");
        let out2 = mgr
            .evaluate("tester", &config, &fake_args("more", "s1", 10))
            .await
            .expect("turn 2");
        assert_eq!(out2, "result 2", "state must persist across turns: {out2:?}");

        // A named second session is independent state.
        let out3 = mgr
            .evaluate("tester", &config, &fake_args("x", "s2", 10))
            .await
            .expect("session 2");
        assert_eq!(out3, "result 1", "got: {out3:?}");

        // Killing a session resets it on next use.
        let key = format!("tester:{}", "s1");
        assert!(mgr.kill_session(&key).await, "session existed");
        let out4 = mgr
            .evaluate("tester", &config, &fake_args("again", "s1", 10))
            .await
            .expect("recreated");
        assert_eq!(out4, "result 1", "fresh interpreter after kill: {out4:?}");

        mgr.kill_session(&format!("tester:s2")).await;
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn runaway_turn_is_killed_without_touching_other_sessions() {
        if skip_without_bwrap() {
            return;
        }
        let dir = std::env::temp_dir().join(format!(
            "llung-repl-hang-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let script = fake_interp_script(&dir);
        let config = fake_config(&script);

        let mut mgr = ReplManager::default();
        mgr.evaluate("tester", &config, &fake_args("warm up", "ok", 10))
            .await
            .unwrap();

        // HANG wedges this session; the 2s evaluation timeout kills it.
        let err = mgr
            .evaluate("tester", &config, &fake_args("HANG", "bad", 2))
            .await
            .unwrap_err();
        assert!(err.contains("timed out"), "got: {err}");

        // The wedged session was removed; the healthy one still answers.
        let out = mgr
            .evaluate("tester", &config, &fake_args("still", "ok", 10))
            .await
            .expect("other session alive");
        assert_eq!(out, "result 2", "got: {out:?}");

        mgr.kill_session(&format!("tester:ok")).await;
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn python_state_persists_when_available() {
        if skip_without_bwrap() || closure::which("python3").is_none() {
            eprintln!("python3 not on PATH; skipping");
            return;
        }
        let (interp, backend, mut args) =
            parse_args(&serde_json::json!({"interp": "python", "input": "x = 41", "timeout_secs": 15}).to_string())
                .unwrap();
        let config = SessionConfig::for_interp(interp, backend).unwrap();
        manager()
            .lock()
            .await
            .evaluate("test-python", &config, &args)
            .await
            .expect("set x");
        args.input = "x + 1".into();
        let out = manager()
            .lock()
            .await
            .evaluate("test-python", &config, &args)
            .await
            .expect("read x");
        assert_eq!(out.trim(), "42", "state must persist: {out:?}");
        manager().lock().await.kill_session("test-python:python").await;
    }

    #[tokio::test]
    async fn chibi_scheme_state_persists_when_available() {
        if skip_without_bwrap() || closure::which("chibi-scheme").is_none() {
            eprintln!("chibi-scheme not on PATH; skipping");
            return;
        }
        let (interp, backend, mut args) = parse_args(
            &serde_json::json!({"interp": "chibi-scheme", "input": "(define y 10)", "timeout_secs": 15})
                .to_string(),
        )
        .unwrap();
        let config = SessionConfig::for_interp(interp, backend).unwrap();
        manager()
            .lock()
            .await
            .evaluate("test-chibi", &config, &args)
            .await
            .expect("define y");
        args.input = "y".into();
        let out = manager()
            .lock()
            .await
            .evaluate("test-chibi", &config, &args)
            .await
            .expect("read y");
        assert_eq!(out.trim(), "10", "got: {out:?}");
        manager().lock().await.kill_session("test-chibi:chibi").await;
    }

    #[tokio::test]
    async fn futhark_repl_evaluates_when_available() {
        if skip_without_bwrap() || closure::which("futhark").is_none() {
            eprintln!("futhark not on PATH; skipping");
            return;
        }
        let (interp, backend, mut args) = parse_args(
            &serde_json::json!({"interp": "futhark", "input": "1 + 1", "timeout_secs": 60})
                .to_string(),
        )
        .unwrap();
        let config = SessionConfig::for_interp(interp, backend).unwrap();
        let out = manager()
            .lock()
            .await
            .evaluate("test-futhark", &config, &args)
            .await
            .expect("1+1");
        assert!(out.contains('2'), "got: {out:?}");
        // Idle-read: a definition produces no value output but must
        // still return (FIRST_OUTPUT_GRACE path).
        args.input = "def f (x: i32) = x + 1".into();
        manager()
            .lock()
            .await
            .evaluate("test-futhark", &config, &args)
            .await
            .expect("definition turn returns");
        args.input = "f 41".into();
        let out = manager()
            .lock()
            .await
            .evaluate("test-futhark", &config, &args)
            .await
            .expect("call f");
        assert!(out.contains('2'), "42 as i32 starts with 2: {out:?}");
        manager().lock().await.kill_session("test-futhark:futhark").await;
    }

    #[tokio::test]
    async fn arg_parsing_and_unknown_names() {
        let (interp, backend, args) = parse_args(
            &serde_json::json!({"interp": "futhark", "input": "1", "backend": "opencl"}).to_string(),
        )
        .unwrap();
        assert_eq!(interp, Interp::Futhark);
        assert_eq!(backend, Backend::OpenCl);
        assert_eq!(args.timeout_secs, None, "default filled at evaluate time");
        // Missing input is a schema error.
        assert!(parse_args(&serde_json::json!({"interp": "python"}).to_string())
            .unwrap_err()
            .contains("input"));
        assert!(Interp::parse("cobol").unwrap_err().contains("unknown interpreter"));
        assert!(Backend::parse("metal").unwrap_err().contains("unknown backend"));
        // Torch parses under all its aliases and never routes through
        // the plain repl executor.
        assert_eq!(Interp::parse("torch").unwrap(), Interp::Torch);
        assert_eq!(Interp::parse("pytorch").unwrap(), Interp::Torch);
        assert_eq!(Interp::parse("python-torch").unwrap(), Interp::Torch);
        assert_eq!(Interp::Torch.default_session(), "torch");
        assert!(run("tester", &serde_json::json!({"interp": "torch", "input": "1"}).to_string())
            .await
            .unwrap_err()
            .contains("gpu:torch"));
    }

    /// Phase 0 acceptance (RESEARCH-HARNESS-PLAN.md 0.1): a torch REPL
    /// session survives across calls and a matmul runs on the iGPU
    /// (HIP/`cuda`) inside the GPU-enabled bwrap sandbox.
    #[tokio::test]
    async fn torch_repl_runs_on_the_gpu_when_available() {
        if skip_without_bwrap() || closure::which("torch-python").is_none() {
            eprintln!("torch-python not on PATH; skipping");
            return;
        }
        let config = SessionConfig::for_interp(Interp::Torch, Backend::C).unwrap();
        let mut args = ReplArgs {
            interp: "torch".into(),
            input: "import torch; assert torch.cuda.is_available(), 'no GPU visible inside the sandbox'; dev = 'cuda'; x = torch.zeros(8, 8, device=dev); print('dev', dev, torch.__version__, torch.cuda.get_device_name(0))".into(),
            session: Some("acc".into()),
            timeout_secs: Some(90),
            backend: None,
        };
        let out = manager()
            .lock()
            .await
            .evaluate("test-torch", &config, &args)
            .await
            .expect("import torch on GPU");
        assert!(out.contains("dev cuda"), "the iGPU must be visible inside the sandbox: {out:?}");
        // State persists: x is still there; matmul on the iGPU.
        args.input = "y = x @ torch.ones(8, 8, device=dev); print('matmul_sum', y.sum().item())".into();
        let out = manager()
            .lock()
            .await
            .evaluate("test-torch", &config, &args)
            .await
            .expect("matmul");
        assert!(out.contains("matmul_sum 0.0"), "8x0 @ 8x1 must sum to 0: {out:?}");
        manager().lock().await.kill_session("test-torch:acc").await;
    }

    #[test]
    fn torch_config_uses_gpu_and_mounts_experiments_rw() {
        let cfg = SessionConfig::for_interp(Interp::Torch, Backend::C).unwrap();
        assert!(cfg.gpu, "torch sessions always spawn in the GPU sandbox");
        assert_eq!(cfg.program, "torch-python");
        assert_eq!(cfg.sentinel, SentinelStyle::Print, "it IS python");
        assert_eq!(cfg.rw_subdirs, vec![Path::new("experiments")]);
        assert!(
            cfg.deps.contains(&"gcc"),
            "triton compiles runtime shims at kernel-launch time"
        );
    }

    /// Triton acceptance: custom kernels compile AND run on the iGPU
    /// inside the torch sandbox. Triton compiles through a C toolchain
    /// at RUNTIME — the sandbox must let that happen (writable /tmp
    /// for the cache, compiler on PATH via the closure).
    #[tokio::test]
    async fn triton_kernel_compiles_inside_the_torch_sandbox() {
        if skip_without_bwrap() || closure::which("torch-python").is_none() {
            eprintln!("torch-python not on PATH; skipping");
            return;
        }
        let config = SessionConfig::for_interp(Interp::Torch, Backend::C).unwrap();
        let args = ReplArgs {
            interp: "torch".into(),
            // Write the kernel to a REAL file (triton inspects source),
        // then run it as a subprocess and report back.
            input: r#"
import os, subprocess, sys
os.makedirs("experiments/triton-smoke", exist_ok=True)
src = '''
import torch, triton, triton.language as tl
@triton.jit
def add_k(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)
if __name__ == "__main__":
    n = 1024
    x = torch.randn(n, device="cuda"); y = torch.randn(n, device="cuda"); out = torch.empty_like(x)
    add_k[(1,)](x, y, out, n, BLOCK=1024)
    torch.cuda.synchronize()
    print("triton_ok", torch.allclose(out, x + y))
'''
open("experiments/triton-smoke/k.py", "w").write(src)
r = subprocess.run([sys.executable, "experiments/triton-smoke/k.py"], capture_output=True, text=True, timeout=180)
print(r.stdout.strip() or r.stderr[-500:])
"#
            .into(),
            session: Some("triton".into()),
            timeout_secs: Some(240),
            backend: None,
        };
        let out = manager()
            .lock()
            .await
            .evaluate("test-torch", &config, &args)
            .await
            .expect("triton smoke");
        assert!(
            out.contains("triton_ok True"),
            "custom triton kernels must compile and run in the sandbox: {out:?}"
        );
        manager().lock().await.kill_session("test-torch:triton").await;
        // Workspace fixture cleanup (host side; the sandbox bind is rw).
        let _ = std::fs::remove_dir_all("experiments/triton-smoke");
    }

    #[test]
    fn parse_ro_binds_filters_and_splits() {
        let dir = std::env::temp_dir().join(format!("llung-robinds-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let dir_s = dir.to_string_lossy().into_owned();

        use crate::paths::parse_ro_binds;
        assert!(parse_ro_binds(None).is_empty());
        assert!(parse_ro_binds(Some("")).is_empty());
        // Colon-split, empties dropped, nonexistent paths skipped (the
        // session binds each under the SAME absolute path, so configs
        // can reference them verbatim).
        let parsed = parse_ro_binds(Some(&format!(":/nonexistent-xyz:{dir_s}::")));
        assert_eq!(parsed, vec![dir.clone()]);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// LLUNG_RO_BINDS data is actually readable inside a session — the
    /// streaming-traces path. SAFETY: scoped env override; the worst a
    /// concurrent session spawn picks up is one extra ro temp-dir bind.
    #[tokio::test]
    async fn ro_bind_env_path_is_readable_inside_a_session() {
        if skip_without_bwrap() || closure::which("python3").is_none() {
            eprintln!("python3/bwrap not available; skipping");
            return;
        }
        // NOT under /tmp: the sandbox mounts a fresh tmpfs over /tmp,
        // which would shadow the bind (same rule as argv()'s note).
        let dir = std::env::current_dir()
            .unwrap()
            .join("target")
            .join(format!(
                "llung-robind-live-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("trace shard 0001.bin"), b"trace-bytes").unwrap();

        unsafe { std::env::set_var("LLUNG_RO_BINDS", &dir) };
        let (interp, backend, mut args) = parse_args(
            &serde_json::json!({"interp": "python", "input": "pass", "timeout_secs": 15})
                .to_string(),
        )
        .unwrap();
        // The path inside the sandbox is the SAME absolute path.
        args.input = format!(
            "print(open({:?}).read())",
            dir.join("trace shard 0001.bin").to_string_lossy()
        );
        let config = SessionConfig::for_interp(interp, backend).unwrap();
        let out = manager()
            .lock()
            .await
            .evaluate("test-robind", &config, &args)
            .await
            .expect("read through the env bind");
        unsafe { std::env::remove_var("LLUNG_RO_BINDS") };
        assert!(
            out.contains("trace-bytes"),
            "external data must be streamable in the sandbox: {out:?}"
        );
        manager().lock().await.kill_session("test-robind:python").await;
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn shell_join_quotes() {
        assert_eq!(shell_join(&["a".into(), "b-c".into()]), "a b-c");
        assert_eq!(shell_join(&["it's".into()]), "'it'\\''s'");
        assert_eq!(
            shell_join(&["--backend=opencl".into()]),
            "--backend=opencl"
        );
    }
}
