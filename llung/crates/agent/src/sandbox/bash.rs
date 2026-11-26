// crates/agent/src/sandbox/bash.rs
//
// The `proc:bash` tool: run a shell command inside the bubblewrap
// sandbox. Default policy: workspace mounted READ-ONLY (a failed write
// beats a rollback), wall-clock timeout, memory limit, combined output
// cap. `writable=true` is where Phase 5's grant prompt hooks in.

use serde::Deserialize;
use std::time::Duration;
use tokio::io::AsyncReadExt;

use crate::sandbox::bwrap::{MAX_OUTPUT_BYTES, Sandbox};

const DEFAULT_TIMEOUT_SECS: u64 = 30;
const MAX_TIMEOUT_SECS: u64 = 300;
const DEFAULT_MEMORY_MB: u64 = 512;

#[derive(Debug, Deserialize)]
struct BashArgs {
    /// Shell command to run (`sh -c`).
    cmd: String,
    /// Wall-clock seconds before the process is killed (default 30, max 300).
    #[serde(default)]
    timeout_secs: Option<u64>,
    /// Mount the workspace writable (default false = read-only).
    /// Phase 5: requires a human grant.
    #[serde(default)]
    writable: Option<bool>,
    /// Memory limit in MB (default 512).
    #[serde(default)]
    memory_mb: Option<u64>,
}

/// Synchronous argument validation (unit-testable without a runtime).
fn parse_args(arguments_json: &str) -> Result<BashArgs, String> {
    let args: BashArgs = serde_json::from_str(arguments_json)
        .map_err(|e| format!("bash: arguments must be JSON with a 'cmd' string: {e}"))?;
    if args.cmd.trim().is_empty() {
        return Err("bash: 'cmd' must not be empty".into());
    }
    Ok(args)
}

/// Run `cmd` in the sandbox. Mounts come from `places` (see
/// `crate::places`): the workspace (ro unless `writable`), the data
/// directories ro at their own paths, and the runs directory rw —
/// the agent's scratch area, writable without approval. The runs path
/// is also exported as `$LLUNG_RUNS_DIR`.
pub async fn run(arguments_json: &str, places: &crate::places::Places) -> Result<String, String> {
    let args = parse_args(arguments_json)?;
    let timeout_secs = args
        .timeout_secs
        .unwrap_or(DEFAULT_TIMEOUT_SECS)
        .clamp(1, MAX_TIMEOUT_SECS);
    let memory_mb = args.memory_mb.unwrap_or(DEFAULT_MEMORY_MB).clamp(16, 1 << 20);
    let writable = args.writable.unwrap_or(false);
    let workspace = places.workspace().to_path_buf();

    // Escape hatch for hosts without bubblewrap (macOS dev, exotic
    // CI): LLUNG_SANDBOX=off runs the command directly with the same
    // timeout, clearly labeled.
    if std::env::var("LLUNG_SANDBOX").as_deref() == Ok("off") {
        return run_unsandboxed(&args.cmd, Duration::from_secs(timeout_secs)).await;
    }

    let mut sb = Sandbox::baseline("sh", vec!["-c".into(), args.cmd.clone()]);
    sb.workspace(&workspace, writable);
    // Data directories (e.g. ~/Code/data): read-only at their own
    // paths — the same set the repl sessions and plot_device's
    // host-side reader accept.
    for dir in places.data_dirs() {
        sb.bind_ro(&dir, &dir);
    }
    // Runs: writable scratch. A chosen-but-missing runs dir (unmounted
    // drive) is reported in the output, not created.
    let mut runs_note = None;
    match places.runs_ready() {
        Ok(runs) => {
            sb.bind_rw(&runs, &runs);
            sb.setenv("LLUNG_RUNS_DIR", runs.to_string_lossy().into_owned());
        }
        Err(e) => runs_note = Some(e),
    }
    sb.memory_max(format!("{memory_mb}M"));
    let out = sb.run(Duration::from_secs(timeout_secs)).await?;

    let mut text = String::new();
    if let Some(note) = &runs_note {
        text.push_str(&format!("[runs directory not mounted: {note}]\n"));
    }
    if let Some(warn) = &out.degraded {
        text.push_str(&format!("[sandbox: degraded — {warn}]\n"));
    }
    if out.timed_out {
        text.push_str(&format!("bash: timed out after {timeout_secs}s\n"));
        return Err(text);
    }
    let stdout = capped(&out.stdout);
    let stderr = capped(&out.stderr);
    match out.exit_code {
        Some(0) => {
            text.push_str(&stdout);
            if !stderr.is_empty() {
                if !text.is_empty() && !text.ends_with('\n') {
                    text.push('\n');
                }
                text.push_str(&format!("[stderr]\n{stderr}"));
            }
            Ok(text)
        }
        code => {
            text.push_str(&format!("bash: exited with code {code:?}\n"));
            if !stdout.is_empty() {
                text.push_str(&format!("[stdout]\n{stdout}\n"));
            }
            if !stderr.is_empty() {
                text.push_str(&format!("[stderr]\n{stderr}"));
            }
            Err(text)
        }
    }
}

fn capped(bytes: &[u8]) -> String {
    let truncated = bytes.len() > MAX_OUTPUT_BYTES;
    let mut s = String::from_utf8_lossy(&bytes[..bytes.len().min(MAX_OUTPUT_BYTES)]).into_owned();
    if truncated {
        s.push_str(&format!("\n… output truncated at {MAX_OUTPUT_BYTES} bytes"));
    }
    s
}

async fn run_unsandboxed(cmd: &str, timeout: Duration) -> Result<String, String> {
    let mut child = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        // Timeout path just drops the future; make drop kill the child.
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("bash: spawn failed: {e}"))?;
    let mut so = child.stdout.take().expect("stdout is piped");
    let mut se = child.stderr.take().expect("stderr is piped");
    let result = tokio::time::timeout(timeout, async {
        let (mut ob, mut eb) = (Vec::new(), Vec::new());
        let (stdout, stderr) = tokio::join!(
            async {
                so.read_to_end(&mut ob).await.ok();
                ob
            },
            async {
                se.read_to_end(&mut eb).await.ok();
                eb
            },
        );
        let status = child.wait().await;
        (stdout, stderr, status)
    })
    .await;
    let mut text = String::from("[sandbox: OFF — LLUNG_SANDBOX=off, command ran unsandboxed]\n");
    match result {
        Ok((stdout, stderr, Ok(status))) => {
            text.push_str(&String::from_utf8_lossy(&stdout));
            if !status.success() {
                text.push_str(&format!("bash: exited with {:?}\n", status.code()));
                text.push_str(&String::from_utf8_lossy(&stderr));
                return Err(text);
            }
            Ok(text)
        }
        Ok((_, _, Err(e))) => Err(format!("{text}bash: wait failed: {e}")),
        Err(_) => Err(format!("{text}bash: timed out after {}s", timeout.as_secs())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skip_without_bwrap() -> bool {
        if std::env::var("LLUNG_SANDBOX").as_deref() == Ok("off") {
            eprintln!("LLUNG_SANDBOX=off; skipping");
            return true;
        }
        if crate::sandbox::closure::which("bwrap").is_none() {
            eprintln!("bubblewrap not on PATH; skipping");
            return true;
        }
        false
    }

    /// Run the tool end-to-end (argv → bwrap → output), in the current
    /// process cwd as the workspace (read-only).
    async fn bash(cmd: &str) -> Result<String, String> {
        let args = serde_json::json!({"cmd": cmd, "timeout_secs": 10}).to_string();
        run(&args, &crate::places::current()).await
    }

    fn places_with(data: Vec<std::path::PathBuf>, runs: &std::path::Path) -> crate::places::Places {
        crate::places::Places::fixed(std::env::current_dir().unwrap(), runs, data)
    }

    /// The runs directory is writable (no approval) while the workspace
    /// stays read-only, and its path is exported as $LLUNG_RUNS_DIR.
    #[tokio::test]
    async fn runs_dir_is_writable_and_exported() {
        if skip_without_bwrap() {
            return;
        }
        let runs = std::env::current_dir()
            .unwrap()
            .join("target")
            .join(format!("llung-bash-runs-{}", std::process::id()));
        std::fs::create_dir_all(&runs).unwrap();
        let places = places_with(vec![], &runs);
        let args = serde_json::json!({
            "cmd": "echo ok > \"$LLUNG_RUNS_DIR/marker\" && cat \"$LLUNG_RUNS_DIR/marker\"",
            "timeout_secs": 10
        })
        .to_string();
        let out = run(&args, &places).await;
        let written = std::fs::read_to_string(runs.join("marker")).ok();
        std::fs::remove_dir_all(&runs).ok();
        assert!(out.expect("runs is writable").contains("ok"));
        assert_eq!(written.as_deref(), Some("ok\n"));

        // A chosen runs dir that is missing (unmounted drive): the
        // command still runs, the output says why runs isn't there, and
        // nothing is created.
        let gone = runs.join("not-mounted");
        let out = run(
            &serde_json::json!({"cmd": "echo hi", "timeout_secs": 10}).to_string(),
            &places_with(vec![], &gone),
        )
        .await
        .unwrap();
        assert!(out.contains("runs directory not mounted"), "{out}");
        assert!(!gone.exists());
    }

    /// LLUNG_RO_BINDS directories (operator data outside the
    /// workspace, e.g. ~/Code/data) are visible read-only at their own
    /// path. Binds injected directly: no env mutation, no cross-test
    /// interference. The dir lives under /tmp ON PURPOSE — the sandbox's
    /// /tmp tmpfs must not hide it (mount layering in `argv`).
    #[tokio::test]
    async fn ro_binds_are_readable_but_not_writable() {
        if skip_without_bwrap() {
            return;
        }
        let dir = std::env::temp_dir().join(format!("llung-bash-robind-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("flights.csv"), "month,passengers\n1949-01,112\n").unwrap();
        let places = places_with(vec![dir.clone()], &std::env::temp_dir().join("llung-unused-runs"));
        let args = |cmd: String| serde_json::json!({"cmd": cmd, "timeout_secs": 10}).to_string();
        let read = run(&args(format!("head -2 {}/flights.csv", dir.display())), &places).await;
        let write = run(&args(format!("echo x > {}/new.txt", dir.display())), &places).await;
        std::fs::remove_dir_all(&dir).ok();

        let read = read.expect("bound data is readable");
        assert!(read.contains("1949-01,112"), "got: {read}");
        assert!(write.is_err(), "bound data must be read-only: {write:?}");
    }

    #[tokio::test]
    async fn echo_runs_and_returns_stdout() {
        if skip_without_bwrap() {
            return;
        }
        let out = bash("echo hello").await.expect("echo");
        assert!(out.contains("hello"), "got: {out}");
    }

    #[tokio::test]
    async fn nonzero_exit_is_an_error_with_stderr() {
        if skip_without_bwrap() {
            return;
        }
        let err = bash("echo some-stdout; echo some-stderr >&2; exit 3")
            .await
            .unwrap_err();
        assert!(err.contains("exited with code Some(3)"), "got: {err}");
        assert!(err.contains("some-stdout"), "got: {err}");
        assert!(err.contains("some-stderr"), "got: {err}");
    }

    #[tokio::test]
    async fn readonly_workspace_rejects_writes() {
        if skip_without_bwrap() {
            return;
        }
        // The tool's own cwd is mounted ro by default.
        let err = bash("touch llung-bwrap-write-test.tmp")
            .await
            .unwrap_err();
        assert!(err.contains("exited with code"), "got: {err}");
        assert!(
            !std::env::current_dir()
                .unwrap()
                .join("llung-bwrap-write-test.tmp")
                .exists(),
            "read-only default must prevent host writes"
        );
    }

    #[tokio::test]
    async fn timeout_kills_a_hang() {
        if skip_without_bwrap() {
            return;
        }
        let err = bash("sleep 60").await.unwrap_err();
        assert!(err.contains("timed out"), "got: {err}");
    }

    #[tokio::test]
    async fn unsandboxed_escape_hatch_runs_labeled() {
        // Directly exercise the LLUNG_SANDBOX=off code path. The env var
        // itself is read once at the top of run(); flipping process env
        // in a unit test would race with the parallel sandboxed tests.
        let out = run_unsandboxed("echo off-mode", Duration::from_secs(10))
            .await
            .expect("off mode");
        assert!(out.contains("sandbox: OFF"), "got: {out}");
        assert!(out.contains("off-mode"), "got: {out}");
    }

    #[test]
    fn bad_args_are_clean_errors() {
        assert!(parse_args("not json").unwrap_err().contains("must be JSON"));
        // Missing 'cmd' is a JSON-schema error naming the field.
        assert!(parse_args(&serde_json::json!({}).to_string())
            .unwrap_err()
            .contains("cmd"));
        assert!(parse_args(&serde_json::json!({"cmd": "   "}).to_string())
            .unwrap_err()
            .contains("must not be empty"));
        let ok = parse_args(&serde_json::json!({"cmd": "ls"}).to_string()).unwrap();
        assert_eq!(ok.timeout_secs, None);
        assert!(!ok.writable.unwrap_or(false));
    }
}
