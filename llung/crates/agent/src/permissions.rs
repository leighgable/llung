// crates/agent/src/permissions.rs
//
// The human-in-the-loop permission flow (Phase 5). The HARNESS asks —
// the agent is untrusted prompt-driven text, and the kernel runs in the
// harness process. When a manifest's tier `requires_prompt()` and no
// standing grant covers it, the orchestrator sends a `PermissionRequest`
// to the UI and awaits the operator's answer before executing:
//
//   y → Once     run this one manifest (self-signed grant, as before)
//   Y → Standing run it AND pre-authorize (agent, effect) forever
//   n → Deny     record a Denied receipt in the merkle history, tell the
//                model the operator said no
//
// `allow_standing` is false for tiers that never get implicit perpetual
// grants (Gpu): those require an explicit, named `/grant`.

use crate::effects::Tier;
use tokio::sync::oneshot;

/// One permission prompt, delivered to the UI over an mpsc channel.
pub struct PermissionRequest {
    /// Who is asking (agent display name + being id).
    pub agent_name: String,
    pub agent_id: String,
    /// The manifest's target_effect (e.g. "proc:bash", "gpu:torch").
    pub effect: String,
    pub tier: Tier,
    /// Short human-readable summary of the arguments (see
    /// `summarize_arguments`).
    pub summary: String,
    /// False when the tier forbids implicit standing grants (Gpu): the
    /// UI must not offer "always" here — `/grant` is the only path.
    pub allow_standing: bool,
    /// Answer channel; the UI sends exactly one.
    pub respond: oneshot::Sender<PermissionAnswer>,
}

impl std::fmt::Debug for PermissionRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PermissionRequest")
            .field("agent_name", &self.agent_name)
            .field("effect", &self.effect)
            .field("tier", &self.tier)
            .field("summary", &self.summary)
            .field("allow_standing", &self.allow_standing)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionAnswer {
    /// Authorize this one execution.
    Once,
    /// Authorize this execution and add a standing grant.
    Standing,
    /// Refuse; a Denied receipt is recorded.
    Deny,
}

/// Short, human-readable argument summary for the prompt. Never dumps
/// the raw JSON for long fields — a permission prompt is read, not audited.
pub fn summarize_arguments(effect: &crate::effects::Effect<'_>, arguments: &[u8]) -> String {
    let args = String::from_utf8_lossy(arguments);
    let json: serde_json::Value = serde_json::from_str(&args).unwrap_or(serde_json::Value::Null);
    let truncate = |s: &str, n: usize| -> String {
        let mut t: String = s.chars().take(n).collect();
        if s.chars().count() > n {
            t.push('…');
        }
        t
    };
    match effect.tier {
        Tier::ProcBash => json
            .get("cmd")
            .and_then(|c| c.as_str())
            .map(|c| format!("$ {c}"))
            .unwrap_or_else(|| truncate(&args, 120)),
        Tier::Repl => {
            let interp = json
                .get("interp")
                .and_then(|i| i.as_str())
                .unwrap_or(effect.name);
            let input = json
                .get("input")
                .and_then(|i| i.as_str())
                .map(|i| truncate(i, 100))
                .unwrap_or_default();
            format!("{interp}: {input}")
        }
        Tier::Gpu => format!(
            "{} {}",
            effect.name,
            json.get("backend")
                .and_then(|b| b.as_str())
                .map(|b| format!("(backend={b})"))
                .unwrap_or_default()
        ),
        // Host effects don't prompt today, but keep the summary useful
        // for receipts/logs: name the file being published.
        Tier::Host => json
            .get("path")
            .and_then(|p| p.as_str())
            .map(|p| format!("{} {p}", effect.name))
            .unwrap_or_else(|| truncate(&args, 120)),
        _ => truncate(&args, 120),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effects::classify;

    #[test]
    fn summarizes_bash_commands() {
        let e = classify("proc:bash");
        let s = summarize_arguments(&e, br#"{"cmd": "cargo test", "timeout_secs": 30}"#);
        assert_eq!(s, "$ cargo test");
    }

    #[test]
    fn summarizes_repl_input_truncated() {
        let e = classify("repl:python");
        let long = "x".repeat(500);
        let s = summarize_arguments(
            &e,
            format!(r#"{{"interp": "python", "input": "{long}"}}"#).as_bytes(),
        );
        assert!(s.starts_with("python: "), "got: {s}");
        assert!(s.contains('…'), "long input must truncate: {s}");
        assert!(s.chars().count() < 130, "got {} chars", s.chars().count());
    }

    #[test]
    fn summarizes_gpu_and_unknown_args() {
        let e = classify("gpu:futhark");
        let s = summarize_arguments(&e, br#"{"interp": "futhark", "backend": "opencl"}"#);
        assert!(s.contains("futhark"), "got: {s}");
        assert!(s.contains("opencl"), "got: {s}");
        // Non-JSON args degrade to a truncated raw dump.
        let e = classify("proc:bash");
        let s = summarize_arguments(&e, b"not json at all");
        assert!(!s.is_empty());
    }

    #[test]
    fn answer_variants_are_distinct() {
        assert_ne!(PermissionAnswer::Once, PermissionAnswer::Standing);
        assert_ne!(PermissionAnswer::Standing, PermissionAnswer::Deny);
    }
}
