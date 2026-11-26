// crates/agent/src/tools/host/mod.rs
//
// In-process host tools: glob, grep, edit. These run natively inside the
// harness (the `ignore`/`grep-searcher`/`similar` stack — the same
// libraries ripgrep and diff tooling are built on), never spawning
// external processes. Phase 3 adds the bubblewrap-backed `bash` tool.

pub mod edit;
pub mod glob;
pub mod grep;

/// Names of every host tool, also advertised in the tool catalog.
pub const HOST_TOOLS: &[&str] = &["glob", "grep", "edit"];

/// Dispatch to a host tool. Returns `None` when `name` is not a host
/// tool, so the caller falls through to the wasm runtime path.
/// `scope`: the agent's places (confined), or `None` for the operator.
pub fn dispatch(
    name: &str,
    arguments_json: &str,
    scope: Option<&crate::places::Places>,
) -> Option<Result<String, String>> {
    match name {
        "glob" => Some(glob::run_in(arguments_json, scope)),
        "grep" => Some(grep::run_in(arguments_json, scope)),
        "edit" => Some(edit::run_in(arguments_json, scope)),
        _ => None,
    }
}
