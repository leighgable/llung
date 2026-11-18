//! `diagram` — a llung wasm tool.
//!
//! Flowcharts and nested-box diagrams as terminal box-drawing art. The
//! model describes the graph in a small line-based language; the tool
//! lays it out (ranks, crossings, containers) and returns monospace art
//! that renders in any terminal — pure text, no graphics protocol.
//!
//! The language (also in the tool description the model reads):
//!
//! ```text
//! # comments start with #
//! box OUTER: Optional Label {   # containers nest freely
//!   A: First node               # node: `id: label`
//!   box INNER {                 # label defaults to the id
//!     B: Nested node
//!   }
//! }
//! A -> B: optional edge label   # edges; chains (A -> B -> C) work too
//! C                             # bare id = node with that label
//! ```

wit_bindgen::generate!({
    path: "wit",
    world: "universal-tool",
});

mod dsl;
mod render;

/// KEEP IN SYNC with `DIAGRAM_TOOL_SCHEMA` in crates/agent/src/tools.rs
/// (the host-side copy that goes into the LLM's system prompt).
/// (Only referenced by the component export glue, hence wasm-only.)
#[cfg(target_arch = "wasm32")]
const INPUT_SCHEMA: &str = r#"{
  "type": "object",
  "properties": {
    "graph": {"type": "string", "description": "The diagram in diagram-language (see tool description). Lines: 'ID: Label' declares a node; 'box ID: Label { ... }' declares a nested container (close with '}'); 'A -> B: optional label' declares edges (chains A -> B -> C work); '#' starts a comment."},
    "title": {"type": "string", "description": "OPTIONAL short title shown next to the panel (e.g. 'agent loop'). Omit unless useful."},
    "width": {"type": "integer", "default": 64, "description": "Maximum width in characters, 24..=110. Smaller fits narrow chat panels."}
  },
  "required": ["graph"]
}"#;

/// Cap the art to one chat message: even 110×60 ≈ 6.6k cells of mostly
/// single-byte box chars (~8 KiB) plus JSON escaping.
const MAX_WIDTH: usize = 110;
const MIN_WIDTH: usize = 24;

#[cfg(target_arch = "wasm32")]
struct Diagram;

#[cfg(target_arch = "wasm32")]
impl Guest for Diagram {
    fn get_metadata() -> ToolMetadata {
        ToolMetadata {
            name: "diagram".into(),
            description: "Draw a flowchart or nested-box diagram as terminal \
                          box-drawing art. Use when asked to diagram, flowchart, \
                          or draw boxes-and-arrows of a system or process."
                .into(),
            input_schema: INPUT_SCHEMA.into(),
        }
    }

    fn execute(arguments_json: String) -> Result<String, String> {
        run(&arguments_json)
    }
}

#[cfg(target_arch = "wasm32")]
export!(Diagram);

/// Render the graph and return the box-drawing art (plain text — the
/// chat shows it as a wrapped panel; keep `width` within the panel).
pub fn run(arguments_json: &str) -> Result<String, String> {
    let v: serde_json::Value = serde_json::from_str(arguments_json)
        .map_err(|e| format!("arguments must be a JSON object: {e}"))?;
    let graph_src = v
        .get("graph")
        .and_then(|s| s.as_str())
        .ok_or("missing required 'graph' string")?;
    let width = v
        .get("width")
        .and_then(|n| n.as_i64())
        .unwrap_or(64)
        .clamp(MIN_WIDTH as i64, MAX_WIDTH as i64) as usize;

    let graph = dsl::parse(graph_src);
    if graph.nodes.is_empty() && graph.items.is_empty() {
        return Err("the graph is empty — declare nodes like 'A: First step'".into());
    }
    Ok(render::render(&graph, width))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_loop_example_draws() {
        let out = run(
            r#"{"graph": "INTAKE: Intake\nbox INVESTIGATE: Investigate {\n  SUBA: Sub Agent A\n  SUBB: Sub Agent B\n}\nINTAKE -> INVESTIGATE\nSUBA -> PLAN: findings\nSUBB -> PLAN\nPLAN: Draft Plan\nPLAN -> APPROVAL: proposed plan\nAPPROVAL -> EXECUTE: approved\nEXECUTE -> REPORT\nREPORT: Report\n", "width": 72}"#,
        )
        .unwrap();
        for expected in [
            "Intake", "Investigate", "Sub Agent A", "Sub Agent B", "Draft Plan", "Report",
        ] {
            assert!(out.contains(expected), "missing {expected}:\n{out}");
        }
        assert!(out.contains('▼'));
        assert!(out.lines().all(|l| l.chars().count() <= 72));
        println!("{out}");
    }

    #[test]
    fn nested_harness_example_draws() {
        let out = run(
            r#"{"graph": "box HARNESS: Harness {\n  box CONTEXT: Context {\n    box PROMPT: Prompt {\n      box MODEL: Model {\n      }\n    }\n  }\n}\n"}"#,
        )
        .unwrap();
        assert!(out.contains("Harness"));
        assert!(out.contains("Model"));
    }

    #[test]
    fn missing_graph_is_a_clean_error() {
        assert!(run(r#"{"width": 40}"#).unwrap_err().contains("graph"));
        assert!(run("not json").is_err());
    }

    #[test]
    fn width_is_clamped() {
        let narrow = run(r#"{"graph": "A: Alpha\nB: Beta\nA -> B\n", "width": 5}"#).unwrap();
        assert!(narrow.lines().all(|l| l.chars().count() <= 24));
    }
}
