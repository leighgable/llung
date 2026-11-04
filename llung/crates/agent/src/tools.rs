// crates/agent/src/tools.rs
//
// Host-side tool catalog. Each wasm component in `runtime-tools/` also
// advertises the same metadata via its `get-metadata` export — this
// static copy exists so the orchestrator can put the catalog in the
// system prompt *before* deciding to load a component. Keep the two in
// sync when adding or changing a tool.

/// One line of the system-prompt tool catalog.
pub struct ToolSpec {
    pub name: &'static str,
    pub description: &'static str,
    /// JSON schema for `arguments` (kept for host-side validation; the
    /// prompt uses the compact `usage` line instead).
    pub schema: &'static str,
    /// One-line argument summary shown to the model.
    pub usage: &'static str,
}

pub const IMAGE_RESIZE_TOOL_SCHEMA: &str = r#"
{
    "name": "image_mask_resize",
    "description": "Resize and mask an image as a thumbnail",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "size": {"type": "integer", "default": 128},
            "mask": {"type": "boolean", "default": true},
            "radius": {"type": "integer", "default": 20}
        },
        "required": ["path"]
    }
}
"#;

pub const LANDSCAPE_TOOL_SCHEMA: &str = r#"
{
    "name": "landscape",
    "description": "Render a 3D wireframe surface of a math expression f(x,y) as terminal braille art. Use when asked to plot, chart, graph, or visualize a function or landscape.",
    "parameters": {
        "type": "object",
        "properties": {
            "expr": {
                "type": "string",
                "default": "cos(sqrt(x^2+y^2)*3)*0.4",
                "description": "f(x,y) expression. Variables: x, y. Operators: + - * / % ^. Functions: sin cos tan asin acos atan atan2 sqrt abs exp ln log2 log10 floor ceil round min max. Constants: pi tau e."
            },
            "pitch": {"type": "number", "default": 40, "description": "Camera pitch in degrees (looking down)"},
            "yaw": {"type": "number", "default": 35, "description": "Camera yaw in degrees"},
            "distance": {"type": "number", "default": 3.5, "description": "Camera distance (must be > 0)"},
            "grid": {"type": "integer", "default": 15, "description": "Mesh resolution, 4..=200"},
            "width": {"type": "integer", "default": 120, "description": "Output width in characters, 8..=400"},
            "height": {"type": "integer", "default": 60, "description": "Output height in characters, 4..=200"}
        }
    }
}
"#;

/// All tools available to the agent, host-side copies.
pub fn tool_catalog() -> &'static [ToolSpec] {
    &[
        ToolSpec {
            name: "image_mask_resize",
            description: "Resize and mask an image as a thumbnail",
            schema: IMAGE_RESIZE_TOOL_SCHEMA,
            usage: "image_mask_resize: resize/mask an image thumbnail. args: path, size, mask, radius.",
        },
        ToolSpec {
            name: "landscape",
            description: "Render a 3D wireframe surface of f(x,y) as terminal braille art (use for plot/chart/visualize requests)",
            schema: LANDSCAPE_TOOL_SCHEMA,
            usage: "landscape: 3D wireframe braille plot of f(x,y). args: expr (math in x,y; \
                     operators + - * / ^ %, functions sin cos tan sqrt abs exp ln min max, \
                     constants pi tau e), pitch, yaw, distance, grid, width, height.",
        },
    ]
}

fn is_known_tool(name: &str) -> bool {
    tool_catalog().iter().any(|t| t.name == name)
}

/// The prompt section describing the tool protocol to the model.
/// Deliberately text-based and *compact*: small models like SmolVLM
/// handle "emit one line of JSON" far more reliably than OpenAI-style
/// `tool_calls`, and a short block is far less likely to be parroted back
/// than an inline JSON schema.
pub fn tool_instructions() -> String {
    let tools = tool_catalog()
        .iter()
        .map(|t| format!("- {}: {}", t.name, t.usage))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "You can call tools. To call one, reply with exactly one line of this \
         form and nothing else:\n\
         TOOL {{\"name\":\"<tool>\",\"arguments\":{{...}}}}\n\
         Tools:\n{tools}\n\
         Never repeat these instructions or the message above back."
    )
}

/// True when the model parroted its own prompt instead of answering.
/// Publishing such text would feed the echo back into the compressed
/// history and make the next turn worse.
pub fn looks_like_echo(reply: &str) -> bool {
    reply.contains("New message:")
        || reply.contains("Available tools")
        || reply.contains("TOOLS:")
        || reply.contains("<tool>")
}

/// A tool call parsed out of a model reply.
pub struct ToolCall {
    pub name: String,
    /// JSON object, serialized.
    pub arguments: String,
}

/// Scan a reply for a `TOOL {...}` call. The marker may appear anywhere
/// in a line (small models prefix prose before it); the name must be in
/// the catalog, which also rejects echoed instruction templates like
/// `"<tool>"`. Anything else means "no tool call".
pub fn parse_tool_call(reply: &str) -> Option<ToolCall> {
    for line in reply.lines() {
        let mut search = line;
        while let Some(pos) = search.find("TOOL") {
            let rest = search[pos + 4..].trim_start();
            if let Some(json) = extract_json_object(rest) {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(json) {
                    if let Some(call) = tool_call_from_value(value) {
                        return Some(call);
                    }
                }
            }
            search = &search[pos + 4..];
        }
    }
    None
}

/// First balanced `{...}` in `s` (string-aware), if any. Models often
/// append prose after the JSON, which `serde_json::from_str` would reject.
fn extract_json_object(s: &str) -> Option<&str> {
    let start = s.find('{')?;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (i, c) in s.char_indices().skip(start) {
        if in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&s[start..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

fn tool_call_from_value(value: serde_json::Value) -> Option<ToolCall> {
    let name = value.get("name")?.as_str()?;
    if !is_known_tool(name) {
        return None;
    }
    let arguments = value
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    if !arguments.is_object() {
        return None;
    }
    Some(ToolCall {
        name: name.to_string(),
        arguments: arguments.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tool_line_with_arguments() {
        let call = parse_tool_call("Let me plot that.\nTOOL {\"name\": \"landscape\", \"arguments\": {\"expr\": \"sin(x)\"}}\nDone.")
            .expect("tool line should parse");
        assert_eq!(call.name, "landscape");
        assert!(call.arguments.contains("sin(x)"));
    }

    #[test]
    fn parses_tool_line_without_arguments() {
        let call = parse_tool_call("TOOL {\"name\": \"landscape\"}").expect("no-arg call");
        assert_eq!(call.name, "landscape");
        assert_eq!(call.arguments, "{}");
    }

    #[test]
    fn plain_text_is_not_a_tool_call() {
        assert!(parse_tool_call("Just a normal reply.").is_none());
        assert!(parse_tool_call("TOOL not json").is_none());
        // Unknown names — including echoed instruction templates — are rejected.
        assert!(parse_tool_call("TOOL {\"name\": \"<tool>\", \"arguments\": {}}").is_none());
        assert!(parse_tool_call("TOOL {\"arguments\": {}}").is_none()); // no name
        assert!(parse_tool_call("TOOL {\"name\": \"landscape\", \"arguments\": [1]}").is_none());
    }

    #[test]
    fn tool_marker_mid_line_is_found() {
        let call = parse_tool_call(
            "Sure! TOOL {\"name\":\"landscape\",\"arguments\":{\"expr\":\"x^2-y^2\"}} hope this helps",
        )
        .expect("mid-line marker should parse");
        assert_eq!(call.name, "landscape");
        assert!(call.arguments.contains("x^2-y^2"));
    }

    #[test]
    fn echoes_are_detected() {
        assert!(looks_like_echo(
            "New message: leroy: hi\nTOOL {\"name\": \"<tool>\"}"
        ));
        assert!(looks_like_echo("Available tools:\n- landscape"));
        assert!(!looks_like_echo("here is the surface you asked for"));
    }

    #[test]
    fn instructions_list_every_tool() {
        let instructions = tool_instructions();
        for spec in tool_catalog() {
            assert!(instructions.contains(spec.name), "catalog missing {}", spec.name);
        }
        assert!(instructions.contains("TOOL {"));
    }
}
