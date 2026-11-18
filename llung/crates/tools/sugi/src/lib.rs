//! `sugi` — a llung wasm tool.
//!
//! Layered layout for directed graphs (the Sugiyama method, via
//! `rust-sugiyama` on a `petgraph` `StableDiGraph`). The model sends
//! plain JSON — nodes and edges, nothing else to learn — and the tool
//! returns a `sugi-v1` render document: every node's coordinates and
//! size plus the edge list. The HOST (TUI) places the boxes on a
//! character grid and draws orthogonal connectors — the tool does not
//! draw anything itself.
//!
//! Why a render document instead of art (like `diagram`): coordinates
//! let the host re-fit the diagram to the viewer's width at render
//! time, instead of the tool guessing one width up front.

wit_bindgen::generate!({
    path: "wit",
    world: "universal-tool",
});

use std::collections::HashMap;

use petgraph::graph::NodeIndex;
use petgraph::stable_graph::StableDiGraph;
use rust_sugiyama::configure::Config;

/// KEEP IN SYNC with `SUGI_TOOL_SCHEMA` in crates/agent/src/tools.rs
/// (the host-side copy that goes into the LLM's system prompt).
#[cfg(target_arch = "wasm32")]
const INPUT_SCHEMA: &str = r#"{
  "type": "object",
  "properties": {
    "nodes": {"type": "array", "description": "Nodes: strings ['A', ...] or objects [{'id': 'A', 'label': 'Human text'}, ...]. Labels default to the id."},
    "edges": {"type": "array", "description": "Edges as pairs [['A','B'], ...] or objects [{'from': 'A', 'to': 'B', 'label': 'optional'}, ...]. Cycles are laid out best-effort."},
    "title": {"type": "string", "description": "OPTIONAL short title shown next to the panel. Omit unless useful."},
    "width": {"type": "integer", "description": "OPTIONAL maximum panel width in characters (8..=200). Omit unless asked."}
  },
  "required": ["nodes", "edges"]
}"#;

/// Node count cap: the Sugiyama pass is superlinear; a chat diagram is
/// small. The model must split bigger drawings.
const MAX_NODES: usize = 100;
const MAX_EDGES: usize = 300;

/// Node box size in character units: label + horizontal padding, one
/// text row with a blank row above and below.
fn vertex_size(label: &str) -> (f64, f64) {
    ((label.chars().count() as f64 + 4.0).max(8.0), 3.0)
}

struct InputNode {
    id: String,
    label: String,
}

struct InputEdge {
    from: String,
    to: String,
    label: Option<String>,
}

#[cfg(target_arch = "wasm32")]
struct Sugi;

#[cfg(target_arch = "wasm32")]
impl Guest for Sugi {
    fn get_metadata() -> ToolMetadata {
        ToolMetadata {
            name: "sugi".into(),
            description: "Draw a directed graph (flowchart, pipeline, architecture) as a \
                          layered box diagram. Send nodes and edges as plain JSON — sugiyama \
                          layout computes positions; use for any diagram/flowchart request."
                .into(),
            input_schema: INPUT_SCHEMA.into(),
        }
    }

    fn execute(arguments_json: String) -> Result<String, String> {
        run(&arguments_json)
    }
}

#[cfg(target_arch = "wasm32")]
export!(Sugi);

/// One positioned node in the render document (character units: x grows
/// right, y grows DOWN, (x, y) is the top-left corner).
#[derive(Clone, serde::Serialize)]
struct OutNode {
    id: String,
    label: String,
    x: f64,
    y: f64,
    w: f64,
    h: f64,
}

#[derive(Clone, serde::Serialize)]
struct OutEdge {
    from: String,
    to: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    label: Option<String>,
}

#[derive(serde::Serialize)]
struct Doc {
    render: &'static str,
    title: String,
    /// Natural size of the laid-out drawing, in character units.
    width: f64,
    height: f64,
    /// Optional cap the requester asked for (the host fits within it).
    #[serde(skip_serializing_if = "Option::is_none")]
    max_width: Option<f64>,
    nodes: Vec<OutNode>,
    edges: Vec<OutEdge>,
}

/// Layout the graph and return the `sugi-v1` render document.
pub fn run(arguments_json: &str) -> Result<String, String> {
    let (nodes, edges) = parse_arguments(arguments_json)?;
    if nodes.is_empty() {
        return Err("the graph is empty — pass at least one node".into());
    }
    let title = serde_json::from_str::<serde_json::Value>(arguments_json)
        .ok()
        .and_then(|v| v.get("title").and_then(|t| t.as_str()).map(String::from))
        .unwrap_or_default();
    let requested_width = serde_json::from_str::<serde_json::Value>(arguments_json)
        .ok()
        .and_then(|v| v.get("width").and_then(|w| w.as_i64()))
        .map(|w| w.clamp(8, 200) as f64);

    // id -> NodeIndex. NodeIndex(i) == insertion order for a graph with
    // no removals, so sizes can be an index-keyed Vec.
    let mut index_of: HashMap<&str, NodeIndex> = HashMap::new();
    let mut graph: StableDiGraph<(), ()> = StableDiGraph::new();
    for node in &nodes {
        index_of.insert(node.id.as_str(), graph.add_node(()));
    }
    let sizes: Vec<(f64, f64)> = nodes.iter().map(|n| vertex_size(&n.label)).collect();
    for edge in &edges {
        let from = *index_of
            .get(edge.from.as_str())
            .ok_or_else(|| format!("edge references unknown node '{}'", edge.from))?;
        let to = *index_of
            .get(edge.to.as_str())
            .ok_or_else(|| format!("edge references unknown node '{}'", edge.to))?;
        graph.add_edge(from, to, ());
    }

    // Tweaked from Config::default(): minimum_length stays 1 (values >1
    // produced empty layers and panics inside rust-sugiyama 0.4), and
    // the vertex spacing is in character units, so 2 — not the default
    // 10 — keeps diagrams compact in chat.
    let config = Config {
        vertex_spacing: 2.0,
        transpose: nodes.len() <= 60,
        ..Config::default()
    };

    // rust-sugiyama returns one layout per connected component, each
    // with its own local coordinates. Normalize orientation (edges
    // point DOWN) and stack components vertically with a gap.
    let layouts = rust_sugiyama::from_graph(&graph, &|idx, _| sizes[idx.index()], &config);

    let mut out_nodes: Vec<OutNode> = Vec::new();
    let mut max_w = 0.0f64;
    let mut y_cursor = 0.0f64;
    for (layout, width, _height) in layouts {
        let pos_of = |target: NodeIndex| {
            layout
                .iter()
                .find(|(n, _)| *n == target)
                .map(|(_, p)| *p)
        };
        // Orientation: the layer-growth direction is an implementation
        // detail — flip so most edges point downward.
        let mut down = 0usize;
        let mut up = 0usize;
        for edge in &edges {
            if let (Some(a), Some(b)) =
                (index_of.get(edge.from.as_str()), index_of.get(edge.to.as_str()))
            {
                if let (Some(pa), Some(pb)) = (pos_of(*a), pos_of(*b)) {
                    if pa.1 < pb.1 {
                        down += 1;
                    } else if pa.1 > pb.1 {
                        up += 1;
                    }
                }
            }
        }
        let flip = up > down;

        let min_x = layout
            .iter()
            .map(|(_, (x, _))| *x)
            .fold(f64::INFINITY, f64::min);
        let min_y = layout
            .iter()
            .map(|(_, (_, y))| *y)
            .fold(f64::INFINITY, f64::min);
        let max_y = layout
            .iter()
            .map(|(_, (_, y))| *y)
            .fold(f64::NEG_INFINITY, f64::max);

        for (idx, (x, y)) in &layout {
            let n = &nodes[idx.index()];
            let (w, h) = sizes[idx.index()];
            let local_y = if flip { max_y - y } else { y - min_y };
            out_nodes.push(OutNode {
                id: n.id.clone(),
                label: n.label.clone(),
                x: x - min_x,
                y: local_y + y_cursor,
                w,
                h,
            });
        }
        max_w = max_w.max(width);
        y_cursor += (max_y - min_y) + 4.0; // component height + gap
    }

    // Fallback the algorithm shouldn't need: isolated inputs with no
    // edges still produced layouts above; if ever empty, row them out.
    if out_nodes.is_empty() {
        let mut x = 0.0;
        for n in &nodes {
            let (w, h) = vertex_size(&n.label);
            out_nodes.push(OutNode {
                id: n.id.clone(),
                label: n.label.clone(),
                x,
                y: 0.0,
                w,
                h,
            });
            x += w + 2.0;
        }
        max_w = x;
    }

    let height = out_nodes
        .iter()
        .map(|n| n.y + n.h)
        .fold(0.0f64, f64::max);
    let doc = Doc {
        render: "sugi-v1",
        title,
        width: max_w,
        height,
        max_width: requested_width,
        nodes: out_nodes,
        edges: edges
            .iter()
            .map(|e| OutEdge {
                from: e.from.clone(),
                to: e.to.clone(),
                label: e.label.clone(),
            })
            .collect(),
    };
    serde_json::to_string(&doc).map_err(|e| format!("serialization failed: {e}"))
}

fn parse_arguments(arguments_json: &str) -> Result<(Vec<InputNode>, Vec<InputEdge>), String> {
    let v: serde_json::Value = serde_json::from_str(arguments_json)
        .map_err(|e| format!("arguments must be a JSON object: {e}"))?;

    let mut nodes: Vec<InputNode> = Vec::new();
    let mut seen: HashMap<String, usize> = HashMap::new();
    let raw_nodes = v
        .get("nodes")
        .and_then(|n| n.as_array())
        .ok_or("missing required 'nodes' array")?;
    for (i, raw) in raw_nodes.iter().enumerate() {
        if nodes.len() >= MAX_NODES {
            return Err(format!("too many nodes (cap {MAX_NODES}) — split the diagram"));
        }
        let (id, label) = match raw {
            serde_json::Value::String(s) => (s.clone(), s.clone()),
            serde_json::Value::Object(_) => {
                let id = raw
                    .get("id")
                    .and_then(|s| s.as_str())
                    .ok_or_else(|| format!("nodes[{i}] needs an 'id' string"))?;
                let label = raw.get("label").and_then(|s| s.as_str()).unwrap_or(id);
                (id.to_string(), label.to_string())
            }
            _ => return Err(format!("nodes[{i}] must be a string or an object with 'id'")),
        };
        if id.is_empty() {
            return Err(format!("nodes[{i}] has an empty id"));
        }
        if seen.insert(id.clone(), i).is_some() {
            return Err(format!("duplicate node id '{id}'"));
        }
        nodes.push(InputNode { id, label });
    }

    let mut edges: Vec<InputEdge> = Vec::new();
    if let Some(raw_edges) = v.get("edges").and_then(|e| e.as_array()) {
        for (i, raw) in raw_edges.iter().enumerate() {
            if edges.len() >= MAX_EDGES {
                return Err(format!("too many edges (cap {MAX_EDGES}) — split the diagram"));
            }
            let (from, to, label) = match raw {
                serde_json::Value::Array(pair) if pair.len() == 2 => {
                    let a = pair[0]
                        .as_str()
                        .ok_or_else(|| format!("edges[{i}] pair entries must be strings"))?;
                    let b = pair[1]
                        .as_str()
                        .ok_or_else(|| format!("edges[{i}] pair entries must be strings"))?;
                    (a.to_string(), b.to_string(), None)
                }
                serde_json::Value::Object(_) => {
                    let from = raw
                        .get("from")
                        .and_then(|s| s.as_str())
                        .ok_or_else(|| format!("edges[{i}] needs a 'from' string"))?;
                    let to = raw
                        .get("to")
                        .and_then(|s| s.as_str())
                        .ok_or_else(|| format!("edges[{i}] needs a 'to' string"))?;
                    let label = raw.get("label").and_then(|s| s.as_str()).map(String::from);
                    (from.to_string(), to.to_string(), label)
                }
                _ => {
                    return Err(format!(
                        "edges[{i}] must be a [from, to] pair or an object with 'from'/'to'"
                    ))
                }
            };
            edges.push(InputEdge { from, to, label });
        }
    }

    Ok((nodes, edges))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc_of(json: &str) -> serde_json::Value {
        serde_json::from_str(&run(json).expect("layout runs")).unwrap()
    }

    fn node<'a>(doc: &'a serde_json::Value, id: &str) -> &'a serde_json::Value {
        doc["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|n| n["id"].as_str() == Some(id))
            .unwrap_or_else(|| panic!("node {id} missing in {doc}"))
    }

    #[test]
    fn chain_flows_downward() {
        let doc = doc_of(
            r#"{"nodes": [{"id": "A", "label": "Intake"}, "B", "C"], "edges": [["A", "B"], ["B", "C"]]}"#,
        );
        assert_eq!(doc["render"], "sugi-v1");
        let ay = node(&doc, "A")["y"].as_f64().unwrap();
        let by = node(&doc, "B")["y"].as_f64().unwrap();
        let cy = node(&doc, "C")["y"].as_f64().unwrap();
        assert!(ay < by && by < cy, "layers must flow down: {ay} {by} {cy}");
        // Labels and sizes come through for the renderer.
        assert_eq!(node(&doc, "A")["label"], "Intake");
        assert!(node(&doc, "A")["w"].as_f64().unwrap() >= 8.0);
        assert_eq!(doc["edges"].as_array().unwrap().len(), 2);
        assert!(doc["width"].as_f64().unwrap() > 0.0);
        assert!(doc["height"].as_f64().unwrap() > 0.0);
    }

    #[test]
    fn object_edges_carry_labels_and_diamond_lays_out() {
        let doc = doc_of(
            r#"{
              "nodes": ["INTAKE", {"id": "SUBA", "label": "Sub Agent A"}, {"id": "SUBB", "label": "Sub Agent B"}, "PLAN", "REPORT"],
              "edges": [
                {"from": "INTAKE", "to": "SUBA"},
                {"from": "INTAKE", "to": "SUBB"},
                {"from": "SUBA", "to": "PLAN", "label": "findings"},
                {"from": "SUBB", "to": "PLAN"},
                {"from": "PLAN", "to": "REPORT"}
              ],
              "title": "agent loop"
            }"#,
        );
        assert_eq!(doc["title"], "agent loop");
        // Fan-out: SUBA and SUBB share one layer (above PLAN).
        let suba_y = node(&doc, "SUBA")["y"].as_f64().unwrap();
        let subb_y = node(&doc, "SUBB")["y"].as_f64().unwrap();
        let plan_y = node(&doc, "PLAN")["y"].as_f64().unwrap();
        assert_eq!(suba_y, subb_y, "siblings share a layer");
        assert!(suba_y < plan_y);
        // Edge label survives to the document.
        let labeled = doc["edges"]
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e.get("label").is_some())
            .expect("labeled edge");
        assert_eq!(labeled["label"], "findings");
    }

    #[test]
    fn disconnected_components_stack_vertically() {
        let doc = doc_of(
            r#"{"nodes": ["A", "B", "X", "Y"], "edges": [["A", "B"], ["X", "Y"]]}"#,
        );
        let a_y = node(&doc, "A")["y"].as_f64().unwrap();
        let x_y = node(&doc, "X")["y"].as_f64().unwrap();
        assert!(a_y != x_y, "components must not overlap: {a_y} vs {x_y}");
    }

    #[test]
    fn single_node_without_edges_renders() {
        let doc = doc_of(r#"{"nodes": ["ONLY"], "edges": []}"#);
        assert_eq!(doc["nodes"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn bad_input_is_a_clean_model_correctable_error() {
        assert!(run(r#"{"nodes": []}"#).unwrap_err().contains("empty"));
        assert!(run(r#"{"nodes": ["A"], "edges": [["A", "GHOST"]]}"#)
            .unwrap_err()
            .contains("GHOST"));
        assert!(run(r#"{"nodes": ["A", "A"]}"#).unwrap_err().contains("duplicate"));
        assert!(run(r#"{"nodes": [{"noid": 1}]}"#).unwrap_err().contains("id"));
        assert!(run("not json").is_err());
    }

    #[test]
    fn requested_width_is_carried_as_a_cap() {
        let doc = doc_of(r#"{"nodes": ["A", "B"], "edges": [["A", "B"]], "width": 60}"#);
        assert_eq!(doc["max_width"], 60.0);
    }
}
