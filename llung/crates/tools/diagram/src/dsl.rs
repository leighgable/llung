//! The `diagram` mini-language: a lenient, line-based DSL the model can
//! emit reliably. Three constructs:
//!
//! ```text
//! # comment
//! box OUTER: Optional Label {   # open a container (nests freely)
//!   A: First node               # node: `id: label`
//!   box INNER {                 # label defaults to the id
//!     B: Nested node
//!   }
//! }
//! A -> B: optional edge label   # edges; chains (A -> B -> C) work
//! C                             # bare id = node with that label
//! ```
//!
//! Lenient on purpose: edges auto-create undeclared nodes, duplicate
//! declarations keep the first label, and anything unparseable becomes a
//! node (so output is ALWAYS drawable — a small model's typo degrades
//! the diagram instead of failing the turn).

use std::collections::HashMap;

/// One node in the diagram.
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub id: String,
    pub label: String,
}

/// A container box; children may be nodes or nested boxes.
#[derive(Clone, Debug, PartialEq)]
pub struct BoxNode {
    pub id: String,
    pub label: String,
    pub items: Vec<Item>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Item {
    Node(String),
    Box(BoxNode),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Edge {
    pub from: String,
    pub to: String,
    pub label: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Graph {
    /// Top-level items in declaration order.
    pub items: Vec<Item>,
    /// All nodes by id (declaration order preserved by the caller).
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

impl Graph {
    pub fn node(&self, id: &str) -> Option<&Node> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// Every box, flattened, with the path of ancestor box ids (outermost
    /// first) — ordering and drawing both need this.
    pub fn boxes_with_paths(&self) -> Vec<(Vec<String>, &BoxNode)> {
        fn walk<'a>(
            items: &'a [Item],
            path: &mut Vec<String>,
            out: &mut Vec<(Vec<String>, &'a BoxNode)>,
        ) {
            for item in items {
                if let Item::Box(b) = item {
                    path.push(b.id.clone());
                    out.push((path.clone(), b));
                    walk(&b.items, path, out);
                    path.pop();
                }
            }
        }
        let mut out = Vec::new();
        walk(&self.items, &mut Vec::new(), &mut out);
        out
    }

    /// Ids of all nodes inside a box (recursively).
    pub fn box_members(&self, path: &[String]) -> Vec<String> {
        let mut out = Vec::new();
        let mut items = Some(&self.items);
        for segment in path {
            let found = items.and_then(|items| {
                items.iter().find_map(|i| match i {
                    Item::Box(b) if &b.id == segment => Some(&b.items),
                    _ => None,
                })
            });
            items = found;
        }
        fn collect(items: &[Item], out: &mut Vec<String>) {
            for item in items {
                match item {
                    Item::Node(id) => out.push(id.clone()),
                    Item::Box(b) => collect(&b.items, out),
                }
            }
        }
        if let Some(items) = items {
            collect(items, &mut out);
        }
        out
    }
}

pub fn parse(source: &str) -> Graph {
    let mut g = Graph::default();
    // Stack of open box item-lists. Each entry: (id, label, items).
    let mut scope: Vec<(String, String, Vec<Item>)> = Vec::new();
    let mut top_items: Vec<Item> = Vec::new();
    // Ids auto-created by edges: a real declaration later REPLACES the
    // stub label; among real declarations the first wins.
    let mut auto: Vec<String> = Vec::new();

    fn declare_node(g: &mut Graph, auto: &mut Vec<String>, id: &str, label: &str, from_edge: bool) {
        match g.nodes.iter_mut().find(|n| n.id == id) {
            None => {
                g.nodes.push(Node {
                    id: id.to_string(),
                    label: label.to_string(),
                });
                if from_edge {
                    auto.push(id.to_string());
                }
            }
            Some(n) => {
                if from_edge {
                    // Never let an edge stub shadow a real declaration.
                } else if auto.iter().any(|a| a == id) {
                    n.label = label.to_string();
                    auto.retain(|a| a != id);
                }
            }
        }
    }

    for raw in source.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if line == "}" {
            if let Some((id, label, items)) = scope.pop() {
                let b = Item::Box(BoxNode { id, label, items });
                match scope.last_mut() {
                    Some((_, _, items)) => items.push(b),
                    None => top_items.push(b),
                }
            }
            continue;
        }

        // `box ID[: label] {`
        if let Some(rest) = line.strip_prefix("box ") {
            let rest = rest.trim_end_matches('{').trim();
            let (id, label) = split_id_label(rest);
            if !id.is_empty() {
                scope.push((id.to_string(), label.to_string(), Vec::new()));
            }
            continue;
        }

        // Edge: `A -> B (-> C)*[: label]` — an edge line contains `->`.
        if line.contains("->") {
            let (chain_part, label) = match line.split_once(':') {
                Some((c, l)) => (c.trim(), Some(l.trim().to_string())),
                None => (line, None),
            };
            let ids: Vec<&str> = chain_part.split("->").map(str::trim).collect();
            let mut prev: Option<String> = None;
            for id in ids.iter().filter(|s| !s.is_empty()) {
                let id = id.to_string();
                if let Some(p) = prev.take() {
                    g.edges.push(Edge {
                        from: p,
                        to: id.clone(),
                        label: label.clone(),
                    });
                }
                declare_node(&mut g, &mut auto, &id, &id, true);
                prev = Some(id);
            }
            continue;
        }

        // `ID: label` or bare `ID`.
        let (id, label) = split_id_label(line);
        if !id.is_empty() {
            declare_node(&mut g, &mut auto, &id, &label, false);
            let item = Item::Node(id.to_string());
            match scope.last_mut() {
                Some((_, _, items)) => items.push(item),
                None => top_items.push(item),
            }
        }
    }

    // Unclosed boxes: close them all (lenient).
    while let Some((id, label, items)) = scope.pop() {
        let b = Item::Box(BoxNode { id, label, items });
        match scope.last_mut() {
            Some((_, _, items)) => items.push(b),
            None => top_items.push(b),
        }
    }

    g.items = top_items;

    // Edges may name a box: `A -> GROUP` means "into the group's entry
    // node", `GROUP -> B` means "from the group's exit node". Retarget
    // to the first/last member so layout sees a plain node graph.
    let box_members: HashMap<String, Vec<String>> = g
        .boxes_with_paths()
        .into_iter()
        .map(|(p, b)| (b.id.clone(), g.box_members(&p)))
        .collect();
    for e in &mut g.edges {
        if let Some(m) = box_members.get(&e.from) {
            if let Some(last) = m.last() {
                e.from = last.clone();
            }
        }
        if let Some(m) = box_members.get(&e.to) {
            if let Some(first) = m.first() {
                e.to = first.clone();
            }
        }
    }

    // Drop phantom stubs: a node named exactly like a box whose edges
    // were retargeted away (e.g. `A -> GROUP` auto-created GROUP before
    // the box was known). It has no edges and no item — remove it.
    let box_ids: std::collections::HashSet<&str> = box_members.keys().map(String::as_str).collect();
    let referenced: std::collections::HashSet<&str> = g
        .edges
        .iter()
        .flat_map(|e| [&e.from, &e.to])
        .map(String::as_str)
        .collect();
    let itemized: std::collections::HashSet<&str> = g
        .items
        .iter()
        .filter_map(|i| match i {
            Item::Node(id) => Some(id.as_str()),
            _ => None,
        })
        .collect();
    g.nodes
        .retain(|n| !(box_ids.contains(n.id.as_str()) && !referenced.contains(n.id.as_str()) && !itemized.contains(n.id.as_str())));

    g
}

/// Split `ID: label` / `ID :label` / bare `ID`. The id is the first
/// run of non-`:` non-`{`/`}` non-whitespace characters.
fn split_id_label(s: &str) -> (&str, String) {
    let id_end = s
        .find(|c: char| c == ':' || c == '{' || c == '}' || c.is_whitespace())
        .unwrap_or(s.len());
    let id = &s[..id_end];
    let label = s[id_end..]
        .trim_start_matches(':')
        .trim()
        .to_string();
    let label = if label.is_empty() { id.to_string() } else { label };
    (id, label)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nodes_edges_and_labels() {
        let g = parse(
            "# comment\nSTART: Kickoff\nMID: Middle\nA -> B -> C: then\nC\n",
        );
        assert_eq!(g.nodes.len(), 5);
        assert_eq!(g.node("START").unwrap().label, "Kickoff");
        assert_eq!(g.edges.len(), 2);
        assert_eq!(g.edges[0].from, "A");
        assert_eq!(g.edges[0].to, "B");
        assert_eq!(g.edges[1].to, "C");
        assert_eq!(g.edges[0].label.as_deref(), Some("then"));
        // Bare id = node with itself as label.
        assert_eq!(g.node("C").unwrap().label, "C");
        // Edge-only ids are auto-created.
        assert!(g.node("A").is_some());
    }

    #[test]
    fn nested_boxes_and_paths() {
        let g = parse(
            "box HARNESS: Harness {\n  box CONTEXT: Context {\n    PROMPT: Prompt\n    MODEL: Model\n  }\n}\nPROMPT -> MODEL\n",
        );
        let boxes = g.boxes_with_paths();
        assert_eq!(boxes.len(), 2);
        assert_eq!(boxes[0].0, vec!["HARNESS".to_string()]);
        assert_eq!(boxes[1].0, vec!["HARNESS".to_string(), "CONTEXT".to_string()]);
        let members = g.box_members(&boxes[1].0);
        assert_eq!(members, vec!["PROMPT".to_string(), "MODEL".to_string()]);
    }

    #[test]
    fn lenient_unclosed_box_still_draws() {
        let g = parse("box A {\n  B: bee\n");
        assert_eq!(g.boxes_with_paths().len(), 1);
        assert!(g.node("B").is_some());
    }

    #[test]
    fn duplicate_declarations_keep_first_label() {
        let g = parse("A: first\nA: second\n");
        assert_eq!(g.node("A").unwrap().label, "first");
    }

    #[test]
    fn edges_to_a_box_retarget_to_entry_and_exit_nodes() {
        let g = parse(
            "box G: Group {\n  A: first inner\n  B: last inner\n}\nSTART: Start\nEND: End\nSTART -> G\nG -> END\n",
        );
        assert_eq!(g.edges[0].to, "A", "into the box = its first member");
        assert_eq!(g.edges[1].from, "B", "out of the box = its last member");
    }
}
