// crates/agent/src/history.rs
//
// OptChat's "tree": a purely binary summary tree where every parent is the
// merge of exactly its two children. See the gist §3.
//
// Alignment notes (the gist is explicit about these):
//   * All sizes are UTF-8 BYTES, never tokens — a tokenizer changes
//     between models, a byte count never does.
//   * NODE is a TARGET (512 bytes), not a bound anything relies on,
//     because the view measures real sizes.
//   * FREE NODES: if the source already fits in NODE bytes, it IS the
//     node — verbatim, with no model call. Short messages stay word for
//     word forever; two short children merge for free if their join fits.
//
// Not yet implemented from the gist (the real follow-up work):
//   * The background compactor with pump()/first() ordering and JOBS
//     concurrency — we rebuild the tree synchronously per turn.
//   * The TRIES retry loop with cut-at-limit feedback for oversized lines.
//   * The view's cache breakpoints (MARKS) — only matters with a
//     prompt-caching provider.

use llung_core::network::message::{ChatMessage, MessageKind};

/// Target size of one summary line (OptChat NODE).
pub const NODE_BYTES: usize = 512;

/// OptChat message kinds, mapped onto our MessageKind: `user: text`.
fn kind_tag(kind: &MessageKind) -> &'static str {
    match kind {
        MessageKind::Human => "user",
        MessageKind::Agent => "talk",
        MessageKind::ToolCall { .. } => "tool",
        MessageKind::ToolResponse { .. } => "echo",
        // System / Notes carry imported memories and meta announcements.
        MessageKind::System | MessageKind::Notes => "note",
    }
}

/// A node in the binary summary tree.
#[derive(Clone)]
pub struct SummaryNode {
    /// One-line summary (or verbatim source — see `verbatim`).
    pub summary: String,
    /// UTF-8 byte size of `summary` — the budget accounting unit.
    pub size_bytes: usize,
    /// Pointer back to the original message(s).
    pub source_range: (usize, usize), // start..end indices in flat history
    /// Left/right children in the binary tree.
    pub left: Option<Box<SummaryNode>>,
    pub right: Option<Box<SummaryNode>>,
    /// True when `summary` is the source text itself (a free node:
    /// no model call was spent on it).
    pub verbatim: bool,
    /// If true, the agent has "zoomed" and loaded full text.
    pub zoomed: bool,
    /// The full messages (only populated after zoom).
    pub full_text: Option<String>,
}

impl SummaryNode {
    fn leaf(summary: String, source_range: (usize, usize), verbatim: bool) -> Self {
        let size_bytes = summary.len();
        Self {
            summary,
            size_bytes,
            source_range,
            left: None,
            right: None,
            verbatim,
            zoomed: false,
            full_text: None,
        }
    }
}

pub struct HistoryCompressor {
    /// The cheap model used for summarization (could be a tiny ONNX model).
    backend: Box<dyn Summarizer>,
    /// Max BYTES to feed into the expensive model (OptChat VIEW).
    pub context_budget: usize,
}

impl HistoryCompressor {
    pub fn new(backend: Box<dyn Summarizer>, context_budget: usize) -> Self {
        Self {
            backend,
            context_budget,
        }
    }

    /// Build a binary summary tree from a flat slice of messages.
    /// Purely binary: every parent covers exactly its two children.
    pub fn compress(&self, messages: &[ChatMessage]) -> SummaryNode {
        if messages.is_empty() {
            return SummaryNode::leaf("(no history)".into(), (0, 0), true);
        }
        self.compress_range(messages, 0)
    }

    fn compress_range(&self, messages: &[ChatMessage], offset: usize) -> SummaryNode {
        if messages.len() == 1 {
            return self.leaf_for(&messages[0], offset);
        }

        // split in half, recurse
        let mid = messages.len() / 2;
        let left = self.compress_range(&messages[..mid], offset);
        let right = self.compress_range(&messages[mid..], offset + mid);

        let (summary, verbatim) = self.merge_pair(&left, &right);
        let mut node = SummaryNode::leaf(
            summary,
            (left.source_range.0, right.source_range.1),
            verbatim,
        );
        node.left = Some(Box::new(left));
        node.right = Some(Box::new(right));
        node
    }

    /// Level-0 node for one message. Free node when it fits: verbatim,
    /// no model call, full text kept so zoom is free too.
    fn leaf_for(&self, msg: &ChatMessage, offset: usize) -> SummaryNode {
        let line = format!(
            "{}: {}",
            kind_tag(&msg.kind),
            flatten_newlines(&msg.content)
        );

        if line.len() <= NODE_BYTES {
            let mut node = SummaryNode::leaf(line, (offset, offset + 1), true);
            node.full_text = Some(msg.content.clone());
            return node;
        }

        let summary = self.summarize_one(msg);
        let mut node = SummaryNode::leaf(summary, (offset, offset + 1), false);
        node.full_text = Some(msg.content.clone());
        node
    }

    /// Level-0 summary for one message that doesn't fit verbatim.
    fn summarize_one(&self, msg: &ChatMessage) -> String {
        match msg.kind {
            MessageKind::ToolCall {
                ref tool_name,
                ref arguments,
            } => format!(
                "Called tool '{}' with args: {}",
                tool_name,
                arguments.chars().take(60).collect::<String>()
            ),
            MessageKind::ToolResponse { ref tool_name, .. } => {
                format!("Tool '{}' returned result.", tool_name)
            }
            _ => self.backend.summarize(&msg.content),
        }
    }

    /// Merge two sibling nodes into one line. Free merge when the join
    /// fits in NODE bytes — no model call (OptChat §3 "free nodes").
    fn merge_pair(&self, left: &SummaryNode, right: &SummaryNode) -> (String, bool) {
        let joined = format!("{}\n{}", left.summary, right.summary);
        if joined.len() <= NODE_BYTES {
            (joined, true)
        } else {
            (self.backend.merge(&left.summary, &right.summary), false)
        }
    }

    /// Reduce the tree until its flattened view fits `budget` bytes.
    /// Each call merges the cheapest adjacent sibling pair into its parent
    /// (fewer view lines, coarser summaries). Returns false when nothing
    /// more can be merged.
    pub fn merge_adjacent(&self, root: &mut SummaryNode) -> bool {
        // Locate the parent of the mergeable pair with the smallest
        // combined size; `path` records the right/left turns taken.
        fn find<'a>(
            node: &'a SummaryNode,
            best: &mut Option<(usize, Vec<bool>)>,
            path: &mut Vec<bool>,
        ) {
            if let (Some(l), Some(r)) = (&node.left, &node.right) {
                if l.left.is_none() && l.right.is_none() && r.left.is_none() && r.right.is_none() {
                    let cost = l.size_bytes + r.size_bytes;
                    if best.as_ref().is_none_or(|(c, _)| cost < *c) {
                        *best = Some((cost, path.clone()));
                    }
                }
                path.push(false);
                find(l, best, path);
                path.pop();
                path.push(true);
                find(r, best, path);
                path.pop();
            }
        }

        let mut best = None;
        find(root, &mut best, &mut Vec::new());
        let Some((_, path)) = best else {
            return false;
        };

        let mut cur = root;
        for go_right in path {
            cur = if go_right {
                cur.right.as_mut().expect("path is valid")
            } else {
                cur.left.as_mut().expect("path is valid")
            };
        }

        let left = cur.left.take().expect("found pair");
        let right = cur.right.take().expect("found pair");
        let (summary, verbatim) = self.merge_pair(&left, &right);
        cur.summary = summary;
        cur.size_bytes = cur.summary.len();
        cur.source_range = (left.source_range.0, right.source_range.1);
        cur.verbatim = verbatim;
        cur.zoomed = false;
        cur.full_text = None;
        true
    }

    /// Flatten the tree into the view: node lines, oldest first.
    /// Zoomed nodes contribute their full text instead of a summary.
    pub fn flatten(&self, root: &SummaryNode) -> String {
        let mut parts = Vec::new();
        self.flatten_inner(root, &mut parts);
        parts.join("\n")
    }

    fn flatten_inner(&self, node: &SummaryNode, parts: &mut Vec<String>) {
        if let Some(ref text) = node.full_text {
            parts.push(text.clone());
            return;
        }
        if let Some(ref left) = node.left {
            self.flatten_inner(left, parts);
        }
        parts.push(node.summary.clone());
        if let Some(ref right) = node.right {
            self.flatten_inner(right, parts);
        }
    }

    /// "zoom" into a single summary node: replace summary with full text
    pub fn zoom(&mut self, node: &mut SummaryNode, messages: &[ChatMessage]) {
        zoom_node(node, messages);
    }
}

/// Zoom every node fully inside `[start, end)` — the agent-side of
/// OptChat's `zoom(id, n)` tool, range-addressed instead of id+n.
pub fn zoom_range(node: &mut SummaryNode, messages: &[ChatMessage], start: usize, end: usize) {
    if node.source_range.0 >= start && node.source_range.1 <= end {
        zoom_node(node, messages);
        return;
    }
    if let Some(left) = &mut node.left {
        zoom_range(left, messages, start, end);
    }
    if let Some(right) = &mut node.right {
        zoom_range(right, messages, start, end);
    }
}

fn zoom_node(node: &mut SummaryNode, messages: &[ChatMessage]) {
    if node.zoomed {
        return;
    }
    let (start, end) = node.source_range;
    let full: Vec<String> = messages
        .get(start..end)
        .unwrap_or(&[])
        .iter()
        .map(|m| format!("[{}]: {}", m.sender_name, m.content))
        .collect();
    node.full_text = Some(full.join("\n"));
    node.zoomed = true;
    node.size_bytes = node.full_text.as_ref().map(|t| t.len()).unwrap_or(0);
}

fn flatten_newlines(text: &str) -> String {
    text.replace('\n', " ")
}

/// Summarizer trait: a cheap model or a rule-based fallback.
pub trait Summarizer: Send + Sync {
    fn summarize(&self, text: &str) -> String;
    fn merge(&self, left: &str, right: &str) -> String;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rule-based backend: passthrough "summaries" that we can count
    /// calls on (via the shared counter, since the backend is boxed).
    struct FakeBackend {
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }
    impl FakeBackend {
        fn new(calls: std::sync::Arc<std::sync::atomic::AtomicUsize>) -> Self {
            Self { calls }
        }
    }
    impl Summarizer for FakeBackend {
        fn summarize(&self, text: &str) -> String {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            text.chars().take(NODE_BYTES).collect()
        }
        fn merge(&self, left: &str, right: &str) -> String {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            format!("{left} + {right}")
        }
    }

    fn chat_msg(kind: MessageKind, content: &str) -> ChatMessage {
        ChatMessage {
            id: String::new(),
            topic: "t".into(),
            sender_id: "p".into(),
            sender_name: "peer".into(),
            parent_id: None,
            kind,
            content: content.into(),
            timestamp: 0,
        }
    }

    #[test]
    fn short_messages_are_free_nodes() {
        // OptChat §3: source that fits in NODE bytes IS the node,
        // verbatim, with no model call.
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let c = HistoryCompressor::new(Box::new(FakeBackend::new(calls.clone())), 128_000);

        let messages = vec![
            chat_msg(MessageKind::Human, "hello there"),
            chat_msg(MessageKind::Agent, "hi! how can I help?"),
        ];
        let tree = c.compress(&messages);

        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "free nodes must not call the model"
        );
        assert!(tree.verbatim, "two short children join for free");
        assert!(tree.summary.contains("user: hello there"));
        assert!(tree.summary.contains("talk: hi! how can I help?"));
    }

    #[test]
    fn long_messages_cost_model_calls() {
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let calls_clone = std::sync::Arc::clone(&calls);
        let c = HistoryCompressor::new(Box::new(FakeBackend::new(calls)), 128_000);

        let long = "x".repeat(NODE_BYTES * 2);
        let messages = vec![chat_msg(MessageKind::Human, &long)];
        let tree = c.compress(&messages);

        assert_eq!(
            std::sync::Arc::clone(&calls_clone).load(std::sync::atomic::Ordering::SeqCst),
            1
        );
        assert!(!tree.verbatim);
    }

    #[test]
    fn sizes_are_bytes_not_tokens() {
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let c = HistoryCompressor::new(Box::new(FakeBackend::new(calls)), 128_000);
        let messages = vec![chat_msg(
            MessageKind::Human,
            "héllo wörld", // multibyte chars
        )];
        let tree = c.compress(&messages);
        assert_eq!(tree.size_bytes, tree.summary.len());
    }

    #[test]
    fn merge_adjacent_reduces_flatten_size() {
        let c = HistoryCompressor::new(
            Box::new(FakeBackend::new(std::sync::Arc::new(
                std::sync::atomic::AtomicUsize::new(0),
            ))),
            128_000,
        );

        let long = "y".repeat(NODE_BYTES * 2);
        let messages = vec![
            chat_msg(MessageKind::Human, &long),
            chat_msg(MessageKind::Human, &long),
            chat_msg(MessageKind::Human, &long),
            chat_msg(MessageKind::Human, &long),
        ];
        let mut tree = c.compress(&messages);
        let before = c.flatten(&tree).len();

        assert!(c.merge_adjacent(&mut tree));
        let after = c.flatten(&tree).len();
        assert!(
            after < before,
            "merging must shrink the view: {before} -> {after}"
        );

        // Eventually nothing left to merge.
        while c.merge_adjacent(&mut tree) {}
        assert!(tree.left.is_none() && tree.right.is_none());
    }

    #[test]
    fn zoom_range_loads_full_text() {
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut c = HistoryCompressor::new(Box::new(FakeBackend::new(calls)), 128_000);
        let long = "z".repeat(NODE_BYTES * 2);
        let messages = vec![
            chat_msg(MessageKind::Human, &long),
            chat_msg(MessageKind::Human, &long),
        ];
        let mut tree = c.compress(&messages);

        zoom_range(&mut tree, &messages, 0, 2);
        let flat = c.flatten(&tree);
        assert!(flat.contains(&long), "zoomed nodes show full text");
    }
}
