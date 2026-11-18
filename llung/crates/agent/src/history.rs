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

use std::future::Future;
use std::pin::Pin;

use llung_core::network::message::{ChatMessage, MessageKind};

use crate::interface::Backend;

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
        MessageKind::Media(_) => "media",
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

    /// Whether a model does the compressing (see [`Summarizer::is_model_backed`]).
    pub fn is_model_backed(&self) -> bool {
        self.backend.is_model_backed()
    }

    /// Adopt a chat backend for compression when the embedder configured
    /// a rule-based summarizer only. No-op when already model-backed.
    pub fn upgrade_to_model(&mut self, backend: Box<dyn Backend>) {
        if !self.is_model_backed() {
            self.backend = Box::new(ModelSummarizer::new(backend));
        }
    }

    /// Build a binary summary tree from a flat slice of messages.
    /// Purely binary: every parent covers exactly its two children.
    pub async fn compress(&self, messages: &[ChatMessage]) -> SummaryNode {
        if messages.is_empty() {
            return SummaryNode::leaf("(no history)".into(), (0, 0), true);
        }
        self.compress_range(messages, 0).await
    }

    /// Boxed because it recurses (async fn recursion is not allowed).
    fn compress_range<'a>(
        &'a self,
        messages: &'a [ChatMessage],
        offset: usize,
    ) -> Pin<Box<dyn Future<Output = SummaryNode> + Send + 'a>> {
        Box::pin(async move {
            if messages.len() == 1 {
                return self.leaf_for(&messages[0], offset).await;
            }

            // split in half, recurse
            let mid = messages.len() / 2;
            let left = self.compress_range(&messages[..mid], offset).await;
            let right = self.compress_range(&messages[mid..], offset + mid).await;

            let (summary, verbatim) = self.merge_pair(&left, &right).await;
            let mut node = SummaryNode::leaf(
                summary,
                (left.source_range.0, right.source_range.1),
                verbatim,
            );
            node.left = Some(Box::new(left));
            node.right = Some(Box::new(right));
            node
        })
    }

    /// Level-0 node for one message. Free node when it fits: verbatim,
    /// no model call, full text kept so zoom is free too.
    async fn leaf_for(&self, msg: &ChatMessage, offset: usize) -> SummaryNode {
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

        // No full_text here: the flattened view must show the SUMMARY,
        // or zooming could never reveal anything new. `zoom_node`
        // recomputes full text from the messages when asked.
        SummaryNode::leaf(self.summarize_one(msg).await, (offset, offset + 1), false)
    }

    /// Level-0 summary for one message that doesn't fit verbatim.
    async fn summarize_one(&self, msg: &ChatMessage) -> String {
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
            // NODE is a target the summarizer may miss (a backend that
            // rambles): enforce it so the flattened cut never
            // re-expands a "merged" line.
            _ => head(&self.backend.summarize(&msg.content).await, NODE_BYTES),
        }
    }

    /// Merge two sibling nodes into one line. Free merge when the join
    /// fits in NODE bytes — no model call (OptChat §3 "free nodes").
    async fn merge_pair(&self, left: &SummaryNode, right: &SummaryNode) -> (String, bool) {
        let joined = format!("{}\n{}", left.summary, right.summary);
        if joined.len() <= NODE_BYTES {
            (joined, true)
        } else {
            let merged = self.backend.merge(&left.summary, &right.summary).await;
            (head(&merged, NODE_BYTES), false)
        }
    }

    /// Reduce the tree until its flattened view fits `budget` bytes.
    /// Each call merges the cheapest adjacent sibling pair into its parent
    /// (fewer view lines, coarser summaries). Returns false when nothing
    /// more can be merged.
    pub async fn merge_adjacent(&self, root: &mut SummaryNode) -> bool {
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
        let (summary, verbatim) = self.merge_pair(&left, &right).await;
        cur.summary = summary;
        cur.size_bytes = cur.summary.len();
        cur.source_range = (left.source_range.0, right.source_range.1);
        cur.verbatim = verbatim;
        cur.zoomed = false;
        cur.full_text = None;
        true
    }

    /// Flatten the tree into the view: a budget-aware cut, oldest first.
    /// A node whose line fits NODE_BYTES is emitted whole; a larger
    /// internal node is expanded into its children, down to the leaves.
    /// Zoomed nodes (full_text) always emit their full text. Every line
    /// carries its `[start]` / `[start-end)` source-range prefix — the
    /// addressing the model's `zoom` tool uses to expand a compressed
    /// range back to full text.
    pub fn flatten(&self, root: &SummaryNode) -> String {
        let mut parts = Vec::new();
        self.flatten_cut(root, &mut parts);
        parts.join("\n")
    }

    /// The OptChat VIEW: the coarsest cut of the tree whose total
    /// flattened size fits `budget` BYTES, expanding NEWEST-first —
    /// recent history keeps full detail while older turns stay
    /// summarized, and the whole view grows into the budget instead of
    /// collapsing to a single root summary. Zoomed nodes (full_text) are
    /// never expanded further and always emit in full.
    pub fn flatten_budgeted(&self, root: &SummaryNode, budget: usize) -> String {
        let line_size = |n: &SummaryNode| {
            let prefix = if n.source_range.1 > n.source_range.0 {
                if n.source_range.1 == n.source_range.0 + 1 {
                    6 // "[123] "
                } else {
                    12 // "[123-456] "
                }
            } else {
                0
            };
            prefix
                + n.full_text
                    .as_ref()
                    .map(|t| t.len())
                    .unwrap_or_else(|| n.summary.len())
        };

        // Start from the root; repeatedly expand the NEWEST internal
        // node whose expansion still fits the budget.
        let mut cut: Vec<&SummaryNode> = vec![root];
        loop {
            let total: usize = cut.iter().map(|n| line_size(n)).sum::<usize>()
                + cut.len().saturating_sub(1); // newlines
            let mut best: Option<usize> = None;
            for (i, n) in cut.iter().enumerate() {
                if n.full_text.is_some() || n.left.is_none() {
                    continue; // zoomed or leaf: nothing to expand
                }
                let children: usize = n
                    .left
                    .as_ref()
                    .map(|l| line_size(l))
                    .unwrap_or(0)
                    + n.right
                        .as_ref()
                        .map(|r| line_size(r))
                        .unwrap_or(0)
                    + 1; // extra newline
                let delta = children.saturating_sub(line_size(n));
                let fits = total.saturating_add(delta) <= budget;
                let newer = best.is_none_or(|b| {
                    n.source_range.1 > cut[b].source_range.1
                });
                if fits && newer {
                    best = Some(i);
                }
            }
            let Some(i) = best else { break };
            let n = cut.remove(i);
            if let Some(l) = n.left.as_deref() {
                cut.push(l);
            }
            if let Some(r) = n.right.as_deref() {
                cut.push(r);
            }
        }

        // Emit oldest first.
        cut.sort_by_key(|n| (n.source_range.0, n.source_range.1));
        let mut parts = Vec::with_capacity(cut.len());
        for n in cut {
            self.emit_node(n, &mut parts);
        }
        parts.join("\n")
    }

    fn flatten_cut(&self, node: &SummaryNode, parts: &mut Vec<String>) {
        if node.full_text.is_some() {
            self.emit_node(node, parts);
            return;
        }
        let is_leaf = node.left.is_none() && node.right.is_none();
        if is_leaf || node.size_bytes <= NODE_BYTES {
            self.emit_node(node, parts);
            return;
        }
        // Too big and not zoomed: expand into the children.
        if let Some(ref left) = node.left {
            self.flatten_cut(left, parts);
        }
        if let Some(ref right) = node.right {
            self.flatten_cut(right, parts);
        }
    }

    /// One emitted view line for a node: source-range prefix plus the
    /// zoomed full text or the summary.
    fn emit_node(&self, node: &SummaryNode, parts: &mut Vec<String>) {
        let prefix = if node.source_range.1 > node.source_range.0 {
            if node.source_range.1 == node.source_range.0 + 1 {
                format!("[{}] ", node.source_range.0)
            } else {
                format!("[{}-{}] ", node.source_range.0, node.source_range.1)
            }
        } else {
            String::new()
        };
        if let Some(ref text) = node.full_text {
            parts.push(format!("{prefix}{text}"));
        } else {
            parts.push(format!("{prefix}{}", node.summary));
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

/// Summarizer trait: the model compressing history, or a rule-based
/// fallback. Async because a model-backed summarizer calls the same chat
/// backend as the agent (with a different system prompt); the rule-based
/// fallback just resolves immediately. Output is plain `String` — any
/// failure must be handled inside the implementation by falling back to
/// rule-based truncation, so compression can never break a turn.
pub trait Summarizer: Send + Sync {
    fn summarize<'a>(
        &'a self,
        text: &'a str,
    ) -> Pin<Box<dyn Future<Output = String> + Send + 'a>>;
    fn merge<'a>(
        &'a self,
        left: &'a str,
        right: &'a str,
    ) -> Pin<Box<dyn Future<Output = String> + Send + 'a>>;

    /// True when a model does the compressing (vs. rule-based
    /// truncation). Lets an agent adopt its own chat backend for
    /// compression when the embedder configured none.
    fn is_model_backed(&self) -> bool {
        false
    }
}

/// Model-backed Summarizer: asks the agent's chat backend (e.g. SmolVLM
/// on llama-server) to compress history with a dedicated system prompt.
/// Suspicious output (empty, rambling, prompt-echo) and backend errors
/// fall back to rule-based truncation via `head`.
pub struct ModelSummarizer {
    backend: Box<dyn Backend>,
    /// Target byte size for one summary line.
    budget: usize,
}

impl ModelSummarizer {
    pub fn new(backend: Box<dyn Backend>) -> Self {
        Self {
            backend,
            budget: NODE_BYTES,
        }
    }

    /// One compression request; None means "fall back to rules".
    async fn ask(&self, system: String, user: String) -> Option<String> {
        let prompt = vec![ChatMessage::system(system), ChatMessage::user(user)];
        let out = self.backend.chat(&prompt).await.ok()?;
        let out = out.trim().replace('\n', " ");
        if out.is_empty() || out.len() > self.budget * 2 {
            return None; // empty or rambling/echoing — let the caller fall back
        }
        Some(head(&out, self.budget))
    }
}

impl Summarizer for ModelSummarizer {
    fn is_model_backed(&self) -> bool {
        true
    }

    fn summarize<'a>(
        &'a self,
        text: &'a str,
    ) -> Pin<Box<dyn Future<Output = String> + Send + 'a>> {
        Box::pin(async move {
            let system = format!(
                "You are a history compressor inside a chat agent. Compress the \
                 given text into ONE line of at most {} bytes, keeping who did \
                 what and the key facts. Reply with ONLY that line — no \
                 commentary, no quotes.",
                self.budget
            );
            self.ask(system, text.to_string())
                .await
                .unwrap_or_else(|| head(text, self.budget))
        })
    }

    fn merge<'a>(
        &'a self,
        left: &'a str,
        right: &'a str,
    ) -> Pin<Box<dyn Future<Output = String> + Send + 'a>> {
        Box::pin(async move {
            let system = format!(
                "You are a history compressor inside a chat agent. Merge the two \
                 given history lines into ONE line of at most {} bytes, keeping \
                 the key facts of both. Reply with ONLY the merged line — no \
                 commentary, no quotes.",
                self.budget
            );
            let user = format!("Line 1: {left}\nLine 2: {right}");
            match self.ask(system, user).await {
                Some(merged) => merged,
                None => format!(
                    "{}\n{}",
                    head(left, self.budget / 2),
                    head(right, self.budget - self.budget / 2)
                ),
            }
        })
    }
}

/// Rule-based Summarizer fallback: no model, just keeps the head of the
/// text within the node budget. Only invoked for messages or joins that
/// exceed NODE_BYTES — shorter ones are free nodes and never reach this.
pub struct HeadlineSummarizer;

impl HeadlineSummarizer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for HeadlineSummarizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Summarizer for HeadlineSummarizer {
    fn summarize<'a>(
        &'a self,
        text: &'a str,
    ) -> Pin<Box<dyn Future<Output = String> + Send + 'a>> {
        Box::pin(async move { head(text, NODE_BYTES) })
    }
    fn merge<'a>(
        &'a self,
        left: &'a str,
        right: &'a str,
    ) -> Pin<Box<dyn Future<Output = String> + Send + 'a>> {
        Box::pin(async move {
            // Keep the head of each side, split roughly in half.
            let half = NODE_BYTES / 2;
            format!("{}\n{}", head(left, half), head(right, NODE_BYTES - half))
        })
    }
}

/// First `budget` UTF-8 bytes of `text` (cut on a char boundary), with a
/// marker when anything was dropped.
fn head(text: &str, budget: usize) -> String {
    if text.len() <= budget {
        return text.to_string();
    }
    let cut = text
        .char_indices()
        .take_while(|(i, _)| *i < budget.saturating_sub(1)) // room for the ellipsis
        .last()
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(0);
    format!("{}…", &text[..cut])
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
        fn summarize<'a>(
            &'a self,
            text: &'a str,
        ) -> Pin<Box<dyn Future<Output = String> + Send + 'a>> {
            let calls = self.calls.clone();
            Box::pin(async move {
                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                text.chars().take(NODE_BYTES).collect()
            })
        }
        fn merge<'a>(
            &'a self,
            left: &'a str,
            right: &'a str,
        ) -> Pin<Box<dyn Future<Output = String> + Send + 'a>> {
            let calls = self.calls.clone();
            Box::pin(async move {
                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                format!("{left} + {right}")
            })
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

    #[tokio::test]
    async fn short_messages_are_free_nodes() {
        // OptChat §3: source that fits in NODE bytes IS the node,
        // verbatim, with no model call.
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let c = HistoryCompressor::new(Box::new(FakeBackend::new(calls.clone())), 128_000);

        let messages = vec![
            chat_msg(MessageKind::Human, "hello there"),
            chat_msg(MessageKind::Agent, "hi! how can I help?"),
        ];
        let tree = c.compress(&messages).await;

        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "free nodes must not call the model"
        );
        assert!(tree.verbatim, "two short children join for free");
        assert!(tree.summary.contains("user: hello there"));
        assert!(tree.summary.contains("talk: hi! how can I help?"));
    }

    #[tokio::test]
    async fn long_messages_cost_model_calls() {
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let calls_clone = std::sync::Arc::clone(&calls);
        let c = HistoryCompressor::new(Box::new(FakeBackend::new(calls)), 128_000);

        let long = "x".repeat(NODE_BYTES * 2);
        let messages = vec![chat_msg(MessageKind::Human, &long)];
        let tree = c.compress(&messages).await;

        assert_eq!(
            std::sync::Arc::clone(&calls_clone).load(std::sync::atomic::Ordering::SeqCst),
            1
        );
        assert!(!tree.verbatim);
    }

    #[tokio::test]
    async fn sizes_are_bytes_not_tokens() {
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let c = HistoryCompressor::new(Box::new(FakeBackend::new(calls)), 128_000);
        let messages = vec![chat_msg(
            MessageKind::Human,
            "héllo wörld", // multibyte chars
        )];
        let tree = c.compress(&messages).await;
        assert_eq!(tree.size_bytes, tree.summary.len());
    }

    #[tokio::test]
    async fn budgeted_view_expands_newest_first_and_respects_budget() {
        // Hand-built tree: root over [old-internal (two old leaves), new leaf].
        let old1 = SummaryNode::leaf("old-one-text".repeat(10), (0, 1), true);
        let old2 = SummaryNode::leaf("old-two-text".repeat(10), (1, 2), true);
        let mut left = SummaryNode::leaf("old summary".into(), (0, 2), false);
        left.left = Some(Box::new(old1));
        left.right = Some(Box::new(old2));
        let new = SummaryNode::leaf("NEW-UNIQUE-TEXT".repeat(10), (2, 3), true);
        let mut root = SummaryNode::leaf("root".into(), (0, 3), false);
        root.left = Some(Box::new(left));
        root.right = Some(Box::new(new));

        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let c = HistoryCompressor::new(Box::new(FakeBackend::new(calls)), 128_000);

        // Tight budget: the newest leaf is expanded, the old pair stays
        // summarized — even though expanding the old pair ALONE would fit.
        let tight = c.flatten_budgeted(&root, 200);
        assert!(tight.len() <= 200, "{tight}");
        assert!(tight.contains("NEW-UNIQUE-TEXT"), "{tight}");
        assert!(tight.contains("old summary"), "{tight}");
        assert!(!tight.contains("old-one-text"), "{tight}");

        // Generous budget: everything expands.
        let full = c.flatten_budgeted(&root, 64_000);
        assert!(full.contains("old-one-text"), "{full}");
        assert!(full.contains("NEW-UNIQUE-TEXT"), "{full}");

        // The cut only grows with budget.
        assert!(tight.len() <= full.len());
    }

    #[tokio::test]
    async fn merge_adjacent_reduces_flatten_size() {
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
        let mut tree = c.compress(&messages).await;
        let before = c.flatten(&tree).len();

        assert!(c.merge_adjacent(&mut tree).await);
        let after = c.flatten(&tree).len();
        assert!(
            after < before,
            "merging must shrink the view: {before} -> {after}"
        );

        // Eventually nothing left to merge.
        while c.merge_adjacent(&mut tree).await {}
        assert!(tree.left.is_none() && tree.right.is_none());
    }

    #[tokio::test]
    async fn zoom_range_loads_full_text() {
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut c = HistoryCompressor::new(Box::new(FakeBackend::new(calls)), 128_000);
        let long = "z".repeat(NODE_BYTES * 2);
        let messages = vec![
            chat_msg(MessageKind::Human, &long),
            chat_msg(MessageKind::Human, &long),
        ];
        let mut tree = c.compress(&messages).await;

        zoom_range(&mut tree, &messages, 0, 2);
        let flat = c.flatten(&tree);
        assert!(flat.contains(&long), "zoomed nodes show full text");
    }
}
