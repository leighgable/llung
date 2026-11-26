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

use crate::config::{AgentConfig, BackendKind};
use crate::interface::Backend;

/// Target size of one summary line (OptChat NODE).
pub const NODE_BYTES: usize = 512;

/// Default ceiling (BYTES) for the history view fed to the agent's
/// model (OptChat VIEW).
pub const DEFAULT_CONTEXT_BUDGET: usize = 32_000;

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
    /// Which model compresses (e.g. "SmolVLM"), for UI attribution;
    /// see [`Self::display_name`].
    label: Option<String>,
    /// Rule-based on purpose ([`Self::rules_only`]): never upgraded to
    /// the agent's own model.
    pinned_rules: bool,
}

impl HistoryCompressor {
    pub fn new(backend: Box<dyn Summarizer>, context_budget: usize) -> Self {
        Self {
            backend,
            context_budget,
            label: None,
            pinned_rules: false,
        }
    }

    /// Rule-based truncation ONLY, deliberately: unlike a bare
    /// [`Self::new`] with [`HeadlineSummarizer`], the agent loop will not
    /// upgrade it to the agent's own model. For expensive models with a
    /// budget large enough that compression rarely triggers.
    pub fn rules_only(context_budget: usize) -> Self {
        let mut c = Self::new(Box::new(HeadlineSummarizer::new()), context_budget);
        c.pinned_rules = true;
        c
    }

    /// Set the view's byte budget (see [`crate::config::ModelProfile`]).
    pub fn with_budget(mut self, context_budget: usize) -> Self {
        self.context_budget = context_budget;
        self
    }

    /// The whole history verbatim — every message in full, `[i]`
    /// numbered like the tree view — when it fits the budget. No tree,
    /// no summarizer calls: with a large budget (frontier models) most
    /// turns never pay for compression. `None` when it doesn't fit.
    pub fn verbatim_view(&self, messages: &[ChatMessage]) -> Option<String> {
        if messages.is_empty() {
            let empty = "(no history)";
            return (empty.len() <= self.context_budget).then(|| empty.into());
        }
        let mut parts = Vec::with_capacity(messages.len());
        let mut total = 0usize;
        for (i, msg) in messages.iter().enumerate() {
            let line = format!("[{i}] {}", render_message(msg));
            total += line.len() + 1;
            if total > self.context_budget + 1 {
                return None;
            }
            parts.push(line);
        }
        Some(parts.join("\n"))
    }

    /// Name the model doing the compression (e.g. "SmolVLM") so the UI
    /// can attribute compression work to it, not to the agent.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Who is working while this compressor runs, for UI spinners:
    /// `HistoryCompressor (SmolVLM)`, or bare `HistoryCompressor`.
    pub fn display_name(&self) -> String {
        match &self.label {
            Some(label) => format!("HistoryCompressor ({label})"),
            None => "HistoryCompressor".into(),
        }
    }

    /// Model-backed compressor with the [`DEFAULT_CONTEXT_BUDGET`]
    /// ceiling. Pick the model with [`uses_smolvlm`]: SmolVLM
    /// (`OpenAiCompatible::smolvlm`) on the local llamacpp backend, the
    /// agent's own backend otherwise. If the model is unreachable,
    /// `ModelSummarizer` falls back to rule-based truncation —
    /// compression never breaks a turn.
    pub fn with_model(model: Box<dyn Backend>) -> Self {
        Self::new(
            Box::new(ModelSummarizer::new(model)),
            DEFAULT_CONTEXT_BUDGET,
        )
    }

    /// Whether a model does the compressing (see [`Summarizer::is_model_backed`]).
    pub fn is_model_backed(&self) -> bool {
        self.backend.is_model_backed()
    }

    /// Adopt a chat backend for compression when the embedder configured
    /// a rule-based summarizer only. No-op when already model-backed.
    pub fn upgrade_to_model(&mut self, backend: Box<dyn Backend>) {
        if !self.is_model_backed() && !self.pinned_rules {
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
        let rendered = render_message(msg);
        let line = flatten_newlines(&rendered);

        if line.len() <= NODE_BYTES {
            let mut node = SummaryNode::leaf(line, (offset, offset + 1), true);
            node.full_text = Some(rendered);
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
            // Tool traffic: rule-based head + tail, never a model call.
            // The head says what ran; the tail is where results and
            // errors are (a status table, a traceback's last line).
            MessageKind::ToolCall { .. } | MessageKind::ToolResponse { .. } => {
                let line = flatten_newlines(&render_message(msg));
                head(&head_tail(&line, NODE_BYTES), NODE_BYTES)
            }
            // NODE is a target the summarizer may miss (a backend that
            // rambles): enforce it so the flattened cut never
            // re-expands a "merged" line. The speaker prefix survives
            // summarization: the model must know who said what.
            _ => {
                let rendered = render_message(msg);
                let summary = self.backend.summarize(&rendered).await;
                let label = speaker_label(msg);
                let summary = if summary.starts_with(&label) {
                    summary
                } else {
                    format!("{label}: {summary}")
                };
                head(&summary, NODE_BYTES)
            }
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

/// True when an agent's history compression should run on SmolVLM
/// (llama-server router mode) instead of its own model: the local
/// llamacpp backend without a `model_url` override.
pub fn uses_smolvlm(config: &AgentConfig) -> bool {
    config.backend == BackendKind::LlamaCpp && config.model_url.is_none()
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
        .map(render_message)
        .collect();
    node.full_text = Some(full.join("\n"));
    node.zoomed = true;
    node.size_bytes = node.full_text.as_ref().map(|t| t.len()).unwrap_or(0);
}

/// Who a history line is from: `luigi`, `sensei`, or `note`.
fn speaker_label(msg: &ChatMessage) -> String {
    match msg.kind {
        MessageKind::System | MessageKind::Notes => "note".into(),
        _ if msg.sender_name.is_empty() => kind_tag(&msg.kind).into(),
        _ => msg.sender_name.clone(),
    }
}

/// One message as the agent sees it in its history view: always
/// attributed, tool traffic in a neutral `name(args)` form.
///
/// Neutral on purpose: history lines act as few-shot examples. A tool
/// call shown in some concrete call syntax teaches that syntax, and an
/// agent reply that was published as text because its tool markup
/// didn't parse would teach the model to repeat the broken markup —
/// observed live (sensenova copied its own malformed `<tool name=…>`
/// three turns in a row). Such replies are replaced by a placeholder.
pub fn render_message(msg: &ChatMessage) -> String {
    let who = speaker_label(msg);
    match &msg.kind {
        MessageKind::ToolCall { tool_name, arguments } => {
            format!("{who} ran tool {tool_name}({arguments})")
        }
        MessageKind::ToolResponse { tool_name, error, .. } => {
            let status = if *error { "FAILED" } else { "returned" };
            format!("tool {tool_name} {status}: {}", msg.content.trim_end())
        }
        MessageKind::Agent if crate::tools::looks_like_tool_markup(&msg.content) => {
            format!("{who}: (a malformed tool call — it was not executed)")
        }
        MessageKind::Media(att) => {
            let caption = msg.content.trim();
            if caption.is_empty() {
                format!("{who} sent image {}", att.filename)
            } else {
                format!("{who} sent image {}: {caption}", att.filename)
            }
        }
        _ => format!("{who}: {}", msg.content),
    }
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

/// Model-backed Summarizer: asks the agent's chat backend (e.g.
/// SenseNova on llama-server) to compress history with a dedicated
/// system prompt.
/// Suspicious output (empty, rambling, prompt-echo) and backend errors
/// fall back to rule-based truncation via `head`.
pub struct ModelSummarizer {
    backend: Box<dyn Backend>,
    /// Target byte size for one summary line.
    budget: usize,
    /// Successful answers by request hash. The tree is rebuilt for every
    /// new message, so the same long message (or the same pair of
    /// lines) is asked about again and again; with a paid compressor
    /// model each repeat costs money and latency. Fallbacks are not
    /// memoized, so a model that was down gets asked again later.
    memo: std::sync::Mutex<std::collections::HashMap<u64, String>>,
}

/// Memo entries kept before the memo is cleared (bounded memory).
const MEMO_CAPACITY: usize = 4_096;

impl ModelSummarizer {
    pub fn new(backend: Box<dyn Backend>) -> Self {
        Self {
            backend,
            budget: NODE_BYTES,
            memo: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// One compression request; None means "fall back to rules".
    async fn ask(&self, system: String, user: String) -> Option<String> {
        use std::hash::{Hash, Hasher};
        let key = {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            system.hash(&mut h);
            user.hash(&mut h);
            h.finish()
        };
        if let Some(hit) = self.memo.lock().ok().and_then(|m| m.get(&key).cloned()) {
            return Some(hit);
        }
        let prompt = vec![ChatMessage::system(system), ChatMessage::user(user)];
        let out = self.backend.chat(&prompt).await.ok()?;
        let out = out.trim().replace('\n', " ");
        if out.is_empty() || out.len() > self.budget * 2 {
            return None; // empty or rambling/echoing — let the caller fall back
        }
        let out = head(&out, self.budget);
        if let Ok(mut memo) = self.memo.lock() {
            if memo.len() >= MEMO_CAPACITY {
                memo.clear();
            }
            memo.insert(key, out.clone());
        }
        Some(out)
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

/// `text` cut to about `budget` bytes keeping BOTH ends: ~40% head,
/// ~60% tail, joined by ` … `. Char-boundary safe.
pub(crate) fn head_tail(text: &str, budget: usize) -> String {
    if text.len() <= budget {
        return text.to_string();
    }
    const JOIN: &str = " … ";
    let room = budget.saturating_sub(JOIN.len());
    let head_bytes = room * 2 / 5;
    let tail_bytes = room - head_bytes;
    let mut head_end = head_bytes.min(text.len());
    while !text.is_char_boundary(head_end) {
        head_end -= 1;
    }
    let mut tail_start = text.len().saturating_sub(tail_bytes);
    while !text.is_char_boundary(tail_start) {
        tail_start += 1;
    }
    format!("{}{JOIN}{}", &text[..head_end], &text[tail_start..])
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
            attachment: None,
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
        assert!(tree.summary.contains("peer: hello there"));
        assert!(tree.summary.contains("peer: hi! how can I help?"));
    }

    /// History lines are attributed, tool traffic is rendered neutrally,
    /// and agent replies that were published tool markup (unparsed
    /// calls) never reappear as examples for the model to copy.
    #[test]
    fn rendering_attributes_and_neutralizes_tool_markup() {
        let mut call = chat_msg(
            MessageKind::ToolCall { tool_name: "glob".into(), arguments: r#"{"pattern":"*.csv"}"#.into() },
            r#"{"pattern":"*.csv"}"#,
        );
        call.sender_name = "sensei".into();
        assert_eq!(render_message(&call), r#"sensei ran tool glob({"pattern":"*.csv"})"#);
        let resp = chat_msg(
            MessageKind::ToolResponse { tool_name: "glob".into(), result_summary: String::new(), error: true },
            "no such dir\n",
        );
        assert_eq!(render_message(&resp), "tool glob FAILED: no such dir");
        let mut bad = chat_msg(
            MessageKind::Agent,
            "<tool_call>\n<tool name=\"bash\" arguments='{}'>\n</tool_call>",
        );
        bad.sender_name = "sensei".into();
        let r = render_message(&bad);
        assert!(r.starts_with("sensei: (a malformed tool call"), "{r}");
        assert!(!r.contains("<tool"), "{r}");
        let mut note = chat_msg(MessageKind::System, "luigi joined");
        note.sender_name = "luigi".into();
        assert_eq!(render_message(&note), "note: luigi joined");
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

    #[test]
    fn verbatim_view_when_it_fits_and_none_when_not() {
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let long = "y".repeat(NODE_BYTES * 3);
        let msgs = vec![chat_msg(MessageKind::Human, &long), chat_msg(MessageKind::Human, "short")];
        let roomy = HistoryCompressor::new(Box::new(FakeBackend::new(calls.clone())), 10_000);
        let view = roomy.verbatim_view(&msgs).expect("fits");
        assert!(view.starts_with("[0] ") && view.contains(&long), "full text, indexed");
        assert!(view.contains("\n[1] "));
        let tight = HistoryCompressor::new(Box::new(FakeBackend::new(calls.clone())), 1_000);
        assert!(tight.verbatim_view(&msgs).is_none());
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 0, "never a model call");
    }

    #[tokio::test]
    async fn long_tool_output_keeps_head_and_tail() {
        let c = HistoryCompressor::new(Box::new(HeadlineSummarizer::new()), 64_000);
        let output = format!("epoch 1 loss 2.31\n{}\nFINAL val_acc=0.97", "noise ".repeat(300));
        let mut m = chat_msg(MessageKind::ToolResponse {
            tool_name: "gpu:torch".into(),
            result_summary: String::new(),
            error: false,
        }, &output);
        m.sender_name = String::new();
        let line = c.summarize_one(&m).await;
        assert!(line.len() <= NODE_BYTES, "{}", line.len());
        assert!(line.contains("gpu:torch"), "what ran: {line}");
        assert!(line.contains("epoch 1"), "head kept: {line}");
        assert!(line.contains("FINAL val_acc=0.97"), "tail kept: {line}");
    }

    #[test]
    fn rules_only_is_never_upgraded() {
        struct Never;
        impl Backend for Never {
            fn chat(
                &self,
                _m: &[ChatMessage],
            ) -> Pin<Box<dyn Future<Output = llung_core::network::message::AgentResult<String>> + Send + '_>> {
                Box::pin(async { Ok(String::new()) })
            }
        }
        let mut pinned = HistoryCompressor::rules_only(1_000);
        pinned.upgrade_to_model(Box::new(Never));
        assert!(!pinned.is_model_backed());
        let mut plain = HistoryCompressor::new(Box::new(HeadlineSummarizer::new()), 1_000);
        plain.upgrade_to_model(Box::new(Never));
        assert!(plain.is_model_backed(), "an unpinned rule-based compressor still upgrades");
    }

    #[tokio::test]
    async fn model_summaries_are_memoized() {
        struct Counting(std::sync::Arc<std::sync::atomic::AtomicUsize>);
        impl Backend for Counting {
            fn chat(
                &self,
                _m: &[ChatMessage],
            ) -> Pin<Box<dyn Future<Output = llung_core::network::message::AgentResult<String>> + Send + '_>> {
                self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Box::pin(async { Ok("a summary".to_string()) })
            }
        }
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let s = ModelSummarizer::new(Box::new(Counting(calls.clone())));
        assert_eq!(s.summarize("some long text").await, "a summary");
        assert_eq!(s.summarize("some long text").await, "a summary");
        assert_eq!(s.merge("l", "r").await, "a summary");
        assert_eq!(s.merge("l", "r").await, "a summary");
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2, "one call per distinct request");
    }

    #[test]
    fn head_tail_is_char_boundary_safe() {
        let text = "é".repeat(400);
        let cut = head_tail(&text, 101);
        assert!(cut.len() <= 101 + 4, "{}", cut.len());
        assert!(cut.contains(" … "));
    }
}

