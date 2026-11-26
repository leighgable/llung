//! Anthropic backend: Claude over the native Messages API
//! (`POST {base}/messages`, SSE streaming).
//!
//! Native rather than Anthropic's OpenAI-compatibility layer: the
//! Messages API is the supported surface (extended thinking, image
//! blocks, precise stop reasons). Shape differences from
//! [`super::OpenAiCompatible`], all handled in [`convert_messages`]:
//! - `system` is a top-level field, not a message;
//! - only `user` / `assistant` roles, and the conversation must open
//!   on a `user` turn — the harness's back-to-back user messages
//!   (context, trigger, feedback) are merged into one turn of several
//!   text blocks;
//! - images are `image` blocks with base64 `source`, user turns only.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use tokio::sync::mpsc;

use llung_core::network::message::{AgentResult, AttachmentKind, ChatMessage};

use super::{
    Backend, MediaLookupFn, PermanentError, StreamChunk, check, http_client, model_idle_timeout,
    stall_aware,
};
use crate::config::{AgentConfig, ThinkingLevel};

/// Model id when a spawn line names none.
pub const ANTHROPIC_DEFAULT_MODEL: &str = "claude-sonnet-4-5";

/// History compression for Anthropic agents: the small, cheap model
/// (the agent's own model would bill Sonnet/Opus rates for summaries).
/// Override with `LLUNG_ANTHROPIC_COMPRESSOR_MODEL`.
pub const ANTHROPIC_COMPRESSOR_MODEL: &str = "claude-haiku-4-5";

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";
const API_VERSION: &str = "2023-06-01";

/// `max_tokens` is REQUIRED by the Messages API and covers thinking +
/// answer. 32k fits every current Claude model's output ceiling.
/// Override with `LLUNG_ANTHROPIC_MAX_TOKENS`.
const DEFAULT_MAX_TOKENS: u64 = 32_000;
/// Tokens always left for the answer after the thinking budget.
const MIN_ANSWER_TOKENS: u64 = 4_096;
/// The API's floor for `thinking.budget_tokens`.
const MIN_THINKING_BUDGET: u64 = 1_024;

/// Image media types the Messages API accepts.
const IMAGE_TYPES: &[&str] = &["image/jpeg", "image/png", "image/gif", "image/webp"];

/// Claude via the Messages API. Cheap to clone (strings, an `Arc`, a
/// shared connection pool). Deliberately NOT `Debug`: it holds the API
/// key, which must never reach a log.
#[derive(Clone)]
pub struct Anthropic {
    base_url: String,
    model: String,
    /// Sent as `x-api-key`. In memory only — loaded from the
    /// keyring/env at spawn, never stored in `AgentConfig`.
    api_key: Option<String>,
    thinking: Option<ThinkingLevel>,
    max_tokens: u64,
    /// Mark the end of each request as a cache breakpoint (see
    /// [`Anthropic::request_body`]). Off for one-shot callers (the
    /// history compressor), where a cache write is pure overhead.
    prompt_caching: bool,
    media_lookup: Option<Arc<MediaLookupFn>>,
    client: reqwest::Client,
    idle_timeout: std::time::Duration,
}

fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|s| !s.is_empty())
}

impl Anthropic {
    /// Claude `model` on the default endpoint (`LLUNG_ANTHROPIC_BASE_URL`
    /// overrides it, e.g. for a proxy).
    pub fn new(model: impl Into<String>) -> Self {
        let base = env_nonempty("LLUNG_ANTHROPIC_BASE_URL").unwrap_or_else(|| DEFAULT_BASE_URL.into());
        Self::with_idle_timeout(base, model, model_idle_timeout())
    }

    /// [`Self::new`] with an explicit endpoint and stall limit (tests).
    pub(crate) fn with_idle_timeout(
        base_url: impl Into<String>,
        model: impl Into<String>,
        idle_timeout: std::time::Duration,
    ) -> Self {
        let max_tokens = env_nonempty("LLUNG_ANTHROPIC_MAX_TOKENS")
            .and_then(|s| s.parse::<u64>().ok())
            .filter(|n| *n > 0)
            .unwrap_or(DEFAULT_MAX_TOKENS);
        Self {
            base_url: base_url.into(),
            model: model.into(),
            api_key: None,
            thinking: None,
            max_tokens,
            prompt_caching: true,
            media_lookup: None,
            client: http_client(idle_timeout),
            idle_timeout,
        }
    }

    /// From an agent config: an explicit `model_url` replaces the base
    /// URL. The key is NOT resolved here (see `OpenAiCompatible::kimi`):
    /// the caller attaches it with [`Self::with_api_key`].
    pub fn from_config(config: &AgentConfig) -> Self {
        let mut backend = Self::new(config.model.clone());
        if let Some(url) = &config.model_url {
            backend.base_url = url.clone();
        }
        backend.thinking = config.thinking;
        backend
    }

    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    pub fn with_thinking(mut self, level: ThinkingLevel) -> Self {
        self.thinking = Some(level);
        self
    }

    /// Multimodal input, as `OpenAiCompatible::with_media_lookup`.
    pub fn with_media_lookup(
        mut self,
        lookup: impl Fn(&str) -> Option<Vec<u8>> + Send + Sync + 'static,
    ) -> Self {
        self.media_lookup = Some(Arc::new(lookup));
        self
    }

    /// The same endpoint and key on the cheap model, without thinking:
    /// the history compressor for Anthropic agents.
    pub fn compressor(&self) -> Self {
        let mut c = self.clone();
        c.model = env_nonempty("LLUNG_ANTHROPIC_COMPRESSOR_MODEL")
            .unwrap_or_else(|| ANTHROPIC_COMPRESSOR_MODEL.into());
        c.thinking = None;
        // Every compressor call has a different prompt: nothing to reuse.
        c.prompt_caching = false;
        c
    }

    /// Toggle prompt caching (on by default for agent turns).
    pub fn with_prompt_caching(mut self, on: bool) -> Self {
        self.prompt_caching = on;
        self
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    fn url(&self, path: &str) -> String {
        format!("{}/{path}", self.base_url.trim_end_matches('/'))
    }

    fn authed(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let req = req.header("anthropic-version", API_VERSION);
        match &self.api_key {
            Some(key) => req.header("x-api-key", key),
            None => req,
        }
    }

    /// `GET {base}/models`: proves the endpoint is up AND the key is
    /// accepted (a bad key is a 401 here, not on the first turn).
    pub async fn probe(&self) -> Result<(), String> {
        let req = self
            .authed(self.client.get(self.url("models")))
            .timeout(std::time::Duration::from_secs(5));
        check(req.send().await, std::time::Duration::from_secs(5))
            .await
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    /// The Messages API request body (always streaming: [`Backend::chat`]
    /// is built on the stream too, so long generations never sit
    /// silently behind one big response body).
    fn request_body(&self, messages: &[ChatMessage]) -> serde_json::Value {
        let (system, mut turns) = convert_messages(self.media_lookup.as_deref(), messages);
        // Prompt caching: a breakpoint on the LAST block caches the whole
        // prompt. The harness re-asks with the same prefix (system,
        // history, trigger + tool instructions) on every tool step of a
        // turn, only appending the latest tool result — the next request
        // finds this prefix and pays ~10% for it. Prompts under the
        // model's cache minimum are simply not cached (no error).
        if let Some(block) = turns
            .last_mut()
            .filter(|_| self.prompt_caching)
            .and_then(|t| t["content"].as_array_mut())
            .and_then(|blocks| blocks.last_mut())
        {
            block["cache_control"] = serde_json::json!({"type": "ephemeral"});
        }
        let mut max_tokens = self.max_tokens;
        let mut body = serde_json::json!({
            "model": self.model,
            "messages": turns,
            "stream": true,
        });
        if !system.is_empty() {
            body["system"] = system.into();
        }
        if let Some(level) = self.thinking {
            max_tokens = max_tokens.max(MIN_THINKING_BUDGET + MIN_ANSWER_TOKENS);
            body["thinking"] = serde_json::json!({
                "type": "enabled",
                "budget_tokens": thinking_budget(level, max_tokens),
            });
        }
        body["max_tokens"] = max_tokens.into();
        body
    }
}

/// The thinking budget for a level, clamped so `budget < max_tokens`
/// with [`MIN_ANSWER_TOKENS`] to spare (the API rejects budgets at or
/// above `max_tokens`) and never under the API floor.
fn thinking_budget(level: ThinkingLevel, max_tokens: u64) -> u64 {
    level
        .budget_tokens()
        .min(max_tokens.saturating_sub(MIN_ANSWER_TOKENS))
        .max(MIN_THINKING_BUDGET)
}

/// Split llung's prompt into the Messages API's `system` string and
/// alternating `user`/`assistant` turns (see the module docs).
fn convert_messages(
    lookup: Option<&MediaLookupFn>,
    messages: &[ChatMessage],
) -> (String, Vec<serde_json::Value>) {
    use base64::Engine as _;
    let mut system: Vec<&str> = Vec::new();
    let mut turns: Vec<(&'static str, Vec<serde_json::Value>)> = Vec::new();
    for m in messages {
        let (role, mut text) = match m.agent_role() {
            // Leading system messages are THE system prompt.
            "system" if turns.is_empty() => {
                if !m.content.trim().is_empty() {
                    system.push(&m.content);
                }
                continue;
            }
            // Later ones have no slot of their own: a labelled user note.
            "system" => ("user", format!("[system note]\n{}", m.content)),
            "tool" => ("user", format!("[tool result]\n{}", m.content)),
            "assistant" => ("assistant", m.content.clone()),
            _ => ("user", m.content.clone()),
        };
        let mut blocks = Vec::new();
        if let Some(att) = m.image_attachment().filter(|a| a.kind == AttachmentKind::Image) {
            let bytes = lookup.and_then(|f| f(&att.cid));
            match bytes {
                // Images go first: Claude reads them before the question.
                Some(bytes)
                    if role == "user"
                        && bytes.len() <= llung_core::media::image::MAX_IMAGE_BYTES
                        && IMAGE_TYPES.contains(&att.mime_type.as_str()) =>
                {
                    blocks.push(serde_json::json!({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": att.mime_type,
                            "data": base64::engine::general_purpose::STANDARD.encode(&bytes),
                        },
                    }));
                }
                _ => text.push_str(&format!("\n[image attached: {}]", att.filename)),
            }
        }
        // The API rejects empty text blocks.
        if !text.trim().is_empty() {
            blocks.push(serde_json::json!({"type": "text", "text": text}));
        }
        if blocks.is_empty() {
            continue;
        }
        match turns.last_mut() {
            Some((last, existing)) if *last == role => existing.extend(blocks),
            _ => turns.push((role, blocks)),
        }
    }
    let note = |t: &str| vec![serde_json::json!({"type": "text", "text": t})];
    if turns.first().map(|(r, _)| *r) != Some("user") {
        turns.insert(0, ("user", note("(conversation start)")));
    }
    // A trailing assistant turn would be a prefill — the model would
    // continue it rather than answer (and thinking forbids it).
    if turns.last().map(|(r, _)| *r) == Some("assistant") {
        turns.push(("user", note("(continue)")));
    }
    let turns = turns
        .into_iter()
        .map(|(role, content)| serde_json::json!({"role": role, "content": content}))
        .collect();
    (system.join("\n\n"), turns)
}

/// What one SSE `data:` payload means to the harness.
#[derive(Debug, PartialEq, Eq)]
enum Event {
    Text(String),
    Thought(String),
    /// `message_delta.stop_reason`: end_turn, max_tokens, refusal, ...
    Stop(String),
    /// `message_stop`: the stream completed normally.
    Done,
    /// An `error` event mid-stream (e.g. overloaded_error).
    Error(String),
    /// ping, message_start, block start/stop, signature deltas, ...
    Other,
}

fn parse_event(data: &str) -> Event {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(data) else {
        return Event::Other;
    };
    let str_at = |p: &str| v.pointer(p).and_then(|x| x.as_str()).map(str::to_string);
    match v["type"].as_str() {
        Some("content_block_delta") => match v["delta"]["type"].as_str() {
            Some("text_delta") => str_at("/delta/text").map_or(Event::Other, Event::Text),
            Some("thinking_delta") => str_at("/delta/thinking").map_or(Event::Other, Event::Thought),
            _ => Event::Other,
        },
        Some("message_delta") => str_at("/delta/stop_reason").map_or(Event::Other, Event::Stop),
        Some("message_stop") => Event::Done,
        Some("error") => Event::Error(format!(
            "{}: {}",
            str_at("/error/type").unwrap_or_else(|| "error".into()),
            str_at("/error/message").unwrap_or_default()
        )),
        _ => Event::Other,
    }
}

/// The verdict on a completed stream. Like `super::finish_check`:
/// truncation is an error even with partial text — a cut-off plan or
/// half a TOOL line must not be published as a finished reply.
fn finish(full: String, stop: Option<String>) -> AgentResult<String> {
    match stop.as_deref() {
        // Permanent: a retry would be cut (or refused) again, at full price.
        Some("max_tokens") => {
            return Err(Box::new(PermanentError(
                "model response truncated (stop_reason=max_tokens; \
                 raise LLUNG_ANTHROPIC_MAX_TOKENS or lower thinking=)"
                    .into(),
            )));
        }
        Some("refusal") => {
            return Err(Box::new(PermanentError(
                "model declined to answer (stop_reason=refusal)".into(),
            )));
        }
        _ => {}
    }
    if full.trim().is_empty() {
        return Err("model backend returned no content".into());
    }
    Ok(full)
}

impl Backend for Anthropic {
    fn chat(
        &self,
        messages: &[ChatMessage],
    ) -> Pin<Box<dyn Future<Output = AgentResult<String>> + Send + '_>> {
        let messages = messages.to_vec();
        Box::pin(async move {
            // Streaming underneath; nobody listens to the deltas.
            let (tx, rx) = mpsc::channel(1);
            drop(rx);
            self.chat_stream(&messages, tx).await
        })
    }

    fn chat_stream<'a>(
        &'a self,
        messages: &'a [ChatMessage],
        deltas: mpsc::Sender<StreamChunk>,
    ) -> Pin<Box<dyn Future<Output = AgentResult<String>> + Send + 'a>> {
        Box::pin(async move {
            let idle = self.idle_timeout;
            let req = self
                .authed(self.client.post(self.url("messages")))
                .json(&self.request_body(messages));
            let mut response = check(req.send().await, idle).await?;

            // SSE: `event: <type>\ndata: {json}\n\n`. The JSON repeats
            // the type, so only `data:` lines matter. A chunk boundary
            // can fall mid-line and mid-UTF-8: split complete lines out
            // of a byte buffer.
            let mut full = String::new();
            let mut stop: Option<String> = None;
            let mut buf: Vec<u8> = Vec::new();
            while let Some(chunk) = response.chunk().await.map_err(|e| stall_aware(e, idle))? {
                buf.extend_from_slice(&chunk);
                while let Some(pos) = buf.iter().position(|b| *b == b'\n') {
                    let line: Vec<u8> = buf.drain(..=pos).collect();
                    let line = String::from_utf8_lossy(&line);
                    let Some(data) = line.trim().strip_prefix("data:") else {
                        continue;
                    };
                    match parse_event(data.trim()) {
                        Event::Text(t) => {
                            full.push_str(&t);
                            let _ = deltas.send(StreamChunk::Text(t)).await;
                        }
                        Event::Thought(t) => {
                            let _ = deltas.send(StreamChunk::Thought(t)).await;
                        }
                        Event::Stop(reason) => stop = Some(reason),
                        Event::Done => return finish(full, stop),
                        Event::Error(e) => return Err(format!("anthropic stream error: {e}").into()),
                        Event::Other => {}
                    }
                }
            }
            // Closed without `message_stop` (proxy drop, network): the
            // text may be cut anywhere, so it is never a reply.
            Err(format!(
                "anthropic stream ended before message_stop ({} chars received)",
                full.len()
            )
            .into())
        })
    }

    fn try_duplicate(&self) -> Option<Box<dyn Backend>> {
        Some(Box::new(self.clone()))
    }

    /// Claude handles the XML parameter format best: raw multi-line
    /// values (Python for the torch REPL) need no JSON escaping, and
    /// XML-ish tags are what it is trained to emit.
    fn preferred_tool_protocol(&self) -> Option<crate::tools::ToolProtocol> {
        Some(crate::tools::ToolProtocol::QwenXml)
    }

    fn probe(&self) -> Pin<Box<dyn Future<Output = Result<(), String>> + Send + '_>> {
        Box::pin(Anthropic::probe(self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llung_core::network::message::{Attachment, MessageKind};

    fn assistant(text: &str) -> ChatMessage {
        let mut m = ChatMessage::user(text);
        m.kind = MessageKind::Agent;
        m
    }

    fn image(mime: &str) -> Attachment {
        Attachment {
            cid: "cid1".into(),
            filename: "loss.png".into(),
            mime_type: mime.into(),
            size_bytes: 3,
            kind: AttachmentKind::Image,
            width: None,
            height: None,
        }
    }

    fn texts(turn: &serde_json::Value) -> Vec<String> {
        turn["content"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|b| b["text"].as_str().map(str::to_string))
            .collect()
    }

    #[test]
    fn harness_prompt_maps_to_system_plus_one_user_turn() {
        // The orchestrator's prompt shape: system, context, trigger, feedback.
        let msgs = vec![
            ChatMessage::system("You are scout."),
            ChatMessage::user("history..."),
            ChatMessage::user("New message: train it"),
            ChatMessage::user("feedback"),
        ];
        let (system, turns) = convert_messages(None, &msgs);
        assert_eq!(system, "You are scout.");
        assert_eq!(turns.len(), 1, "consecutive user messages merge: {turns:?}");
        assert_eq!(turns[0]["role"], "user");
        assert_eq!(texts(&turns[0]), ["history...", "New message: train it", "feedback"]);
    }

    #[test]
    fn roles_alternate_and_open_and_close_on_user() {
        let mut tool = ChatMessage::user("exit 0");
        tool.kind = MessageKind::ToolResponse {
            tool_name: "bash".into(),
            result_summary: String::new(),
            error: false,
        };
        let msgs = vec![
            ChatMessage::system("sys"),
            assistant("I ran it."),
            tool,
            ChatMessage::system("late note"),
            assistant("done"),
        ];
        let (_, turns) = convert_messages(None, &msgs);
        let roles: Vec<&str> = turns.iter().map(|t| t["role"].as_str().unwrap()).collect();
        assert_eq!(roles, ["user", "assistant", "user", "assistant", "user"]);
        assert_eq!(texts(&turns[2]), ["[tool result]\nexit 0", "[system note]\nlate note"]);
        // Empty messages never become (rejected) empty text blocks.
        let (_, turns) = convert_messages(None, &[ChatMessage::user("  ")]);
        assert_eq!(texts(&turns[0]), ["(conversation start)"]);
    }

    #[test]
    fn images_become_base64_blocks_on_user_turns_only() {
        let lookup = |_: &str| Some(vec![1u8, 2, 3]);
        let msg = ChatMessage::user("trend?").with_attachment(image("image/png"));
        let (_, turns) = convert_messages(Some(&lookup), &[msg]);
        let blocks = turns[0]["content"].as_array().unwrap();
        assert_eq!(blocks[0]["type"], "image", "image before the text");
        assert_eq!(blocks[0]["source"]["media_type"], "image/png");
        assert_eq!(blocks[0]["source"]["data"], "AQID");
        assert_eq!(blocks[1]["text"], "trend?");

        // Unsupported type, missing bytes, or an assistant turn: placeholder.
        let svg = ChatMessage::user("x").with_attachment(image("image/svg+xml"));
        let own = assistant("my plot").with_attachment(image("image/png"));
        let (_, turns) = convert_messages(Some(&lookup), &[svg, own]);
        assert!(texts(&turns[0])[0].contains("[image attached: loss.png]"));
        assert!(texts(&turns[1])[0].contains("[image attached: loss.png]"));
        let (_, turns) = convert_messages(None, &[ChatMessage::user("y").with_attachment(image("image/png"))]);
        assert!(texts(&turns[0])[0].contains("[image attached:"));
    }

    #[test]
    fn request_body_carries_required_fields_and_clamped_thinking() {
        let msgs = [ChatMessage::system("s"), ChatMessage::user("hi")];
        let plain = Anthropic::with_idle_timeout("http://x/v1", "claude-sonnet-4-5", std::time::Duration::from_secs(1));
        let body = plain.request_body(&msgs);
        assert_eq!(body["model"], "claude-sonnet-4-5");
        assert_eq!(body["system"], "s");
        assert_eq!(body["stream"], true);
        assert!(body["max_tokens"].as_u64().unwrap() > 0, "max_tokens is required");
        assert!(body.get("thinking").is_none());

        let thinking = plain.clone().with_thinking(ThinkingLevel::High);
        let body = thinking.request_body(&msgs);
        let budget = body["thinking"]["budget_tokens"].as_u64().unwrap();
        let max = body["max_tokens"].as_u64().unwrap();
        assert_eq!(body["thinking"]["type"], "enabled");
        assert!(budget + MIN_ANSWER_TOKENS <= max, "budget {budget} must leave room under {max}");

        assert_eq!(thinking_budget(ThinkingLevel::Low, 32_000), 2_048);
        assert_eq!(thinking_budget(ThinkingLevel::High, 32_000), 32_000 - MIN_ANSWER_TOKENS);
        assert_eq!(thinking_budget(ThinkingLevel::High, 100), MIN_THINKING_BUDGET);
    }

    #[test]
    fn cache_breakpoint_marks_the_last_block_only() {
        let msgs = [
            ChatMessage::system("s"),
            ChatMessage::user("history"),
            ChatMessage::user("trigger"),
            assistant("TOOL call"),
            ChatMessage::user("result"),
        ];
        let agent = Anthropic::with_idle_timeout("http://x/v1", "m", std::time::Duration::from_secs(1));
        let body = agent.request_body(&msgs);
        let marked: Vec<String> = body["messages"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|t| t["content"].as_array().unwrap().clone())
            .filter(|b| b.get("cache_control").is_some())
            .map(|b| b["text"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(marked, ["result"], "one breakpoint, at the end");

        let body = agent.compressor().request_body(&msgs);
        assert!(!body.to_string().contains("cache_control"), "compressor never caches");
    }

    #[test]
    fn prefers_the_xml_tool_protocol() {
        let agent = Anthropic::new("m");
        assert_eq!(agent.preferred_tool_protocol(), Some(crate::tools::ToolProtocol::QwenXml));
    }

    #[test]
    fn compressor_is_the_cheap_model_without_thinking() {
        let agent = Anthropic::new("claude-opus-4-1").with_thinking(ThinkingLevel::High).with_api_key("k");
        let c = agent.compressor();
        if std::env::var("LLUNG_ANTHROPIC_COMPRESSOR_MODEL").is_err() {
            assert_eq!(c.model, ANTHROPIC_COMPRESSOR_MODEL);
        }
        assert!(c.thinking.is_none());
        assert_eq!(c.api_key.as_deref(), Some("k"));
    }

    #[test]
    fn parses_stream_events() {
        assert_eq!(
            parse_event(r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}"#),
            Event::Text("Hi".into())
        );
        assert_eq!(
            parse_event(r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}}"#),
            Event::Thought("hmm".into())
        );
        assert_eq!(
            parse_event(r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}"#),
            Event::Stop("end_turn".into())
        );
        assert_eq!(parse_event(r#"{"type":"message_stop"}"#), Event::Done);
        assert_eq!(
            parse_event(r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#),
            Event::Error("overloaded_error: Overloaded".into())
        );
        assert_eq!(parse_event(r#"{"type":"ping"}"#), Event::Other);
        assert_eq!(
            parse_event(r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"x"}}"#),
            Event::Other
        );
        assert_eq!(parse_event("not json"), Event::Other);
    }

    #[test]
    fn finish_rejects_truncation_refusal_and_empty() {
        assert_eq!(finish("ok".into(), Some("end_turn".into())).unwrap(), "ok");
        assert!(finish("partial".into(), Some("max_tokens".into())).is_err());
        assert!(finish("no".into(), Some("refusal".into())).is_err());
        assert!(finish(" ".into(), Some("end_turn".into())).is_err());
    }

    /// One-shot Messages API stub: captures the request head, answers
    /// with `events` as SSE (each its own `data:` line).
    async fn stub(events: Vec<&'static str>) -> (String, tokio::sync::oneshot::Receiver<String>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (head_tx, head_rx) = tokio::sync::oneshot::channel();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut req = Vec::new();
            let mut buf = vec![0u8; 8192];
            while !req.windows(4).any(|w| w == b"\r\n\r\n") {
                let n = socket.read(&mut buf).await.unwrap();
                if n == 0 {
                    break;
                }
                req.extend_from_slice(&buf[..n]);
            }
            let _ = head_tx.send(String::from_utf8_lossy(&req).to_string());
            let _ = socket
                .write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n")
                .await;
            for e in events {
                let _ = socket.write_all(format!("event: x\ndata: {e}\n\n").as_bytes()).await;
            }
        });
        (format!("http://{addr}/v1"), head_rx)
    }

    const START: &str = r#"{"type":"message_start","message":{"id":"m","content":[]}}"#;
    const THINK: &str = r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"plan"}}"#;
    const T1: &str = r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Hello "}}"#;
    const T2: &str = r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"world"}}"#;
    const END: &str = r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#;
    const STOP: &str = r#"{"type":"message_stop"}"#;

    #[tokio::test]
    async fn streams_text_and_thoughts_with_auth_headers() {
        let (base, head) = stub(vec![START, r#"{"type":"ping"}"#, THINK, T1, T2, END, STOP]).await;
        let backend = Anthropic::with_idle_timeout(base, "claude-sonnet-4-5", std::time::Duration::from_secs(5))
            .with_api_key("sk-ant-test");
        let (tx, mut rx) = mpsc::channel(16);
        let reply = backend.chat_stream(&[ChatMessage::user("hi")], tx).await.unwrap();
        assert_eq!(reply, "Hello world");
        assert_eq!(rx.recv().await, Some(StreamChunk::Thought("plan".into())));
        assert_eq!(rx.recv().await, Some(StreamChunk::Text("Hello ".into())));

        let head = head.await.unwrap().to_ascii_lowercase();
        assert!(head.starts_with("post /v1/messages "), "{head}");
        assert!(head.contains("x-api-key: sk-ant-test"), "{head}");
        assert!(head.contains(&format!("anthropic-version: {API_VERSION}")), "{head}");
        assert!(!head.contains("authorization:"), "no bearer auth on Anthropic");
    }

    #[tokio::test]
    async fn chat_without_listener_still_returns_the_reply() {
        let (base, _) = stub(vec![START, T1, T2, END, STOP]).await;
        let backend = Anthropic::with_idle_timeout(base, "m", std::time::Duration::from_secs(5));
        assert_eq!(backend.chat(&[ChatMessage::user("hi")]).await.unwrap(), "Hello world");
    }

    #[tokio::test]
    async fn truncated_cut_and_errored_streams_fail() {
        let max = r#"{"type":"message_delta","delta":{"stop_reason":"max_tokens"}}"#;
        let err = r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#;
        for (events, want) in [
            (vec![START, T1, max, STOP], "truncated"),
            (vec![START, T1], "before message_stop"),
            (vec![START, T1, err], "overloaded_error"),
        ] {
            let (base, _) = stub(events).await;
            let backend = Anthropic::with_idle_timeout(base, "m", std::time::Duration::from_secs(5));
            let e = backend.chat(&[ChatMessage::user("hi")]).await.unwrap_err().to_string();
            assert!(e.contains(want), "want {want}: {e}");
        }
    }
}
