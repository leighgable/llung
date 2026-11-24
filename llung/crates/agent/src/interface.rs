use std::future::Future;
use std::pin::Pin;

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use llung_core::network::message::{AgentResult, ChatMessage};

use crate::config::{AgentConfig, BackendKind, ThinkingLevel};

/// Multimodal input: resolve an attachment's content id to its cached
/// bytes.
type MediaLookupFn = dyn Fn(&str) -> Option<Vec<u8>> + Send + Sync;

/// Display fallback for the local llama-server's model id. llama-server
/// identifies models by their full GGUF path, so this friendly name is
/// only used until `discover_model()` (or a failed spawn's `probe()`
/// warning) replaces it with the live id. Override with
/// `LLUNG_LLAMA_MODEL`.
pub const LLAMA_FALLBACK_MODEL: &str = "sensenova-9b";

/// Fallback model id for SmolVLM — the default history-compressor
/// model for the llamacpp backend, served by the SAME llama-server as
/// the agent (router mode, `scripts/serve-sensenova.sh`) and selected
/// by the request's `model` field. Matches the `[smolvlm]` preset
/// name; `discover_smolvlm()` replaces it with the live id (e.g. an
/// HF-cache name). Override with `LLUNG_SMOLVLM_MODEL`.
pub const SMOLVLM_FALLBACK_MODEL: &str = "smolvlm";

/// True for model ids that name a SmolVLM model.
fn is_smolvlm_id(id: &str) -> bool {
    id.to_ascii_lowercase().contains("smolvlm")
}

/// Query an OpenAI-compatible `{base}/models` and return the first
/// model id. llama-server answers with the full GGUF path it serves.
/// Shared by `discover_model` and the agent_chat REPL.
pub async fn first_model_at(base_url: &str) -> Option<String> {
    models_at(base_url).await.into_iter().next()
}

/// All model ids an OpenAI-compatible `{base}/models` lists — one for
/// a single-model llama-server, every configured model in router mode.
/// Empty when the server can't be queried.
pub async fn models_at(base_url: &str) -> Vec<String> {
    #[derive(serde::Deserialize)]
    struct ModelList {
        data: Vec<ModelEntry>,
    }
    #[derive(serde::Deserialize)]
    struct ModelEntry {
        id: String,
    }

    let url = format!("{}/models", base_url.trim_end_matches('/'));
    let Ok(resp) = reqwest::get(url).await else {
        return Vec::new();
    };
    let Ok(list) = resp.json::<ModelList>().await else {
        return Vec::new();
    };
    list.data.into_iter().map(|m| m.id).collect()
}

/// One incremental chunk of a streaming reply. `Thought` is the
/// reasoning-model scratchpad (e.g. Kimi's `reasoning_content`): the
/// UI shows it as activity (spinner / elided trace), never as the
/// answer itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StreamChunk {
    Text(String),
    Thought(String),
}

pub trait Backend: Send + Sync {
    fn chat(
        &self,
        messages: &[ChatMessage],
    ) -> Pin<Box<dyn Future<Output = AgentResult<String>> + Send + '_>>;

    /// Like `chat`, but forwards each response chunk as it arrives.
    /// The default falls back to `chat` and delivers the whole reply as
    /// one text chunk — streaming is an optimization, never required.
    fn chat_stream<'a>(
        &'a self,
        messages: &'a [ChatMessage],
        deltas: mpsc::Sender<StreamChunk>,
    ) -> Pin<Box<dyn Future<Output = AgentResult<String>> + Send + 'a>> {
        Box::pin(async move {
            let full = self.chat(messages).await?;
            let _ = deltas.send(StreamChunk::Text(full.clone())).await;
            Ok(full)
        })
    }

    /// An independent handle to the same backend, when the
    /// implementation can provide one (connection pool shared). Used to
    /// give the history compressor its own handle to the agent's model.
    fn try_duplicate(&self) -> Option<Box<dyn Backend>> {
        None
    }
}

pub struct OpenAiCompatible {
    base_url: String,
    model: String,
    /// Bearer token for cloud backends (Kimi). Held in memory only —
    /// loaded from the keyring/env at spawn, never logged, and NOT in
    /// `Debug` output anywhere (this struct doesn't derive Debug).
    api_key: Option<String>,
    /// Reasoning effort for thinking models; None omits the parameter.
    thinking: Option<ThinkingLevel>,
    /// Multimodal input: resolve an attachment's content id to bytes.
    /// The harness wires this to the media cache; without it, image
    /// attachments degrade to a text placeholder so text-only backends
    /// keep working unchanged.
    media_lookup: Option<std::sync::Arc<MediaLookupFn>>,
    client: reqwest::Client,
    /// Longest tolerated backend silence (see `model_idle_timeout`);
    /// kept for error messages.
    idle_timeout: std::time::Duration,
}

/// Cheap to clone: strings plus a shared connection pool. Lets the
/// TUI hand one backend to the agent's chat loop and another to the
/// history compressor. (Cloning does copy the in-memory API key, same
/// as any reqwest client clone.)
impl Clone for OpenAiCompatible {
    fn clone(&self) -> Self {
        Self {
            base_url: self.base_url.clone(),
            model: self.model.clone(),
            api_key: self.api_key.clone(),
            thinking: self.thinking,
            media_lookup: self.media_lookup.clone(),
            client: self.client.clone(),
            idle_timeout: self.idle_timeout,
        }
    }
}

/// Longest silence tolerated from a model backend: before the first
/// byte (prompt processing on a slow local GPU can take minutes) and
/// between streamed chunks. `LLUNG_MODEL_IDLE_TIMEOUT_SECS` overrides
/// the 300 s default.
fn model_idle_timeout() -> std::time::Duration {
    let secs = std::env::var("LLUNG_MODEL_IDLE_TIMEOUT_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|s| *s > 0)
        .unwrap_or(300);
    std::time::Duration::from_secs(secs)
}

/// Turn a reqwest timeout into a message that says what happened
/// (reqwest reports a mid-body stall as "error decoding response body").
fn stall_aware(
    e: reqwest::Error,
    idle: std::time::Duration,
) -> Box<dyn std::error::Error + Send + Sync> {
    if e.is_timeout() {
        format!(
            "model backend went silent for {}s (stalled; set \
             LLUNG_MODEL_IDLE_TIMEOUT_SECS to wait longer)",
            idle.as_secs()
        )
        .into()
    } else {
        Box::new(e)
    }
}

impl OpenAiCompatible {
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self::with_idle_timeout(base_url, model, model_idle_timeout())
    }

    /// [`Self::new`] with an explicit stall limit (tests use seconds).
    pub(crate) fn with_idle_timeout(
        base_url: impl Into<String>,
        model: impl Into<String>,
        idle_timeout: std::time::Duration,
    ) -> Self {
        // Bound STALLS, not total time: a backend that accepts but goes
        // silent (e.g. a busy llama-server slot) would otherwise hang
        // the turn forever (reqwest has no default timeout). An overall
        // request timeout also killed slow-but-healthy streams: a long
        // generation on an iGPU legitimately streams for minutes. So:
        // the limit is on SILENCE between reads (`read_timeout`); a
        // stream that keeps producing tokens may run as long as it
        // needs. For non-streaming calls the body arrives in one piece
        // at the end, so there it still bounds the whole generation.
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(10))
            .read_timeout(idle_timeout)
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            base_url: base_url.into(),
            model: model.into(),
            api_key: None,
            thinking: None,
            media_lookup: None,
            client,
            idle_timeout,
        }
    }

    /// Attach a bearer API key (cloud backends).
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Attach a reasoning-effort level (thinking models).
    pub fn with_thinking(mut self, level: ThinkingLevel) -> Self {
        self.thinking = Some(level);
        self
    }

    /// Wire multimodal input: `lookup(cid)` returns the attachment
    /// bytes from the media cache. With a lookup, image attachments on
    /// user/assistant turns travel as OpenAI image_url content parts
    /// (data URLs); without one they degrade to a `[image attached: …]`
    /// placeholder in the text.
    pub fn with_media_lookup(
        mut self,
        lookup: impl Fn(&str) -> Option<Vec<u8>> + Send + Sync + 'static,
    ) -> Self {
        self.media_lookup = Some(std::sync::Arc::new(lookup));
        self
    }

    /// Quick liveness check (`GET {base_url}/models`) so embedders can
    /// warn at spawn time when the model server isn't reachable, instead
    /// of the agent silently failing on its first turn.
    pub async fn probe(&self) -> Result<(), String> {
        let url = format!("{}/models", self.base_url.trim_end_matches('/'));
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(3))
            .build()
            .map_err(|e| e.to_string())?;
        let mut req = client.get(url);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }
        check(req.send().await, std::time::Duration::from_secs(3))
            .await
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn llama_server(port: u16) -> Self {
        Self::new(format!("http://localhost:{port}/v1"), Self::llama_fallback_model())
    }

    /// Fallback model id for the local llama-server, when the server
    /// can't be queried (yet). `LLUNG_LLAMA_MODEL` overrides it
    /// (mirrors `LLM_MODEL` in agent_chat.rs). Discovery at spawn
    /// (`discover_model`) normally replaces this with the live id —
    /// llama-server identifies models by their full GGUF path.
    pub fn llama_fallback_model() -> String {
        std::env::var("LLUNG_LLAMA_MODEL")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| LLAMA_FALLBACK_MODEL.into())
    }

    /// llama.cpp llama-server on the sensenova-loop fork (the default
    /// local backend): SenseNova U1.5 8B, vision-capable via --mmproj.
    pub fn llamacpp(port: u16) -> Self {
        Self::llama_server(port)
    }

    /// SmolVLM on the local llama-server in router mode (same port as
    /// the agent's model, picked by model id): the small, fast model
    /// that compresses history for llamacpp agents, so the main model's
    /// slot stays free for the actual turn.
    pub fn smolvlm(port: u16) -> Self {
        let model = std::env::var("LLUNG_SMOLVLM_MODEL")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| SMOLVLM_FALLBACK_MODEL.into());
        Self::new(format!("http://localhost:{port}/v1"), model)
    }

    /// Query `{base}/models` and remember the FIRST model id —
    /// llama-server identifies models by their full GGUF path, so
    /// asking beats guessing. Returns true when discovery replaced the
    /// stored id; on failure (server down at spawn) the stored fallback
    /// stays, and `probe()` reports the outage right below.
    ///
    /// Router mode lists several models (presets plus the HF cache):
    /// the stored id wins when it is listed verbatim (the preset name,
    /// e.g. `sensenova-9b`); otherwise the first non-SmolVLM id —
    /// SmolVLM is the history compressor, not the agent.
    pub async fn discover_model(&mut self) -> bool {
        let ids = models_at(&self.base_url).await;
        let pick = ids
            .iter()
            .find(|id| **id == self.model)
            .or_else(|| ids.iter().find(|id| !is_smolvlm_id(id)))
            .or_else(|| ids.first());
        if let Some(id) = pick {
            self.model = id.clone();
            return true;
        }
        false
    }

    /// Like `discover_model`, but picks the served SmolVLM id. Returns
    /// false (keeping the fallback id) when the server is down or
    /// serves no SmolVLM model — history compression then degrades to
    /// rule-based truncation.
    pub async fn discover_smolvlm(&mut self) -> bool {
        let ids = models_at(&self.base_url).await;
        let pick = ids
            .iter()
            .find(|id| **id == self.model)
            .or_else(|| ids.iter().find(|id| is_smolvlm_id(id)));
        if let Some(id) = pick {
            self.model = id.clone();
            return true;
        }
        false
    }

    /// Lemonade server: OpenAI-compatible API, default port 13305.
    /// Model id comes from `GET /v1/models` (e.g. LFM2.5-8B-A1B).
    pub fn lemonade(port: u16) -> Self {
        Self::new(format!("http://localhost:{port}/v1"), "LFM2.5-8B-A1B")
    }

    pub fn from_kind(kind: BackendKind) -> Self {
        match kind {
            BackendKind::LlamaCpp => Self::llamacpp(9931),
            BackendKind::Lemonade => Self::lemonade(13305),
            BackendKind::Kimi => Self::kimi("kimi-k2-0905-preview", None),
            BackendKind::KimiCoding => Self::kimi_coding("kimi-for-coding", None),
        }
    }

    /// Moonshot Kimi: OpenAI-compatible chat completions over HTTPS.
    /// Base URL overridable with `LLUNG_KIMI_BASE_URL` (e.g. the .cn
    /// endpoint). The API key is NOT resolved here — the caller loads
    /// it via `llung_core::secrets::load_api_key("kimi")` and attaches
    /// it with `with_api_key`, so the key never sits in `AgentConfig`
    /// (which is Debug-printed).
    pub fn kimi(model: impl Into<String>, thinking: Option<ThinkingLevel>) -> Self {
        let base = std::env::var("LLUNG_KIMI_BASE_URL")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "https://api.moonshot.ai/v1".into());
        let mut backend = Self::new(base, model);
        backend.thinking = thinking;
        backend
    }

    /// "Kimi for Coding" subscription: same OpenAI-compatible shape as
    /// `kimi`, but a different host (api.kimi.com/coding/v1) and model
    /// namespace. Keys from the coding plan are NOT accepted by
    /// api.moonshot.ai — that's the 401 "Invalid Authentication" this
    /// variant exists to avoid. Base URL overridable with
    /// `LLUNG_KIMI_CODING_BASE_URL`.
    pub fn kimi_coding(model: impl Into<String>, thinking: Option<ThinkingLevel>) -> Self {
        let base = std::env::var("LLUNG_KIMI_CODING_BASE_URL")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "https://api.kimi.com/coding/v1".into());
        let mut backend = Self::new(base, model);
        backend.thinking = thinking;
        backend
    }

    /// Build a backend from an agent config: an explicit `model_url`
    /// wins, otherwise the configured `backend` picks the endpoint.
    pub fn from_config(config: &AgentConfig) -> Self {
        match &config.model_url {
            Some(url) => Self::new(url.clone(), config.model.clone()),
            None => match config.backend {
                BackendKind::Kimi => Self::kimi(config.model.clone(), config.thinking),
                BackendKind::KimiCoding => Self::kimi_coding(config.model.clone(), config.thinking),
                BackendKind::LlamaCpp => {
                    let mut backend = Self::llamacpp(9931);
                    // An explicit `model=` on the spawn line overrides
                    // the fallback id (the env override inside
                    // llamacpp() applies only when the config still
                    // carries the default model).
                    if config.model != AgentConfig::default().model {
                        backend.model = config.model.clone();
                    }
                    backend
                }
                kind => Self::from_kind(kind),
            },
        }
    }
}

/// Extract the text delta from one SSE `data:` payload, if any.
/// Servers send a first chunk with only `role` (content null) — skipped.
fn extract_delta(payload: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(payload).ok()?;
    let content = value["choices"][0]["delta"]["content"].as_str()?;
    Some(content.to_string())
}

/// Reasoning-model scratchpad (e.g. LFM2.5's `reasoning_content`), sent
/// as preview-only deltas — never part of the accumulated reply.
fn extract_reasoning(payload: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(payload).ok()?;
    let reasoning = value["choices"][0]["delta"]["reasoning_content"].as_str()?;
    Some(reasoning.to_string())
}

/// `choices[0].finish_reason` of one SSE payload, when present. OpenAI-
/// compatible servers send it on the final chunk before `[DONE]`; `stop`
/// means complete, `length` means the output was cut by the token budget.
fn extract_finish_reason(payload: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(payload).ok()?;
    let fr = value["choices"][0]["finish_reason"].as_str()?;
    Some(fr.to_string())
}

/// The terminal verdict on a completed stream. Truncation (`length`) is
/// an error even with partial text: a cut-off plan or half a TOOL line
/// would otherwise be published as if the model had finished.
fn finish_check(full: String, finish: Option<String>) -> AgentResult<String> {
    if full.is_empty() {
        return Err("model backend returned no content".into());
    }
    if finish.as_deref() == Some("length") {
        return Err("model response truncated (finish_reason=length)".into());
    }
    Ok(full)
}

/// llung-specific fields (topic, sender, DAG parent, ...); the endpoint
/// only wants a role and content, so map explicitly instead of
/// serializing ChatMessage verbatim.
#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<RequestMessage>,
    stream: bool,
    /// Reasoning effort for thinking models; omitted entirely when not
    /// configured so plain models never see an unknown parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<serde_json::Value>,
}

/// The `thinking` request parameter for a level. Centralized because
/// providers keep moving this shape around: Moonshot documents the
/// budget-token form for Kimi K2 Thinking; if your endpoint expects
/// `{"reasoning_effort": "high"}` instead, this is the one place to change.
fn thinking_param(level: ThinkingLevel) -> serde_json::Value {
    serde_json::json!({
        "type": "enabled",
        "budget_tokens": level.budget_tokens(),
    })
}

#[derive(Serialize)]
struct RequestMessage {
    role: &'static str,
    /// A plain string, or an OpenAI content-parts array
    /// (`[{type:"text",…}, {type:"image_url",…}]`) for multimodal
    /// input.
    content: serde_json::Value,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
    /// `length` means the output was cut by the token budget; the
    /// partial text must not be treated as a complete reply.
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct ResponseMessage {
    content: Option<String>,
}

/// Convert a chat message to the OpenAI request shape. Multimodal
/// input: an image attachment becomes an `image_url` content part
/// (base64 data URL) when the media lookup resolves its cid and the
/// bytes fit the image cap; otherwise the text degrades to an
/// `[image attached: …]` placeholder so text-only backends keep their
/// context. Images never ride system turns (OpenAI rejects that).
fn message_to_request(lookup: Option<&MediaLookupFn>, m: &ChatMessage) -> RequestMessage {
    use base64::Engine as _;
    let role = m.agent_role();
    let mut text = m.content.clone();
    let mut parts: Vec<serde_json::Value> = Vec::new();
    if role != "system" {
        if let Some(att) = m.image_attachment() {
            if att.kind == llung_core::network::message::AttachmentKind::Image {
                let bytes = lookup.and_then(|f| f(&att.cid));
                match bytes {
                    Some(bytes)
                        if bytes.len() <= llung_core::media::image::MAX_IMAGE_BYTES =>
                    {
                        let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
                        parts.push(serde_json::json!({
                            "type": "image_url",
                            "image_url": {
                                "url": format!("data:{};base64,{b64}", att.mime_type)
                            },
                        }));
                    }
                    _ => text.push_str(&format!("\n[image attached: {}]", att.filename)),
                }
            }
        }
    }
    let content = if parts.is_empty() {
        serde_json::Value::String(text)
    } else {
        if !text.is_empty() {
            parts.insert(0, serde_json::json!({"type": "text", "text": text}));
        }
        serde_json::Value::Array(parts)
    };
    RequestMessage { role, content }
}

/// Send, then fail with the server's body included on non-2xx. A bare
/// reqwest status error ("401 Unauthorized for url") sends users down
/// the wrong debugging path; providers put the actual reason in the
/// JSON body ("Invalid Authentication", "model not found", ...).
async fn check(
    sent: Result<reqwest::Response, reqwest::Error>,
    idle: std::time::Duration,
) -> Result<reqwest::Response, Box<dyn std::error::Error + Send + Sync>> {
    // Headers wait out prompt processing: a stall there is a timeout.
    let response = sent.map_err(|e| stall_aware(e, idle))?;
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    let body = response.text().await.unwrap_or_default();
    let body = body.trim();
    let detail = if body.is_empty() {
        String::new()
    } else {
        format!(" — {body}")
    };
    Err(format!("{status}{detail}").into())
}

impl OpenAiCompatible {
    /// Map prompt messages to the request shape, resolving image
    /// attachments through the media lookup when one is wired.
    fn request_messages(&self, messages: &[ChatMessage]) -> Vec<RequestMessage> {
        let lookup = self.media_lookup.as_deref();
        messages
            .iter()
            .map(|m| message_to_request(lookup, m))
            .collect()
    }

    fn chat_url(&self) -> String {
        format!("{}/chat/completions", self.base_url.trim_end_matches('/'))
    }

    fn chat_request<'a>(&'a self, messages: &'a [ChatMessage], stream: bool) -> ChatRequest<'a> {
        ChatRequest {
            model: &self.model,
            messages: self.request_messages(messages),
            stream,
            thinking: self.thinking.map(thinking_param),
        }
    }
}

impl Backend for OpenAiCompatible {
    fn try_duplicate(&self) -> Option<Box<dyn Backend>> {
        // Cheap to clone: two strings plus a shared connection pool.
        Some(Box::new(self.clone()))
    }

    fn chat_stream<'a>(
        &'a self,
        messages: &'a [ChatMessage],
        deltas: mpsc::Sender<StreamChunk>,
    ) -> Pin<Box<dyn Future<Output = AgentResult<String>> + Send + 'a>> {
        // Clone what the request needs so the future owns its data and
        // no borrowed lifetimes leak into the boxed type.
        let model = self.model.clone();
        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let thinking = self.thinking;
        let messages = messages.to_vec();
        let url = self.chat_url();
        let idle = self.idle_timeout;

        Box::pin(async move {
            let request = ChatRequest {
                model: &model,
                messages: self.request_messages(&messages),
                stream: true,
                thinking: thinking.map(thinking_param),
            };
            let mut req = client.post(url).json(&request);
            if let Some(key) = &api_key {
                req = req.bearer_auth(key);
            }
            let mut response = check(req.send().await, idle).await?;

            // SSE: `data: {json}\n\n` chunks, terminated by `data: [DONE]`.
            // Read byte chunks, split complete lines out of the buffer —
            // a chunk boundary can fall mid-line and mid-UTF-8.
            let mut full = String::new();
            let mut finish: Option<String> = None;
            let mut buf: Vec<u8> = Vec::new();
            while let Some(chunk) = response.chunk().await.map_err(|e| stall_aware(e, idle))? {
                buf.extend_from_slice(&chunk);
                while let Some(pos) = buf.iter().position(|b| *b == b'\n') {
                    let line: Vec<u8> = buf.drain(..=pos).collect();
                    let line = String::from_utf8_lossy(&line);
                    let Some(data) = line.trim().strip_prefix("data:") else {
                        continue;
                    };
                    let data = data.trim();
                    if data == "[DONE]" {
                        return finish_check(full, finish);
                    }
                    if let Some(fr) = extract_finish_reason(data) {
                        finish = Some(fr);
                    }
                    if let Some(delta) = extract_delta(data) {
                        full.push_str(&delta);
                        let _ = deltas.send(StreamChunk::Text(delta)).await;
                    } else if let Some(reasoning) = extract_reasoning(data) {
                        // The reasoning scratchpad is activity, not
                        // answer: the UI shows a spinner/elided trace.
                        let _ = deltas.send(StreamChunk::Thought(reasoning)).await;
                    }
                }
            }
            // Stream closed without [DONE] (proxy drop, idle timeout):
            // `finish_check` still applies — empty means "no content",
            // an explicit `length` means truncated.
            finish_check(full, finish)
        })
    }

    fn chat(
        &self,
        messages: &[ChatMessage],
    ) -> Pin<Box<dyn Future<Output = AgentResult<String>> + Send + '_>> {
        // Clone what the request needs so the future owns its data and
        // no borrowed lifetimes leak into the boxed type.
        let model = self.model.clone();
        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let thinking = self.thinking;
        let messages = messages.to_vec();
        let url = self.chat_url();
        let idle = self.idle_timeout;

        Box::pin(async move {
            let request = ChatRequest {
                model: &model,
                messages: self.request_messages(&messages),
                stream: false,
                thinking: thinking.map(thinking_param),
            };
            let mut req = client.post(url).json(&request);
            if let Some(key) = &api_key {
                req = req.bearer_auth(key);
            }
            let response = check(req.send().await, idle).await?;

            let body: ChatResponse = response.json().await.map_err(|e| stall_aware(e, idle))?;

            let choice = body.choices.into_iter().next();
            let content = choice
                .as_ref()
                .and_then(|c| c.message.content.clone())
                .filter(|c| !c.is_empty());
            let finish = choice.and_then(|c| c.finish_reason);
            match content {
                Some(text) => finish_check(text, finish),
                None => Err("model backend returned no content".into()),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llung_core::network::message::{Attachment, AttachmentKind};

    fn png(cid: &str) -> Attachment {
        Attachment {
            cid: cid.into(),
            filename: "plot.png".into(),
            mime_type: "image/png".into(),
            size_bytes: 3,
            kind: AttachmentKind::Image,
            width: None,
            height: None,
        }
    }

    #[test]
    fn multimodal_input_becomes_an_image_url_part() {
        use base64::Engine as _;
        let msg = ChatMessage::user("what trend do you see?").with_attachment(png("cid1"));
        let lookup = |_cid: &str| Some(vec![1u8, 2, 3]);
        let req = message_to_request(Some(&lookup), &msg);
        let parts = req.content.as_array().expect("content parts array");
        assert_eq!(parts[0]["type"], "text");
        assert!(parts[0]["text"].as_str().unwrap().contains("what trend"));
        assert_eq!(parts[1]["type"], "image_url");
        let url = parts[1]["image_url"]["url"].as_str().unwrap();
        let b64 = url.strip_prefix("data:image/png;base64,").unwrap();
        assert_eq!(
            base64::engine::general_purpose::STANDARD.decode(b64).unwrap(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn image_input_degrades_to_a_text_placeholder() {
        let msg = ChatMessage::user("look").with_attachment(png("cid1"));
        // No lookup wired (text-only backend).
        let req = message_to_request(None, &msg);
        assert!(
            req.content.as_str().unwrap().contains("[image attached: plot.png]"),
            "got: {}",
            req.content
        );
        // Lookup wired but the bytes aren't cached.
        let miss = |_cid: &str| None;
        let req = message_to_request(Some(&miss), &msg);
        assert!(req.content.as_str().unwrap().contains("[image attached:"));
        // Bytes over the image cap also degrade (never a 5 MB data URL).
        let big = |_cid: &str| Some(vec![0u8; llung_core::media::image::MAX_IMAGE_BYTES + 1]);
        let req = message_to_request(Some(&big), &msg);
        assert!(req.content.as_str().unwrap().contains("[image attached:"));
    }

    #[test]
    fn system_turns_never_carry_images() {
        let msg = ChatMessage::system("you are helpful").with_attachment(png("cid1"));
        let lookup = |_cid: &str| Some(vec![1u8]);
        let req = message_to_request(Some(&lookup), &msg);
        assert!(
            req.content.is_string(),
            "OpenAI rejects image parts on system turns: {}",
            req.content
        );
    }

    #[test]
    fn media_kind_payload_converts_without_the_top_level_field() {
        // Reloaded messages (DB keeps the payload in the kind column).
        let att = png("cid9");
        let mut msg = ChatMessage::user("my plot");
        msg.kind = llung_core::network::message::MessageKind::Media(att);
        let lookup = |cid: &str| (cid == "cid9").then(|| vec![7u8]);
        let req = message_to_request(Some(&lookup), &msg);
        assert!(
            req.content.to_string().contains("image_url"),
            "Media-kind payload is image input after a reload: {}",
            req.content
        );
    }

    /// One-shot HTTP stub answering `GET /models` like llama-server
    /// does: the served model id is the FULL GGUF path.
    async fn models_stub(model_id: &str) -> String {
        models_stub_many(&[model_id], 1).await
    }

    /// Like `models_stub`, but lists several ids (router mode) and
    /// answers `requests` connections.
    /// One-shot SSE stub: streams `chunks` text deltas `gap` apart, then
    /// (unless `stall` is set, in which case it goes silent instead)
    /// `[DONE]`. Like llama-server, headers go out first.
    async fn sse_stub(chunks: usize, gap: std::time::Duration, stall: Option<std::time::Duration>) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 8192];
            let _ = socket.read(&mut buf).await; // the request; content irrelevant
            // Close-delimited body (no content-length): chunks arrive as written.
            let _ = socket
                .write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n")
                .await;
            for i in 0..chunks {
                let event = format!("data: {{\"choices\":[{{\"delta\":{{\"content\":\"t{i} \"}}}}]}}\n\n");
                let _ = socket.write_all(event.as_bytes()).await;
                tokio::time::sleep(gap).await;
            }
            if let Some(stall) = stall {
                tokio::time::sleep(stall).await;
                return;
            }
            let _ = socket.write_all(b"data: [DONE]\n\n").await;
        });
        format!("http://{addr}/v1")
    }

    #[tokio::test]
    async fn slow_steady_streams_outlive_the_idle_limit() {
        // 6 chunks 400 ms apart = ~2.4 s total under a 1 s SILENCE limit:
        // a total-request timeout would have killed this healthy stream.
        let base = sse_stub(6, std::time::Duration::from_millis(400), None).await;
        let backend =
            OpenAiCompatible::with_idle_timeout(base, "m", std::time::Duration::from_secs(1));
        let (tx, _rx) = mpsc::channel(32);
        let reply = backend
            .chat_stream(&[ChatMessage::user("hi".to_string())], tx)
            .await
            .expect("steady stream completes");
        assert_eq!(reply, "t0 t1 t2 t3 t4 t5 ");
    }

    #[tokio::test]
    async fn stalled_streams_fail_with_a_clear_message() {
        let base = sse_stub(1, std::time::Duration::ZERO, Some(std::time::Duration::from_secs(5))).await;
        let backend =
            OpenAiCompatible::with_idle_timeout(base, "m", std::time::Duration::from_secs(1));
        let (tx, _rx) = mpsc::channel(32);
        let err = backend
            .chat_stream(&[ChatMessage::user("hi".to_string())], tx)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("went silent for 1s"), "{err}");
    }

    async fn models_stub_many(model_ids: &[&str], requests: usize) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let entries: Vec<String> = model_ids
            .iter()
            .map(|id| format!(r#"{{"id":{id:?}}}"#))
            .collect();
        let body = format!(r#"{{"data":[{}]}}"#, entries.join(","));
        tokio::spawn(async move {
            for _ in 0..requests {
                let body = body.clone();
                let (socket, _) = listener.accept().await.unwrap();
                let mut socket = socket;
                use tokio::io::{AsyncReadExt, AsyncWriteExt};
                let mut buf = vec![0u8; 4096];
                let _ = socket.read(&mut buf).await; // the request; content irrelevant
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                socket.write_all(response.as_bytes()).await.unwrap();
            }
        });
        format!("http://{addr}/v1")
    }

    #[tokio::test]
    async fn router_mode_splits_agent_model_and_smolvlm() {
        // Router mode: HF-cache entries plus presets. The agent keeps
        // its preset id; the compressor finds SmolVLM.
        let ids = [
            "ggml-org/SmolVLM2-2.2B-Instruct-GGUF:Q4_K_M",
            "LiquidAI/LFM2.5-8B-A1B-GGUF:Q4_K_M",
            "sensenova-9b",
        ];
        let base = models_stub_many(&ids, 3).await;

        let mut agent = OpenAiCompatible::new(base.clone(), "sensenova-9b");
        assert!(agent.discover_model().await);
        assert_eq!(agent.model, "sensenova-9b", "listed preset id wins");

        let mut agent = OpenAiCompatible::new(base.clone(), "not-listed");
        assert!(agent.discover_model().await);
        assert_eq!(agent.model, ids[1], "SmolVLM is never the agent");

        let mut smol = OpenAiCompatible::new(base, SMOLVLM_FALLBACK_MODEL);
        assert!(smol.discover_smolvlm().await);
        assert_eq!(smol.model, ids[0]);
    }

    #[tokio::test]
    async fn first_model_at_reads_the_live_model_list() {
        let base = models_stub("/models/SenseNova-U1.5-8B-MoT-Q4_K_M.gguf").await;
        assert_eq!(
            first_model_at(&base).await.as_deref(),
            Some("/models/SenseNova-U1.5-8B-MoT-Q4_K_M.gguf")
        );
        // Unreachable server: None, never a panic.
        assert!(first_model_at("http://127.0.0.1:1/v1").await.is_none());
    }

    #[tokio::test]
    async fn discover_model_replaces_the_fallback_and_survives_outages() {
        let base = models_stub("/models/exp07.gguf").await;
        let mut backend = OpenAiCompatible::new(base.clone(), LLAMA_FALLBACK_MODEL);
        assert!(backend.discover_model().await, "stub answers");
        assert_eq!(backend.model, "/models/exp07.gguf");

        // Server down at spawn: keep the stored id, report failure —
        // probe() carries the warning to the UI from here.
        let mut backend = OpenAiCompatible::new("http://127.0.0.1:1/v1", "sensenova-9b");
        assert!(!backend.discover_model().await);
        assert_eq!(backend.model, "sensenova-9b");
    }

    #[test]
    fn llama_fallback_model_is_sensenova_with_env_override() {
        // Default: the display fallback. (Reads the env var, which no
        // other test in this binary sets.)
        if std::env::var("LLUNG_LLAMA_MODEL").is_ok() {
            eprintln!("LLUNG_LLAMA_MODEL set in the environment; skipping the default half");
        } else {
            assert_eq!(OpenAiCompatible::llama_fallback_model(), LLAMA_FALLBACK_MODEL);
            assert_eq!(LLAMA_FALLBACK_MODEL, "sensenova-9b");
        }
        // SAFETY: scoped env override, removed immediately; no other
        // test reads LLUNG_LLAMA_MODEL concurrently in a way that
        // asserts on it (llamacpp() users only need SOME model id).
        unsafe { std::env::set_var("LLUNG_LLAMA_MODEL", "/models/custom.gguf") };
        assert_eq!(OpenAiCompatible::llama_fallback_model(), "/models/custom.gguf");
        assert_eq!(OpenAiCompatible::llama_server(1).model, "/models/custom.gguf");
        unsafe { std::env::remove_var("LLUNG_LLAMA_MODEL") };
    }

    #[test]
    fn from_config_lamacpp_respects_explicit_model_and_defaults_otherwise() {
        let default = AgentConfig::default();
        assert_eq!(default.backend, BackendKind::LlamaCpp);
        // Default config: the fallback id comes from llamacpp() (env
        // override applies; none is set in the test environment).
        let b = OpenAiCompatible::from_config(&default);
        assert_eq!(b.base_url, "http://localhost:9931/v1");

        // Explicit model= on the spawn line wins over the fallback.
        let explicit = AgentConfig {
            model: "/models/exp07.gguf".into(),
            ..Default::default()
        };
        assert_eq!(OpenAiCompatible::from_config(&explicit).model, "/models/exp07.gguf");
    }

    #[test]
    fn extracts_content_deltas_only() {
        assert_eq!(
            extract_delta(r#"{"choices":[{"delta":{"content":"Hel"}}]}"#).as_deref(),
            Some("Hel")
        );
        // The first SSE chunk usually carries only the role — skipped.
        assert_eq!(
            extract_delta(r#"{"choices":[{"delta":{"role":"assistant"}}]}"#),
            None
        );
        assert_eq!(extract_delta("not json"), None);
        assert_eq!(extract_delta(""), None);
    }

    #[test]
    fn finish_reason_is_tracked_and_length_is_an_error() {
        let chunk = r#"{"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}"#;
        assert_eq!(extract_finish_reason(chunk).as_deref(), Some("stop"));
        let chunk = r#"{"choices":[{"delta":{},"finish_reason":"length"}]}"#;
        assert_eq!(extract_finish_reason(chunk).as_deref(), Some("length"));
        let chunk = r#"{"choices":[{"delta":{"content":"x"}}]}"#;
        assert_eq!(extract_finish_reason(chunk), None);

        // Complete replies pass; truncation errors even with partial text
        // (a cut-off plan must not be published as a finished answer).
        assert!(finish_check("full".into(), Some("stop".into())).is_ok());
        assert!(finish_check("full".into(), None).is_ok());
        assert!(finish_check("partial".into(), Some("length".into())).is_err());
        assert!(finish_check(String::new(), Some("stop".into())).is_err());
    }

    #[test]
    fn extracts_reasoning_separately_from_content() {
        let reasoning = r#"{"choices":[{"delta":{"reasoning_content":"\nThe"}}]}"#;
        assert_eq!(extract_reasoning(reasoning).as_deref(), Some("\nThe"));
        // Reasoning chunks carry no content and vice versa.
        assert_eq!(extract_delta(reasoning), None);
        let content = r#"{"choices":[{"delta":{"content":"Hi"}}]}"#;
        assert_eq!(extract_reasoning(content), None);
    }

    struct NoopBackend;

    impl Backend for NoopBackend {
        fn chat(
            &self,
            _messages: &[ChatMessage],
        ) -> Pin<Box<dyn Future<Output = AgentResult<String>> + Send + '_>> {
            Box::pin(async { Ok("whole reply".to_string()) })
        }
    }

    #[test]
    fn thinking_param_maps_levels_to_budgets() {
        let low = thinking_param(ThinkingLevel::Low);
        assert_eq!(low["type"], "enabled");
        assert_eq!(low["budget_tokens"], ThinkingLevel::Low.budget_tokens());
        assert!(
            thinking_param(ThinkingLevel::High)["budget_tokens"]
                .as_u64()
                .unwrap()
                > thinking_param(ThinkingLevel::Medium)["budget_tokens"]
                    .as_u64()
                    .unwrap()
        );
    }

    #[test]
    fn chat_request_omits_thinking_unless_configured() {
        let msgs = vec![ChatMessage::user("hi".to_string())];
        let plain_backend = OpenAiCompatible::new("http://x/v1", "m");
        let plain = plain_backend.chat_request(&msgs, true);
        let value = serde_json::to_value(&plain).unwrap();
        assert!(
            value.get("thinking").is_none(),
            "plain models must not see the param"
        );

        let thinking_backend = OpenAiCompatible::new("http://x/v1", "kimi-k2-thinking")
            .with_thinking(ThinkingLevel::High);
        let thinking = thinking_backend.chat_request(&msgs, false);
        let value = serde_json::to_value(&thinking).unwrap();
        assert_eq!(value["thinking"]["budget_tokens"], 32_768);
        assert_eq!(value["model"], "kimi-k2-thinking");
        assert_eq!(value["stream"], false);
    }

    #[test]
    fn kimi_backend_uses_cloud_endpoint_and_keeps_key_out_of_config() {
        let config = AgentConfig {
            model: "kimi-k2-thinking".into(),
            backend: BackendKind::Kimi,
            thinking: Some(ThinkingLevel::Medium),
            ..Default::default()
        };
        // from_config must not resolve keys: the secret never sits in
        // AgentConfig (Debug-printed), only on the backend.
        let backend = OpenAiCompatible::from_config(&config);
        assert_eq!(backend.chat_request(&[], true).model, "kimi-k2-thinking");
        // Base URL comes from the env override or the Moonshot default.
        let default = OpenAiCompatible::kimi("m", None);
        let _ = default;
        let value = serde_json::to_value(backend.chat_request(&[], false)).unwrap();
        assert_eq!(value["thinking"]["budget_tokens"], 8_192);
    }

    #[tokio::test]
    async fn default_stream_delivers_the_reply_as_one_delta() {
        let (tx, mut rx) = mpsc::channel(4);
        let full = NoopBackend.chat_stream(&[], tx).await.unwrap();
        assert_eq!(full, "whole reply");
        assert_eq!(
            rx.recv().await,
            Some(StreamChunk::Text("whole reply".into()))
        );
    }
}
