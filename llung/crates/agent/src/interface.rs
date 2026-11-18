use std::future::Future;
use std::pin::Pin;

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use llung_core::network::message::{AgentResult, ChatMessage};

use crate::config::{AgentConfig, BackendKind, ThinkingLevel};

pub trait Backend: Send + Sync {
    fn chat(
        &self,
        messages: &[ChatMessage],
    ) -> Pin<Box<dyn Future<Output = AgentResult<String>> + Send + '_>>;

    /// Like `chat`, but forwards each response chunk as it arrives.
    /// The default falls back to `chat` and delivers the whole reply as
    /// one delta — streaming is an optimization, never required.
    fn chat_stream<'a>(
        &'a self,
        messages: &'a [ChatMessage],
        deltas: mpsc::Sender<String>,
    ) -> Pin<Box<dyn Future<Output = AgentResult<String>> + Send + 'a>> {
        Box::pin(async move {
            let full = self.chat(messages).await?;
            let _ = deltas.send(full.clone()).await;
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
    client: reqwest::Client,
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
            client: self.client.clone(),
        }
    }
}

impl OpenAiCompatible {
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        // Bound every stage: a backend that accepts but stalls (e.g. a
        // busy llama-server slot) would otherwise hang the agent's turn
        // forever — no timeout is reqwest's default.
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(10))
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            base_url: base_url.into(),
            model: model.into(),
            api_key: None,
            thinking: None,
            client,
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
        check(req.send().await).await.map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn llama_server(port: u16) -> Self {
        Self::new(format!("http://localhost:{port}/v1"), "SmolVLM")
    }

    /// llama.cpp llama-server hosting SmolVLM Q4 (the default backend).
    pub fn llamacpp(port: u16) -> Self {
        Self::llama_server(port)
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
                BackendKind::KimiCoding => {
                    Self::kimi_coding(config.model.clone(), config.thinking)
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

/// llung-specific fields (topic, sender, DAG parent, ...); the endpoint
/// only wants a role and content, so map explicitly instead of
/// serializing ChatMessage verbatim.
#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<RequestMessage<'a>>,
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
struct RequestMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Deserialize)]
struct ResponseMessage {
    content: Option<String>,
}

fn to_request_messages(messages: &[ChatMessage]) -> Vec<RequestMessage<'_>> {
    messages
        .iter()
        .map(|m| RequestMessage {
            role: m.agent_role(),
            content: &m.content,
        })
        .collect()
}

/// Send, then fail with the server's body included on non-2xx. A bare
/// reqwest status error ("401 Unauthorized for url") sends users down
/// the wrong debugging path; providers put the actual reason in the
/// JSON body ("Invalid Authentication", "model not found", ...).
async fn check(
    sent: Result<reqwest::Response, reqwest::Error>,
) -> Result<reqwest::Response, Box<dyn std::error::Error + Send + Sync>> {
    let response = sent?;
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
    fn chat_url(&self) -> String {
        format!("{}/chat/completions", self.base_url.trim_end_matches('/'))
    }

    fn chat_request<'a>(&'a self, messages: &'a [ChatMessage], stream: bool) -> ChatRequest<'a> {
        ChatRequest {
            model: &self.model,
            messages: to_request_messages(messages),
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
        deltas: mpsc::Sender<String>,
    ) -> Pin<Box<dyn Future<Output = AgentResult<String>> + Send + 'a>> {
        // Clone what the request needs so the future owns its data and
        // no borrowed lifetimes leak into the boxed type.
        let model = self.model.clone();
        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let thinking = self.thinking;
        let messages = messages.to_vec();
        let url = self.chat_url();

        Box::pin(async move {
            let request = ChatRequest {
                model: &model,
                messages: to_request_messages(&messages),
                stream: true,
                thinking: thinking.map(thinking_param),
            };
            let mut req = client.post(url).json(&request);
            if let Some(key) = &api_key {
                req = req.bearer_auth(key);
            }
            let mut response = check(req.send().await).await?;

            // SSE: `data: {json}\n\n` chunks, terminated by `data: [DONE]`.
            // Read byte chunks, split complete lines out of the buffer —
            // a chunk boundary can fall mid-line and mid-UTF-8.
            let mut full = String::new();
            let mut buf: Vec<u8> = Vec::new();
            while let Some(chunk) = response.chunk().await? {
                buf.extend_from_slice(&chunk);
                while let Some(pos) = buf.iter().position(|b| *b == b'\n') {
                    let line: Vec<u8> = buf.drain(..=pos).collect();
                    let line = String::from_utf8_lossy(&line);
                    let Some(data) = line.trim().strip_prefix("data:") else {
                        continue;
                    };
                    let data = data.trim();
                    if data == "[DONE]" {
                        return if full.is_empty() {
                            Err("model backend returned no content".into())
                        } else {
                            Ok(full)
                        };
                    }
                    if let Some(delta) = extract_delta(data) {
                        full.push_str(&delta);
                        let _ = deltas.send(delta).await;
                    } else if let Some(reasoning) = extract_reasoning(data) {
                        // Preview-only: keeps the chat alive during the
                        // thinking phase; the final reply replaces it.
                        let _ = deltas.send(reasoning).await;
                    }
                }
            }
            if full.is_empty() {
                Err("model backend stream ended without content".into())
            } else {
                Ok(full)
            }
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

        Box::pin(async move {
            let request = ChatRequest {
                model: &model,
                messages: to_request_messages(&messages),
                stream: false,
                thinking: thinking.map(thinking_param),
            };
            let mut req = client.post(url).json(&request);
            if let Some(key) = &api_key {
                req = req.bearer_auth(key);
            }
            let response = check(req.send().await).await?;

            let body: ChatResponse = response.json().await?;

            body.choices
                .into_iter()
                .next()
                .and_then(|c| c.message.content)
                .filter(|c| !c.is_empty())
                .ok_or_else(|| "model backend returned no content".into())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            thinking_param(ThinkingLevel::High)["budget_tokens"].as_u64().unwrap()
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
        assert!(value.get("thinking").is_none(), "plain models must not see the param");

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
        assert_eq!(rx.recv().await.as_deref(), Some("whole reply"));
    }
}
