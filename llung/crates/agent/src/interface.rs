use serde::{Deserialize, Serialize};

use llung_core::network::message::{AgentResult, Backend, ChatMessage};

/// A pluggable LLM backend. Implementations must be safe to call from
/// the agent bridge task. Boxed (not `impl Future`) so the trait stays
/// dyn-compatible for `Box<dyn LlmBackend>`.
pub trait LlmBackend: Send + Sync {
    fn chat(
        &self,
        messages: &[ChatMessage],
    ) -> std::pin::Pin<Box<dyn Future<Output = AgentResult<String>> + Send + '_>>;
}

/// Talks to any OpenAI-compatible `/v1/chat/completions` endpoint.
/// One implementation covers ollama (http://localhost:11434/v1),
/// vLLM, and llama.cpp's llama-server — they differ only in
/// base URL and model name.
pub struct OpenAiCompatible {
    base_url: String,
    model: String,
    client: reqwest::Client,
}

impl OpenAiCompatible {
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            model: model.into(),
            client: reqwest::Client::new(),
        }
    }
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    stream: bool,
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
    content: String,
}

impl LlmBackend for OpenAiCompatible {
    fn chat(
        &self,
        messages: &[ChatMessage],
    ) -> std::pin::Pin<Box<dyn Future<Output = AgentResult<String>> + Send + '_>> {
        // Clone what the request needs so the future owns its data and
        // no borrowed lifetimes leak into the boxed type.
        let model = self.model.clone();
        let base_url = self.base_url.clone();
        let client = self.client.clone();
        let messages = messages.to_vec();

        Box::pin(async move {
            let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));

            let response = client
                .post(url)
                .json(&ChatRequest {
                    model: &model,
                    messages: &messages,
                    stream: false,
                })
                .send()
                .await?
                .error_for_status()?;

            let body: ChatResponse = response.json().await?;

            body.choices
                .into_iter()
                .next()
                .map(|c| c.message.content)
                .ok_or_else(|| "LLM backend returned no choices".into())
        })
    }
}
