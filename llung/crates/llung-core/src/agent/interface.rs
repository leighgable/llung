use serde::{Deserialize, Serialize};

pub type AgentResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
        }
    }
}

/// A pluggable LLM backend. Implementations must be safe to call from
/// the agent bridge task.
pub trait LlmBackend: Send + Sync {
    fn chat(&self, messages: &[ChatMessage]) -> impl Future<Output = AgentResult<String>> + Send;
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
    async fn chat(&self, messages: &[ChatMessage]) -> AgentResult<String> {
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));

        let response = self
            .client
            .post(url)
            .json(&ChatRequest {
                model: &self.model,
                messages,
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
    }
}
