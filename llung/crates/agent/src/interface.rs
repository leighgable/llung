use std::future::Future;
use std::pin::Pin;

use serde::{Deserialize, Serialize};

use llung_core::network::message::{AgentResult, ChatMessage};

use crate::config::AgentConfig;

pub trait Backend: Send + Sync {
    fn chat(
        &self,
        messages: &[ChatMessage],
    ) -> Pin<Box<dyn Future<Output = AgentResult<String>> + Send + '_>>;
}

pub struct OpenAiCompatible {
    base_url: String,
    model: String,
    client: reqwest::Client,
}

/// Cheap to clone: two strings plus a shared connection pool. Lets the
/// TUI hand one backend to the agent's chat loop and another to the
/// history compressor.
impl Clone for OpenAiCompatible {
    fn clone(&self) -> Self {
        Self {
            base_url: self.base_url.clone(),
            model: self.model.clone(),
            client: self.client.clone(),
        }
    }
}

impl OpenAiCompatible {
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            model: model.into(),
            client: reqwest::Client::new(),
        }
    }

    pub fn llama_server(port: u16) -> Self {
        Self::new(format!("http://localhost:{port}/v1"), "SmolVLM")
    }

    /// Build a backend from an agent config: an explicit `model_url`
    /// wins, otherwise fall back to the default llama-server port.
    pub fn from_config(config: &AgentConfig) -> Self {
        match &config.model_url {
            Some(url) => Self::new(url.clone(), config.model.clone()),
            None => Self::llama_server(9931),
        }
    }
}

/// Wire format for `/v1/chat/completions`. ChatMessage carries a lot of
/// llung-specific fields (topic, sender, DAG parent, ...); the endpoint
/// only wants a role and content, so map explicitly instead of
/// serializing ChatMessage verbatim.
#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<RequestMessage<'a>>,
    stream: bool,
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

impl Backend for OpenAiCompatible {
    fn chat(
        &self,
        messages: &[ChatMessage],
    ) -> Pin<Box<dyn Future<Output = AgentResult<String>> + Send + '_>> {
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
                    messages: to_request_messages(&messages),
                    stream: false,
                })
                .send()
                .await?
                .error_for_status()?;

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
