use llung_core::network::message::{AgentResult, Backend, ChatMessage};

pub struct OrtSession {
    model_path: String,
}

impl Backend for OrtSession {
    async fn chat(&self, messages: &[ChatMessage]) -> AgentResult<String> {
        // Build a simple prompt from the unified ChatMessage history
        let prompt = messages
            .iter()
            .map(|m| format!("{}: {}", m.agent_role(), m.content))
            .collect::<Vec<_>>()
            .join("\n");

        // TODO: tokenize, run ort inference, decode
        Ok(format!(
            "ONNX placeholder: processed {} messages",
            messages.len()
        ))
    }
}
