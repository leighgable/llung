#[derive(Clone, Debug)]
pub struct AgentConfig {
    pub model: String,
    pub name: String,
    pub model_url: Option<String>,
    pub max_history: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            model: "SmolVLM".into(),
            name: "llung-agent".into(),
            // llama.cpp's llama-server hosting SmolVLM Q4.
            model_url: Some("http://localhost:9931/v1".into()),
            max_history: 20,
        }
    }
}
