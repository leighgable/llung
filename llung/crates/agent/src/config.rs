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
            model: "olmo-7B-think".into(),
            name: "llung-agent".into(),
            model_url: None,
            max_history: 20,
        }
    }
}
