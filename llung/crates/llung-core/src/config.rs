use crate::agent::AgentConfig;
use std::path::PathBuf;

pub struct CoreConfig {
    pub db_path: PathBuf,
    pub agent: Option<AgentConfig>,
    pub chat_topic: String,
}

impl CoreConfig {
    pub fn new(db_path: PathBuf, agent: Option<AgentConfig>, chat_topic: String) -> Self {
        CoreConfig {
            db_path,
            agent,
            chat_topic,
        }
    }
}
