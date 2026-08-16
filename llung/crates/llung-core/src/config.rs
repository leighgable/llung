use std::path::PathBuf;
use crate::agent::AgentConfig

pub struct CoreConfig {
    pub db_path: PathBuf,
    pub agent: Option<AgentConfig>,
    pub chat_topic: String,
}
