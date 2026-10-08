use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct CoreConfig {
    pub db_path: PathBuf,
    pub chat_topic: String,
}

impl CoreConfig {
    pub fn new(db_path: PathBuf, chat_topic: String) -> Self {
        CoreConfig {
            db_path,
            chat_topic,
        }
    }
}
