use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct CoreConfig {
    pub db_path: PathBuf,
    pub chat_topic: String,
    /// Relay servers to stay dialed to (multiaddr strings, e.g.
    /// `/ip4/1.2.3.4/tcp/4001/p2p/<relay-peer-id>`). Relays carry
    /// gossipsub between NAT'd peers, enable hole punching, and cache
    /// lightweight images (avatars, message images).
    pub relays: Vec<String>,
}

impl CoreConfig {
    pub fn new(db_path: PathBuf, chat_topic: String) -> Self {
        CoreConfig {
            db_path,
            chat_topic,
            relays: Vec::new(),
        }
    }

    pub fn with_relays(mut self, relays: Vec<String>) -> Self {
        self.relays = relays;
        self
    }

    /// Relay multiaddrs from `LLUNG_RELAY` (comma-separated), the cheap
    /// way to point every node at the same relay on a €5 VPS.
    pub fn relays_from_env() -> Vec<String> {
        std::env::var("LLUNG_RELAY")
            .unwrap_or_default()
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect()
    }
}
