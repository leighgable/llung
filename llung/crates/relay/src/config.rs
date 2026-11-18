// crates/relay/src/config.rs
//
// Environment-driven configuration. Everything has a sane default so the
// binary runs bare on a fresh VPS; env vars tune it.

use std::path::PathBuf;
use std::time::Duration;

pub struct RelayConfig {
    /// TCP + QUIC listen port.
    pub port: u16,
    /// Persistent server identity (PEM-ish protobuf libp2p encoding).
    /// Generate once, back it up: the PeerId is baked into every
    /// client's relay multiaddr.
    pub key_path: PathBuf,
    /// SQLite image cache (avatars, message images).
    pub db_path: PathBuf,
    /// Maximum cached images; oldest-expiring evicted past this.
    pub max_images: usize,
    /// How long a cached image lives without being re-put.
    pub image_ttl: Duration,
    /// Circuit relay v2 limits (per server / per peer).
    pub max_reservations: usize,
    pub max_circuits_per_peer: usize,
    /// PUT floor: one peer can push at most this many images per
    /// connection minute... kept simple: a per-peer burst cap.
    pub max_puts_per_peer: u32,
}

fn data_dir() -> PathBuf {
    if let Ok(custom) = std::env::var("LLUNG_DATA_DIR") {
        if !custom.is_empty() {
            return PathBuf::from(custom);
        }
    }
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("llung")
}

fn env_u16(name: &str, default: u16) -> u16 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_secs(name: &str, default: u64) -> Duration {
    Duration::from_secs(
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default),
    )
}

impl RelayConfig {
    pub fn from_env() -> Self {
        let dir = data_dir().join("relay");
        Self {
            port: env_u16("LLUNG_RELAY_PORT", 4001),
            key_path: std::env::var("LLUNG_RELAY_KEY")
                .map(PathBuf::from)
                .unwrap_or_else(|_| dir.join("relay-key.pb")),
            db_path: std::env::var("LLUNG_RELAY_DB")
                .map(PathBuf::from)
                .unwrap_or_else(|_| dir.join("images.db")),
            max_images: env_usize("LLUNG_RELAY_MAX_IMAGES", 4096),
            image_ttl: env_secs("LLUNG_RELAY_IMAGE_TTL_SECS", 30 * 24 * 3600),
            max_reservations: env_usize("LLUNG_RELAY_MAX_RESERVATIONS", 512),
            max_circuits_per_peer: env_usize("LLUNG_RELAY_MAX_CIRCUITS_PER_PEER", 8),
            max_puts_per_peer: env_u16("LLUNG_RELAY_MAX_PUTS_PER_PEER", 60) as u32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_sane() {
        // No env guarantees in a test environment: just ensure parsing
        // doesn't panic and the struct is buildable.
        let cfg = RelayConfig::from_env();
        assert!(cfg.port > 0);
        assert!(cfg.max_images > 0);
        assert!(cfg.image_ttl.as_secs() > 0);
        assert!(cfg.max_puts_per_peer > 0);
    }
}
