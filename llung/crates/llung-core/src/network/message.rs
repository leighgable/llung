use ::serde;
use libp2p::PeerId;
use sha2::{Digest, Sha256};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Attachment {
    pub cid: String,       // IPFS CID or SHA256 content hash
    pub filename: String,  // e.g., "diagram.png"
    pub mime_type: String, // e.g., "image/png"
    pub size_bytes: u64,   // Enables "Click to download (X MB)" UX
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ChatMessage {
    pub id: String, // SHA256 of the message contents
    pub topic: String,
    pub sender_id: String, // base58 PeerId of the author
    pub parent_id: Option<String>, // <--- creates branches
    pub content: String,
    pub timestamp: u64,
}

impl ChatMessage {
    pub fn new(topic: &str, sender: PeerId, parent_id: Option<String>, content: String) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut msg = Self {
            id: String::new(),
            topic: topic.to_string(),
            sender_id: sender.to_base58(),
            parent_id,
            content,
            timestamp,
        };
        msg.id = msg.compute_id();
        msg
    }

    /// Content-addressed ID over everything that makes this message unique.
    /// The whole serialized message is signed by gossipsub, so the
    /// self-reported sender_id can't be forged.
    pub fn compute_id(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.topic.as_bytes());
        hasher.update(self.sender_id.as_bytes());
        hasher.update(self.parent_id.as_deref().unwrap_or("").as_bytes());
        hasher.update(self.content.as_bytes());
        hasher.update(self.timestamp.to_be_bytes());
        hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }
}
