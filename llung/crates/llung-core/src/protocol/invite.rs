use serde::{Deserialize, Serialize};

/// Payload sent over an encrypted direct message when inviting a peer
/// to join a topic/chat.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvitePayload {
    pub topic_id: String,
    pub encryption_key: String,
    pub display_name: String,
    pub invited_by: String,
    pub timestamp: i64,
}
