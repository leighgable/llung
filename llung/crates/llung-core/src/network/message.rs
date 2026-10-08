use ::serde::{Deserialize, Serialize};
use libp2p::PeerId;
use rusqlite::{
    ToSql,
    types::{FromSql, FromSqlError, FromSqlResult, ToSqlOutput, ValueRef},
};
use sha2::{Digest, Sha256};
use std::time::{SystemTime, UNIX_EPOCH};

pub type AgentResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

pub trait Backend: Send + Sync {
    fn chat(&self, messages: &[ChatMessage]) -> impl Future<Output = AgentResult<String>> + Send;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Attachment {
    pub cid: String,       // IPFS CID or SHA256 content hash
    pub filename: String,  // e.g., "diagram.png"
    pub mime_type: String, // e.g., "image/png"
    pub size_bytes: u64,   // Enables "Click to download (X MB)" UX
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageKind {
    Human,
    Agent,
    ToolCall {
        tool_name: String,
        arguments: String,
    },
    ToolResponse {
        tool_name: String,
        result_summary: String,
    },
    System, // joins, leaves, topic changes
    Notes,
}

// Stored in SQLite as JSON text, matching how BeingStatus is persisted.
impl ToSql for MessageKind {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        let json_str = serde_json::to_string(self)
            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
        Ok(ToSqlOutput::Owned(rusqlite::types::Value::Text(json_str)))
    }
}

impl FromSql for MessageKind {
    fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
        serde_json::from_str(value.as_str()?)
            .map_err(|e| FromSqlError::Other(Box::new(e)))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub id: String, // SHA256 of the message contents
    pub topic: String,
    pub sender_id: String, // base58 PeerId of the author
    pub sender_name: String,
    pub parent_id: Option<String>, // <--- creates branches
    pub kind: MessageKind,
    pub content: String,
    pub timestamp: u64,
}

impl ChatMessage {
    /// Prompt-only message (never stored or gossiped): a system instruction.
    pub fn system(content: impl Into<String>) -> Self {
        Self::prompt_message(MessageKind::System, content)
    }

    /// Prompt-only message (never stored or gossiped): user/turn input.
    pub fn user(content: impl Into<String>) -> Self {
        Self::prompt_message(MessageKind::Human, content)
    }

    fn prompt_message(kind: MessageKind, content: impl Into<String>) -> Self {
        Self {
            id: String::new(),
            topic: String::new(),
            sender_id: String::new(),
            sender_name: String::new(),
            parent_id: None,
            kind,
            content: content.into(),
            timestamp: 0,
        }
    }

    pub fn new(
        topic: String,
        sender: PeerId,
        parent_id: Option<String>,
        kind: MessageKind,
        sender_name: String,
        content: String,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut msg = Self {
            id: String::new(),
            topic: topic,
            sender_id: sender.to_base58(),
            sender_name,
            parent_id,
            content,
            timestamp,
            kind,
        };
        msg.id = msg.compute_id();
        msg
    }

    pub fn agent_role(&self) -> &'static str {
        match self.kind {
            MessageKind::System => "system",
            MessageKind::Human => "user",
            MessageKind::Agent => "assistant",
            MessageKind::ToolCall { .. } => "assistant",
            MessageKind::ToolResponse { .. } => "tool",
            MessageKind::Notes => "system",
        }
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectMessage(pub Vec<u8>);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectMessageResponse;
