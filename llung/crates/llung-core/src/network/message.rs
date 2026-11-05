use ::serde::{Deserialize, Serialize};
use libp2p::PeerId;
use mime_guess;
use rusqlite::{
    ToSql,
    types::{FromSql, FromSqlError, FromSqlResult, ToSqlOutput, ValueRef},
};
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    time::{SystemTime, UNIX_EPOCH},
};
pub type AgentResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub enum AttachmentKind {
    Image,
    Audio,
    Video,
    Document,
    Unknown,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Attachment {
    pub cid: String,       // IPFS CID or SHA256 content hash
    pub filename: String,  // e.g., "diagram.png"
    pub mime_type: String, // e.g., "image/png"
    pub size_bytes: u64,   // Enables "Click to download (X MB)" UX
    pub kind: AttachmentKind,
    pub width: Option<u32>,
    pub height: Option<u32>,
}

impl Attachment {
    pub fn from_path(path: &str, cid: String) -> Result<Self, Box<dyn Error>> {
        let metadata = std::fs::metadata(path)?;
        let mime = mime_guess::from_path(path)
            .first_or_octet_stream()
            .to_string();
        let (width, height) = if mime.starts_with("image/") {
            let img = image::open(path)?;
            (Some(img.width()), Some(img.height()))
        } else {
            (None, None)
        };
        Ok(Self {
            cid,
            filename: std::path::Path::new(path)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string(),
            mime_type: mime.clone(),
            size_bytes: metadata.len(),
            kind: Self::detect_kind(&mime),
            width,
            height,
        })
    }

    fn detect_kind(mime: &str) -> AttachmentKind {
        if mime.starts_with("image/") {
            AttachmentKind::Image
        } else if mime.starts_with("audio/") {
            AttachmentKind::Audio
        } else if mime.starts_with("video/") {
            AttachmentKind::Video
        } else if mime.starts_with("application/pdf") || mime == "text/plain" {
            AttachmentKind::Document
        } else {
            AttachmentKind::Unknown
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageKind {
    Human,
    Agent,
    ToolCall {
        tool_name: String,
        arguments: String,
    },
    /// Result of a tool run. `parent_id` of the carrying `ChatMessage`
    /// points at the `ToolCall` message, so viewers can replace the
    /// pending call with its result. `content` holds the tool output
    /// (e.g. a `braille-v1` render document) or the error text.
    ToolResponse {
        tool_name: String,
        result_summary: String,
        /// The tool failed; `content` is the error. Defaulted so stored
        /// pre-existing responses still deserialize.
        #[serde(default)]
        error: bool,
    },
    System, // joins, leaves, topic changes
    Notes,
    Media(Attachment),
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
        serde_json::from_str(value.as_str()?).map_err(|e| FromSqlError::Other(Box::new(e)))
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
            MessageKind::Media(_) => "user",
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
