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
    /// Optional multimodal input: ONE image alongside the text.
    /// Multimodal backends (SenseNova + mmproj) render it as an image
    /// input; text-only backends degrade to a placeholder and never
    /// see the key. `skip_serializing_if` keeps the wire format
    /// unchanged for text-only messages, so old peers stay compatible
    /// both directions (serde `default` fills `None` on read).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attachment: Option<Attachment>,
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

    /// Prompt-only message (never stored or gossiped): the agent's own
    /// earlier output within a turn (e.g. the tool call it made).
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::prompt_message(MessageKind::Agent, content)
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
            attachment: None,
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
            attachment: None,
            timestamp,
            kind,
        };
        msg.id = msg.compute_id();
        msg
    }

    /// Attach an image (multimodal input) and recompute the id, so two
    /// messages differing only by their attachment never share a DAG
    /// node.
    pub fn with_attachment(mut self, attachment: Attachment) -> Self {
        self.attachment = Some(attachment);
        self.id = self.compute_id();
        self
    }

    /// The image this message carries, if any: the multimodal input
    /// field first, else the payload of a Media-kind message (the
    /// chat-rendered form — see tui::send_attachment, which sets both).
    pub fn image_attachment(&self) -> Option<&Attachment> {
        if let Some(att) = &self.attachment {
            return Some(att);
        }
        match &self.kind {
            MessageKind::Media(att) if att.kind == AttachmentKind::Image => Some(att),
            _ => None,
        }
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
        if let Some(att) = &self.attachment {
            // The cid commits to the bytes; hashing it ties the
            // attachment into the message id.
            hasher.update(att.cid.as_bytes());
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn png() -> Attachment {
        Attachment {
            cid: "cid-abc".into(),
            filename: "plot.png".into(),
            mime_type: "image/png".into(),
            size_bytes: 100,
            kind: AttachmentKind::Image,
            width: Some(640),
            height: Some(480),
        }
    }

    #[test]
    fn attachment_field_is_optional_on_the_wire() {
        let msg = ChatMessage::user("hello");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(
            !json.contains("attachment"),
            "text-only messages keep the old wire shape: {json}"
        );
        // Old peers' payloads (no field at all) deserialize with None.
        let old = r#"{"id":"","topic":"","sender_id":"","sender_name":"","parent_id":null,"kind":"Human","content":"hi","timestamp":0}"#;
        let parsed: ChatMessage = serde_json::from_str(old).unwrap();
        assert!(parsed.attachment.is_none());
    }

    #[test]
    fn with_attachment_recomputes_the_id_and_roundtrips() {
        let base = ChatMessage::user("look");
        let with = ChatMessage::user("look").with_attachment(png());
        assert_ne!(base.id, with.id, "the attachment ties into the DAG id");
        assert_eq!(with.image_attachment().unwrap().cid, "cid-abc");
        let json = serde_json::to_string(&with).unwrap();
        let back: ChatMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(back.attachment.unwrap().filename, "plot.png");
    }

    #[test]
    fn image_attachment_falls_back_to_the_media_kind() {
        // Reloaded messages: the DB keeps the payload inside the kind
        // column (see Database::row_to_message), so the top-level field
        // is None after a reload — the kind fallback must cover it.
        let msg = ChatMessage::user("caption").with_attachment(png());
        let mut reloaded = msg.clone();
        reloaded.attachment = None;
        reloaded.kind = MessageKind::Media(png());
        assert_eq!(reloaded.image_attachment().unwrap().cid, "cid-abc");
        // No attachment anywhere -> None.
        assert!(ChatMessage::user("plain").image_attachment().is_none());
    }
}
