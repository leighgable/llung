use ::serde;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Attachment {
    pub cid: String,       // IPFS CID or SHA256 content hash
    pub filename: String,  // e.g., "diagram.png"
    pub mime_type: String, // e.g., "image/png"
    pub size_bytes: u64,   // Enables "Click to download (X MB)" UX
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ChatMessage {
    pub id: String, // e.g., SHA256 of the payload
    pub topic: String,
    pub sender_id: String,
    pub parent_id: Option<String>, // <--- creates branches
    pub content: String,
    pub timestamp: u64,
}

// CREATE TABLE messages (
//     id TEXT PRIMARY KEY,       -- A hash of the message content + parent_id
//     topic TEXT NOT NULL,
//     sender_peer_id TEXT NOT NULL,
//     parent_id TEXT,            -- NULL if it's the root of a conversation
//     content TEXT NOT NULL,
//     timestamp INTEGER NOT NULL
// );

// ** Get your conversation back! **
// WITH RECURSIVE branch_path AS (
//     -- Base case: start with the specific message
//     SELECT id, parent_id, content, timestamp
//     FROM messages
//     WHERE id = ?

//     UNION ALL

//     -- Recursive step: join with the parent message
//     SELECT m.id, m.parent_id, m.content, m.timestamp
//     FROM messages m
//     JOIN branch_path bp ON m.id = bp.parent_id
// )
// SELECT * FROM branch_path ORDER BY timestamp ASC;

// -- Main message store
// CREATE TABLE messages (
//     id TEXT PRIMARY KEY,
//     topic TEXT NOT NULL,
//     sender_peer_id TEXT NOT NULL,
//     parent_id TEXT,
//     content TEXT NOT NULL,
//     timestamp INTEGER NOT NULL
// );

// -- Separate asset store for managing P2P transfers and disk cache
// CREATE TABLE media_assets (
//     cid TEXT PRIMARY KEY,
//     filename TEXT NOT NULL,
//     mime_type TEXT NOT NULL,
//     size_bytes INTEGER NOT NULL,
//     local_path TEXT,          -- NULL if not downloaded yet
//     download_status TEXT NOT NULL DEFAULT 'remote' -- 'remote', 'downloading', 'ready', 'failed'
// );

// -- Many-to-many relationship between messages and media
// CREATE TABLE message_attachments (
//     message_id TEXT NOT NULL,
//     cid TEXT NOT NULL,
//     PRIMARY KEY (message_id, cid),
//     FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE,
//     FOREIGN KEY (cid) REFERENCES media_assets(cid) ON DELETE CASCADE
// );
