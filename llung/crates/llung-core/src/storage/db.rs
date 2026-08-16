use crate::identity::being::{Being, BeingKind, BeingStatus};
use crate::network::message::ChatMessage;
use crate::storage::schema;
use libp2p::{PeerId, identity::Keypair};
use rusqlite::{Connection, Result, params};
use std::{
    str::FromStr,
    time::{SystemTime, UNIX_EPOCH},
};

pub struct Database {
    conn: Connection,
}

impl Database {
    pub fn new(db_path: &str) -> Result<(Self, Keypair)> {
        let mut conn = Connection::open(db_path)?;

        let tx = conn.transaction()?;

        schema::run_migrations(&tx)?;

        let keypair = Self::load_or_create_keypair(&tx, "node_identity")?;

        tx.commit()?;

        Ok((Self { conn }, keypair))
    }

    /// Open an additional connection to an existing database.
    /// Useful for async tasks (e.g. the agent bridge), since a rusqlite
    /// Connection cannot be shared across threads.
    pub fn open(db_path: &str) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        schema::run_migrations(&conn)?;
        Ok(Self { conn })
    }

    fn load_or_create_keypair(conn: &Connection, table: &str) -> Result<Keypair> {
        let mut stmt = conn.prepare(&format!("SELECT secret_bytes FROM {table} WHERE id = 1"))?;
        let mut rows = stmt.query([])?;

        if let Some(row) = rows.next()? {
            let secret_bytes: Vec<u8> = row.get(0)?;

            match Keypair::from_protobuf_encoding(&secret_bytes) {
                Ok(keypair) => Ok(keypair),
                Err(err) => {
                    eprintln!("⚠️ WARNING: Invalid identity in {table}: {err}");
                    eprintln!("⚠️ Generating new keypair and overwriting...");

                    let new_keypair = Keypair::generate_ed25519();
                    let new_bytes = new_keypair
                        .to_protobuf_encoding()
                        .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;

                    conn.execute(
                        &format!(
                            "INSERT OR REPLACE INTO {table} (id, secret_bytes) VALUES (1, ?1)"
                        ),
                        params![new_bytes],
                    )?;

                    Ok(new_keypair)
                }
            }
        } else {
            // First run: generate and persist fresh keypair
            let keypair = Keypair::generate_ed25519();
            let bytes = keypair
                .to_protobuf_encoding()
                .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;

            conn.execute(
                &format!("INSERT INTO {table} (id, secret_bytes) VALUES (1, ?1)"),
                params![bytes],
            )?;

            Ok(keypair)
        }
    }

    pub fn save_being(&self, being: &Being) -> rusqlite::Result<()> {
        self.conn.execute(
            "INSERT INTO beings (peer_id, human_name, kind, status, avatar_cid)
            VALUES (?1, ?2, ?3, ?4, ?5)
            ON CONFLICT(peer_id) DO UPDATE SET
                human_name = excluded.human_name,
                kind = excluded.kind,
                status = excluded.status,
                avatar_cid = excluded.avatar_cid",
            params![
                being.peer_id.to_base58(),
                being.human_name,
                being.kind,
                being.status,
                being.avatar_cid
            ],
        )?;
        Ok(())
    }
    /// Fetches a Being by PeerId if it exists in SQLite
    pub fn get_being(&self, peer_id: &PeerId) -> Result<Option<Being>> {
        let mut stmt = self.conn.prepare(
            "SELECT peer_id, human_name, kind, status, avatar_cid FROM beings WHERE peer_id = ?1",
        )?;

        let mut rows = stmt.query(params![peer_id.to_base58()])?;

        if let Some(row) = rows.next()? {
            let peer_id_str: String = row.get(0)?;
            let peer_id = PeerId::from_str(&peer_id_str).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })?;
            let human_name: String = row.get(1)?;
            let kind: BeingKind = row.get(2)?;
            let status: BeingStatus = row.get(3)?;
            let avatar_cid: Option<String> = row.get(4)?;

            Ok(Some(Being {
                peer_id,
                human_name,
                kind,
                status,
                avatar_cid,
            }))
        } else {
            Ok(None)
        }
    }
    /// Persist a chat message. INSERT OR IGNORE dedupes redeliveries
    /// (e.g. both the UI task and the agent task saving the same gossip).
    pub fn save_message(&self, msg: &ChatMessage) -> rusqlite::Result<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO messages (id, topic, sender_peer_id, parent_id, content, timestamp)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                msg.id,
                msg.topic,
                msg.sender_id,
                msg.parent_id,
                msg.content,
                msg.timestamp as i64
            ],
        )?;
        Ok(())
    }

    /// All messages for a topic, oldest first — feed into
    /// `ChatTree::build_from_flat_list`.
    pub fn get_messages(&self, topic: &str) -> Result<Vec<ChatMessage>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, topic, sender_peer_id, parent_id, content, timestamp
             FROM messages WHERE topic = ?1 ORDER BY timestamp ASC",
        )?;
        let rows = stmt.query_map(params![topic], Self::row_to_message)?;
        rows.collect()
    }

    /// The path from a message up to its root (i.e. one branch of the
    /// conversation), oldest first.
    pub fn get_branch(&self, message_id: &str) -> Result<Vec<ChatMessage>> {
        let mut stmt = self.conn.prepare(schema::BRANCH_SELECT)?;
        let rows = stmt.query_map(params![message_id], Self::row_to_message)?;
        rows.collect()
    }

    fn row_to_message(row: &rusqlite::Row<'_>) -> rusqlite::Result<ChatMessage> {
        Ok(ChatMessage {
            id: row.get(0)?,
            topic: row.get(1)?,
            sender_id: row.get(2)?,
            parent_id: row.get(3)?,
            content: row.get(4)?,
            timestamp: row.get::<_, i64>(5)? as u64,
        })
    }

    pub fn get_or_create_keypair(&self) -> rusqlite::Result<libp2p::identity::Keypair> {
        Self::load_or_create_keypair(&self.conn, "node_identity")
    }

    /// Identity keypair for the LLM agent Being, giving it its own
    /// PeerId (and thus its own gossipsub identity) distinct from the
    /// human operator's node.
    pub fn get_or_create_agent_keypair(&self) -> rusqlite::Result<libp2p::identity::Keypair> {
        Self::load_or_create_keypair(&self.conn, "agent_identity")
    }
    pub fn save_avatar_cache(&self, cid: &str, bytes: &[u8]) -> Result<()> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        self.conn.execute(
            "INSERT INTO avatar_cache (cid, image_bytes, created_at)
             VALUES (?1, ?2, ?3)
             ON CONFLICT(cid) DO NOTHING",
            params![cid, bytes, now],
        )?;
        Ok(())
    }
}
