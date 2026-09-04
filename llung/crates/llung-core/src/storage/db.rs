use crate::identity::{
    being::{Being, BeingKind, BeingStatus},
    machine::MachineIdentity,
};
use crate::network::message::ChatMessage;
use crate::storage::schema;
use libp2p::{PeerId, identity::Keypair};
use rusqlite::{Connection, Result, params};
use std::{
    error::Error,
    time::{SystemTime, UNIX_EPOCH},
};

pub struct Database {
    conn: Connection,
}

impl Database {
    pub fn new(db_path: &str) -> Result<(Self, MachineIdentity), Box<dyn Error>> {
        let mut conn: Connection = Connection::open(db_path)?;

        let tx = conn.transaction()?;

        schema::run_migrations(&tx)?;

        let machine: MachineIdentity = Self::get_or_create_machine_identity(&tx)?;

        tx.commit()?;

        Ok((Self { conn }, machine))
    }

    pub fn create_machine(&self, device_name: &str) -> Result<MachineIdentity, Box<dyn Error>> {
        let keypair = Keypair::generate_ed25519();
        let peer_id = keypair.public().to_peer_id();
        let secret = keypair.to_protobuf_encoding()?;

        self.conn.execute(
            "INSERT INTO machines (peer_id, being_id, device_name, secret_key)
            VALUES (?1, NULL, ?2, ?3)",
            (&peer_id.to_base58(), device_name, &secret),
        )?;

        Ok(MachineIdentity {
            keypair,
            peer_id,
            device_name: device_name.to_string(),
            being_id: String::new(), // filled later
        })
    }

    pub fn create_being_for_machine(
        &self,
        peer_id: &str,
        human_name: &str,
        kind: BeingKind,
    ) -> Result<Being, Box<dyn Error>> {
        let keypair = Keypair::generate_ed25519();
        let being_id = keypair.public().to_peer_id().to_base58();
        let secret = keypair.to_protobuf_encoding()?;

        self.conn.execute(
            "INSERT INTO beings (being_id, human_name, kind, status, avatar_cid, secret_key)
            VALUES (1?, 2?, 3?, 4?, NULL, 75)",
            (&being_id, human_name, kind, BeingStatus::Available, &secret),
        )?;

        self.conn.execute(
            "UPDATE machines SET being_id = ?1 WHERE peer_id = ?2",
            (&being_id, peer_id),
        )?;

        Ok(Being {
            being_id,
            human_name: human_name.to_string(),
            kind,
            status: BeingStatus::Available,
            avatar_cid: None,
        })
    }

    fn get_or_create_machine_identity(
        conn: &Connection,
    ) -> Result<MachineIdentity, Box<dyn Error>> {
        // Try to load an existing device keypair for this DB file
        let mut stmt = conn.prepare("SELECT peer_id, secret_key FROM machines LIMIT 1")?;
        if let Some(row) = stmt
            .query_map([], |row| {
                let peer_id: String = row.get(0)?;
                let secret: Vec<u8> = row.get(1)?;
                Ok((peer_id, secret))
            })?
            .next()
        {
            let (peer_id_str, secret) = row?;
            let keypair = Keypair::from_protobuf_encoding(&secret)?; // or ed25519 bytes
            let peer_id = keypair.public().to_peer_id();
            return Ok(MachineIdentity {
                keypair,
                peer_id,
                device_name: "default".into(), // or load from db
                being_id: String::new(),       // filled later by session.rs
            });
        }
        // Generate fresh device keypair
        let keypair = Keypair::generate_ed25519();
        let peer_id = keypair.public().to_peer_id();
        let secret = keypair
            .to_protobuf_encoding()
            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?; // or ed25519 bytes
        conn.execute(
            &format!("INSERT INTO machines (peer_id, being_id, machine_name, secret_key) VALUES (?1, NULL, 'default', ?2)"),
            (&peer_id.to_base58(), &secret),
        )?;

        Ok(MachineIdentity {
            keypair,
            peer_id,
            device_name: "default".into(),
            being_id: String::new(),
        })
    }

    pub fn create_being(
        &self,
        human_name: &str,
        kind: BeingKind,
        avatar_cid: Option<&str>,
    ) -> Result<Being, Box<dyn Error>> {
        let keypair: Keypair = Keypair::generate_ed25519();
        let being_id: String = keypair.public().to_peer_id().to_base58();
        let secret = keypair.to_protobuf_encoding()?;

        self.conn.execute(
            "INSERT INTO beings (being_id, human_name, kind, status, avatar_cid, secret_key)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            (
                &being_id,
                human_name,
                kind,
                BeingStatus::Available,
                avatar_cid,
                &secret,
            ),
        )?;
        self.conn.execute(
            "UPDATE machines SET being_id = ?1 WHERE being_id IS NULL",
            [&being_id],
        )?;

        Ok(Being {
            being_id,
            human_name: human_name.to_string(),
            kind,
            status: BeingStatus::Available,
            avatar_cid: avatar_cid.map(|s| s.to_string()),
        })
    }

    /// Loads existing user or returns None (caller must run registration flow).
    ///
    /// Resolve a network PeerId to the stable Being that owns it.
    pub fn get_being_by_peer_id(
        &self,
        peer_id: &PeerId,
    ) -> Result<Option<Being>, Box<dyn std::error::Error>> {
        let mut stmt = self.conn.prepare(
            "SELECT b.being_id, b.human_name, b.kind, b.status, b.avatar_cid
             FROM beings b
             JOIN machines d ON b.being_id = d.being_id
             WHERE d.peer_id = ?1",
        )?;

        let mut rows = stmt.query_map([&peer_id.to_base58()], |row| {
            Ok(Being {
                being_id: row.get(0)?,
                human_name: row.get(1)?,
                kind: row.get(2)?,
                status: row.get(3)?,
                avatar_cid: row.get(4)?,
            })
        })?;

        rows.next().transpose().map_err(|e| e.into())
    }

    pub fn load_being(&self) -> Result<Option<Being>, Box<dyn Error>> {
        let result = self.conn.query_row(
            "SELECT being_id, human_name, kind, status, avatar_cid FROM beings LIMIT 1",
            [], // no query parameters
            |row| {
                Ok(Being {
                    being_id: row.get(0)?,
                    human_name: row.get(1)?,
                    kind: row.get(2)?,
                    status: row.get(3)?,
                    avatar_cid: row.get(4)?,
                })
            },
        );

        match result {
            Ok(being) => Ok(Some(being)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(Box::new(e)),
        }
    }

    /// Called once during registration.
    pub fn create_being_identity(
        &self,
        name: &str,
        avatar_cid: Option<&str>,
    ) -> Result<Being, Box<dyn Error>> {
        let keypair = Keypair::generate_ed25519(); // master user key
        let being_id = keypair.public().to_peer_id().to_base58(); // reuse base58 for convenience
        let secret = keypair
            .to_protobuf_encoding()
            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?; // or ed25519 bytes

        self.conn.execute(
            "INSERT INTO beings (being_id, human_name, kind, status, avatar_cid, secret_key)
             VALUES (?1, ?2, 'human', '{\"type\":\"available\"}', ?3, ?4)",
            (&being_id, name, &avatar_cid, &secret),
        )?;

        // Link the current device to this user
        self.conn.execute(
            "UPDATE machines SET being_id = ?1 WHERE being_id = ''",
            [&being_id],
        )?;

        Ok(Being {
            being_id,
            human_name: name.into(),
            kind: BeingKind::Human,
            status: BeingStatus::Available,
            avatar_cid: avatar_cid.map(|s| s.into()),
        })
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
                    tracing::info!("⚠️ WARNING: Invalid identity in {table}: {err}");
                    tracing::info!("⚠️ Generating new keypair and overwriting...");

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
            "INSERT INTO beings (being_id, human_name, kind, status, avatar_cid)
            VALUES (?1, ?2, ?3, ?4, ?5)
            ON CONFLICT(being_id) DO UPDATE SET
                human_name = excluded.human_name,
                kind = excluded.kind,
                status = excluded.status,
                avatar_cid = excluded.avatar_cid",
            params![
                being.being_id,
                being.human_name,
                being.kind,
                being.status,
                being.avatar_cid
            ],
        )?;
        Ok(())
    }
    /// Fetches a Being by PeerId if it exists in SQLite
    pub fn get_being(&self, being_id: &str) -> Result<Option<Being>> {
        let mut stmt = self.conn.prepare(
            "SELECT peer_id, human_name, kind, status, avatar_cid FROM beings WHERE being_id = ?1",
        )?;

        let mut rows = stmt.query(params![being_id])?;

        if let Some(row) = rows.next()? {
            let being_id: String = row.get(0)?;
            let human_name: String = row.get(1)?;
            let kind: BeingKind = row.get(2)?;
            let status: BeingStatus = row.get(3)?;
            let avatar_cid: Option<String> = row.get(4)?;

            Ok(Some(Being {
                being_id,
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

    pub fn get_avatar_cache(
        &self,
        cid: &str,
    ) -> Result<Option<Vec<u8>>, Box<dyn std::error::Error>> {
        let mut stmt = self
            .conn
            .prepare("SELECT bytes FROM avatar_cache WHERE cid = ?1")?;
        let mut rows = stmt.query_map([cid], |row| row.get::<_, Vec<u8>>(0))?;
        Ok(rows.next().transpose()?)
    }
}
