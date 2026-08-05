use crate::identity::being::Being;
use crate::storage::schema;
use libp2p::PeerId;
use rusqlite::{Connection, Result, params};
use std::error::Error;
use std::str::FromStr;

pub struct Database {
    conn: Connection,
}

impl Database {
    pub fn new(db_path: &str) -> Result<Self> {
        let conn = Connection::open(db_path)?;

        schema::run_migrations(&conn)?;
        Ok(Self { conn })
    }

    pub fn save_being(&self, being: &Being) -> rusqlite::Result<()> {
        self.conn.execute(
            "INSERT INTO beings (peer_id, human_name, is_agent, status, avatar_cid)
            VALUES (?1, ?2, ?3, ?4, ?5)
            ON CONFLICT(peer_id) DO UPDATE SET
                human_name = excluded.human_name,
                is_agent = excluded.is_agent,
                status = excluded.status,
                avatar_cid = excluded.avatar_cid",
            params![
                being.peer_id.to_base58(),
                being.human_name,
                being.is_agent,
                being.status,
                being.avatar_cid
            ],
        )?;
        Ok(())
    }
    /// Fetches a Being by PeerId if it exists in SQLite
    pub fn get_being(&self, peer_id: &PeerId) -> Result<Option<Being>> {
        let mut stmt = self.conn.prepare(
            "SELECT peer_id, human_name, is_agent, status, avatar_cid FROM beings WHERE peer_id = ?1",
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

            Ok(Some(Being {
                peer_id,
                human_name: row.get(1)?,
                is_agent: row.get(2)?,
                status: row.get(3)?,
                avatar_cid: row.get(4)?,
            }))
        } else {
            Ok(None)
        }
    }
}
