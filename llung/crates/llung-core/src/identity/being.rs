use crate::storage::db::Database;
use crate::utils::{generate_cid_from_bytes, process_avatar_thumbnail};
use libp2p::{
    PeerId,
    gossipsub::{IdentTopic, PublishError},
};
use rusqlite::types::{FromSql, FromSqlError, FromSqlResult, ToSql, ToSqlOutput, ValueRef};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{
    error::Error,
    io::{self, Write},
    str::FromStr,
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BeingKind {
    Human,
    Agent,
}

impl Default for BeingKind {
    fn default() -> Self {
        Self::Human
    }
}

impl std::fmt::Display for BeingKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Human => write!(f, "human"),
            Self::Agent => write!(f, "agent"),
        }
    }
}

// Stored in SQLite as a plain "human"/"agent" string (not JSON),
// so the column stays readable and the 'human' DEFAULT stays simple.
impl ToSql for BeingKind {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        Ok(ToSqlOutput::Owned(rusqlite::types::Value::Text(
            self.to_string(),
        )))
    }
}

impl FromSql for BeingKind {
    fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
        match value.as_str()? {
            "human" => Ok(Self::Human),
            "agent" => Ok(Self::Agent),
            other => Err(FromSqlError::Other(
                format!("unknown being kind: {other}").into(),
            )),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type", content = "value")]
pub enum BeingStatus {
    Available,
    Away,
    Busy,
    Offline,
    Custom(String),
}

impl Default for BeingStatus {
    fn default() -> Self {
        Self::Available
    }
}

impl std::fmt::Display for BeingStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Available => write!(f, "available"),
            Self::Away => write!(f, "away"),
            Self::Busy => write!(f, "busy"),
            Self::Offline => write!(f, "offline"),
            Self::Custom(msg) => write!(f, "{msg}"),
        }
    }
}

impl ToSql for BeingStatus {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        let json_str = serde_json::to_string(self)
            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
        Ok(ToSqlOutput::Owned(rusqlite::types::Value::Text(json_str)))
    }
}

impl FromSql for BeingStatus {
    fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
        let text = value.as_str()?;
        serde_json::from_str(text).map_err(|e| FromSqlError::Other(Box::new(e)))
    }
}

pub fn broadcast_presence(
    swarm: &mut libp2p::Swarm<crate::network::LlungBehaviour>,
    local_being: &Being,
    topic: &IdentTopic,
) -> Result<(), Box<dyn Error>> {
    let payload = PresenceMessage::from(local_being);
    let bytes = serde_json::to_vec(&payload)?;

    // Publish message to the Gossipsub topic
    match swarm
        .behaviour_mut()
        .gossipsub
        .publish(topic.clone(), bytes)
    {
        Ok(msg_id) => {
            println!(
                "Broadcasted presence for {} (id: {})",
                local_being.human_name, msg_id
            );
        }
        Err(PublishError::Duplicate) => {
            // Silently ignore if this exact presence message was already published
        }
        Err(e) => {
            eprintln!("Failed to publish presence: {e:?}");
        }
    }

    Ok(())
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PresenceMessage {
    pub peer_id: String,
    pub human_name: String,
    pub kind: BeingKind,
    pub status: BeingStatus,
    pub avatar_cid: Option<String>,
}

impl From<&Being> for PresenceMessage {
    fn from(being: &Being) -> Self {
        Self {
            peer_id: being.peer_id.to_base58(),
            human_name: being.human_name.clone(),
            kind: being.kind.clone(),
            status: being.status.clone(),
            avatar_cid: being.avatar_cid.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Being {
    #[serde(with = "peer_id_serde")]
    pub being_id: PeerId,
    pub human_name: String,
    pub kind: BeingKind,
    pub status: BeingStatus,
    pub avatar_cid: Option<String>,
}

mod peer_id_serde {
    use super::*;

    pub fn serialize<S>(peer_id: &PeerId, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&peer_id.to_base58())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<PeerId, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        PeerId::from_str(&s).map_err(serde::de::Error::custom)
    }
}

/// Create a brand-new local Being, optionally processing an avatar image
/// into a thumbnail that is stored locally and referenced by CID.
///
/// This only stages the bytes locally (avatar_cache + beings tables).
/// Announcing the CID to the network is the caller's job (e.g. by sending
/// `NetworkCommand::ProvideMediaCid` once the engine is running).
pub fn create_local_being(
    db: &Database,
    local_peer_id: PeerId,
    human_name: String,
    avatar_path: Option<&str>,
    kind: BeingKind,
) -> Result<Being, Box<dyn std::error::Error>> {
    let mut avatar_cid = None;

    if let Some(path) = avatar_path {
        let thumbnail_bytes = process_avatar_thumbnail(path, 128)?;
        let cid = generate_cid_from_bytes(&thumbnail_bytes)?;

        // Keep the actual bytes local; peers will fetch them by CID.
        db.save_avatar_cache(&cid, &thumbnail_bytes)?;
        println!("Avatar staged with CID: {cid}");

        avatar_cid = Some(cid);
    }

    let being = Being {
        peer_id: local_peer_id,
        human_name,
        kind,
        status: BeingStatus::Available,
        avatar_cid,
    };

    db.save_being(&being)?;
    println!("Saved.\n");

    Ok(being)
}

pub fn load_local_being(
    db: &Database,
    local_peer_id: PeerId,
) -> Result<Being, Box<dyn std::error::Error>> {
    match db.get_being(&local_peer_id)? {
        Some(being) => {
            println!("Loading {}...", being.human_name);
            println!("Peer Id: {}", being.peer_id);
            Ok(being)
        }
        None => {
            Err("Identity data missing from the database. The database may be corrupted.".into())
        }
    }
}

pub fn create_local_being_interactive(
    db: &Database,
    local_peer_id: PeerId,
    kind: BeingKind,
) -> Result<Being, Box<dyn std::error::Error>> {
    println!("\n==========================================");
    println!("  -------- Welcome to Llung —--------     ");
    println!("==========================================");

    print!("Enter your Display Name: ");
    io::stdout().flush()?;
    let mut human_name = String::new();
    io::stdin().read_line(&mut human_name)?;
    let human_name = human_name.trim().to_string();

    print!("Enter Avatar image path (optional, press Enter to skip): ");
    io::stdout().flush()?;
    let mut avatar_input = String::new();
    io::stdin().read_line(&mut avatar_input)?;
    let avatar_path = match avatar_input.trim() {
        "" => None,
        path => Some(path),
    };

    create_local_being(db, local_peer_id, human_name, avatar_path, kind)
}

pub fn get_or_create_local_being(
    db: &Database,
    local_peer_id: PeerId,
) -> Result<Being, Box<dyn std::error::Error>> {
    if let Some(existing) = db.get_being(&local_peer_id)? {
        println!("Welcome, {}.", existing.human_name);
        println!("Peer Id: {}", existing.peer_id);
        return Ok(existing);
    }
    create_local_being_interactive(db, local_peer_id, BeingKind::Human)
}
