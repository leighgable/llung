use crate::storage::db::Database;
use libp2p::PeerId;
use rusqlite::types::{FromSql, FromSqlError, FromSqlResult, ToSql, ToSqlOutput, ValueRef};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{
    io::{self, Write},
    str::FromStr,
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type", content = "value")]
pub enum BeingStatus {
    Online,
    Away,
    Busy,
    Offline,
    Custom(String),
}

impl Default for BeingStatus {
    fn default() -> Self {
        Self::Online
    }
}

impl std::fmt::Display for BeingStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Online => write!(f, "online"),
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Being {
    #[serde(with = "peer_id_serde")]
    pub peer_id: PeerId,
    pub human_name: String,
    pub is_agent: bool,
    pub status: String,
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

pub fn get_or_create_local_being(
    db: &Database,
    local_peer_id: PeerId,
) -> Result<Being, Box<dyn std::error::Error>> {
    // 1. Check if local profile exists
    if let Some(existing_being) = db.get_being(&local_peer_id)? {
        println!("Loaded profile for: {}", existing_being.human_name);
        return Ok(existing_being);
    }

    // 2. Profile does not exist, query user for input
    println!("\n=== Initial Setup: ");

    print!("Enter your Display Name: ");
    io::stdout().flush()?;
    let mut human_name = String::new();
    io::stdin().read_line(&mut human_name)?;
    let human_name = human_name.trim().to_string();

    print!("Enter initial Status message (e.g., 'Online'): ");
    io::stdout().flush()?;
    let mut status = String::new();
    io::stdin().read_line(&mut status)?;
    let status = status.trim().to_string();

    let new_being = Being {
        peer_id: local_peer_id,
        human_name,
        is_agent: false, // Default to human user for local CLI profile
        status: if status.is_empty() {
            "Available".to_string()
        } else {
            status
        },
        avatar_cid: None,
    };

    // 3. Persist new profile to SQLite
    db.save_being(&new_being)?;
    println!("Profile saved to local database!\n");

    Ok(new_being)
}
