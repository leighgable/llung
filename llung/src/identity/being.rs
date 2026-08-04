use libp2p::{Multiaddr, PeerId};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::str::FromStr;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Being {
    #[serde(with = "peer_id_serde")]
    pub peer_id: PeerId,
    pub current_addr: Option<Multiaddr>,
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
