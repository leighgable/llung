// llung-core/src/network/sync.rs
//
// History exchange: the `/llung/sync/1.0.0` request-response protocol.
// Gossipsub carries LIVE messages; this fills the DAG backwards when two
// nodes meet — most importantly the devices of ONE account, which find
// each other through relay-circuit addresses advertised in presence.
//
// - `Chain` walks a branch's ancestors newest-first, `limit` messages
//   per request; `more_parent` lets the requester continue the walk
//   until genesis. Depth-capped by the engine so a loop can't run away.
// - `Missing` fetches concrete ids — siblings on parallel branches the
//   walk doesn't traverse.
//
// Trust boundary: history is served to any directly-connected peer —
// the same peers who already receive every live message in the topic.

use serde::{Deserialize, Serialize};

use crate::network::message::ChatMessage;

pub const SYNC_PROTOCOL: &str = "/llung/sync/1.0.0";
/// Per-request message cap (clamped server-side too).
pub const MAX_SYNC_MESSAGES: usize = 200;
/// Hard cap on continuation depth: 200 × 25 = 5000 messages per sync.
pub const MAX_SYNC_DEPTH: u32 = 25;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncRequest {
    /// Walk ancestors of `from` (or the topic tip when None), newest
    /// first, at most `limit` messages.
    Chain {
        topic: String,
        from: Option<String>,
        limit: usize,
    },
    /// Fetch concrete messages by id; absent ones are silently skipped.
    Missing { ids: Vec<String> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncResponse {
    /// `more_parent`: the parent of the oldest returned message — the
    /// cursor for the next `Chain` request. `None` = genesis reached
    /// (or nothing to send).
    Chain {
        topic: String,
        messages: Vec<ChatMessage>,
        more_parent: Option<String>,
    },
    Missing { messages: Vec<ChatMessage> },
    /// The peer doesn't serve history (no store attached).
    Unavailable,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_response_shapes_round_trip() {
        let req = SyncRequest::Chain {
            topic: "introductions".into(),
            from: Some("m1".into()),
            limit: 50,
        };
        let back: serde_json::Value =
            serde_json::from_value(serde_json::to_value(&req).unwrap()).unwrap();
        assert_eq!(serde_json::to_value(&req).unwrap(), back);

        let resp = SyncResponse::Chain {
            topic: "t".into(),
            messages: vec![],
            more_parent: None,
        };
        let back: serde_json::Value =
            serde_json::from_value(serde_json::to_value(&resp).unwrap()).unwrap();
        assert_eq!(serde_json::to_value(&resp).unwrap(), back);
    }

    #[test]
    fn sync_caps_are_sane() {
        assert!(MAX_SYNC_MESSAGES > 0);
        assert!(MAX_SYNC_DEPTH > 0);
    }
}
