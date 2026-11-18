use libp2p::PeerId;
use tokio::sync::oneshot;

use crate::identity::being::{Being, BeingStatus};
use crate::media::status::DownloadStatus;
use crate::network::sync::SyncResponse;

/// Inbound history-sync request: the app (which owns the DB) answers.
pub struct HistoryQuery {
    pub request: crate::network::sync::SyncRequest,
    pub respond: oneshot::Sender<SyncResponse>,
}

/// Inbound `ImageRequest::Get` callback: the app (which owns the DB)
/// answers from its local avatar/media cache. `(mime_type, bytes)` or None.
pub struct ImageQuery {
    pub cid: String,
    pub respond: oneshot::Sender<Option<(String, Vec<u8>)>>,
}

#[derive(Debug)]
pub enum NetworkEvent {
    MessageReceived {
        topic: String,
        sender: PeerId,
        data: Vec<u8>,
    },
    DirectMessageReceived {
        sender: PeerId,
        payload: Vec<u8>,
    },
    PeerDiscovered {
        peer_id: PeerId,
    },
    ProfileUpdated {
        peer_id: PeerId,
        profile: Being,
    },
    MediaCidFound {
        cid: String,
        providers: Vec<PeerId>,
    },
    StatusChanged {
        peer_id: PeerId,
        status: BeingStatus,
    },
    MediaProgress {
        cid: String,
        status: DownloadStatus,
    },
    /// A peer (usually a relay) pushed a validated lightweight image
    /// into our cache. The app persists it (avatar/media cache).
    ImagePushed {
        from: PeerId,
        cid: String,
        mime_type: String,
        data: Vec<u8>,
    },
    /// History sync returned messages (one event per Chain/Missing
    /// response). The app saves them idempotently.
    HistoryReceived {
        from: PeerId,
        messages: Vec<crate::network::message::ChatMessage>,
    },
}
