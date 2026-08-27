use libp2p::PeerId;

use crate::identity::being::{Being, BeingStatus};
use crate::media::status::DownloadStatus;

#[derive(Debug)]
pub enum NetworkEvent {
    MessageReceived {
        topic: String,
        sender: PeerId,
        data: Vec<u8>,
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
}
