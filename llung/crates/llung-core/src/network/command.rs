use crate::identity::being::Being;
use libp2p::{PeerId, gossipsub::IdentTopic};
use tokio::sync::oneshot;

pub enum NetworkCommand {
    PublishMessage {
        topic: IdentTopic,
        contents: Vec<u8>,
    },
    SendDirectMessage {
        target: PeerId,
        message: Vec<u8>,
    },
    PutProfile {
        profile: Being,
    },
    FetchProfile {
        peer_id: PeerId,
        responder: oneshot::Sender<Option<Being>>,
    },
    ProvideMediaCid {
        cid: String,
    },
    GetMediaProviders {
        cid: String,
    },
}
