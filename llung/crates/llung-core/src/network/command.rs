use crate::identity::being::Being;
use libp2p::{Multiaddr, PeerId, gossipsub::IdentTopic};
use tokio::sync::oneshot;

pub enum NetworkCommand {
    PublishMessage {
        topic: IdentTopic,
        contents: Vec<u8>,
    },
    /// Connect to a specific address, bypassing discovery. Same-process
    /// nodes (TUI + spawned agent) use this because mDNS relies on
    /// multicast, which some hosts/routers filter entirely.
    Dial {
        multiaddr: Multiaddr,
    },
    /// Current listen addresses, for handing to a sibling node.
    GetListenAddrs {
        responder: oneshot::Sender<Vec<Multiaddr>>,
    },
    SendDirectMessage {
        target: PeerId,
        payload: Vec<u8>,
    },
    SubscribeTopic {
        topic: IdentTopic,
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
    /// Push a lightweight image (avatar / message image, ≤512KB) into
    /// the caches of every connected relay. Fire-and-forget.
    PublishImage {
        cid: String,
        mime_type: String,
        data: Vec<u8>,
    },
    /// Fetch a lightweight image by content id from the relays.
    /// `None` when no relay has it.
    GetImage {
        cid: String,
        responder: oneshot::Sender<Option<(String, Vec<u8>)>>,
    },
}
