use libp2p::{
    dcutr, gossipsub, identify, kad, mdns, relay,
    request_response::cbor::Behaviour as CborBehaviour, swarm::NetworkBehaviour,
};

use crate::media::image::{ImageRequest, ImageResponse};
use crate::network::{
    event::NetworkEvent,
    message::{DirectMessage, DirectMessageResponse},
    sync::{SyncRequest, SyncResponse},
};

pub type DirectMessageBehaviour = CborBehaviour<DirectMessage, DirectMessageResponse>;
pub type ImageTransferBehaviour = CborBehaviour<ImageRequest, ImageResponse>;
pub type HistorySyncBehaviour = CborBehaviour<SyncRequest, SyncResponse>;

#[derive(NetworkBehaviour)]
pub struct LlungBehaviour {
    pub gossipsub: gossipsub::Behaviour,
    pub mdns: mdns::tokio::Behaviour,
    pub kademlia: kad::Behaviour<kad::store::MemoryStore>,
    pub relay_client: relay::client::Behaviour,
    pub dcutr: dcutr::Behaviour,
    pub identify: identify::Behaviour,
    pub direct_message: DirectMessageBehaviour,
    pub image_transfer: ImageTransferBehaviour,
    pub history_sync: HistorySyncBehaviour,
}

impl LlungBehaviour {
    pub fn handle_event(&mut self, event: LlungBehaviourEvent) -> Option<NetworkEvent> {
        match event {
            LlungBehaviourEvent::Mdns(mdns::Event::Discovered(list)) => {
                for (peer_id, multiaddr) in list {
                    self.gossipsub.add_explicit_peer(&peer_id);
                    self.kademlia.add_address(&peer_id, multiaddr);
                }
                None
            }

            LlungBehaviourEvent::Mdns(mdns::Event::Expired(list)) => {
                for (peer_id, _multiaddr) in list {
                    self.gossipsub.remove_explicit_peer(&peer_id);
                }
                None
            }
            // Transform raw Gossipsub messages into high-level NetworkEvents for app/UI
            LlungBehaviourEvent::Gossipsub(gossipsub::Event::Message {
                propagation_source,
                message,
                ..
            }) => Some(NetworkEvent::MessageReceived {
                topic: message.topic.to_string(),
                sender: propagation_source,
                data: message.data,
            }),

            // Ignore unused internal protocol events (DCUtR, Relay).
            // Kademlia OutboundQueryProgressed events are intercepted by
            // NetworkEngine, which owns the pending-query map.
            _ => None,
        }
    }
}
