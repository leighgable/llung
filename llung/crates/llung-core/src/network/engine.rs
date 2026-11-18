use std::collections::{HashMap, HashSet};
use std::time::Duration;

use futures::StreamExt;
use libp2p::{
    PeerId, Swarm, gossipsub::IdentTopic, gossipsub::TopicHash, identify, kad, mdns,
    request_response, swarm::SwarmEvent,
};
use tokio::sync::{mpsc, oneshot};

use crate::identity::being::PresenceMessage;
use crate::media::image::{validate_image_put, ImageRequest, ImageResponse, RELAY_AGENT_VERSION};
use crate::network::{
    behaviour::{LlungBehaviour, LlungBehaviourEvent},
    command::NetworkCommand,
    event::{HistoryQuery, ImageQuery, NetworkEvent},
    message::{DirectMessage, DirectMessageResponse},
    sync::{SyncRequest, SyncResponse, MAX_SYNC_DEPTH},
};

/// Tracks in-flight Kademlia queries so results can be correlated
/// with the command that started them.
enum PendingQuery {
    StartProviding { cid: String },
    GetProviders { cid: String },
}

/// One in-flight `GetImage`: try each known relay in turn until one
/// returns the bytes (or the list is exhausted).
struct PendingImageGet {
    cid: String,
    responder: oneshot::Sender<Option<(String, Vec<u8>)>>,
    /// Relays not yet tried.
    remaining: Vec<PeerId>,
}

/// Images fan out to at most this many connected peers per publish —
/// enough for a healthy group mesh, bounded on big chats.
const MAX_IMAGE_FANOUT: usize = 12;
/// Peers tried per GetImage before falling back to relays.
const MAX_IMAGE_PEER_PROBES: usize = 8;
/// Images published while nobody is connected wait here and flush on
/// the next connection — capped, oldest dropped (startup burst guard).
const MAX_PENDING_IMAGE_PUBLISHES: usize = 32;

#[derive(Clone)]
struct PendingImagePublish {
    cid: String,
    mime_type: String,
    data: Vec<u8>,
}

/// One in-flight history `Chain` walk (continuation included).
struct PendingSync {
    peer: PeerId,
    topic: String,
    depth: u32,
    /// Resend attempts after stream failures (fresh connections need a
    /// moment before request streams negotiate).
    attempts: u32,
}
const MAX_SYNC_ATTEMPTS: u32 = 3;

pub struct NetworkEngine {
    swarm: Swarm<LlungBehaviour>,
    command_rx: mpsc::Receiver<NetworkCommand>,
    event_tx: mpsc::Sender<NetworkEvent>,
    pending_queries: HashMap<kad::QueryId, PendingQuery>,
    local_presence: PresenceMessage,
    /// Every topic we have joined — presence heartbeats go to all of them
    /// so peers in invite-only topics keep working name resolution.
    subscribed_topics: HashSet<TopicHash>,
    /// Relay servers (detected by their identify agent_version). Image
    /// puts go to all of them; gets try them in turn.
    relays: HashSet<PeerId>,
    /// In-flight `GetImage` requests, by request-response RequestId.
    pending_image_gets: HashMap<request_response::OutboundRequestId, PendingImageGet>,
    /// Answers inbound `Get` requests from the local image cache; owned
    /// by the app layer (it has the DB).
    image_source: Option<mpsc::Sender<ImageQuery>>,
    /// Relay multiaddrs to dial at startup (from `CoreConfig.relays`).
    relay_addrs: Vec<libp2p::Multiaddr>,
    /// Our addresses as seen through relay reservations — advertised in
    /// presence so our other devices can dial us back.
    relayed_addrs: Vec<libp2p::Multiaddr>,
    /// Images waiting for a connection to be published to.
    pending_image_publishes: std::collections::VecDeque<PendingImagePublish>,
    /// Answers inbound history-sync requests from local storage; owned
    /// by the app layer (it has the DB).
    history_source: Option<mpsc::Sender<HistoryQuery>>,
    /// (peer, topic) pairs already synced on this connection — sync
    /// once per connection, not per gossipsub mesh event.
    synced: HashSet<(PeerId, TopicHash)>,
    /// Currently connected peers, tracked from establish/closed events
    /// (Swarm::connected_peers proved unreliable at subscribe time).
    connected: HashSet<PeerId>,
    /// In-flight history walks, by request id.
    pending_syncs: HashMap<request_response::OutboundRequestId, PendingSync>,
    /// The presence heartbeat topic (not a chat topic — never synced).
    presence_topic: TopicHash,
}

impl NetworkEngine {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        swarm: Swarm<LlungBehaviour>,
        command_rx: mpsc::Receiver<NetworkCommand>,
        event_tx: mpsc::Sender<NetworkEvent>,
        local_presence: PresenceMessage,
        presence_topic: IdentTopic,
        image_source: Option<mpsc::Sender<ImageQuery>>,
        relay_addrs: Vec<libp2p::Multiaddr>,
        history_source: Option<mpsc::Sender<HistoryQuery>>,
    ) -> Self {
        let mut subscribed_topics = HashSet::new();
        let presence_topic: TopicHash = presence_topic.clone().into();
        subscribed_topics.insert(presence_topic.clone());
        Self {
            swarm,
            command_rx,
            event_tx,
            pending_queries: HashMap::new(),
            local_presence,
            subscribed_topics,
            relays: HashSet::new(),
            pending_image_gets: HashMap::new(),
            image_source,
            relay_addrs,
            relayed_addrs: Vec::new(),
            pending_image_publishes: std::collections::VecDeque::new(),
            history_source,
            synced: HashSet::new(),
            connected: HashSet::new(),
            pending_syncs: HashMap::new(),
            presence_topic,
        }
    }

    /// The main network event loop. Runs on its own spawned Tokio task.
    pub async fn run(mut self) {
        // Re-announce ourselves so peers that join later can still
        // resolve our human_name/avatar.
        let mut presence_tick = tokio::time::interval(Duration::from_secs(15));
        // Stay dialed to the configured relays: they carry gossipsub
        // across NATs, coordinate hole punching, and cache images.
        for addr in self.relay_addrs.clone() {
            tracing::info!(%addr, "dialing relay");
            if let Err(e) = self.swarm.dial(addr.clone()) {
                tracing::warn!(%addr, "relay dial failed: {e}");
            }
        }
        loop {
            tokio::select! {
                Some(cmd) = self.command_rx.recv() => {
                    let _ = self.handle_command(cmd).await;
                }

                // 2. Process incoming libp2p network events
                event = self.swarm.select_next_some() => {
                    self.handle_swarm_event(event).await;
                }

                // 3. Presence heartbeat on every joined topic. Addresses
                // are refreshed every beat: direct listeners plus any
                // relay-circuit addresses we've been granted — that's how
                // our other devices find us.
                _ = presence_tick.tick() => {
                    let mut presence = self.local_presence.clone();
                    presence.listen_addrs = self
                        .swarm
                        .listeners()
                        .map(|a| a.to_string())
                        .chain(self.relayed_addrs.iter().map(|a| a.to_string()))
                        .collect();
                    if let Ok(payload) = serde_json::to_vec(&presence) {
                        let topics: Vec<TopicHash> =
                            self.subscribed_topics.iter().cloned().collect();
                        for topic in topics {
                            if let Err(e) = self
                                .swarm
                                .behaviour_mut()
                                .gossipsub
                                .publish(topic, payload.clone())
                            {
                                tracing::debug!("Presence heartbeat failed: {e:?}");
                            }
                        }
                    }
                }
            }
        }
    }

    async fn handle_command(
        &mut self,
        cmd: NetworkCommand,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match cmd {
            NetworkCommand::PublishMessage { topic, contents } => {
                if let Err(e) = self
                    .swarm
                    .behaviour_mut()
                    .gossipsub
                    .publish(topic, contents)
                {
                    tracing::info!("Failed to publish message: {e:?}");
                }
            }
            NetworkCommand::Dial { multiaddr } => {
                if let Err(e) = self.swarm.dial(multiaddr) {
                    tracing::warn!("Dial failed: {e}");
                }
            }
            NetworkCommand::GetListenAddrs { responder } => {
                let _ = responder.send(self.swarm.listeners().cloned().collect());
            }
            NetworkCommand::SubscribeTopic { topic } => {
                let is_new = self.swarm.behaviour_mut().gossipsub.subscribe(&topic)?;
                self.subscribed_topics.insert(topic.clone().into());

                if is_new {
                    tracing::info!("Subscribed to topic: {}", topic);
                    match serde_json::to_vec(&self.local_presence) {
                        Ok(payload) => {
                            if let Err(e) =
                                self.swarm.behaviour_mut().gossipsub.publish(topic, payload)
                            {
                                // No mesh yet (we're alone): not an error.
                                tracing::debug!("Initial presence publish failed: {e:?}");
                            }
                        }
                        Err(e) => tracing::warn!("presence serialization failed: {e}"),
                    }
                }

                // Late join: peers we are ALREADY connected to never got
                // a sync trigger for this topic — fire it now (the
                // per-(peer, topic) `synced` set dedupes against the
                // connection-established trigger).
                for peer in self.connected.iter().copied().collect::<Vec<_>>() {
                    if !self.relays.contains(&peer) {
                        self.trigger_history_sync(peer);
                    }
                }
            }
            NetworkCommand::SendDirectMessage { target, payload } => {
                let request_id = self
                    .swarm
                    .behaviour_mut()
                    .direct_message
                    .send_request(&target, DirectMessage(payload));
                tracing::debug!("Send direct message {request_id} to {target}");
            }
            NetworkCommand::PutProfile { profile } => {
                // TODO: Insert profile into Kademlia DHT
            }
            NetworkCommand::FetchProfile { peer_id, responder } => {
                // TODO: Query Kademlia DHT and send back result via responder
            }
            NetworkCommand::ProvideMediaCid { cid } => {
                let key = kad::RecordKey::new(&cid);
                match self.swarm.behaviour_mut().kademlia.start_providing(key) {
                    Ok(id) => {
                        self.pending_queries
                            .insert(id, PendingQuery::StartProviding { cid });
                    }
                    Err(e) => tracing::info!("Failed to start providing {cid}: {e:?}"),
                }
            }
            NetworkCommand::GetMediaProviders { cid } => {
                let key = kad::RecordKey::new(&cid);
                let id = self.swarm.behaviour_mut().kademlia.get_providers(key);
                self.pending_queries
                    .insert(id, PendingQuery::GetProviders { cid });
            }
            NetworkCommand::PublishImage {
                cid,
                mime_type,
                data,
            } => {
                // Self-validate before wasting anyone's bandwidth.
                if let Err(reason) = validate_image_put(&cid, &mime_type, &data) {
                    tracing::warn!(%cid, %reason, "PublishImage rejected locally");
                    return Ok(());
                }
                // Relays cache for offline/device catch-up; connected
                // peers ALSO get the Put directly — group images should
                // live in the group, not behind the relay.
                let targets = self.image_targets();
                if targets.is_empty() {
                    // Nobody yet: hold the publish and flush it when a
                    // peer or relay connects.
                    if !self
                        .pending_image_publishes
                        .iter()
                        .any(|p| p.cid == cid)
                    {
                        self.pending_image_publishes.push_back(PendingImagePublish {
                            cid,
                            mime_type,
                            data,
                        });
                        while self.pending_image_publishes.len() > MAX_PENDING_IMAGE_PUBLISHES {
                            self.pending_image_publishes.pop_front();
                        }
                        tracing::debug!("PublishImage: queued until a connection exists");
                    }
                    return Ok(());
                }
                let request = ImageRequest::Put {
                    cid,
                    mime_type,
                    data,
                };
                for target in targets {
                    self.send_image_put(target, &request);
                }
            }
            NetworkCommand::GetImage { cid, responder } => {
                // Peers first (p2p is the preferred source), relays as
                // the catch-all fallback, each tried in turn.
                let mut peers: Vec<PeerId> = self
                    .connected
                    .iter()
                    .copied()
                    .filter(|p| !self.relays.contains(p))
                    .collect();
                peers.sort();
                peers.truncate(MAX_IMAGE_PEER_PROBES);
                let mut remaining: Vec<PeerId> = peers;
                remaining.extend(self.relays.iter().copied());
                if remaining.is_empty() {
                    let _ = responder.send(None);
                    return Ok(());
                }
                let first = remaining[0];
                let id = self.send_image_get(first, &cid);
                self.pending_image_gets.insert(
                    id,
                    PendingImageGet {
                        cid,
                        responder,
                        remaining: remaining[1..].to_vec(),
                    },
                );
            }
        }
        Ok(())
    }

    /// Current Put targets: relays + connected chat peers (capped).
    fn image_targets(&self) -> Vec<PeerId> {
        let mut targets: Vec<PeerId> = self.relays.iter().copied().collect();
        let mut peers: Vec<PeerId> = self
            .connected
            .iter()
            .copied()
            .filter(|p| !self.relays.contains(p))
            .collect();
        peers.sort();
        targets.extend(peers.into_iter().take(MAX_IMAGE_FANOUT));
        targets
    }

    fn send_image_put(&mut self, target: PeerId, request: &ImageRequest) {
        let id = self
            .swarm
            .behaviour_mut()
            .image_transfer
            .send_request(&target, request.clone());
        tracing::debug!(%target, %id, "image Put sent");
    }

    /// A new peer/relay appeared: everything that was published while
    /// alone gets another shot at this target.
    fn flush_pending_image_publishes(&mut self, peer: PeerId) {
        while let Some(pending) = self.pending_image_publishes.pop_front() {
            self.send_image_put(
                peer,
                &ImageRequest::Put {
                    cid: pending.cid,
                    mime_type: pending.mime_type,
                    data: pending.data,
                },
            );
        }
    }

    fn send_image_get(
        &mut self,
        relay: PeerId,
        cid: &str,
    ) -> request_response::OutboundRequestId {
        self.swarm
            .behaviour_mut()
            .image_transfer
            .send_request(&relay, ImageRequest::Get { cid: cid.into() })
    }

    async fn handle_swarm_event(&mut self, event: SwarmEvent<LlungBehaviourEvent>) {
        match event {
            // Kademlia query results are handled here (rather than in
            // LlungBehaviour) because the pending-query map lives in the engine.
            SwarmEvent::Behaviour(LlungBehaviourEvent::Kademlia(
                kad::Event::OutboundQueryProgressed { id, result, .. },
            )) => {
                if let Some(app_event) = self.handle_kad_result(id, result) {
                    let _ = self.event_tx.send(app_event).await;
                }
            }
            // Identify does two jobs for us: learn listen addresses for
            // Kademlia, and spot relay servers (their agent_version) so
            // image puts/fetches have a destination. Handled HERE, not
            // in LlungBehaviour, because the relay set lives in the engine.
            SwarmEvent::Behaviour(LlungBehaviourEvent::Identify(
                identify::Event::Received { peer_id, info, .. },
            )) => {
                for addr in info.listen_addrs {
                    self.swarm.behaviour_mut().kademlia.add_address(&peer_id, addr);
                }
                if info.agent_version == RELAY_AGENT_VERSION && self.relays.insert(peer_id) {
                    tracing::info!(%peer_id, "relay connected");
                }
            }

            // mDNS discovery only *reports* peers — nothing connects to
            // them unless we dial. Without this, two LAN nodes discover
            // each other but never link, gossipsub never forms a mesh,
            // and every publish fails with NoPeersSubscribedToTopic
            // (silently, from the user's point of view).
            SwarmEvent::Behaviour(LlungBehaviourEvent::Mdns(mdns::Event::Discovered(list))) => {
                let self_id = *self.swarm.local_peer_id();
                let list: Vec<_> = list;
                tracing::debug!(count = list.len(), "mDNS discovered peers");
                for (peer_id, multiaddr) in list {
                    if peer_id == self_id {
                        continue;
                    }
                    match self.swarm.dial(multiaddr.clone()) {
                        Ok(()) => tracing::debug!(%peer_id, %multiaddr, "dialing discovered peer"),
                        Err(e) => tracing::warn!(%peer_id, %multiaddr, "mDNS dial failed: {e}"),
                    }
                    self.swarm.behaviour_mut().gossipsub.add_explicit_peer(&peer_id);
                    self.swarm.behaviour_mut().kademlia.add_address(&peer_id, multiaddr);
                }
            }

            SwarmEvent::ConnectionEstablished {
                peer_id, endpoint, ..
            } => {
                tracing::info!(%peer_id, ?endpoint, "connection established");
                self.connected.insert(peer_id);
                // History sync happens over DIRECT (or relay-circuit)
                // connections to chat peers — relays themselves serve no
                // history. One sync per (peer, topic) per connection.
                if !self.relays.contains(&peer_id) {
                    self.trigger_history_sync(peer_id);
                }
                self.flush_pending_image_publishes(peer_id);
            }
            SwarmEvent::ConnectionClosed { peer_id, .. } => {
                self.connected.remove(&peer_id);
                self.synced.retain(|(p, _)| *p != peer_id);
            }
            // Relay reservations: advertise the circuit address so our
            // other devices can dial us through the relay. The event
            // carries only the relay's PeerId, so reconstruct the base
            // address from the configured relay multiaddr (which is the
            // public one clients dial anyway).
            SwarmEvent::Behaviour(LlungBehaviourEvent::RelayClient(
                libp2p::relay::client::Event::ReservationReqAccepted {
                    relay_peer_id, ..
                },
            )) => {
                let base = self.relay_addrs.iter().find_map(|addr| {
                    let mut base = addr.clone();
                    if let Some(libp2p::multiaddr::Protocol::P2p(p)) = base.pop() {
                        if p == relay_peer_id {
                            return Some(base);
                        }
                    }
                    None
                });
                match base {
                    Some(base) => {
                        let circuit = base
                            .with(libp2p::multiaddr::Protocol::P2p(relay_peer_id))
                            .with(libp2p::multiaddr::Protocol::P2pCircuit)
                            .with(libp2p::multiaddr::Protocol::P2p(*self.swarm.local_peer_id()));
                        if self.relayed_addrs.contains(&circuit) {
                            tracing::info!(%circuit, "relay reservation renewed");
                        } else {
                            tracing::info!(%circuit, "relay reservation granted");
                            self.relayed_addrs.push(circuit);
                        }
                    }
                    None => {
                        tracing::debug!(%relay_peer_id, "reservation granted by unknown relay; not advertising");
                    }
                }
            }
            SwarmEvent::OutgoingConnectionError { peer_id, error, .. } => {
                tracing::warn!(?peer_id, "outgoing connection error: {error}");
            }

            // Lightweight image transfer (avatars, message images).
            SwarmEvent::Behaviour(LlungBehaviourEvent::ImageTransfer(event)) => {
                self.handle_image_event(event).await;
            }

            // History exchange.
            SwarmEvent::Behaviour(LlungBehaviourEvent::HistorySync(event)) => {
                self.handle_sync_event(event).await;
            }

            SwarmEvent::Behaviour(LlungBehaviourEvent::DirectMessage(event)) => match event {
                request_response::Event::Message { peer, message, .. } => match message {
                    request_response::Message::Request {
                        request, channel, ..
                    } => {
                        let _ = self
                            .event_tx
                            .send(NetworkEvent::DirectMessageReceived {
                                sender: peer,
                                payload: request.0,
                            })
                            .await;
                        let _ = self
                            .swarm
                            .behaviour_mut()
                            .direct_message
                            .send_response(channel, DirectMessageResponse);
                    }
                    request_response::Message::Response { .. } => {}
                },
                request_response::Event::OutboundFailure { peer, error, .. } => {
                    tracing::warn!("Direct message to {peer} failed: {error}");
                }
                request_response::Event::InboundFailure { peer, error, .. } => {
                    tracing::warn!("Direct message from {peer} failed: {error}");
                }
                _ => {}
            },

            SwarmEvent::Behaviour(behaviour_event) => {
                // Delegate protocol routing to LlungBehaviour
                if let Some(app_event) = self.swarm.behaviour_mut().handle_event(behaviour_event) {
                    let _ = self.event_tx.send(app_event).await;
                }
            }
            SwarmEvent::NewListenAddr { address, .. } => {
                tracing::info!("Node listening on: {address}");
            }
            _ => {}
        }
    }

    /// Serve inbound image requests from the local cache (via the app's
    /// DB-backed `image_source`), ack validated Puts with an app event.
    async fn handle_image_event(
        &mut self,
        event: request_response::Event<ImageRequest, ImageResponse>,
    ) {
        match event {
            request_response::Event::Message { peer, message, .. } => match message {
                request_response::Message::Request {
                    request, channel, ..
                } => match request {
                    ImageRequest::Get { cid } => {
                        let response = match &self.image_source {
                            Some(src) => {
                                let (tx, rx) = oneshot::channel();
                                if src.send(ImageQuery { cid: cid.clone(), respond: tx }).await.is_ok()
                                {
                                    match tokio::time::timeout(Duration::from_secs(5), rx).await {
                                        Ok(Ok(Some((mime, data)))) => ImageResponse::Got {
                                            mime_type: mime,
                                            data,
                                        },
                                        _ => ImageResponse::NotFound,
                                    }
                                } else {
                                    ImageResponse::NotFound
                                }
                            }
                            None => ImageResponse::NotFound,
                        };
                        let _ = self
                            .swarm
                            .behaviour_mut()
                            .image_transfer
                            .send_response(channel, response);
                    }
                    ImageRequest::Put {
                        cid,
                        mime_type,
                        data,
                    } => {
                        let response =
                            match validate_image_put(&cid, &mime_type, &data) {
                                Ok(()) => {
                                    let _ = self
                                        .event_tx
                                        .send(NetworkEvent::ImagePushed {
                                            from: peer,
                                            cid: cid.clone(),
                                            mime_type: mime_type.clone(),
                                            data,
                                        })
                                        .await;
                                    ImageResponse::PutAck {
                                        accepted: true,
                                        reason: None,
                                    }
                                }
                                Err(reason) => ImageResponse::PutAck {
                                    accepted: false,
                                    reason: Some(reason),
                                },
                            };
                        let _ = self
                            .swarm
                            .behaviour_mut()
                            .image_transfer
                            .send_response(channel, response);
                    }
                },
                request_response::Message::Response {
                    request_id,
                    response,
                } => {
                    let Some(pending) = self.pending_image_gets.remove(&request_id) else {
                        return;
                    };
                    match response {
                        ImageResponse::Got { mime_type, data } => {
                            let _ = pending.responder.send(Some((mime_type, data)));
                        }
                        _ => self.try_next_relay(pending),
                    }
                }
            },
            request_response::Event::OutboundFailure {
                request_id, error, ..
            } => {
                tracing::debug!(%error, "image Get failed");
                if let Some(pending) = self.pending_image_gets.remove(&request_id) {
                    self.try_next_relay(pending);
                }
            }
            request_response::Event::InboundFailure { peer, error, .. } => {
                tracing::debug!(%peer, %error, "inbound image request failed");
            }
            _ => {}
        }
    }

    /// A relay answered NotFound or failed: try the next one, or finish
    /// the request with None.
    fn try_next_relay(&mut self, pending: PendingImageGet) {
        if pending.remaining.is_empty() {
            let _ = pending.responder.send(None);
            return;
        }
        let (next, rest) = pending.remaining.split_first().expect("non-empty");
        let id = self.send_image_get(*next, &pending.cid);
        self.pending_image_gets.insert(
            id,
            PendingImageGet {
                cid: pending.cid,
                responder: pending.responder,
                remaining: rest.to_vec(),
            },
        );
    }

    /// Kick off history sync with a freshly connected peer: one Chain
    /// walk per subscribed chat topic (presence topic excluded).
    fn trigger_history_sync(&mut self, peer: PeerId) {
        let topics: Vec<TopicHash> = self
            .subscribed_topics
            .iter()
            .filter(|t| **t != self.presence_topic)
            .cloned()
            .collect();
        for topic in topics {
            if !self.synced.insert((peer, topic.clone())) {
                continue;
            }
            let name = topic.to_string();
            let request = SyncRequest::Chain {
                topic: name.clone(),
                from: None,
                limit: crate::network::sync::MAX_SYNC_MESSAGES,
            };
            let id = self
                .swarm
                .behaviour_mut()
                .history_sync
                .send_request(&peer, request);
            tracing::debug!(%peer, topic = %name, "history sync triggered");
            self.pending_syncs.insert(
                id,
                PendingSync {
                    peer,
                    topic: name,
                    depth: 0,
                    attempts: 0,
                },
            );
        }
    }

    /// Resend a Chain walk from scratch (idempotent — the merge is
    /// INSERT OR IGNORE on the receiving side).
    fn resend_sync(&mut self, pending: PendingSync) {
        let request = SyncRequest::Chain {
            topic: pending.topic.clone(),
            from: None,
            limit: crate::network::sync::MAX_SYNC_MESSAGES,
        };
        let id = self
            .swarm
            .behaviour_mut()
            .history_sync
            .send_request(&pending.peer, request);
        self.pending_syncs.insert(id, pending);
    }

    async fn handle_sync_event(
        &mut self,
        event: request_response::Event<SyncRequest, SyncResponse>,
    ) {
        match event {
            request_response::Event::Message { peer, message, .. } => match message {
                request_response::Message::Request {
                    request, channel, ..
                } => {
                    let response = match &self.history_source {
                        Some(src) => {
                            let (tx, rx) = oneshot::channel();
                            if src
                                .send(HistoryQuery {
                                    request: request.clone(),
                                    respond: tx,
                                })
                                .await
                                .is_ok()
                            {
                                tokio::time::timeout(Duration::from_secs(15), rx)
                                    .await
                                    .ok()
                                    .and_then(|r| r.ok())
                                    .unwrap_or(SyncResponse::Unavailable)
                            } else {
                                SyncResponse::Unavailable
                            }
                        }
                        None => SyncResponse::Unavailable,
                    };
                    let _ = self
                        .swarm
                        .behaviour_mut()
                        .history_sync
                        .send_response(channel, response);
                }
                request_response::Message::Response {
                    request_id,
                    response,
                } => {
                    let Some(pending) = self.pending_syncs.remove(&request_id) else {
                        return;
                    };
                    match response {
                        SyncResponse::Chain {
                            topic,
                            messages,
                            more_parent,
                        } => {
                            if !messages.is_empty() {
                                let _ = self
                                    .event_tx
                                    .send(NetworkEvent::HistoryReceived {
                                        from: peer,
                                        messages,
                                    })
                                    .await;
                            }
                            // Continue the walk unless genesis or depth cap.
                            if let Some(parent) = more_parent {
                                if pending.depth + 1 < MAX_SYNC_DEPTH {
                                    let request = SyncRequest::Chain {
                                        topic: topic.clone(),
                                        from: Some(parent),
                                        limit: crate::network::sync::MAX_SYNC_MESSAGES,
                                    };
                                    let id = self
                                        .swarm
                                        .behaviour_mut()
                                        .history_sync
                                        .send_request(&pending.peer, request);
                                    self.pending_syncs.insert(
                                        id,
                                        PendingSync {
                                            peer: pending.peer,
                                            topic,
                                            depth: pending.depth + 1,
                                            attempts: pending.attempts,
                                        },
                                    );
                                } else {
                                    tracing::debug!(%topic, "sync depth cap reached");
                                }
                            }
                        }
                        SyncResponse::Missing { messages } => {
                            if !messages.is_empty() {
                                let _ = self
                                    .event_tx
                                    .send(NetworkEvent::HistoryReceived {
                                        from: peer,
                                        messages,
                                    })
                                    .await;
                            }
                        }
                        SyncResponse::Unavailable => {
                            tracing::debug!(%peer, "peer serves no history");
                        }
                    }
                }
            },
            request_response::Event::OutboundFailure {
                request_id, error, ..
            } => {
                let Some(pending) = self.pending_syncs.remove(&request_id) else {
                    return;
                };
                if pending.attempts + 1 < MAX_SYNC_ATTEMPTS {
                    tracing::debug!(%error, "history sync request failed; retrying");
                    // Fresh connections occasionally drop the first
                    // stream; give negotiation a beat before resending.
                    tokio::time::sleep(Duration::from_millis(400)).await;
                    self.resend_sync(PendingSync {
                        attempts: pending.attempts + 1,
                        ..pending
                    });
                } else {
                    tracing::warn!(%error, topic = %pending.topic, "history sync gave up");
                }
            }
            request_response::Event::InboundFailure { peer, error, .. } => {
                tracing::debug!(%peer, %error, "inbound history request failed");
            }
            _ => {}
        }
    }

    fn handle_kad_result(
        &mut self,
        id: kad::QueryId,
        result: kad::QueryResult,
    ) -> Option<NetworkEvent> {
        match result {
            kad::QueryResult::StartProviding(Ok(_)) => {
                if let Some(PendingQuery::StartProviding { cid }) = self.pending_queries.remove(&id)
                {
                    tracing::info!("DHT: now providing media cid {cid}");
                }
                None
            }
            kad::QueryResult::StartProviding(Err(e)) => {
                if let Some(PendingQuery::StartProviding { cid }) = self.pending_queries.remove(&id)
                {
                    tracing::info!("DHT: failed to provide {cid}: {e:?}");
                }
                None
            }
            kad::QueryResult::GetProviders(Ok(kad::GetProvidersOk::FoundProviders {
                key,
                providers,
            })) => {
                let cid = String::from_utf8_lossy(key.as_ref()).into_owned();
                Some(NetworkEvent::MediaCidFound {
                    cid,
                    providers: providers.into_iter().collect(),
                })
            }
            kad::QueryResult::GetProviders(Ok(
                kad::GetProvidersOk::FinishedWithNoAdditionalRecord { .. },
            )) => {
                self.pending_queries.remove(&id);
                None
            }
            kad::QueryResult::GetProviders(Err(e)) => {
                if let Some(PendingQuery::GetProviders { cid }) = self.pending_queries.remove(&id) {
                    tracing::info!("DHT: provider lookup for {cid} failed: {e:?}");
                }
                None
            }
            _ => None,
        }
    }
}
