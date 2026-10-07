use std::collections::HashMap;

use futures::StreamExt;
use libp2p::{Swarm, kad, request_response, swarm::SwarmEvent};
use tokio::sync::mpsc;

use crate::identity::{
    being::{Being, PresenceMessage, broadcast_presence},
    machine::MachineIdentity,
};
use crate::network::{
    behaviour::{LlungBehaviour, LlungBehaviourEvent},
    command::NetworkCommand,
    event::NetworkEvent,
    message::{DirectMessage, DirectMessageResponse},
};

/// Tracks in-flight Kademlia queries so results can be correlated
/// with the command that started them.
enum PendingQuery {
    StartProviding { cid: String },
    GetProviders { cid: String },
}

pub struct NetworkEngine {
    swarm: Swarm<LlungBehaviour>,
    command_rx: mpsc::Receiver<NetworkCommand>,
    event_tx: mpsc::Sender<NetworkEvent>,
    pending_queries: HashMap<kad::QueryId, PendingQuery>,
    local_presence: PresenceMessage,
}

impl NetworkEngine {
    pub fn new(
        swarm: Swarm<LlungBehaviour>,
        command_rx: mpsc::Receiver<NetworkCommand>,
        event_tx: mpsc::Sender<NetworkEvent>,
        local_presence: PresenceMessage,
    ) -> Self {
        Self {
            swarm,
            command_rx,
            event_tx,
            pending_queries: HashMap::new(),
            local_presence,
        }
    }

    /// The main network event loop. Runs on its own spawned Tokio task.
    pub async fn run(mut self) {
        loop {
            tokio::select! {
                Some(cmd) = self.command_rx.recv() => {
                    self.handle_command(cmd).await;
                }

                // 2. Process incoming libp2p network events
                event = self.swarm.select_next_some() => {
                    self.handle_swarm_event(event).await;
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
            NetworkCommand::SubscribeTopic { topic } => {
                let is_new = self.swarm.behaviour_mut().gossipsub.subscribe(&topic)?;

                if is_new {
                    tracing::info!("Subscribed to topic: {}", topic);
                    let payload = serde_json::to_vec(&self.local_presence)?;
                    self.swarm
                        .behaviour_mut()
                        .gossipsub
                        .publish(topic, payload)?;
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
        }
        Ok(())
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
