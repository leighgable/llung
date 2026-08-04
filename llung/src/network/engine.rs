use futures::StreamExt;
use libp2p::{Swarm, swarm::SwarmEvent};
use tokio::sync::mpsc;

use crate::network::{behaviour::LlungBehaviour, command::NetworkCommand, event::NetworkEvent};

pub struct NetworkEngine {
    swarm: Swarm<LlungBehaviour>,
    command_rx: mpsc::Receiver<NetworkCommand>,
    event_tx: mpsc::Sender<NetworkEvent>,
}

impl NetworkEngine {
    pub fn new(
        swarm: Swarm<LlungBehaviour>,
        command_rx: mpsc::Receiver<NetworkCommand>,
        event_tx: mpsc::Sender<NetworkEvent>,
    ) -> Self {
        Self {
            swarm,
            command_rx,
            event_tx,
        }
    }

    /// The main network event loop. Runs on its own spawned Tokio task.
    pub async fn run(mut self) {
        loop {
            tokio::select! {
                // 1. Process outgoing commands from the application / UI
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

    async fn handle_command(&mut self, cmd: NetworkCommand) {
        match cmd {
            NetworkCommand::PublishMessage { topic, contents } => {
                if let Err(e) = self
                    .swarm
                    .behaviour_mut()
                    .gossipsub
                    .publish(topic, contents)
                {
                    eprintln!("Failed to publish message: {e:?}");
                }
            }
            NetworkCommand::SendDirectMessage { target, message } => {
                // TODO: Route via request_response behaviour
            }
            NetworkCommand::PutProfile { profile } => {
                // TODO: Insert profile into Kademlia DHT
            }
            NetworkCommand::FetchProfile { peer_id, responder } => {
                // TODO: Query Kademlia DHT and send back result via responder
            }
            NetworkCommand::ProvideMediaCid { cid } => {
                let _ = self
                    .swarm
                    .behaviour_mut()
                    .kademlia
                    .start_providing(cid.into_bytes().into());
            }
        }
    }

    async fn handle_swarm_event(
        &mut self,
        event: SwarmEvent<crate::network::behaviour::LlungBehaviourEvent>,
    ) {
        match event {
            SwarmEvent::Behaviour(behaviour_event) => {
                // Delegate protocol routing to LlungBehaviour
                if let Some(app_event) = self.swarm.behaviour_mut().handle_event(behaviour_event) {
                    // Send actionable events back to the UI / DB task
                    let _ = self.event_tx.send(app_event).await;
                }
            }
            SwarmEvent::NewListenAddr { address, .. } => {
                println!("Node listening on: {address}");
            }
            _ => {}
        }
    }
}
