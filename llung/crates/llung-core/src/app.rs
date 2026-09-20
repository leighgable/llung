use crate::{
    config::CoreConfig,
    identity::{being::Being, machine::MachineIdentity},
    network::{
        NetworkCommand, NetworkEngine, NetworkEvent,
        behaviour::{DirectMessageBehaviour, LlungBehaviour},
    },
};
use libp2p::{
    PeerId, StreamProtocol, dcutr, gossipsub,
    gossipsub::IdentTopic,
    identify,
    identity::Keypair,
    kad, mdns, noise,
    request_response::{Config as ReqResConfig, ProtocolSupport},
    swarm::Swarm,
    tcp, yamux,
};
use std::{
    collections::hash_map::DefaultHasher,
    error::Error,
    hash::{Hash, Hasher},
    time::Duration,
};
use tokio::{io, sync::mpsc};

pub fn build_swarm(keypair: Keypair) -> Result<Swarm<LlungBehaviour>, Box<dyn Error>> {
    let swarm = libp2p::SwarmBuilder::with_existing_identity(keypair)
        .with_tokio()
        .with_tcp(
            tcp::Config::default(),
            noise::Config::new,
            yamux::Config::default,
        )?
        .with_quic()
        .with_dns()?
        .with_relay_client(noise::Config::new, yamux::Config::default)?
        .with_behaviour(|key, relay_client| {
            let peer_id = key.public().to_peer_id();

            let message_id_fn = |message: &gossipsub::Message| {
                let mut s = DefaultHasher::new();
                // Include source + sequence number so that identical
                // payloads (e.g. two people typing "hi") don't collide
                // and get deduplicated by gossipsub.
                message.source.hash(&mut s);
                message.sequence_number.hash(&mut s);
                message.data.hash(&mut s);
                gossipsub::MessageId::from(s.finish().to_string())
            };

            let gossipsub_config = gossipsub::ConfigBuilder::default()
                .heartbeat_interval(Duration::from_secs(10))
                .validation_mode(gossipsub::ValidationMode::Strict)
                .message_id_fn(message_id_fn)
                .build()
                .map_err(io::Error::other)?;

            let gossipsub = gossipsub::Behaviour::new(
                gossipsub::MessageAuthenticity::Signed(key.clone()),
                gossipsub_config,
            )?;

            let mdns = mdns::tokio::Behaviour::new(mdns::Config::default(), peer_id)?;
            let store = kad::store::MemoryStore::new(peer_id);
            let kademlia = kad::Behaviour::new(peer_id, store);
            let dcutr = dcutr::Behaviour::new(peer_id);
            let identify = identify::Behaviour::new(identify::Config::new(
                "/llung/1.0.0".to_string(),
                key.public(),
            ));
            let direct_message = DirectMessageBehaviour::new(
                [(
                    StreamProtocol::new("/llung/direct/1.0.0"),
                    ProtocolSupport::Full,
                )],
                ReqResConfig::default(),
            );

            Ok(LlungBehaviour {
                gossipsub,
                mdns,
                kademlia,
                relay_client,
                dcutr,
                identify,
                direct_message,
            })
        })?
        .build();

    Ok(swarm)
}

pub struct LlungApp {
    pub local_being: Being,
    pub local_peer_id: PeerId,
    cmd_tx: mpsc::Sender<NetworkCommand>,
}

impl LlungApp {
    /// Initialize the node. Assumes the identity/Being already exists in the DB.
    /// The UI layer is responsible for login/registration before calling this.
    pub async fn init(
        config: CoreConfig,
        machine: MachineIdentity,
        being: Being,
    ) -> Result<(Self, mpsc::Receiver<NetworkEvent>), Box<dyn std::error::Error>> {
        let local_peer_id = machine.peer_id;
        let local_being = being;

        let mut swarm = build_swarm(machine.keypair)?;
        swarm
            .behaviour_mut()
            .kademlia
            .set_mode(Some(kad::Mode::Server));

        let topic = IdentTopic::new(&config.chat_topic);
        swarm.behaviour_mut().gossipsub.subscribe(&topic)?;
        swarm.listen_on("/ip4/0.0.0.0/udp/0/quic-v1".parse()?)?;
        swarm.listen_on("/ip4/0.0.0.0/tcp/0".parse()?)?;

        let (cmd_tx, cmd_rx) = mpsc::channel::<NetworkCommand>(32);
        let (event_tx, event_rx) = mpsc::channel::<NetworkEvent>(32);

        let engine = NetworkEngine::new(swarm, cmd_rx, event_tx);
        tokio::spawn(async move { engine.run().await });

        // Re-announce avatar on startup
        if let Some(cid) = &local_being.avatar_cid {
            let _ = cmd_tx
                .send(NetworkCommand::ProvideMediaCid { cid: cid.clone() })
                .await;
        }

        Ok((
            LlungApp {
                local_being,
                local_peer_id,
                cmd_tx,
            },
            event_rx,
        ))
    }

    pub fn command_tx(&self) -> mpsc::Sender<NetworkCommand> {
        self.cmd_tx.clone()
    }
}
