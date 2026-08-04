use libp2p::{
    dcutr, gossipsub, gossipsub::IdentTopic, identify, kad, mdns, noise, swarm::Swarm, tcp, yamux,
};
use std::{
    collections::hash_map::DefaultHasher,
    error::Error,
    hash::{Hash, Hasher},
    time::Duration,
};
use tokio::{io, io::AsyncBufReadExt, sync::mpsc};
use tracing_subscriber::EnvFilter;

mod agent;
mod app;
mod identity;
mod media;
mod network;
mod storage;

use crate::network::{NetworkCommand, NetworkEngine, NetworkEvent, behaviour::LlungBehaviour};

pub fn build_swarm() -> Result<Swarm<LlungBehaviour>, Box<dyn Error>> {
    let swarm = libp2p::SwarmBuilder::with_new_identity()
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

            // Gossipsub setup
            let message_id_fn = |message: &gossipsub::Message| {
                let mut s = DefaultHasher::new();
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

            // Discovery and protocols setup
            let mdns = mdns::tokio::Behaviour::new(mdns::Config::default(), peer_id)?;
            let store = kad::store::MemoryStore::new(peer_id);
            let kademlia = kad::Behaviour::new(peer_id, store);
            let dcutr = dcutr::Behaviour::new(peer_id);
            let identify = identify::Behaviour::new(identify::Config::new(
                "/llung/1.0.0".to_string(),
                key.public(),
            ));

            Ok(LlungBehaviour {
                gossipsub,
                mdns,
                kademlia,
                relay_client,
                dcutr,
                identify,
            })
        })?
        .build();

    Ok(swarm)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .try_init();

    let mut swarm = build_swarm()?;
    let chat_topic = IdentTopic::new("introductions");
    swarm.behaviour_mut().gossipsub.subscribe(&chat_topic)?;
    swarm.listen_on("/ip4/0.0.0.0/udp/0/quic-v1".parse()?)?;
    swarm.listen_on("/ip4/0.0.0.0/tcp/0".parse()?)?;

    // 3. Create communication channels
    let (cmd_tx, cmd_rx) = mpsc::channel::<NetworkCommand>(32);
    let (event_tx, mut event_rx) = mpsc::channel::<NetworkEvent>(32);

    // 4. Start Network Engine actor task
    let engine = NetworkEngine::new(swarm, cmd_rx, event_tx);
    tokio::spawn(async move {
        engine.run().await;
    });

    // 5. User Input / Interface loop
    let mut stdin = io::BufReader::new(io::stdin()).lines();
    println!("====> Network initialized. Start typing below:");

    loop {
        tokio::select! {
            Ok(Some(line)) = stdin.next_line() => {
                let cmd = NetworkCommand::PublishMessage {
                    topic: chat_topic.clone(),
                    contents: line.into_bytes(),
                };
                cmd_tx.send(cmd).await?;
            }
            Some(event) = event_rx.recv() => {
                match event {
                    NetworkEvent::MessageReceived { topic, sender, data } => {
                        println!("[{sender} on {topic}]: {}", String::from_utf8_lossy(&data));
                    }
                    _ => {}
                }
            }
        }
    }
}
