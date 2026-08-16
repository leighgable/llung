use libp2p::{
    dcutr, gossipsub, gossipsub::IdentTopic, identify, identity::Keypair, kad, mdns, noise,
    swarm::Swarm, tcp, yamux,
};
use std::{
    collections::hash_map::DefaultHasher,
    error::Error,
    hash::{Hash, Hasher},
    path::PathBuf,
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
mod utils;

use crate::identity::{
    being::{BeingKind, create_local_being_interactive, load_local_being},
    session::{SessionChoice, prompt_session},
};

use crate::storage::db::Database;

use crate::{
    agent::{AgentConfig, interface::OpenAiCompatible},
    identity::being::{Being, get_or_create_local_being},
    network::{NetworkCommand, NetworkEngine, NetworkEvent, behaviour::LlungBehaviour},
};

struct AgentCli {
    model: String,
    name: String,
    llm_url: String,
}

/// Usage: llung [db_path] [--agent] [--model NAME] [--agent-name NAME] [--llm-url URL]
fn parse_args() -> (Option<String>, Option<AgentCli>) {
    let mut explicit_db_path: Option<String> = None;
    let mut agent_enabled = false;
    let mut cli = AgentCli {
        model: "SmolLMv3".into(),
        name: "llung-bot".into(),
        llm_url: "http://localhost:11434/v1".into(),
    };

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--agent" => agent_enabled = true,
            "--model" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    cli.model = v.clone();
                }
            }
            "--agent-name" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    cli.name = v.clone();
                }
            }
            "--llm-url" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    cli.llm_url = v.clone();
                }
            }
            s if !s.starts_with("--") => explicit_db_path = Some(s.to_string()),
            other => eprintln!("Ignoring unknown argument: {other}"),
        }
        i += 1;
    }

    (explicit_db_path, agent_enabled.then_some(cli))
}

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
    let (explicit_db_path, agent_cli) = parse_args();

    let (db_path, is_registration, is_legacy) = match explicit_db_path {
        Some(path) => (PathBuf::from(path), false, true),
        None => match prompt_session()? {
            SessionChoice::Login { db_path } => (db_path, false, false),
            SessionChoice::Register { db_path } => (db_path, true, false),
        },
    };

    let db_path_str = db_path
        .to_str()
        .ok_or("Database path contains invalid UTF-8")?;
    let (db, keypair) = Database::new(db_path_str)?;
    let local_peer_id = keypair.public().to_peer_id();

    let local_being: Being = if is_legacy {
        get_or_create_local_being(&db, local_peer_id)?
    } else if is_registration {
        create_local_being_interactive(&db, local_peer_id, BeingKind::Human)?
    } else {
        load_local_being(&db, local_peer_id)?
    };

    println!(
        "Node initialized for {} ({})",
        local_being.human_name, local_peer_id
    );

    let mut swarm = build_swarm(keypair)?;

    // Act as a DHT server so this node stores/serves records (e.g.
    // media provider records) instead of being a query-only client.
    swarm
        .behaviour_mut()
        .kademlia
        .set_mode(Some(kad::Mode::Server));

    let _ = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .try_init();

    let chat_topic = IdentTopic::new("introductions");
    swarm.behaviour_mut().gossipsub.subscribe(&chat_topic)?;
    swarm.listen_on("/ip4/0.0.0.0/udp/0/quic-v1".parse()?)?;
    swarm.listen_on("/ip4/0.0.0.0/tcp/0".parse()?)?;

    let (cmd_tx, cmd_rx) = mpsc::channel::<NetworkCommand>(32);
    let (event_tx, mut event_rx) = mpsc::channel::<NetworkEvent>(32);

    let engine = NetworkEngine::new(swarm, cmd_rx, event_tx);
    tokio::spawn(async move {
        engine.run().await;
    });

    // Announce our avatar to the DHT so peers can fetch it by CID.
    // Done on every startup because provider records expire.
    if let Some(cid) = &local_being.avatar_cid {
        cmd_tx
            .send(NetworkCommand::ProvideMediaCid { cid: cid.clone() })
            .await?;
    }

    // Optionally co-host an LLM agent Being in this process. It gets its
    // own keypair/PeerId/engine, so to the network it is just another peer.
    if let Some(agent_cli) = agent_cli {
        let agent_keypair = db.get_or_create_agent_keypair()?;
        let agent_peer_id = agent_keypair.public().to_peer_id();
        let agent_being =
            crate::agent::get_or_create_agent_being(&db, agent_peer_id, &agent_cli.name)?;
        println!(
            "Agent initialized for {} ({})",
            agent_being.human_name, agent_peer_id
        );

        let mut agent_swarm = build_swarm(agent_keypair)?;
        agent_swarm
            .behaviour_mut()
            .kademlia
            .set_mode(Some(kad::Mode::Server));
        agent_swarm
            .behaviour_mut()
            .gossipsub
            .subscribe(&chat_topic)?;
        agent_swarm.listen_on("/ip4/0.0.0.0/udp/0/quic-v1".parse()?)?;
        agent_swarm.listen_on("/ip4/0.0.0.0/tcp/0".parse()?)?;

        let (agent_cmd_tx, agent_cmd_rx) = mpsc::channel::<NetworkCommand>(32);
        let (agent_event_tx, agent_event_rx) = mpsc::channel::<NetworkEvent>(32);

        let agent_engine = NetworkEngine::new(agent_swarm, agent_cmd_rx, agent_event_tx);
        tokio::spawn(async move {
            agent_engine.run().await;
        });

        let backend = OpenAiCompatible::new(agent_cli.llm_url, agent_cli.model);
        let agent_db = Database::open(&db_path_str)?;
        let config = AgentConfig {
            name: agent_being.human_name.clone(),
            ..Default::default()
        };
        tokio::spawn(crate::agent::run_agent_loop(
            config,
            backend,
            agent_db,
            agent_cmd_tx,
            agent_event_rx,
            chat_topic.clone(),
            agent_peer_id,
        ));
    }

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
                    NetworkEvent::MediaCidFound { cid, providers } => {
                        println!("Providers for {cid}: {providers:?}");
                    }
                    _ => {}
                }
            }
        }
    }
}
