// crates/relay/src/main.rs
//
// The llung relay server — the always-on friend on a €5 VPS:
//
// - Circuit Relay v2: NAT'd peers reserve slots here and tunnel traffic
//   until DCUtR hole punching connects them directly.
// - Kademlia server mode: a stable DHT peer for discovery.
// - Image cache (`/llung/img/1.0.0`): lightweight avatars and message
//   image attachments, validated puts, TTL + count caps, SQLite-backed.
// - Persistent identity: the PeerId MUST stay stable, so wrap the key
//   file into your backup/monitoring.
//
// Run: LLUNG_RELAY_PORT=4001 relay
// Clients point at it with: LLUNG_RELAY=/ip4/<host>/tcp/4001/p2p/<peer-id>

mod config;
mod image_store;

use std::collections::HashSet;
use std::error::Error;
use std::time::Duration;

use futures::StreamExt;
use libp2p::kad::store::MemoryStore;
use libp2p::request_response::{self, Config as ReqResConfig, ProtocolSupport};
use libp2p::swarm::NetworkBehaviour;
use libp2p::{
    Multiaddr, PeerId, StreamProtocol, SwarmBuilder, identify,
    identity::Keypair, kad, ping, relay, swarm::SwarmEvent,
};
use llung_core::media::image::{ImageRequest, ImageResponse, RELAY_AGENT_VERSION};

use crate::config::RelayConfig;
use crate::image_store::ImageStore;

type ImageBehaviour = libp2p::request_response::cbor::Behaviour<ImageRequest, ImageResponse>;

#[derive(NetworkBehaviour)]
struct RelayServerBehaviour {
    kademlia: kad::Behaviour<kad::store::MemoryStore>,
    relay: relay::Behaviour,
    identify: identify::Behaviour,
    ping: ping::Behaviour,
    image_transfer: ImageBehaviour,
}

/// Load the server keypair from disk, generating + persisting it on
/// first run. A fresh keypair every boot would rotate the PeerId and
/// strand every client's relay multiaddr.
fn load_or_create_key(path: &std::path::Path) -> Result<Keypair, Box<dyn Error>> {
    if let Ok(bytes) = std::fs::read(path) {
        if let Ok(keypair) = Keypair::from_protobuf_encoding(&bytes) {
            return Ok(keypair);
        }
        tracing::warn!("key file at {} is unreadable; generating a new one", path.display());
    }
    let keypair = Keypair::generate_ed25519();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, keypair.to_protobuf_encoding()?)?;
    tracing::info!("new relay key written to {}", path.display());
    Ok(keypair)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "llung_relay=info,info".into()),
        )
        .init();

    let config = RelayConfig::from_env();
    let keypair = load_or_create_key(&config.key_path)?;
    let local_peer_id = PeerId::from(keypair.public());
    tracing::info!(%local_peer_id, port = config.port, "llung relay starting");

    let mut store = ImageStore::open(&config.db_path, config.max_images, config.image_ttl)?;
    tracing::info!(images = store.len(), db = %config.db_path.display(), "image cache open");

    let mut swarm = SwarmBuilder::with_existing_identity(keypair)
        .with_tokio()
        .with_tcp(
            libp2p::tcp::Config::default(),
            libp2p::noise::Config::new,
            libp2p::yamux::Config::default,
        )?
        .with_quic()
        .with_behaviour(|key| {
            let store_mem = MemoryStore::new(local_peer_id);
            let mut kademlia = kad::Behaviour::new(local_peer_id, store_mem);
            kademlia.set_mode(Some(kad::Mode::Server));

            let relay = relay::Behaviour::new(
                local_peer_id,
                relay::Config {
                    max_reservations: config.max_reservations,
                    max_circuits_per_peer: config.max_circuits_per_peer,
                    reservation_duration: Duration::from_secs(3600),
                    ..Default::default()
                },
            );

            // agent_version marks us as a relay so clients route image
            // traffic here; protocol_version stays the shared llung one.
            let identify = identify::Behaviour::new(
                identify::Config::new("/llung/1.0.0".into(), key.public())
                    .with_agent_version(RELAY_AGENT_VERSION.to_string()),
            );

            let image_transfer = ImageBehaviour::new(
                [(
                    StreamProtocol::new(llung_core::media::image::IMAGE_PROTOCOL),
                    ProtocolSupport::Full,
                )],
                ReqResConfig::default().with_request_timeout(Duration::from_secs(30)),
            );

            Ok(RelayServerBehaviour {
                kademlia,
                relay,
                identify,
                ping: ping::Behaviour::default(),
                image_transfer,
            })
        })?
        .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(120)))
        .build();

    // TCP + QUIC on the public interface. QUIC matters: DCUtR hole
    // punching needs UDP paths.
    swarm.listen_on(format!("/ip4/0.0.0.0/tcp/{}", config.port).parse::<Multiaddr>()?)?;
    swarm.listen_on(format!("/ip4/0.0.0.0/udp/{}/quic-v1", config.port).parse::<Multiaddr>()?)?;

    let mut connected: HashSet<PeerId> = HashSet::new();
    let mut stats_tick = tokio::time::interval(Duration::from_secs(60));
    let mut sweep_tick = tokio::time::interval(Duration::from_secs(3600));
    // Per-peer PUT burst counter, reset each interval.
    let mut puts_this_interval: std::collections::HashMap<PeerId, u32> =
        std::collections::HashMap::new();

    loop {
        tokio::select! {
            event = swarm.select_next_some() => match event {
                SwarmEvent::NewListenAddr { address, .. } => {
                    let p2p_addr = address.with(libp2p::multiaddr::Protocol::P2p(local_peer_id));
                    tracing::info!(%p2p_addr, "listening");
                    println!("🌐 Relay multiaddr: {p2p_addr}");
                }
                SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                    connected.insert(peer_id);
                }
                SwarmEvent::ConnectionClosed { peer_id, .. } => {
                    connected.remove(&peer_id);
                    puts_this_interval.remove(&peer_id);
                }
                SwarmEvent::Behaviour(RelayServerBehaviourEvent::Identify(
                    identify::Event::Received { peer_id, info, .. },
                )) => {
                    for addr in info.listen_addrs {
                        swarm.behaviour_mut().kademlia.add_address(&peer_id, addr);
                    }
                }
                SwarmEvent::Behaviour(RelayServerBehaviourEvent::Relay(relay_event)) => {
                    // Reservation/circuit lifecycle; logged at debug so a
                    // busy server doesn't flood the journal.
                    tracing::debug!(?relay_event, "relay");
                }
                SwarmEvent::Behaviour(RelayServerBehaviourEvent::ImageTransfer(event)) => {
                    handle_image_event(
                        event,
                        &mut swarm.behaviour_mut().image_transfer,
                        &mut store,
                        &config,
                        &mut puts_this_interval,
                    )
                    .await;
                }
                _ => {}
            },
            _ = stats_tick.tick() => {
                tracing::info!(
                    peers = connected.len(),
                    images = store.len(),
                    reservations = config.max_reservations,
                    "relay stats"
                );
                puts_this_interval.clear();
            }
            _ = sweep_tick.tick() => {
                let removed = store.sweep();
                if removed > 0 {
                    tracing::info!(removed, "expired images swept");
                }
            }
        }
    }
}

async fn handle_image_event(
    event: request_response::Event<ImageRequest, ImageResponse>,
    behaviour: &mut ImageBehaviour,
    store: &mut ImageStore,
    config: &RelayConfig,
    puts: &mut std::collections::HashMap<PeerId, u32>,
) {
    use request_response::Event;
    match event {
        Event::Message { peer, message, .. } => match message {
            request_response::Message::Request {
                request, channel, ..
            } => {
                let response = match request {
                    ImageRequest::Put {
                        cid,
                        mime_type,
                        data,
                    } => {
                        let count = puts.entry(peer).or_insert(0);
                        let response = if *count >= config.max_puts_per_peer {
                            tracing::warn!(%peer, "PUT burst cap hit; dropping");
                            ImageResponse::PutAck {
                                accepted: false,
                                reason: Some("rate limited".into()),
                            }
                        } else {
                            *count += 1;
                            match store.put(&cid, &mime_type, &data) {
                                Ok(()) => {
                                    tracing::debug!(%peer, %cid, bytes = data.len(), "image cached");
                                    ImageResponse::PutAck {
                                        accepted: true,
                                        reason: None,
                                    }
                                }
                                Err(reason) => ImageResponse::PutAck {
                                    accepted: false,
                                    reason: Some(reason),
                                },
                            }
                        };
                        response
                    }
                    ImageRequest::Get { cid } => match store.get(&cid) {
                        Some((mime, data)) => {
                            tracing::debug!(%peer, %cid, bytes = data.len(), "image served");
                            ImageResponse::Got {
                                mime_type: mime,
                                data,
                            }
                        }
                        None => ImageResponse::NotFound,
                    },
                };
                let _ = behaviour.send_response(channel, response);
            }
            request_response::Message::Response { .. } => {}
        },
        Event::OutboundFailure { peer, error, .. } => {
            tracing::debug!(%peer, %error, "image request outbound failure");
        }
        Event::InboundFailure { peer, error, .. } => {
            tracing::debug!(%peer, %error, "image request inbound failure");
        }
        _ => {}
    }
}


