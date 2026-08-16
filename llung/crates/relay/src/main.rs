use std::error::Error;
use std::time::Duration;

use libp2p::kad::store::MemoryStore;
use libp2p::swarm::NetworkBehaviour;
use libp2p::{
    Multiaddr, PeerId, SwarmBuilder, identify, identity::Keypair, kad, ping, relay,
    swarm::SwarmEvent,
};

#[derive(NetworkBehaviour)]
pub struct RelayServerBehaviour {
    pub kademlia: kad::Behaviour<kad::store::MemoryStore>,
    pub relay: relay::Behaviour,
    pub identify: identify::Behaviour,
    pub ping: ping::Behaviour,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt::init();

    // 1. Generate persistent or ephemeral server keypair
    let keypair = Keypair::generate_ed25519();
    let local_peer_id = PeerId::from(keypair.public());
    println!("Starting Llung Relay Server...");
    println!("Server Peer ID: {local_peer_id}");

    // 2. Build libp2p Swarm with TCP + Noise + Yamux
    let mut swarm = SwarmBuilder::with_existing_identity(keypair)
        .with_tokio()
        .with_tcp(
            libp2p::tcp::Config::default(),
            libp2p::noise::Config::new,
            libp2p::yamux::Config::default,
        )?
        .with_behaviour(|key| {
            // Configure Kademlia in Server Mode (essential for DHT caching)
            let store = MemoryStore::new(local_peer_id);
            let mut kad_config = kad::Config::default();
            kad_config.set_protocol_names(vec![std::borrow::Cow::Borrowed(b"/llung/kad/1.0.0")]);

            let mut kademlia = kad::Behaviour::with_config(local_peer_id, store, kad_config);
            kademlia.set_mode(Some(kad::Mode::Server));

            // Configure Circuit Relay v2 Server
            let relay_config = relay::Config {
                max_reservations: 128,
                max_circuits_per_peer: 8,
                reservation_duration: Duration::from_secs(3600),
                ..Default::default()
            };
            let relay = relay::Behaviour::new(local_peer_id, relay_config);

            // Configure Identify (Required for peers to discover their observed NAT address)
            let identify = identify::Behaviour::new(identify::Config::new(
                "/llung/1.0.0".into(),
                key.public(),
            ));

            let ping = ping::Behaviour::default();

            Ok(RelayServerBehaviour {
                kademlia,
                relay,
                identify,
                ping,
            })
        })?
        .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(60)))
        .build();

    // 3. Listen on public TCP interface
    let listen_addr: Multiaddr = "/ip4/0.0.0.0/tcp/4001".parse()?;
    swarm.listen_on(listen_addr.clone())?;

    println!("📡 Listening on port 4001...");

    // 4. Main Event Loop
    loop {
        match swarm.select_next_some().await {
            SwarmEvent::NewListenAddr { address, .. } => {
                let p2p_addr = address.with(libp2p::multiaddr::Protocol::P2p(local_peer_id));
                println!("🌐 Relay Server Multiaddr: {p2p_addr}");
            }

            SwarmEvent::Behaviour(RelayServerBehaviourEvent::Identify(
                identify::Event::Received { peer_id, info },
            )) => {
                // Add peer's listening addresses to Kademlia routing table
                for addr in info.listen_addrs {
                    swarm.behaviour_mut().kademlia.add_address(&peer_id, addr);
                }
            }

            SwarmEvent::Behaviour(RelayServerBehaviourEvent::Relay(relay_event)) => {
                println!("🔄 Relay Event: {relay_event:?}");
            }

            _ => {}
        }
    }
}
