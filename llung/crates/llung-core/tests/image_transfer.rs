//! End-to-end image transfer over `/llung/img/1.0.0`: a node advertising
//! the relay agent_version caches a Put; a second node Gets it back.
//! Exercises relay detection (identify agent_version), the Put/Get
//! flow, validation, and the image_source callback.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use libp2p::{Multiaddr, PeerId, gossipsub::IdentTopic, identity::Keypair};
use llung_core::app::build_swarm_with_agent_version;
use llung_core::identity::being::{BeingKind, BeingStatus, PresenceMessage};
use llung_core::media::image::{image_cid, RELAY_AGENT_VERSION};
use llung_core::network::dns::DnsSetup;
use llung_core::network::{NetworkCommand, NetworkEngine, NetworkEvent};
use tokio::sync::{mpsc, oneshot};

fn presence(name: &str, peer_id: PeerId) -> PresenceMessage {
    PresenceMessage {
        being_id: peer_id.to_base58(),
        machine_id: peer_id,
        human_name: name.into(),
        kind: BeingKind::Human,
        status: BeingStatus::Available,
        avatar_cid: None,
        enc_public_key: None,
        listen_addrs: Vec::new(),
    }
}

/// A test node: engine + channels. Returns (commands, events, peer_id).
fn spawn_node(
    name: &str,
    agent_version: &str,
    image_source: Option<mpsc::Sender<llung_core::network::event::ImageQuery>>,
) -> (
    mpsc::Sender<NetworkCommand>,
    mpsc::Receiver<NetworkEvent>,
    PeerId,
) {
    let mut swarm = build_swarm_with_agent_version(
        Keypair::generate_ed25519(),
        DnsSetup::none("test"),
        agent_version,
    )
    .expect("swarm builds");
    let peer_id = *swarm.local_peer_id();
    swarm
        .listen_on("/ip4/127.0.0.1/tcp/0".parse().unwrap())
        .expect("tcp listen");
    let (cmd_tx, cmd_rx) = mpsc::channel(32);
    let (event_tx, event_rx) = mpsc::channel(32);
    let engine = NetworkEngine::new(
        swarm,
        cmd_rx,
        event_tx,
        presence(name, peer_id),
        IdentTopic::new("img-test"),
        image_source,
        Vec::new(),
        None,
    );
    tokio::spawn(async move { engine.run().await });
    (cmd_tx, event_rx, peer_id)
}

async fn listen_addrs(cmd: &mpsc::Sender<NetworkCommand>) -> Vec<Multiaddr> {
    for _ in 0..50 {
        let (tx, rx) = oneshot::channel();
        if cmd.send(NetworkCommand::GetListenAddrs { responder: tx }).await.is_ok() {
            if let Ok(addrs) = rx.await {
                if !addrs.is_empty() {
                    return addrs;
                }
            }
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("node never listened");
}

/// Ask the client for an image; Ok(Some) means the relay set served it.
async fn get_image(
    client: &mpsc::Sender<NetworkCommand>,
    cid: &str,
) -> Option<(String, Vec<u8>)> {
    let (tx, rx) = oneshot::channel();
    client
        .send(NetworkCommand::GetImage {
            cid: cid.into(),
            responder: tx,
        })
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(10), rx)
        .await
        .ok()?
        .ok()?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn put_then_get_round_trip_via_relay_marked_peer() {
    let _ = tracing_subscriber::fmt().with_test_writer().try_init();

    // Shared stand-in for the relay's SQLite store: seeded with one
    // entry (serves pre-detection Get probes), grown via ImagePushed
    // events (the real relay path: engine validates the Put, emits the
    // event, the store layer persists).
    let cache: Arc<Mutex<HashMap<String, (String, Vec<u8>)>>> = Arc::new(Mutex::new(
        HashMap::from([("seed".to_string(), ("image/png".to_string(), b"seeded".to_vec()))]),
    ));
    let cache2 = cache.clone();

    let (query_tx, mut query_rx) =
        mpsc::channel::<llung_core::network::event::ImageQuery>(8);
    tokio::spawn(async move {
        while let Some(q) = query_rx.recv().await {
            let _ = q.respond.send(cache2.lock().unwrap().get(&q.cid).cloned());
        }
    });

    let (relay_cmd, mut relay_events, relay_peer) =
        spawn_node("relay", RELAY_AGENT_VERSION, Some(query_tx));
    let (client_cmd, _client_events, _client_peer) = spawn_node("client", "/llung/1.0.0", None);

    // Relay consumes its events: ImagePushed entries land in the store.
    let cache3 = cache.clone();
    tokio::spawn(async move {
        while let Some(ev) = relay_events.recv().await {
            if let NetworkEvent::ImagePushed {
                cid, mime_type, data, ..
            } = ev
            {
                cache3.lock().unwrap().insert(cid, (mime_type, data));
            }
        }
    });

    // Client dials the relay by its full /p2p/ multiaddr.
    for addr in listen_addrs(&relay_cmd).await {
        client_cmd
            .send(NetworkCommand::Dial {
                multiaddr: addr.with(libp2p::multiaddr::Protocol::P2p(relay_peer)),
            })
            .await
            .expect("dial command");
    }

    // Phase 1: wait for identify to mark the peer as a relay (the seed
    // Get returns Some only once the relay set is populated).
    let mut detected = false;
    for _ in 0..50 {
        if get_image(&client_cmd, "seed").await.is_some() {
            detected = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    assert!(detected, "relay peer was never detected via identify");

    // Phase 2: Put a real image, wait for it to land in the relay's
    // store via the ImagePushed event, then Get it back.
    let data = b"\x89PNG\r\n\x1a\nend-to-end".to_vec();
    let cid = image_cid(&data);
    client_cmd
        .send(NetworkCommand::PublishImage {
            cid: cid.clone(),
            mime_type: "image/png".into(),
            data: data.clone(),
        })
        .await
        .unwrap();

    let mut served = None;
    for _ in 0..50 {
        if let Some((mime, bytes)) = get_image(&client_cmd, &cid).await {
            served = Some((mime, bytes));
            break;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    let (mime, bytes) = served.expect("put image must be served back by the relay");
    assert_eq!(mime, "image/png");
    assert_eq!(bytes, data);
}
