//! Peer-to-peer image transfer with NO relay in the loop: A publishes
//! an image to its connected peer B (direct Put), B caches it via the
//! ImagePushed event, and C fetches it from B. Group images should live
//! in the group — relays are the fallback, not the path.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use libp2p::{Multiaddr, PeerId, gossipsub::IdentTopic, identity::Keypair};
use llung_core::app::build_swarm;
use llung_core::identity::being::{BeingKind, BeingStatus, PresenceMessage};
use llung_core::media::image::image_cid;
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

/// Node with a DB-free image cache: inbound Gets answer from `cache`,
/// inbound Puts land in `cache` via the ImagePushed event.
fn spawn_node(
    name: &str,
    cache: Arc<Mutex<HashMap<String, (String, Vec<u8>)>>>,
) -> (
    mpsc::Sender<NetworkCommand>,
    mpsc::Receiver<NetworkEvent>,
    PeerId,
) {
    let mut swarm = build_swarm(Keypair::generate_ed25519(), DnsSetup::none("test"))
        .expect("swarm builds");
    let peer_id = *swarm.local_peer_id();
    swarm
        .listen_on("/ip4/127.0.0.1/tcp/0".parse().unwrap())
        .expect("tcp listen");

    let (query_tx, mut query_rx) =
        mpsc::channel::<llung_core::network::event::ImageQuery>(16);
    let cache2 = cache.clone();
    tokio::spawn(async move {
        while let Some(q) = query_rx.recv().await {
            let _ = q.respond.send(cache2.lock().unwrap().get(&q.cid).cloned());
        }
    });

    let (cmd_tx, cmd_rx) = mpsc::channel(32);
    let (event_tx, event_rx) = mpsc::channel(32);
    let engine = NetworkEngine::new(
        swarm,
        cmd_rx,
        event_tx,
        presence(name, peer_id),
        IdentTopic::new("img-p2p-test"),
        Some(query_tx),
        Vec::new(),
        None,
    );
    tokio::spawn(async move { engine.run().await });

    // Inbound Puts arrive on the event receiver the caller gets back;
    // the test body drives caching from there (like the real app).
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

async fn dial(cmd: &mpsc::Sender<NetworkCommand>, addrs: Vec<Multiaddr>, peer: PeerId) {
    for addr in addrs {
        cmd.send(NetworkCommand::Dial {
            multiaddr: addr.with(libp2p::multiaddr::Protocol::P2p(peer)),
        })
        .await
        .expect("dial");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn image_flows_peer_to_peer_without_relay() {
    let _ = tracing_subscriber::fmt().with_test_writer().try_init();

    // Shared cache for B (the middle peer). A and C never share stores.
    let cache_b: Arc<Mutex<HashMap<String, (String, Vec<u8>)>>> =
        Arc::new(Mutex::new(HashMap::new()));
    let cache_a: Arc<Mutex<HashMap<String, (String, Vec<u8>)>>> =
        Arc::new(Mutex::new(HashMap::new()));
    let cache_c: Arc<Mutex<HashMap<String, (String, Vec<u8>)>>> =
        Arc::new(Mutex::new(HashMap::new()));
    let cache_c_check = cache_c.clone();

    let (a_cmd, mut a_events, _a_peer) = spawn_node("a", cache_a);
    let (b_cmd, mut b_events, b_peer) = spawn_node("b", cache_b.clone());
    let (c_cmd, mut c_events, _c_peer) = spawn_node("c", cache_c);

    // Topology: A—B and C—B. A and C never connect to each other.
    dial(&a_cmd, listen_addrs(&b_cmd).await, b_peer).await;
    dial(&c_cmd, listen_addrs(&b_cmd).await, b_peer).await;

    // A publishes; B must receive the Put via the ImagePushed event and
    // cache it (its Gets are served from that cache).
    let data = b"\x89PNG\r\n\x1a\np2p-image".to_vec();
    let cid = image_cid(&data);
    a_cmd
        .send(NetworkCommand::PublishImage {
            cid: cid.clone(),
            mime_type: "image/png".into(),
            data: data.clone(),
        })
        .await
        .unwrap();

    // Drive B's cache from its events until the Put lands.
    let cache = cache_b.clone();
    let cid2 = cid.clone();
    let b_fed = tokio::spawn(async move {
        let deadline = tokio::time::sleep(Duration::from_secs(15));
        tokio::pin!(deadline);
        loop {
            tokio::select! {
                Some(ev) = b_events.recv() => {
                    if let NetworkEvent::ImagePushed { cid, mime_type, data, .. } = ev {
                        cache.lock().unwrap().insert(cid, (mime_type, data));
                        if cache.lock().unwrap().contains_key(&cid2) {
                            return true;
                        }
                    }
                }
                _ = &mut deadline => return false,
            }
        }
    });
    assert!(b_fed.await.unwrap(), "B never received the image Put");

    // C fetches: peers (B) first, no relay involved anywhere.
    let (tx, rx) = oneshot::channel();
    c_cmd
        .send(NetworkCommand::GetImage {
            cid: cid.clone(),
            responder: tx,
        })
        .await
        .unwrap();
    let (mime, bytes) = tokio::time::timeout(Duration::from_secs(15), rx)
        .await
        .expect("timed out")
        .expect("responder dropped")
        .expect("no peer had the image");
    assert_eq!(mime, "image/png");
    assert_eq!(bytes, data);

    // A and C's own caches must still be empty — the bytes flowed B→C
    // peer-to-peer, not through any shared store on A/C.
    drop(a_events);
    drop(c_events);
    assert!(cache_c_check.lock().unwrap().is_empty(), "C served itself");
}
