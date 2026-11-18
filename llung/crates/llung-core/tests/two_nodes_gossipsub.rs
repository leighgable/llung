//! Two engine nodes in one process (exactly what the TUI + spawned agent
//! do): both subscribe to the same topic, one publishes, the other must
//! receive. Reproduces the user's "agent never answers" scenario at the
//! network layer — if this hangs, the bug is below the agent.

use libp2p::{PeerId, gossipsub::IdentTopic, identity::Keypair};
use llung_core::identity::being::{BeingKind, BeingStatus, PresenceMessage};
use llung_core::app::build_swarm;
use llung_core::network::{
    NetworkCommand, NetworkEngine, NetworkEvent,
    dns::DnsSetup,
};
use std::time::Duration;
use tokio::sync::mpsc;

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

fn spawn_node(
    name: &str,
) -> (
    mpsc::Sender<NetworkCommand>,
    mpsc::Receiver<NetworkEvent>,
    PeerId,
) {
    let mut swarm = build_swarm(Keypair::generate_ed25519(), DnsSetup::none("test"))
        .expect("swarm builds");
    let peer_id = *swarm.local_peer_id();
    let presence = presence(name, peer_id);
    swarm
        .listen_on("/ip4/0.0.0.0/tcp/0".parse().unwrap())
        .expect("tcp listen");
    let (cmd_tx, cmd_rx) = mpsc::channel(32);
    let (event_tx, event_rx) = mpsc::channel(32);
    let engine = NetworkEngine::new(
        swarm,
        cmd_rx,
        event_tx,
        presence,
        IdentTopic::new("presence-test"),
        None,
        Vec::new(),
        None,
    );
    tokio::spawn(async move { engine.run().await });
    (cmd_tx, event_rx, peer_id)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_local_nodes_exchange_topic_messages() {
    let _ = tracing_subscriber::fmt()
        .with_test_writer()
        .with_max_level(tracing::Level::DEBUG)
        .try_init();

    let (tx_a, mut rx_a, peer_a) = spawn_node("alice");
    let (tx_b, mut rx_b, peer_b) = spawn_node("bob");

    let topic = IdentTopic::new("plotting");
    tx_a.send(NetworkCommand::SubscribeTopic { topic: topic.clone() })
        .await
        .unwrap();
    tx_b.send(NetworkCommand::SubscribeTopic { topic: topic.clone() })
        .await
        .unwrap();

    // Same-process wiring, exactly what the TUI does for a spawned agent:
    // hand B our listen addresses instead of relying on mDNS multicast
    // (broken on hosts that filter multicast).
    let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
    tx_a.send(NetworkCommand::GetListenAddrs { responder: resp_tx })
        .await
        .unwrap();
    let addrs = resp_rx.await.unwrap();
    assert!(!addrs.is_empty(), "node A must have listen addresses");
    for addr in addrs {
        tx_b.send(NetworkCommand::Dial { multiaddr: addr }).await.unwrap();
    }

    // Give mDNS discovery, dial, and gossipsub mesh formation time.
    // The mesh grafts on heartbeat (10s), so allow several beats.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(60);
    let mut received = false;
    let mut publishes = 0;
    while !received && tokio::time::Instant::now() < deadline {
        publishes += 1;
        tx_a.send(NetworkCommand::PublishMessage {
            topic: topic.clone(),
            contents: format!("hello from alice, attempt {publishes}").into_bytes(),
        })
        .await
        .unwrap();

        // Drain B's events looking for our message.
        let wait = tokio::time::sleep(Duration::from_secs(2));
        tokio::pin!(wait);
        loop {
            tokio::select! {
                _ = &mut wait => break,
                maybe = rx_b.recv() => {
                    match maybe {
                        Some(NetworkEvent::MessageReceived { data, sender, .. }) => {
                            let text = String::from_utf8_lossy(&data).into_owned();
                            if text.contains("attempt") {
                                assert_eq!(sender, peer_a);
                                received = true;
                                break;
                            }
                        }
                        Some(_) => {} // presence etc.
                        None => panic!("node B's engine task died"),
                    }
                }
            }
        }
    }

    assert!(
        received,
        "node B never received a published message after {publishes} attempts"
    );

    // And the reverse direction, once the mesh exists, should be quick.
    tx_b.send(NetworkCommand::PublishMessage {
        topic: topic.clone(),
        contents: b"reply from bob".to_vec(),
    })
    .await
    .unwrap();
    let mut got_reply = false;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
    while !got_reply && tokio::time::Instant::now() < deadline {
        let wait = tokio::time::sleep(Duration::from_millis(500));
        tokio::pin!(wait);
        loop {
            tokio::select! {
                _ = &mut wait => break,
                maybe = rx_a.recv() => match maybe {
                    Some(NetworkEvent::MessageReceived { data, sender, .. }) => {
                        if data == b"reply from bob" {
                            assert_eq!(sender, peer_b);
                            got_reply = true;
                            break;
                        }
                    }
                    Some(_) => {}
                    None => panic!("node A's engine task died"),
                },
            }
        }
    }
    assert!(got_reply, "node A never received bob's reply");
}
