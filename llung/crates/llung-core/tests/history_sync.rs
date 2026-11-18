//! End-to-end history exchange: node A holds a two-message branch;
//! node B connects and must receive both via HistoryReceived — exactly
//! what two devices of one account do when they find each other.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use libp2p::{Multiaddr, PeerId, gossipsub::IdentTopic, identity::Keypair};
use llung_core::app::build_swarm;
use llung_core::identity::being::{BeingKind, BeingStatus, PresenceMessage};
use llung_core::network::dns::DnsSetup;
use llung_core::network::message::{ChatMessage, MessageKind};
use llung_core::network::sync::{SyncRequest, SyncResponse, MAX_SYNC_MESSAGES};
use llung_core::network::{NetworkCommand, NetworkEngine, NetworkEvent};
use tokio::sync::{mpsc, oneshot};

const PRESENCE_TOPIC: &str = "presence-test";
const CHAT_TOPIC: &str = "history-sync-test";

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

/// In-memory history store + SyncRequest answerer (same logic as the
/// app's DB-backed responder; keeps the engine DB-free).
struct Store {
    messages: HashMap<String, ChatMessage>,
    tips: HashMap<String, String>,
}

impl Store {
    fn insert(&mut self, msg: ChatMessage) {
        self.tips.insert(msg.topic.clone(), msg.id.clone());
        self.messages.insert(msg.id.clone(), msg);
    }

    fn answer(&self, request: &SyncRequest) -> SyncResponse {
        match request {
            SyncRequest::Chain { topic, from, limit } => {
                let limit = (*limit).clamp(1, MAX_SYNC_MESSAGES);
                let mut messages = Vec::new();
                let mut cursor = from.clone().or_else(|| self.tips.get(topic).cloned());
                while messages.len() < limit {
                    let Some(id) = cursor.clone() else { break };
                    let Some(node) = self.messages.get(&id) else {
                        cursor = None;
                        break;
                    };
                    if node.topic != *topic {
                        cursor = None;
                        break;
                    }
                    messages.push(node.clone());
                    cursor = node.parent_id.clone();
                }
                SyncResponse::Chain {
                    topic: topic.clone(),
                    messages,
                    more_parent: cursor,
                }
            }
            SyncRequest::Missing { ids } => SyncResponse::Missing {
                messages: ids
                    .iter()
                    .filter_map(|id| self.messages.get(id).cloned())
                    .collect(),
            },
        }
    }
}

fn spawn_node(
    name: &str,
    store: Arc<Mutex<Store>>,
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

    // History source: answer from the shared store.
    let (history_tx, mut history_rx) =
        mpsc::channel::<llung_core::network::event::HistoryQuery>(16);
    tokio::spawn(async move {
        while let Some(q) = history_rx.recv().await {
            let resp = store.lock().unwrap().answer(&q.request);
            let _ = q.respond.send(resp);
        }
    });

    let (cmd_tx, cmd_rx) = mpsc::channel(32);
    let (event_tx, event_rx) = mpsc::channel(32);
    let engine = NetworkEngine::new(
        swarm,
        cmd_rx,
        event_tx,
        presence(name, peer_id),
        IdentTopic::new(PRESENCE_TOPIC),
        None,
        Vec::new(),
        Some(history_tx),
    );
    tokio::spawn(async move { engine.run().await });

    // Subscribe to the chat topic so the engine syncs it on connect.
    let tx = cmd_tx.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        let _ = tx
            .send(NetworkCommand::SubscribeTopic {
                topic: IdentTopic::new(CHAT_TOPIC),
            })
            .await;
    });

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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_nodes_sync_branch_history_on_connect() {
    let _ = tracing_subscriber::fmt().with_test_writer().with_max_level(tracing::Level::DEBUG).try_init();

    // A holds a branch: m1 <- m2. B starts empty.
    let store_a = Arc::new(Mutex::new(Store {
        messages: HashMap::new(),
        tips: HashMap::new(),
    }));
    let sender = PeerId::from(Keypair::generate_ed25519().public());
    let m1 = ChatMessage::new(
        CHAT_TOPIC.into(),
        sender,
        None,
        MessageKind::Human,
        "alice".into(),
        "first".into(),
    );
    let m2 = ChatMessage::new(
        CHAT_TOPIC.into(),
        sender,
        Some(m1.id.clone()),
        MessageKind::Human,
        "alice".into(),
        "second".into(),
    );
    store_a.lock().unwrap().insert(m1.clone());
    store_a.lock().unwrap().insert(m2.clone());

    let store_b = Arc::new(Mutex::new(Store {
        messages: HashMap::new(),
        tips: HashMap::new(),
    }));

    let (a_cmd, _a_events, a_peer) = spawn_node("a", store_a);
    let (_b_cmd, mut b_events, _b_peer) = spawn_node("b", store_b);

    // B dials A → both fire history sync on ConnectionEstablished.
    for addr in listen_addrs(&a_cmd).await {
        _b_cmd
            .send(NetworkCommand::Dial {
                multiaddr: addr.with(libp2p::multiaddr::Protocol::P2p(a_peer)),
            })
            .await
            .expect("dial command");
    }

    // B must receive both messages (order across pages can vary, but
    // this branch fits one Chain page: newest first).
    let mut received = Vec::new();
    let deadline = tokio::time::sleep(Duration::from_secs(20));
    tokio::pin!(deadline);
    loop {
        tokio::select! {
            Some(ev) = b_events.recv() => {
                if let NetworkEvent::HistoryReceived { messages, .. } = ev {
                    received.extend(messages);
                    if received.iter().any(|m| m.id == m1.id)
                        && received.iter().any(|m| m.id == m2.id)
                    {
                        break;
                    }
                }
            }
            _ = &mut deadline => panic!("timed out waiting for history sync (got {received:?})"),
        }
    }
    let contents: Vec<&str> = received.iter().map(|m| m.content.as_str()).collect();
    assert!(contents.contains(&"first"), "got {contents:?}");
    assert!(contents.contains(&"second"), "got {contents:?}");
    // Newest first within the page.
    let pos: HashMap<&str, usize> = contents
        .iter()
        .enumerate()
        .map(|(i, c)| (*c, i))
        .collect();
    assert!(pos["second"] < pos["first"], "newest first: {contents:?}");
}
