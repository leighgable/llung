use crate::{
    config::CoreConfig,
    identity::{
        being::{Being, PresenceMessage},
        machine::MachineIdentity,
    },
    network::{
        NetworkCommand, NetworkEngine, NetworkEvent,
        behaviour::{
            DirectMessageBehaviour, HistorySyncBehaviour, ImageTransferBehaviour, LlungBehaviour,
        },
        dns::{self, DnsSetup},
        event::{HistoryQuery, ImageQuery},
        sync::SYNC_PROTOCOL,
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

/// Answer a history-sync request from the local DAG. `Chain` walks
/// parents newest-first and stops at the limit or at a hole (returning
/// `more_parent: None` there — an unknown node can't be traversed, and
/// re-asking would loop).
fn answer_sync(
    dag: &crate::storage::dag::ChatDag,
    request: &crate::network::sync::SyncRequest,
) -> crate::network::sync::SyncResponse {
    use crate::network::sync::{SyncRequest, SyncResponse, MAX_SYNC_MESSAGES};
    match request {
        SyncRequest::Chain { topic, from, limit } => {
            let limit = (*limit).clamp(1, MAX_SYNC_MESSAGES);
            let mut messages = Vec::new();
            let mut cursor = from.clone().or_else(|| dag.tip_of(topic));
            while messages.len() < limit {
                let Some(id) = cursor.clone() else { break };
                let Some(node) = dag.nodes.get(&id) else {
                    // Hole: stop and signal genesis so the requester
                    // doesn't re-ask a node we can't traverse.
                    cursor = None;
                    break;
                };
                if node.message.topic != *topic {
                    // Parent escaped the topic (legacy unparented
                    // traffic); the branch ends here.
                    cursor = None;
                    break;
                }
                messages.push(node.message.clone());
                cursor = node.message.parent_id.clone();
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
                .filter_map(|id| dag.nodes.get(id).map(|n| n.message.clone()))
                .collect(),
        },
    }
}

/// Build the swarm. `dns` comes from [`dns::system`]; when DNS is
/// unavailable it's a resolver without nameservers, so only `/dns*` dials
/// fail (see `network::dns`).
pub fn build_swarm(
    keypair: Keypair,
    dns: DnsSetup,
) -> Result<Swarm<LlungBehaviour>, Box<dyn Error>> {
    build_swarm_with_agent_version(keypair, dns, "/llung/1.0.0")
}

/// Like [`build_swarm`], with an explicit identify agent_version. Relay
/// servers advertise `media::image::RELAY_AGENT_VERSION` so clients route
/// image traffic to them; tests use this to stage a "relay".
pub fn build_swarm_with_agent_version(
    keypair: Keypair,
    dns: DnsSetup,
    agent_version: &str,
) -> Result<Swarm<LlungBehaviour>, Box<dyn Error>> {
    let swarm = libp2p::SwarmBuilder::with_existing_identity(keypair)
        .with_tokio()
        .with_tcp(
            tcp::Config::default(),
            noise::Config::new,
            yamux::Config::default,
        )?
        .with_quic()
        .with_dns_config(dns.config, dns.opts)
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
            let identify = identify::Behaviour::new(
                identify::Config::new("/llung/1.0.0".to_string(), key.public())
                    .with_agent_version(agent_version.to_string()),
            );
            let direct_message = DirectMessageBehaviour::new(
                [(
                    StreamProtocol::new("/llung/direct/1.0.0"),
                    ProtocolSupport::Full,
                )],
                ReqResConfig::default(),
            );
            let image_transfer = ImageTransferBehaviour::new(
                [(
                    StreamProtocol::new(crate::media::image::IMAGE_PROTOCOL),
                    ProtocolSupport::Full,
                )],
                ReqResConfig::default().with_request_timeout(Duration::from_secs(30)),
            );
            let history_sync = HistorySyncBehaviour::new(
                [(StreamProtocol::new(SYNC_PROTOCOL), ProtocolSupport::Full)],
                ReqResConfig::default().with_request_timeout(Duration::from_secs(30)),
            );

            Ok(LlungBehaviour {
                gossipsub,
                mdns,
                kademlia,
                relay_client,
                dcutr,
                identify,
                direct_message,
                image_transfer,
                history_sync,
            })
        })?
        .build();

    Ok(swarm)
}

pub struct LlungApp {
    pub local_being: Being,
    pub local_peer_id: PeerId,
    /// `false` when the node started without DNS (offline, or the system
    /// resolver config was empty): local peers via mDNS still work,
    /// `/dns*` addresses don't, until restart.
    pub dns_available: bool,
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

        // Publish our DM encryption key so peers can send us invites.
        let enc_public_key = config
            .db_path
            .to_str()
            .and_then(|p| crate::storage::db::Database::open(p).ok())
            .and_then(|db| db.get_human_enc_public_key().ok().flatten());

        let local_presence = PresenceMessage {
            being_id: being.being_id.clone(),
            machine_id: machine.peer_id,
            human_name: being.human_name.clone(),
            kind: being.kind,
            status: being.status.clone(),
            avatar_cid: being.avatar_cid.clone(),
            enc_public_key,
            // Empty here (the swarm doesn't exist yet); the engine
            // fills live listeners + relay-circuit addrs on every
            // heartbeat.
            listen_addrs: Vec::new(),
        };

        let dns = dns::system(dns::DEFAULT_WAIT).await;
        let dns_available = dns.available();
        let mut swarm = build_swarm(machine.keypair, dns)?;
        swarm
            .behaviour_mut()
            .kademlia
            .set_mode(Some(kad::Mode::Server));

        let topic = IdentTopic::new(&config.chat_topic);
        swarm.behaviour_mut().gossipsub.subscribe(&topic)?;

        // Announce ourselves immediately so peers can resolve our
        // human_name/avatar. The engine also re-broadcasts on a heartbeat.
        let presence_bytes = serde_json::to_vec(&local_presence)?;
        if let Err(e) = swarm.behaviour_mut().gossipsub.publish(topic.clone(), presence_bytes) {
            tracing::debug!("Initial presence publish failed (no mesh yet): {e:?}");
        }

        swarm.listen_on("/ip4/0.0.0.0/udp/0/quic-v1".parse()?)?;
        swarm.listen_on("/ip4/0.0.0.0/tcp/0".parse()?)?;

        let (cmd_tx, cmd_rx) = mpsc::channel::<NetworkCommand>(32);
        let (event_tx, event_rx) = mpsc::channel::<NetworkEvent>(32);

        // Answer inbound image Gets from the local avatar/media cache.
        // A per-query DB open keeps the engine DB-free (same pattern as
        // the enc-key lookup above); image Gets are rare.
        let (image_query_tx, mut image_query_rx) = mpsc::channel::<ImageQuery>(16);
        {
            let db_path = config.db_path.clone();
            tokio::spawn(async move {
                while let Some(query) = image_query_rx.recv().await {
                    let answer = db_path
                        .to_str()
                        .and_then(|p| crate::storage::db::Database::open(p).ok())
                        .and_then(|db| db.get_avatar_cache(&query.cid).ok().flatten())
                        .map(|bytes| {
                            let mime = image::guess_format(&bytes)
                                .map(|f| format!("image/{f:?}").to_lowercase())
                                .unwrap_or_else(|_| "application/octet-stream".to_string());
                            (mime, bytes)
                        });
                    let _ = query.respond.send(answer);
                }
            });
        }

        let relay_addrs: Vec<libp2p::Multiaddr> = config
            .relays
            .iter()
            .filter_map(|s| match s.parse() {
                Ok(addr) => Some(addr),
                Err(e) => {
                    tracing::warn!(%s, "ignoring bad relay multiaddr: {e}");
                    None
                }
            })
            .collect();

        // Answer inbound history-sync requests from the local DAG.
        let (history_query_tx, mut history_query_rx) = mpsc::channel::<HistoryQuery>(16);
        {
            let db_path = config.db_path.clone();
            tokio::spawn(async move {
                while let Some(query) = history_query_rx.recv().await {
                    let response = db_path
                        .to_str()
                        .and_then(|p| crate::storage::db::Database::open(p).ok())
                        .and_then(|db| db.load_dag().ok())
                        .map(|dag| answer_sync(&dag, &query.request))
                        .unwrap_or(crate::network::sync::SyncResponse::Unavailable);
                    let _ = query.respond.send(response);
                }
            });
        }

        let engine = NetworkEngine::new(
            swarm,
            cmd_rx,
            event_tx,
            local_presence,
            topic,
            Some(image_query_tx),
            relay_addrs,
            Some(history_query_tx),
        );
        tokio::spawn(async move { engine.run().await });

        // Re-announce avatar on startup
        if let Some(cid) = &being.avatar_cid {
            let _ = cmd_tx
                .send(NetworkCommand::ProvideMediaCid { cid: cid.clone() })
                .await;
        }

        Ok((
            LlungApp {
                local_being: being,
                local_peer_id,
                dns_available,
                cmd_tx,
            },
            event_rx,
        ))
    }

    pub fn command_tx(&self) -> mpsc::Sender<NetworkCommand> {
        self.cmd_tx.clone()
    }
}

#[cfg(test)]
#[cfg(test)]
mod sync_answer_tests {
    use super::*;
    use crate::network::message::{ChatMessage, MessageKind};
    use crate::network::sync::{SyncRequest, SyncResponse, MAX_SYNC_MESSAGES};
    use libp2p::identity::Keypair;

    fn msg(topic: &str, content: &str, parent: Option<&str>, sender: &PeerId) -> ChatMessage {
        ChatMessage::new(
            topic.into(),
            *sender,
            parent.map(String::from),
            MessageKind::Human,
            "tester".into(),
            content.into(),
        )
    }

    fn dag_of(messages: Vec<ChatMessage>) -> crate::storage::dag::ChatDag {
        crate::storage::dag::ChatDag::build_from_flat_list(messages)
    }

    #[test]
    fn chain_walks_newest_first_and_reports_genesis() {
        let sender = PeerId::from(Keypair::generate_ed25519().public());
        let m1 = msg("t", "one", None, &sender);
        let m2 = msg("t", "two", Some(&m1.id), &sender);
        let m3 = msg("t", "three", Some(&m2.id), &sender);
        let dag = dag_of(vec![m1.clone(), m2.clone(), m3.clone()]);

        let resp = answer_sync(
            &dag,
            &SyncRequest::Chain {
                topic: "t".into(),
                from: None,
                limit: 10,
            },
        );
        let SyncResponse::Chain {
            messages,
            more_parent,
            ..
        } = resp
        else {
            panic!("expected Chain, got {resp:?}");
        };
        assert_eq!(
            messages.iter().map(|m| m.content.as_str()).collect::<Vec<_>>(),
            vec!["three", "two", "one"]
        );
        assert_eq!(more_parent, None, "genesis reached");
    }

    #[test]
    fn chain_honors_limit_and_returns_cursor() {
        let sender = PeerId::from(Keypair::generate_ed25519().public());
        let m1 = msg("t", "one", None, &sender);
        let m2 = msg("t", "two", Some(&m1.id), &sender);
        let m3 = msg("t", "three", Some(&m2.id), &sender);
        let dag = dag_of(vec![m1.clone(), m2.clone(), m3.clone()]);

        let resp = answer_sync(
            &dag,
            &SyncRequest::Chain {
                topic: "t".into(),
                from: None,
                limit: 2,
            },
        );
        let SyncResponse::Chain {
            messages,
            more_parent,
            ..
        } = resp
        else {
            panic!("expected Chain");
        };
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content, "three");
        assert_eq!(more_parent, Some(m1.id.clone()), "cursor for the next page");

        // Continuation from the cursor reaches genesis.
        let resp = answer_sync(
            &dag,
            &SyncRequest::Chain {
                topic: "t".into(),
                from: more_parent,
                limit: 2,
            },
        );
        let SyncResponse::Chain {
            messages,
            more_parent,
            ..
        } = resp
        else {
            panic!("expected Chain");
        };
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].content, "one");
        assert_eq!(more_parent, None);
    }

    #[test]
    fn chain_clamps_the_limit() {
        let sender = PeerId::from(Keypair::generate_ed25519().public());
        let m1 = msg("t", "one", None, &sender);
        let dag = dag_of(vec![m1]);
        let resp = answer_sync(
            &dag,
            &SyncRequest::Chain {
                topic: "t".into(),
                from: None,
                limit: usize::MAX,
            },
        );
        let SyncResponse::Chain { messages, .. } = resp else {
            panic!("expected Chain");
        };
        assert!(messages.len() <= MAX_SYNC_MESSAGES);
    }

    #[test]
    fn chain_stops_cleanly_at_holes_and_topic_edges() {
        let sender = PeerId::from(Keypair::generate_ed25519().public());
        let m1 = msg("t", "one", None, &sender);
        // m2's parent is unknown locally (a hole).
        let m2 = msg("t", "two", Some("missing"), &sender);
        // m3 is in another topic; walking to it must stop.
        let other = msg("elsewhere", "x", None, &sender);
        let m4 = msg("t", "four", Some(&other.id), &sender);
        let dag = dag_of(vec![m1.clone(), m2.clone(), other.clone(), m4.clone()]);

        let resp = answer_sync(
            &dag,
            &SyncRequest::Chain {
                topic: "t".into(),
                from: Some(m2.id.clone()),
                limit: 10,
            },
        );
        let SyncResponse::Chain { more_parent, .. } = resp else {
            panic!("expected Chain");
        };
        assert_eq!(more_parent, None, "hole ends the walk (no re-ask loop)");

        let resp = answer_sync(
            &dag,
            &SyncRequest::Chain {
                topic: "t".into(),
                from: Some(m4.id.clone()),
                limit: 10,
            },
        );
        let SyncResponse::Chain { more_parent, .. } = resp else {
            panic!("expected Chain");
        };
        assert_eq!(more_parent, None, "cross-topic parent ends the walk");
    }

    #[test]
    fn missing_returns_only_known_messages() {
        let sender = PeerId::from(Keypair::generate_ed25519().public());
        let m1 = msg("t", "one", None, &sender);
        let dag = dag_of(vec![m1.clone()]);
        let resp = answer_sync(
            &dag,
            &SyncRequest::Missing {
                ids: vec![m1.id.clone(), "unknown".into()],
            },
        );
        let SyncResponse::Missing { messages } = resp else {
            panic!("expected Missing");
        };
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].id, m1.id);
    }
}

mod dns_offline_tests {
    use super::*;

    /// Offline startup: the full swarm builds with a nameserver-less
    /// resolver (this used to abort with "no nameservers found in config").
    #[tokio::test]
    async fn swarm_builds_without_dns() {
        let swarm = build_swarm(Keypair::generate_ed25519(), DnsSetup::none("offline"));
        assert!(swarm.is_ok(), "{:?}", swarm.err());
    }
}
