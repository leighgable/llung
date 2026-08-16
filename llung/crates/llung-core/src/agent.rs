pub mod interface;

use std::collections::VecDeque;

use libp2p::{PeerId, gossipsub::IdentTopic};
use tokio::sync::mpsc;

use crate::identity::being::{Being, BeingKind, create_local_being};
use crate::network::{NetworkCommand, NetworkEvent};
use crate::storage::db::Database;

use interface::{ChatMessage, LlmBackend};

pub struct AgentConfig {
    /// Display name of the agent Being; also its trigger word.
    pub name: String,
    /// How many recent chat messages to keep as context.
    pub max_history: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            name: "llung-bot".into(),
            max_history: 20,
        }
    }
}

/// Load the agent's Being from the DB, or create a fresh one
/// (kind = Agent, no avatar) on first run.
pub fn get_or_create_agent_being(
    db: &Database,
    agent_peer_id: PeerId,
    name: &str,
) -> Result<Being, Box<dyn std::error::Error>> {
    if let Some(existing) = db.get_being(&agent_peer_id)? {
        return Ok(existing);
    }
    create_local_being(db, agent_peer_id, name.to_string(), None, BeingKind::Agent)
}

fn is_mentioned(text: &str, name: &str) -> bool {
    let text = text.to_lowercase();
    let name = name.to_lowercase();
    text.contains(&format!("@{name}"))
        || text.starts_with(&format!("{name}:"))
        || text.starts_with(&format!("{name},"))
}

/// The agent bridge: sits between the agent's network engine channels
/// and the LLM backend. Receives chat events, decides whether to
/// respond, and publishes replies as the agent's own Being.
pub async fn run_agent_loop(
    config: AgentConfig,
    backend: impl LlmBackend,
    db: Database,
    cmd_tx: mpsc::Sender<NetworkCommand>,
    mut event_rx: mpsc::Receiver<NetworkEvent>,
    topic: IdentTopic,
    self_peer_id: PeerId,
) {
    let topic_str = topic.to_string();
    let mut transcript: VecDeque<ChatMessage> = VecDeque::with_capacity(config.max_history + 1);

    // Agent-to-agent chatter limit: we may respond to an agent-authored
    // message once; after that we stay quiet until a human speaks.
    let mut responded_to_agent = false;

    while let Some(event) = event_rx.recv().await {
        let NetworkEvent::MessageReceived {
            topic: msg_topic,
            sender,
            data,
        } = event
        else {
            continue;
        };

        if msg_topic != topic_str || sender == self_peer_id {
            continue;
        }

        let text = String::from_utf8_lossy(&data).into_owned();

        let sender_being = db.get_being(&sender).ok().flatten();
        let sender_is_agent =
            matches!(sender_being.as_ref().map(|b| &b.kind), Some(BeingKind::Agent));
        let sender_name = sender_being
            .map(|b| b.human_name)
            .unwrap_or_else(|| sender.to_base58().chars().take(8).collect());

        // Every message becomes context, whether or not we respond.
        if transcript.len() == config.max_history {
            transcript.pop_front();
        }
        transcript.push_back(ChatMessage::user(format!("{sender_name}: {text}")));

        if sender_is_agent {
            if responded_to_agent {
                continue; // already used our one agent-to-agent response
            }
        } else {
            responded_to_agent = false; // a human speaking resets the limit
        }

        if !is_mentioned(&text, &config.name) {
            continue;
        }

        let mut messages = vec![ChatMessage::system(format!(
            "You are {}, a helpful agent participating in a peer-to-peer group chat. \
             Messages from participants are prefixed with their display name. \
             Keep replies concise and conversational.",
            config.name
        ))];
        messages.extend(transcript.iter().cloned());

        match backend.chat(&messages).await {
            Ok(reply) => {
                if transcript.len() == config.max_history {
                    transcript.pop_front();
                }
                transcript.push_back(ChatMessage::assistant(reply.clone()));

                let cmd = NetworkCommand::PublishMessage {
                    topic: topic.clone(),
                    contents: reply.into_bytes(),
                };
                if cmd_tx.send(cmd).await.is_err() {
                    eprintln!("Agent: network engine went away, shutting down");
                    return;
                }
                if sender_is_agent {
                    responded_to_agent = true;
                }
            }
            Err(e) => eprintln!("Agent: LLM backend error: {e}"),
        }
    }
}
