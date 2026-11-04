// crates/agent/src/orchestrator.rs
use crate::aukora::kernel::{Grant, KernelState, Manifest};
use crate::aukora::security::secure_execute_agent_tool;
use crate::config::AgentConfig;
use crate::history::HistoryCompressor;
use crate::interface::Backend;
use agent_rt::ToolRequest;
use libp2p::gossipsub::IdentTopic;
use llung_core::{
    network::{
        NetworkCommand, NetworkEvent,
        message::{ChatMessage, MessageKind},
    },
    storage::db::Database,
};
use rand::Rng;
use tokio::sync::mpsc;

pub struct AgentLoopContext {
    pub config: AgentConfig,
    pub backend: Box<dyn Backend>,
    pub compressor: HistoryCompressor,
    pub db: Database,
    pub cmd_tx: mpsc::Sender<NetworkCommand>,
    pub topic: String,
    pub self_being_id: String,
    pub self_name: String,
    // aukora kernel for agent instance
    pub kernel: KernelState,
    // channel to agent-rt wasm worker
    pub wasm_tx: mpsc::Sender<ToolRequest>,
}

impl AgentLoopContext {
    /// Build a context for a freshly-provisioned local agent. Creates the
    /// Aukora kernel and the wasm tool worker internally so embedders
    /// (TUI, cli) don't have to wire up agent-rt themselves.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: AgentConfig,
        backend: Box<dyn Backend>,
        compressor: HistoryCompressor,
        db: Database,
        cmd_tx: mpsc::Sender<NetworkCommand>,
        topic: String,
        self_being_id: String,
        self_name: String,
    ) -> Self {
        let mut kernel = KernelState::default();
        // `create_self_grant` authorizes with the all-zeros placeholder key
        // until real supervisor signing lands; it must be pinned or the
        // kernel rejects every grant.
        kernel.pinned_keys.insert([0u8; 32]);
        Self {
            config,
            backend,
            compressor,
            db,
            cmd_tx,
            topic,
            self_being_id,
            self_name,
            kernel,
            wasm_tx: agent_rt::spawn_wasm_worker(),
        }
    }
}

pub async fn run_agent_loop(mut ctx: AgentLoopContext, mut event_rx: mpsc::Receiver<NetworkEvent>) {
    while let Some(event) = event_rx.recv().await {
        let NetworkEvent::MessageReceived {
            topic,
            sender,
            data,
        } = event
        else {
            continue;
        };

        if topic != ctx.topic || sender.to_base58() == ctx.self_being_id {
            continue;
        }

        let Incoming::Message {
            text,
            kind,
            structured,
        } = decode_incoming(&data, &ctx.topic)
        else {
            continue;
        };

        // Persist structured messages so they join the DAG history the
        // compressor reads. Idempotent (INSERT OR IGNORE), so sharing a
        // DB with the TUI that also saves them is safe.
        if let Some(msg) = structured {
            if let Err(e) = ctx.db.save_message(&msg) {
                tracing::warn!("failed to persist inbound message: {e}");
            }
        }

        // Only conversational content earns a turn; tool traffic, notes,
        // system announcements, and media captions stay silent.
        if !kind_responds(&kind) {
            continue;
        }

        // Gate: only spend a turn when addressed. Whether the turn is a
        // plain reply or a tool call is decided by the model's output.
        if !is_addressed(&text, &ctx.config.name) {
            continue;
        }

        if let Err(e) = generate_reply(&mut ctx, &text).await {
            tracing::error!("Agent turn failed: {e}");
        }
    }
}

/// One inbound gossip payload, decoded.
enum Incoming {
    /// Belongs to another topic — no turn.
    Skip,
    /// Something to consider for a reply. `structured` is the parsed
    /// message when the peer speaks the ChatMessage protocol (TUI, other
    /// llung nodes); raw-text peers produce `None`.
    Message {
        text: String,
        kind: MessageKind,
        structured: Option<ChatMessage>,
    },
}

fn decode_incoming(data: &[u8], topic: &str) -> Incoming {
    match serde_json::from_slice::<ChatMessage>(data) {
        Ok(msg) => {
            if msg.topic != topic {
                return Incoming::Skip;
            }
            let text = if msg.sender_name.is_empty() {
                msg.content.clone()
            } else {
                format!("{}: {}", msg.sender_name, msg.content)
            };
            Incoming::Message {
                text,
                kind: msg.kind.clone(),
                structured: Some(msg),
            }
        }
        // Legacy/plain peer sending raw bytes: treat as human chat text.
        Err(_) => Incoming::Message {
            text: String::from_utf8_lossy(data).into_owned(),
            kind: MessageKind::Human,
            structured: None,
        },
    }
}

/// Which message kinds the agent will respond to.
fn kind_responds(kind: &MessageKind) -> bool {
    matches!(kind, MessageKind::Human | MessageKind::Agent)
}

/// Small models shouldn't answer every message in the topic — only
/// those that @-mention the agent (same trigger the old heuristic used).
fn is_addressed(text: &str, agent_name: &str) -> bool {
    text.contains(&format!("@{agent_name}"))
}

// build manifest, get grant, run through aukora
async fn execute_secured_tool(
    ctx: &mut AgentLoopContext,
    tool_name: String,
    arguments: Vec<u8>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let manifest = Manifest {
        agent_id: ctx.self_being_id.clone(),
        target_effect: tool_name,
        arguments,
        nonce: {
            let mut rng = rand::rand_core::UnwrapErr(rand::rngs::SysRng);
            rng.next_u64() // or atomic counter
        },
    };

    // self-grant: the user who spawned this agent is the supervisor
    // In production, this would be a real signature from the user's pinned key
    let grant = create_self_grant(&manifest, &ctx.config);

    // Build receipt via Aukora kernel + wasm runtime
    let (_receipt, output) =
        secure_execute_agent_tool(
            &mut ctx.kernel,
            &ctx.wasm_tx, // agent-rt wasm worker handle
            manifest,
            grant,
        )
        .await?;

    // The tool's actual output (e.g. braille art) is what the chat sees;
    // the receipt's output commitment stays in the kernel for audit.
    publish_agent_text(ctx, output).await?;

    Ok(())
}

/// Publish `text` to the topic on behalf of the agent and log it in the
/// DAG so it joins the compressed history.
///
/// The payload is a serialized ChatMessage, not raw bytes: every peer
/// (TUI or agent) then gets proper sender attribution instead of falling
/// back to raw-text handling and showing "unknown".
async fn publish_agent_text(
    ctx: &mut AgentLoopContext,
    text: String,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let self_peer: libp2p::PeerId = ctx.self_being_id.parse()?;
    let msg = ChatMessage::new(
        ctx.topic.clone(),
        self_peer,
        None,
        MessageKind::Agent,
        ctx.self_name.clone(),
        text,
    );

    let cmd = NetworkCommand::PublishMessage {
        topic: IdentTopic::new(&ctx.topic),
        contents: serde_json::to_vec(&msg)?,
    };
    ctx.cmd_tx.send(cmd).await?;

    let _ = ctx.db.save_message(&msg);

    Ok(())
}

// compressed history + agent reply
async fn generate_reply(
    ctx: &mut AgentLoopContext,
    trigger_text: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // load_dag boxes a non-Send error; re-box it so the future stays Send.
    let dag = ctx
        .db
        .load_dag()
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
            e.to_string().into()
        })?;
    // DAG gives the canonical OptChat order; the compressor wants owned
    // messages for the flat history indexing.
    let messages: Vec<ChatMessage> = dag.topic_linear(&ctx.topic).into_iter().cloned().collect();

    // Compress history, then fit the view under the byte budget.
    let mut summary_tree = ctx.compressor.compress(&messages).await;
    while summary_tree.size_bytes > ctx.compressor.context_budget {
        if !ctx.compressor.merge_adjacent(&mut summary_tree).await {
            break;
        }
    }

    let context = ctx.compressor.flatten(&summary_tree);

    let prompt = vec![
        ChatMessage::system(format!(
            "You are {}, a helpful agent in a peer-to-peer chat. \
             History is compressed. Respond concisely.",
            ctx.self_name
        )),
        ChatMessage::user(context),
        // Tool protocol goes in the *last* turn, not the system message:
        // small models weight recent context far more, and some VLM chat
        // templates mangle or drop system prompts entirely.
        ChatMessage::user(format!(
            "New message: {}\n\n{}",
            trigger_text,
            crate::tools::tool_instructions(),
        )),
    ];

    // AgentResult is already Box<dyn Error + Send + Sync>.
    let reply = ctx.backend.chat(&prompt).await?;

    // The model may have asked for a tool instead of plain prose.
    if let Some(call) = crate::tools::parse_tool_call(&reply) {
        tracing::info!(agent = %ctx.self_name, tool = %call.name, "model requested tool");
        match execute_secured_tool(&mut *ctx, call.name.clone(), call.arguments.into_bytes())
            .await
        {
            Ok(()) => return Ok(()),
            // Never go silent: if the tool fails (missing wasm, bad args),
            // say so in chat so the failure is visible where it happened.
            Err(e) => {
                let note = format!("tool '{}' failed: {e}", call.name);
                tracing::error!(tool = %call.name, "{note}");
                return publish_agent_text(ctx, note).await;
            }
        }
    }

    // Prompt-parroting (common with tiny models): publishing the echo
    // would feed it back into the compressed history and make the next
    // turn worse. Substitute a short canned reply instead.
    if crate::tools::looks_like_echo(&reply) {
        tracing::warn!(agent = %ctx.self_name, "model echoed its prompt; discarding");
        return publish_agent_text(
            ctx,
            "I got confused by my own instructions — please rephrase the request."
                .to_string(),
        )
        .await;
    }

    publish_agent_text(ctx, reply).await
}

// ── SELF-GRANT (placeholder until real supervisor signing) ──
fn create_self_grant(manifest: &Manifest, _config: &AgentConfig) -> Grant {
    // Use the kernel's canonical hash path so the integrity check in
    // consume_manifest_use_core recomputes the SAME hash.
    let manifest_hash = crate::aukora::kernel::compute_sha256(manifest);

    Grant {
        manifest_hash,
        authorized_by: [0u8; 32], // TODO: user's pinned public key
        signature: vec![],        // TODO: real signature
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::aukora::kernel::{Manifest, ReceiptStatus, compute_sha256};

    fn landscape_wasm() -> Option<std::path::PathBuf> {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../runtime-tools/landscape.wasm");
        p.exists().then_some(p)
    }

    /// The exact path `execute_secured_tool` takes: manifest -> self-grant
    /// -> kernel authorization -> wasm worker -> receipt. Skips when the
    /// landscape component isn't built (see crates/tools/landscape/README.md).
    #[tokio::test]
    async fn secured_tool_path_executes_landscape() {
        let Some(wasm) = landscape_wasm() else {
            eprintln!("runtime-tools/landscape.wasm not built; skipping");
            return;
        };
        // The tool path resolves against LLUNG_TOOLS_DIR; point it at the
        // workspace's runtime-tools/ regardless of the test runner's cwd.
        // (unsafe: env mutation; no other test in this binary reads it.)
        #[allow(unused_unsafe)]
        unsafe {
            std::env::set_var(
                "LLUNG_TOOLS_DIR",
                wasm.parent().expect("wasm has a parent dir"),
            );
        }

        let manifest = Manifest {
            agent_id: "test-agent".into(),
            target_effect: "landscape".into(),
            arguments: br#"{"expr": "sin(x)*cos(y)", "width": 40, "height": 16, "grid": 10}"#
                .to_vec(),
            nonce: 42,
        };
        let grant = create_self_grant(&manifest, &AgentConfig::default());

        let mut kernel = KernelState::default();
        kernel.pinned_keys.insert([0u8; 32]);

        let (receipt, output) = secure_execute_agent_tool(
            &mut kernel,
            &agent_rt::spawn_wasm_worker(),
            manifest.clone(),
            grant,
        )
        .await
        .expect("secured tool execution");

        assert_eq!(receipt.manifest_hash, compute_sha256(&manifest));
        assert_eq!(receipt.status, ReceiptStatus::Executed);
        // Anti-replay: the same nonce must now be rejected.
        let grant2 = create_self_grant(&manifest, &AgentConfig::default());
        let mut kernel2 = KernelState::default();
        kernel2.pinned_keys.insert([0u8; 32]);
        let _ = kernel2.executed_nonces.insert(42); // simulate consumed state
        assert!(secure_execute_agent_tool(
            &mut kernel2,
            &agent_rt::spawn_wasm_worker(),
            manifest,
            grant2,
        )
        .await
        .is_err());
        drop(wasm); // path documented; load happens inside the worker
        assert!(
            output.chars().any(|c| ('\u{2800}'..='\u{28FF}').contains(&c)),
            "expected braille art, got: {output}"
        );
    }

    #[test]
    fn tool_call_line_is_routed_not_published_verbatim() {
        let reply = "TOOL {\"name\": \"landscape\", \"arguments\": {\"expr\": \"x\"}}";
        let call = crate::tools::parse_tool_call(reply).expect("should parse");
        assert_eq!(call.name, "landscape");
        assert!(call.arguments.contains("\"expr\""));
    }

    fn human_msg(topic: &str, name: &str, content: &str) -> ChatMessage {
        ChatMessage {
            id: format!("{topic}:{name}:{content}"),
            topic: topic.into(),
            sender_id: "p".into(),
            sender_name: name.into(),
            parent_id: None,
            kind: MessageKind::Human,
            content: content.into(),
            timestamp: 0,
        }
    }

    #[test]
    fn decodes_structured_chat_message() {
        let msg = human_msg("t", "alice", "@plotter plot x^2");
        let data = serde_json::to_vec(&msg).unwrap();

        let Incoming::Message {
            text,
            kind,
            structured,
        } = decode_incoming(&data, "t")
        else {
            panic!("same-topic message must not be skipped");
        };
        assert_eq!(text, "alice: @plotter plot x^2");
        assert!(matches!(kind, MessageKind::Human));
        assert!(structured.is_some());
    }

    #[test]
    fn skips_structured_message_for_another_topic() {
        let msg = human_msg("elsewhere", "alice", "hi");
        let data = serde_json::to_vec(&msg).unwrap();
        assert!(matches!(decode_incoming(&data, "t"), Incoming::Skip));
    }

    #[test]
    fn falls_back_to_raw_text_for_plain_peers() {
        let Incoming::Message {
            text,
            kind,
            structured,
        } = decode_incoming(b"hello @plotter", "t")
        else {
            panic!("raw text must produce a message");
        };
        assert_eq!(text, "hello @plotter");
        assert!(matches!(kind, MessageKind::Human));
        assert!(structured.is_none());
    }

    #[test]
    fn only_conversational_kinds_get_a_turn() {
        assert!(kind_responds(&MessageKind::Human));
        assert!(kind_responds(&MessageKind::Agent));
        assert!(!kind_responds(&MessageKind::System));
        assert!(!kind_responds(&MessageKind::Notes));
        assert!(!kind_responds(&MessageKind::Media(llung_core::network::message::Attachment {
            cid: "c".into(),
            filename: "img.png".into(),
            mime_type: "image/png".into(),
            size_bytes: 0,
            kind: llung_core::network::message::AttachmentKind::Image,
            width: None,
            height: None,
        })));
        assert!(!kind_responds(&MessageKind::ToolCall {
            tool_name: "landscape".into(),
            arguments: "{}".into(),
        }));
    }
}
