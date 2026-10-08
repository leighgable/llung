// crates/agent/src/orchestrator.rs
use crate::aukora::kernel::{Grant, KernelState, Manifest, Receipt};
use crate::aukora::security::secure_execute_agent_tool;
use crate::config::AgentConfig;
use crate::history::HistoryCompressor;
use crate::interface::AgentBackend;
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
use std::error::Error;
use tokio::sync::mpsc;

pub struct AgentLoopContext {
    pub config: AgentConfig,
    pub backend: Box<dyn AgentBackend>,
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

        let text = String::from_utf8_lossy(&data).into_owned();

        // decide: respond, use tool, or ignore
        match decide_action(&text, &ctx.config.name).await {
            AgentAction::Ignore => continue,
            AgentAction::Reply => match run_agent_turn(&ctx, &text).await {
                Ok(reply) => {
                    let cmd = NetworkCommand::PublishMessage {
                        topic: IdentTopic::new(&ctx.topic),
                        contents: reply.clone().into_bytes(),
                    };
                    if let Err(e) = ctx.cmd_tx.send(cmd).await {
                        tracing::error!("Failed to publish reply: {e}");
                        continue;
                    }
                    let self_peer: PeerId = match ctx.self_being_id.parse() {
                        Ok(p) => p,
                        Err(e) => {
                            tracing::error!("Invalid self_being_id: {e}");
                            continue;
                        }
                    };
                    let reply_msg = ChatMessage::new(
                        ctx.topic.clone(),
                        self_peer,
                        None,
                        MessageKind::Agent,
                        ctx.self_name.clone(),
                        reply,
                    );
                    if let Err(e) = ctx.db.save_message(&reply_msg) {
                        tracing::warn!("Failed to save reply to DB: {e}");
                    }
                }
                Err(e) => {
                    tracing::error!("Agent turn failed: {e}");
                }
            },
            AgentAction::UseTool {
                tool_name,
                arguments,
            } => {
                if let Err(e) = execute_secured_tool(&mut ctx, tool_name, arguments).await {
                    tracing::error!("Tool execution failed: {e}");
                }
            }
        }
    }
}

async fn run_agent_turn(
    ctx: &AgentLoopContext,
    trigger_text: &str,
) -> Result<String, Box<dyn Error>> {
    let dag = ctx.db.load_dag()?;
    let messages: Vec<ChatMessage> = dag.topic_linear(&ctx.topic).into_iter().cloned().collect();
    let mut summary_tree = ctx.compressor.compress(&messages);
    while summary_tree.size_bytes > ctx.compressor.context_budget {
        if !ctx.compressor.merge_adjacent(&mut summary_tree) {
            break;
        }
    }
    let recent = 5usize;
    if messages.len() > recent {
        let start = messages.len() - recent;
        zoom_range(&mut summary_tree, &messages, start, messages.len());
    }
    let context = ctx.compressor.flatten(&summary_tree);

    let prompt = vec![
        ChatMessage::system(format!("You are in a peer-to-peer chat.", ctx.self_name)),
        ChatMessage::user(context),
        ChatMessage::user(format!("New message: {}", trigger_text)),
    ];
    ctx.backend
        .chat(&prompt)
        .await
        .map_err(|e| -> Box<dyn Error> { e.into() })
}

// build manifest, get grant, run through aukora
async fn execute_secured_tool(
    ctx: &mut AgentLoopContext,
    tool_name: String,
    arguments: Vec<u8>,
) -> Result<(), Box<dyn std::error::Error>> {
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
    let mut receipt = secure_execute_agent_tool(
        &mut ctx.kernel,
        &ctx.wasm_tx, // agent-rt wasm worker handle
        manifest,
        grant,
    )
    .await?;

    // tool result back into chat as the agent's response
    let result_text = String::from_utf8_lossy(&receipt.output_commitment).into_owned();
    let cmd = NetworkCommand::PublishMessage {
        topic: IdentTopic::new(&ctx.topic),
        contents: result_text.into_bytes(),
    };
    ctx.cmd_tx.send(cmd).await?;

    Ok(())
}

// ── SELF-GRANT (placeholder until real supervisor signing) ──
fn create_self_grant(manifest: &Manifest, config: &AgentConfig) -> Grant {
    // Use the kernel's canonical hash path so the integrity check in
    // consume_manifest_use_core recomputes the SAME hash.
    let manifest_hash = crate::aukora::kernel::compute_sha256(manifest);

    Grant {
        manifest_hash,
        authorized_by: [0u8; 32], // TODO: user's pinned public key
        signature: vec![],        // TODO: real signature
    }
}

enum AgentAction {
    Ignore,
    Reply,
    UseTool {
        tool_name: String,
        arguments: Vec<u8>,
    },
}

async fn decide_action(text: &str, agent_name: &str) -> AgentAction {
    // Simple heuristic: if mentioned, reply. If [TOOL: name] detected, use tool.
    // In practice, the LLM decides this in the first pass.
    if text.contains(&format!("@{}", agent_name)) {
        AgentAction::Reply
    } else {
        AgentAction::Ignore
    }
}
