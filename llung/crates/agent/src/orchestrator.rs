// crates/agent/src/orchestrator.rs
use crate::aukora::{Grant, KernelState, Manifest, Receipt, secure_execute_agent_tool};
use crate::backend::{ChatMessage, LlmBackend};
use crate::history::{HistoryCompressor, merge_adjacent, zoom_range};
use llung_core::{
    network::{NetworkCommand, NetworkEvent, message::IdentTopic},
    storage::{dag::ChatDag, db::Database},
};
use tokio::sync::mpsc;

pub struct AgentLoopContext {
    pub config: AgentConfig,
    pub backend: Box<dyn LlmBackend>,
    pub compressor: HistoryCompressor,
    pub db: Database,
    pub cmd_tx: mpsc::Sender<NetworkCommand>,
    pub topic: String,
    pub self_being_id: String,
    pub self_name: String,
    // ── NEW: Aukora kernel for this agent instance ──
    pub kernel: KernelState,
    // ── NEW: channel to agent-rt wasm worker ──
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

        // ── DECIDE: respond, use tool, or ignore ──
        match decide_action(&text, &ctx.config.name).await {
            AgentAction::Ignore => continue,
            AgentAction::Reply => {
                if let Err(e) = generate_reply(&mut ctx, &text).await {
                    tracing::error!("Agent reply failed: {e}");
                }
            }
            AgentAction::UseTool {
                tool_name,
                arguments,
            } => {
                if let Err(e) = execute_secured_tool(&mut ctx, tool_name, arguments).await {
                    tracing::error!("Secured tool execution failed: {e}");
                }
            }
        }
    }
}

// ── INTERNAL: build manifest, get grant, run through Aukora ──
async fn execute_secured_tool(
    ctx: &mut AgentLoopContext,
    tool_name: String,
    arguments: Vec<u8>,
) -> Result<(), Box<dyn std::error::Error>> {
    let manifest = Manifest {
        agent_id: ctx.self_being_id.clone(),
        target_effect: tool_name,
        arguments,
        nonce: rand::random(), // or atomic counter
    };

    // Self-grant: the user who spawned this agent is the supervisor
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

    // Feed tool result back into chat as the agent's response
    let result_text = String::from_utf8_lossy(&receipt.output_commitment).into_owned();
    let cmd = NetworkCommand::PublishMessage {
        topic: IdentTopic::new(&ctx.topic),
        contents: result_text.into_bytes(),
    };
    ctx.cmd_tx.send(cmd).await?;

    Ok(())
}

// ── INTERNAL: compressed history + LLM reply ──
async fn generate_reply(
    ctx: &mut AgentLoopContext,
    trigger_text: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let dag = ctx.db.load_dag()?;
    let messages = dag.topic_linear(&ctx.topic);

    // Compress history
    let mut summary_tree = ctx.compressor.compress(&messages);
    while summary_tree.token_count > ctx.compressor.context_budget {
        if !merge_adjacent(&mut summary_tree) {
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
        ChatMessage::user(format!("New message: {}", trigger_text)),
    ];

    let reply = ctx.backend.chat(&prompt).await?;

    let cmd = NetworkCommand::PublishMessage {
        topic: IdentTopic::new(&ctx.topic),
        contents: reply.into_bytes(),
    };
    ctx.cmd_tx.send(cmd).await?;

    Ok(())
}

// ── SELF-GRANT (placeholder until real supervisor signing) ──
fn create_self_grant(manifest: &Manifest, config: &AgentConfig) -> Grant {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(&bincode::serialize(manifest).unwrap());
    let manifest_hash: [u8; 32] = hasher.finalize().into();

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
