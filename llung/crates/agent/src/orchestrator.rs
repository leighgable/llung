// crates/agent/src/orchestrator.rs
use crate::aukora::kernel::{Grant, KernelState, Manifest, StandingGrant};
use crate::aukora::security::secure_execute_agent_tool;
use crate::aukora::supervisor::SupervisorKey;
use crate::permissions::{PermissionAnswer, PermissionRequest};
use crate::config::AgentConfig;
use crate::history::{HistoryCompressor, zoom_range};
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

/// Incremental reply text from a local agent turn, for UI streaming.
/// Local-only: the complete reply still travels as one gossipsub
/// message; remote peers see the block.
pub enum AgentStream {
    /// Append text to the in-progress reply `message_id`.
    Delta {
        topic: String,
        message_id: String,
        being_id: String,
        sender_name: String,
        delta: String,
    },
    /// Withdraw the streamed preview of `message_id`: the "reply" turned
    /// out to be a tool call (or was discarded) and must not stay on
    /// screen as text.
    Retract {
        topic: String,
        message_id: String,
    },
}

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
    // local supervisor identity: signs the grants that authorize this
    // agent's tool executions (the human who spawned the agent)
    pub supervisor: SupervisorKey,
    // channel to agent-rt wasm worker
    pub wasm_tx: mpsc::Sender<ToolRequest>,
    // incremental reply text for a local UI (see AgentStream)
    pub stream_tx: mpsc::Sender<AgentStream>,
    /// Channel to the UI's permission prompt: the harness asks the
    /// operator before running prompt-required effect tiers.
    pub perm_tx: mpsc::Sender<PermissionRequest>,
    /// Background compactor cache: (chain tip message id, flattened
    /// compressed view). When a turn's chain tip matches, the view is
    /// reused instead of rebuilding the summary tree synchronously.
    view_cache: Option<(String, String)>,
}

/// Commands the UI sends into a running agent loop (`/grant`).
pub enum AgentCommand {
    /// Pre-authorize `effect` perpetually for this agent.
    GrantStanding { effect: String },
    /// Remove the standing grant for `effect`.
    Revoke { effect: String },
    /// List this agent's standing grants.
    ListGrants { respond: tokio::sync::oneshot::Sender<Vec<String>> },
}

/// How often the background compactor recompresses the agent's home
/// branch between turns (OptChat's background compactor — the tree stays
/// warm instead of being rebuilt synchronously per turn).
const COMPACT_INTERVAL: std::time::Duration = std::time::Duration::from_secs(300);

/// Upper bound on `zoom` re-asks per turn — the model must not be able
/// to ping-pong context expansions forever.
const MAX_ZOOM_ROUNDS: u32 = 3;

impl AgentLoopContext {
    /// Build a context for a freshly-provisioned local agent. Creates the
    /// Aukora kernel and the wasm tool worker internally so embedders
    /// (TUI, cli) don't have to wire up agent-rt themselves.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: AgentConfig,
        backend: Box<dyn Backend>,
        mut compressor: HistoryCompressor,
        db: Database,
        cmd_tx: mpsc::Sender<NetworkCommand>,
        topic: String,
        self_being_id: String,
        self_name: String,
        stream_tx: mpsc::Sender<AgentStream>,
        perm_tx: mpsc::Sender<PermissionRequest>,
    ) -> Self {
        // No compressor model configured? The agent's own backend takes
        // the job — it IS the history compressor.
        if !compressor.is_model_backed() {
            if let Some(dup) = backend.try_duplicate() {
                compressor.upgrade_to_model(dup);
            }
        }
        // Kernel state is persisted per being-id: consumed nonces, the
        // merkle root, and the nonce counter all survive restarts.
        let mut kernel = crate::aukora::persist::load_kernel_state(&db, &self_being_id);
        // The account's supervisor key (created at registration,
        // persisted in the account DB) authorizes the agent's grants;
        // its public half is pinned so the kernel accepts only its
        // signatures. Same identity across restarts and devices.
        let supervisor = SupervisorKey::from_seed(
            &db.get_or_create_supervisor_seed().unwrap_or_else(|e| {
                // The account DB just opened; a failure here means the
                // account store is corrupt, and everything downstream
                // (messages, kernel state) breaks the same way.
                panic!("aukora supervisor seed unavailable: {e}")
            }),
        );
        kernel.pinned_keys.insert(supervisor.public_pin());
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
            supervisor,
            wasm_tx: agent_rt::spawn_wasm_worker(),
            stream_tx,
            perm_tx,
            view_cache: None,
        }
    }
}

pub async fn run_agent_loop(
    mut ctx: AgentLoopContext,
    mut event_rx: mpsc::Receiver<NetworkEvent>,
    mut cmd_rx: mpsc::Receiver<AgentCommand>,
) {
    // Background compactor: recompress the home branch periodically so
    // turns find a warm cache instead of rebuilding the tree inline.
    let mut compaction = tokio::time::interval(COMPACT_INTERVAL);
    compaction.tick().await; // first tick is immediate; skip it
    loop {
        tokio::select! {
            Some(event) = event_rx.recv() => handle_event(&mut ctx, event).await,
            Some(cmd) = cmd_rx.recv() => handle_agent_command(&mut ctx, cmd).await,
            _ = compaction.tick() => background_compact(&mut ctx).await,
        }
    }
}

/// `/grant` commands from the UI: standing-grant management on this
/// agent's kernel. Persisted immediately so revocations stick.
async fn handle_agent_command(ctx: &mut AgentLoopContext, cmd: AgentCommand) {
    match cmd {
        AgentCommand::GrantStanding { effect } => {
            let grant = StandingGrant {
                agent_id: ctx.self_being_id.clone(),
                effect: effect.clone(),
                authorized_by: ctx.supervisor.public_pin(),
            };
            if ctx.kernel.grant_standing(grant) {
                tracing::info!(agent = %ctx.self_name, effect = %effect, "standing grant added");
            }
            let _ = crate::aukora::persist::save_kernel_state(
                &ctx.db,
                &ctx.self_being_id,
                &ctx.kernel,
            );
        }
        AgentCommand::Revoke { effect } => {
            if ctx.kernel.revoke_standing(&ctx.self_being_id.clone(), &effect) {
                tracing::info!(agent = %ctx.self_name, effect = %effect, "standing grant revoked");
            }
            let _ = crate::aukora::persist::save_kernel_state(
                &ctx.db,
                &ctx.self_being_id,
                &ctx.kernel,
            );
        }
        AgentCommand::ListGrants { respond } => {
            let effects: Vec<String> = ctx
                .kernel
                .standing_grants
                .iter()
                .filter(|g| g.agent_id == ctx.self_being_id)
                .map(|g| g.effect.clone())
                .collect();
            let _ = respond.send(effects);
        }
    }
}

/// One inbound network event: persist, gate, and maybe take a turn.
async fn handle_event(ctx: &mut AgentLoopContext, event: NetworkEvent) {
    let NetworkEvent::MessageReceived {
        topic,
        sender,
        data,
    } = event
    else {
        return;
    };

    if topic != ctx.topic || sender.to_base58() == ctx.self_being_id {
        return;
    }

    let Incoming::Message {
        text,
        kind,
        structured,
    } = decode_incoming(&data, &ctx.topic)
    else {
        return;
    };

    // Persist structured messages so they join the DAG history the
    // compressor reads. Idempotent (INSERT OR IGNORE), so sharing a
    // DB with the TUI that also saves them is safe.
    if let Some(msg) = &structured {
        if let Err(e) = ctx.db.save_message(msg) {
            tracing::warn!("failed to persist inbound message: {e}");
        }
    }

    // Only conversational content earns a turn; tool traffic, notes,
    // system announcements, and media captions stay silent.
    if !kind_responds(&kind) {
        return;
    }

    // Gate: only spend a turn when addressed. Whether the turn is a
    // plain reply or a tool call is decided by the model's output.
    if !is_addressed(&text, &ctx.config.name) {
        return;
    }

    let trigger_id = id_of(&structured);
    if let Err(e) = generate_reply(&mut *ctx, &text, trigger_id.as_deref()).await {
        tracing::error!("Agent turn failed: {e}");
        // Never go silent: a broken model backend (connection
        // refused, timeout, ...) must show up in the chat as an
        // error, not look like the agent is still "thinking".
        // Own messages are skipped by the loop above, so this can't
        // trigger another turn.
        if let Err(pe) = publish_agent_text_on(
            &mut *ctx,
            trigger_id.clone(),
            format!("⚠ my model backend failed on that turn: {e}"),
            None,
        )
        .await
        {
            tracing::error!("failed to publish turn-error notice: {pe}");
        }
    }
}

/// Periodic self-compaction (OptChat's background compactor): compress
/// the agent's home branch and warm the view cache. Errors never
/// propagate — a failed compaction just means the next turn compresses
/// synchronously, as before.
async fn background_compact(ctx: &mut AgentLoopContext) {
    let Ok(dag) = ctx.db.load_dag() else {
        return;
    };
    let Some(tip) = dag.tip_of(&ctx.topic) else {
        return;
    };
    if ctx.view_cache.as_ref().is_some_and(|(t, _)| *t == tip) {
        return; // branch unchanged since the last compression
    }
    let messages: Vec<ChatMessage> = dag.context_chain(&tip).into_iter().cloned().collect();
    let _ = compress_chain(ctx, &tip, &messages).await;
}

/// Compress `messages` into the budgeted view, reusing the cached result
/// when `tip_id` matches (background compactor already did the work).
async fn compress_chain(
    ctx: &mut AgentLoopContext,
    tip_id: &str,
    messages: &[ChatMessage],
) -> String {
    if let Some((cached_tip, view)) = &ctx.view_cache {
        if cached_tip == tip_id {
            return view.clone();
        }
    }
    let tree = ctx.compressor.compress(messages).await;
    // The OptChat view: the coarsest cut of the tree that fits the byte
    // budget, newest-first. (The old code compared the ROOT summary's
    // size against the budget — roots are always ≤ NODE bytes, so the
    // view collapsed to a single summary line for the whole history.)
    let view = ctx
        .compressor
        .flatten_budgeted(&tree, ctx.compressor.context_budget);
    ctx.view_cache = Some((tip_id.to_string(), view.clone()));
    view
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
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let manifest = Manifest {
        agent_id: ctx.self_being_id.clone(),
        target_effect: tool_name,
        arguments: arguments.clone(),
        // Monotonic persisted counter — a restart can never re-issue a
        // nonce the kernel has already consumed (anti-replay).
        nonce: ctx.kernel.alloc_nonce(),
    };

    // Human-in-the-loop gate (Phase 5): prompt-required tiers ask the
    // operator unless a standing grant already covers this (agent,
    // effect). Denials are recorded as Denied receipts and fed back to
    // the model as a tool error.
    let effect = crate::effects::classify(&manifest.target_effect);
    if effect.tier.requires_prompt() && !ctx.kernel.check_standing(&manifest) {
        match ask_permission(ctx, &manifest, &effect, &arguments).await? {
            PermissionAnswer::Once => {}
            PermissionAnswer::Standing => {
                ctx.kernel.grant_standing(StandingGrant {
                    agent_id: ctx.self_being_id.clone(),
                    effect: manifest.target_effect.clone(),
                    authorized_by: ctx.supervisor.public_pin(),
                });
                crate::aukora::persist::save_kernel_state(
                    &ctx.db,
                    &ctx.self_being_id,
                    &ctx.kernel,
                )?;
            }
            PermissionAnswer::Deny => {
                ctx.kernel.record_denial(&manifest);
                crate::aukora::persist::save_kernel_state(
                    &ctx.db,
                    &ctx.self_being_id,
                    &ctx.kernel,
                )?;
                return Err(format!(
                    "permission denied by operator for '{}'",
                    manifest.target_effect
                )
                .into());
            }
        }
    }

    // Self-grant, signed by the supervisor key pinned in the kernel.
    let grant = create_self_grant(&manifest, &ctx.supervisor);

    // Build receipt via Aukora kernel + wasm runtime
    let (_receipt, output) = secure_execute_agent_tool(
        &mut ctx.kernel,
        &ctx.wasm_tx, // agent-rt wasm worker handle
        manifest,
        grant,
    )
    .await?;

    // Persist consumed nonces / merkle root / counter so a restart
    // resumes with the same anti-replay state.
    crate::aukora::persist::save_kernel_state(&ctx.db, &ctx.self_being_id, &ctx.kernel)?;

    // The tool's actual output is what the chat sees (the caller
    // publishes it); the receipt's output commitment stays in the kernel
    // for audit.
    Ok(output)
}

/// One-line description of a tool output for `ToolResponse.result_summary`.
fn summarize_output(output: &str) -> String {
    if let Ok(doc) = serde_json::from_str::<serde_json::Value>(output) {
        if let Some(kind) = doc.get("render").and_then(|v| v.as_str()) {
            let title = doc.get("title").and_then(|v| v.as_str()).unwrap_or("");
            return format!("{kind} render of '{title}'");
        }
    }
    output.lines().next().unwrap_or("").chars().take(80).collect()
}

/// Publish `text` to the topic on behalf of the agent and log it in the
/// DAG so it joins the compressed history.
///
/// The payload is a serialized ChatMessage, not raw bytes: every peer
/// (TUI or agent) then gets proper sender attribution instead of falling
/// back to raw-text handling and showing "unknown".
async fn publish_agent_text_on(
    ctx: &mut AgentLoopContext,
    parent_id: Option<String>,
    text: String,
    id: Option<String>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    publish(ctx, MessageKind::Agent, parent_id, text, id)
        .await
        .map(|_| ())
}

/// Publish a message of any `kind` (optionally replying to `parent_id`)
/// and log it in the DAG. Returns the message id.
async fn publish(
    ctx: &mut AgentLoopContext,
    kind: MessageKind,
    parent_id: Option<String>,
    text: String,
    id: Option<String>,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let self_peer: libp2p::PeerId = ctx.self_being_id.parse()?;
    let mut msg = ChatMessage::new(
        ctx.topic.clone(),
        self_peer,
        parent_id,
        kind,
        ctx.self_name.clone(),
        text,
    );
    // A caller that streamed the reply under a precomputed id must
    // publish under the same id, or the UI would show the text twice.
    if let Some(id) = id {
        msg.id = id;
    }

    let cmd = NetworkCommand::PublishMessage {
        topic: IdentTopic::new(&ctx.topic),
        contents: serde_json::to_vec(&msg)?,
    };
    ctx.cmd_tx.send(cmd).await?;

    let _ = ctx.db.save_message(&msg);

    Ok(msg.id)
}

fn id_of(structured: &Option<ChatMessage>) -> Option<String> {
    structured.as_ref().map(|m| m.id.clone())
}

/// The turn prompt: who the agent is, the (compressed, indexed) history
/// view, then the triggering message with the tool protocol. Tool
/// protocol goes in the *last* turn, not the system message: small
/// models weight recent context far more, and some VLM chat templates
/// mangle or drop system prompts entirely.
fn turn_prompt(ctx: &AgentLoopContext, context: &str, trigger_text: &str) -> Vec<ChatMessage> {
    turn_prompt_inner(ctx, context, trigger_text, None)
}

/// Like [`turn_prompt`], with tool-error feedback appended as the last
/// (most-weighted) turn — small models read recent context best.
fn turn_prompt_feedback(
    ctx: &AgentLoopContext,
    context: &str,
    trigger_text: &str,
    feedback: &str,
) -> Vec<ChatMessage> {
    turn_prompt_inner(ctx, context, trigger_text, Some(feedback))
}

fn turn_prompt_inner(
    ctx: &AgentLoopContext,
    context: &str,
    trigger_text: &str,
    feedback: Option<&str>,
) -> Vec<ChatMessage> {
    let mut v = vec![
        ChatMessage::system(format!(
            "You are {}, a helpful agent in a peer-to-peer chat. \
             History is compressed; lines are numbered like [3] or [2-5] \
             — use the zoom tool to expand a range to its full text. \
             Respond concisely.",
            ctx.self_name
        )),
        ChatMessage::user(context.to_string()),
        ChatMessage::user(format!(
            "New message: {}\n\n{}",
            trigger_text,
            crate::tools::tool_instructions(),
        )),
    ];
    if let Some(f) = feedback {
        v.push(ChatMessage::user(f.to_string()));
    }
    v
}

/// One backend call, streaming deltas to the local UI under `reply_id`.
/// Takes the pieces individually (not `&AgentLoopContext`): the future
/// must stay Send, and `Database` is not Sync.
#[allow(clippy::too_many_arguments)]
async fn ask_streamed(
    backend: &(dyn Backend + Sync),
    stream_tx: &mpsc::Sender<AgentStream>,
    topic: &str,
    being_id: &str,
    sender_name: &str,
    prompt: &[ChatMessage],
    reply_id: &str,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let (delta_tx, mut delta_rx) = mpsc::channel::<String>(256);
    let forward_deltas = async {
        while let Some(delta) = delta_rx.recv().await {
            let _ = stream_tx
                .send(AgentStream::Delta {
                    topic: topic.to_string(),
                    message_id: reply_id.to_string(),
                    being_id: being_id.to_string(),
                    sender_name: sender_name.to_string(),
                    delta,
                })
                .await;
        }
    };
    let (reply, ()) = tokio::join!(backend.chat_stream(prompt, delta_tx), forward_deltas);
    reply
}

/// Zoom tool arguments: the `[start, end)` history range to expand.
fn parse_zoom_args(arguments: &str) -> Result<(usize, usize), String> {
    let v: serde_json::Value =
        serde_json::from_str(arguments).map_err(|e| format!("arguments must be JSON: {e}"))?;
    let start = v["start"].as_u64().ok_or("missing 'start'")? as usize;
    let end = v["end"].as_u64().ok_or("missing 'end'")? as usize;
    if end <= start {
        return Err(format!("end ({end}) must be greater than start ({start})"));
    }
    Ok((start, end))
}

// compressed history + agent reply
async fn generate_reply(
    ctx: &mut AgentLoopContext,
    trigger_text: &str,
    trigger_id: Option<&str>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // load_dag boxes a non-Send error; re-box it so the future stays Send.
    let dag = ctx
        .db
        .load_dag()
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
            e.to_string().into()
        })?;
    // The agent sees the context chain of the message it was addressed
    // on: its branch, with parallel agent replies and tool traffic
    // folded in — but explicit `/branch` forks left out. Agents @'d on
    // the same message therefore see each other's replies (linear chat);
    // only forks isolate context.
    let messages: Vec<ChatMessage> = match trigger_id {
        Some(id) if dag.nodes.contains_key(id) => {
            dag.context_chain(id).into_iter().cloned().collect()
        }
        // Raw-text peer (no structured id): canonical whole-topic order.
        _ => dag.topic_linear(&ctx.topic).into_iter().cloned().collect(),
    };

    // Compress history (reusing the background compactor's cached view
    // when this branch's tip matches), then the prompt gets the flat view.
    let tip_id = messages.last().map(|m| m.id.clone()).unwrap_or_default();
    let mut context = compress_chain(ctx, &tip_id, &messages).await;

    let mut prompt = turn_prompt(ctx, &context, trigger_text);

    // The reply streams to the local UI under a precomputed id while the
    // backend request runs; the final publish (below) reuses the id.
    let reply_id = format!("{}-stream-{:016x}", ctx.self_being_id, {
        let mut rng = rand::rand_core::UnwrapErr(rand::rngs::SysRng);
        rng.next_u64()
    });
    // Field references, not `&ctx`: keeps the awaited futures Send
    // (AgentLoopContext holds a `Database`, which is not Sync).
    let mut reply = ask_streamed(
        &*ctx.backend,
        &ctx.stream_tx,
        &ctx.topic,
        &ctx.self_being_id,
        &ctx.self_name,
        &prompt,
        &reply_id,
    )
    .await?;

    // The model may "zoom" into compressed history: expand the requested
    // range to full text and re-ask, a bounded number of times. A zoom
    // call with bad arguments gets the failure fed back so the model
    // can correct itself — the rounds budget covers retries, like tool
    // feedback in other agent harnesses.
    let mut zoom_rounds = 0u32;
    'zoom: while let Some(call) = crate::tools::parse_tool_call(&reply) {
        if call.name != "zoom" {
            break;
        }
        // The streamed preview was raw tool-call text: withdraw it.
        let _ = ctx
            .stream_tx
            .send(AgentStream::Retract {
                topic: ctx.topic.clone(),
                message_id: reply_id.clone(),
            })
            .await;
        zoom_rounds += 1;
        if zoom_rounds > MAX_ZOOM_ROUNDS {
            return publish_agent_text_on(
                ctx,
                trigger_id.map(str::to_string),
                "⚠ too many zoom attempts in one turn — please work from the summaries."
                    .into(),
                None,
            )
            .await;
        }
        // Resolve the arguments, retrying with feedback on model error.
        let mut args = call.arguments.clone();
        let (start, end) = loop {
            match parse_zoom_args(&args) {
                Ok(range) => break range,
                Err(e) => {
                    zoom_rounds += 1;
                    if zoom_rounds > MAX_ZOOM_ROUNDS {
                        return publish_agent_text_on(
                            ctx,
                            trigger_id.map(str::to_string),
                            format!("⚠ zoom failed: {e} (after {MAX_ZOOM_ROUNDS} attempts)"),
                            None,
                        )
                        .await;
                    }
                    let feedback = format!(
                        "Your last reply was a zoom tool call, but its arguments \
                         were rejected: {e}. The zoom tool takes integer 'start' \
                         and 'end' (end greater than start) — the [n] or [a-b] \
                         indices shown on the history lines. Reply with exactly \
                         one corrected TOOL line, or just answer normally."
                    );
                    prompt = turn_prompt_feedback(ctx, &context, trigger_text, &feedback);
                    reply = ask_streamed(
                        &*ctx.backend,
                        &ctx.stream_tx,
                        &ctx.topic,
                        &ctx.self_being_id,
                        &ctx.self_name,
                        &prompt,
                        &reply_id,
                    )
                    .await?;
                    match crate::tools::parse_tool_call(&reply) {
                        // Corrected zoom: hide it and validate again.
                        Some(c) if c.name == "zoom" => {
                            let _ = ctx
                                .stream_tx
                                .send(AgentStream::Retract {
                                    topic: ctx.topic.clone(),
                                    message_id: reply_id.clone(),
                                })
                                .await;
                            args = c.arguments.clone();
                        }
                        // The model moved on (prose or another tool):
                        // let the zoom loop re-judge the new reply.
                        _ => continue 'zoom,
                    }
                }
            }
        };
        // Rebuild the tree for this turn and expand the range. The
        // zoomed view may exceed the byte budget — an explicit expansion
        // the model asked for, so it stands.
        let mut tree = ctx.compressor.compress(&messages).await;
        zoom_range(&mut tree, &messages, start, end);
        // Same budgeted cut; the zoomed range stands even past budget.
        context = ctx
            .compressor
            .flatten_budgeted(&tree, ctx.compressor.context_budget);
        prompt = turn_prompt(ctx, &context, trigger_text);
        reply = ask_streamed(
            &*ctx.backend,
            &ctx.stream_tx,
            &ctx.topic,
            &ctx.self_being_id,
            &ctx.self_name,
            &prompt,
            &reply_id,
        )
        .await?;
    }

    // The model may have asked for a tool instead of plain prose.
    if let Some(call) = crate::tools::parse_tool_call(&reply) {
        // The streamed preview was raw tool-call text: withdraw it.
        let _ = ctx
            .stream_tx
            .send(AgentStream::Retract {
                topic: ctx.topic.clone(),
                message_id: reply_id.clone(),
            })
            .await;
        tracing::info!(agent = %ctx.self_name, tool = %call.name, "model requested tool");
        // Announce the call first: viewers show a pending tool panel, then
        // swap in the response (whose parent_id points back here). The
        // call itself hangs off the triggering message, so the tool
        // panel lands on the same branch.
        let call_id = publish(
            ctx,
            MessageKind::ToolCall {
                tool_name: call.name.clone(),
                arguments: call.arguments.clone(),
            },
            trigger_id.map(str::to_string),
            call.arguments.clone(),
            None,
        )
        .await?;
        // Host-side data files: wasm has no I/O, so a `path` argument
        // is read HERE and injected as `csv` before the sandboxed call.
        let resolved = crate::tools::resolve_data_file(call.clone()).await;
        let (content, error) = match resolved {
            Ok(call) => {
                match execute_secured_tool(
                    &mut *ctx,
                    call.name.clone(),
                    call.arguments.into_bytes(),
                )
                .await
                {
                    Ok(output) => (output, false),
                    // Never go silent: if the tool fails (missing wasm,
                    // bad args), the error panel shows it where it happened.
                    Err(e) => {
                        tracing::error!(tool = %call.name, "tool failed: {e}");
                        (format!("tool '{}' failed: {e}", call.name), true)
                    }
                }
            }
            Err(e) => {
                tracing::error!("data file resolution failed: {e}");
                (format!("plot_device failed: {e}"), true)
            }
        };
        let response = MessageKind::ToolResponse {
            tool_name: call.name,
            result_summary: summarize_output(&content),
            error,
        };
        return publish(ctx, response, Some(call_id), content, None)
            .await
            .map(|_| ());
    }

    // Prompt-parroting (common with tiny models): publishing the echo
    // would feed it back into the compressed history and make the next
    // turn worse. Substitute a short canned reply instead.
    if crate::tools::looks_like_echo(&reply) {
        tracing::warn!(agent = %ctx.self_name, "model echoed its prompt; discarding");
        let _ = ctx
            .stream_tx
            .send(AgentStream::Retract {
                topic: ctx.topic.clone(),
                message_id: reply_id,
            })
            .await;
        return publish_agent_text_on(
            ctx,
            trigger_id.map(str::to_string),
            "I got confused by my own instructions — please rephrase the request."
                .to_string(),
            None,
        )
        .await;
    }

    publish_agent_text_on(ctx, trigger_id.map(str::to_string), reply, Some(reply_id)).await
}

/// Send a permission request to the UI and await the operator's answer.
async fn ask_permission(
    // &mut, not &: the awaited future is spawned onto the tokio
    // runtime, and a shared reference would make it require
    // `AgentLoopContext: Sync` (Database is not Sync).
    ctx: &mut AgentLoopContext,
    manifest: &Manifest,
    effect: &crate::effects::Effect<'_>,
    arguments: &[u8],
) -> Result<PermissionAnswer, Box<dyn std::error::Error + Send + Sync>> {
    let (respond, rx) = tokio::sync::oneshot::channel();
    let request = PermissionRequest {
        agent_name: ctx.self_name.clone(),
        agent_id: ctx.self_being_id.clone(),
        effect: manifest.target_effect.clone(),
        tier: effect.tier,
        summary: crate::permissions::summarize_arguments(effect, arguments),
        allow_standing: effect.tier.standing_allowed(),
        respond,
    };
    ctx.perm_tx
        .send(request)
        .await
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
            format!("permission UI unreachable: {e}").into()
        })?;
    rx.await.map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
        format!("permission prompt dropped: {e}").into()
    })
}

// ── SELF-GRANT ──
/// Sign the manifest with the supervisor key so the kernel's real
/// ed25519 verification accepts it. Uses the kernel's canonical hash
/// path so the integrity check in `consume_manifest_use_core`
/// recomputes the SAME hash that was signed.
fn create_self_grant(manifest: &Manifest, supervisor: &SupervisorKey) -> Grant {
    let manifest_hash = crate::aukora::kernel::compute_sha256(manifest);
    Grant {
        manifest_hash,
        authorized_by: supervisor.public_pin(),
        signature: supervisor.sign(&manifest_hash),
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
        let supervisor = crate::aukora::supervisor::SupervisorKey::generate();
        let grant = create_self_grant(&manifest, &supervisor);

        let mut kernel = KernelState::default();
        kernel.pinned_keys.insert(supervisor.public_pin());

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
        let grant2 = create_self_grant(&manifest, &supervisor);
        let mut kernel2 = KernelState::default();
        kernel2.pinned_keys.insert(supervisor.public_pin());
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
        let doc: serde_json::Value =
            serde_json::from_str(&output).expect("tool returns a JSON render document");
        assert_eq!(doc["render"], "braille-v1", "got: {output}");
        assert!(
            doc["rows"].as_array().unwrap().iter().any(|r| r
                .as_str()
                .unwrap()
                .chars()
                .any(|c| ('\u{2801}'..='\u{28FF}').contains(&c))),
            "expected braille dots"
        );
        assert_eq!(summarize_output(&output), "braille-v1 render of 'sin(x)*cos(y)'");
    }

    #[test]
    fn summaries() {
        assert_eq!(
            summarize_output(r#"{"render":"braille-v1","title":"sin(x)"}"#),
            "braille-v1 render of 'sin(x)'"
        );
        assert_eq!(summarize_output("tool 'x' failed: boom\nmore"), "tool 'x' failed: boom");
    }

    #[test]
    fn tool_call_line_is_routed_not_published_verbatim() {
        let reply = "TOOL {\"name\": \"landscape\", \"arguments\": {\"expr\": \"x\"}}";
        let call = crate::tools::parse_tool_call(reply).expect("should parse");
        assert_eq!(call.name, "landscape");
        assert!(call.arguments.contains("\"expr\""));
    }

    /// Counting summarizer that claims to be model-backed (so
    /// AgentLoopContext::new leaves it alone).
    struct CountingSummarizer {
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }
    impl crate::history::Summarizer for CountingSummarizer {
        fn summarize<'a>(
            &'a self,
            text: &'a str,
        ) -> std::pin::Pin<Box<dyn Future<Output = String> + Send + 'a>> {
            let calls = self.calls.clone();
            Box::pin(async move {
                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                text.chars().take(64).collect()
            })
        }
        fn merge<'a>(
            &'a self,
            left: &'a str,
            right: &'a str,
        ) -> std::pin::Pin<Box<dyn Future<Output = String> + Send + 'a>> {
            let calls = self.calls.clone();
            Box::pin(async move {
                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                format!("{left} + {right}")
            })
        }
        fn is_model_backed(&self) -> bool {
            true
        }
    }

    fn test_db() -> Database {
        let path = std::env::temp_dir().join(format!(
            "llung-agent-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        Database::open(path.to_str().unwrap()).expect("temp db")
    }

    fn ctx_with(
        compressor: crate::history::HistoryCompressor,
    ) -> AgentLoopContext {
        let (cmd_tx, _cmd_rx) = mpsc::channel(8);
        let (stream_tx, _stream_rx) = mpsc::channel(8);
        let (perm_tx, _perm_rx) = mpsc::channel(8);
        AgentLoopContext::new(
            AgentConfig::default(),
            Box::new(crate::interface::OpenAiCompatible::llamacpp(1)),
            compressor,
            test_db(),
            cmd_tx,
            "t".into(),
            "being".into(),
            "tester".into(),
            stream_tx,
            perm_tx,
        )
    }

    #[tokio::test]
    async fn context_without_compressor_model_adopts_its_own() {
        // A rule-based compressor gets upgraded to the agent's backend.
        let ctx = ctx_with(crate::history::HistoryCompressor::new(
            Box::new(crate::history::HeadlineSummarizer::new()),
            128_000,
        ));
        assert!(
            ctx.compressor.is_model_backed(),
            "the agent's own model should take over history compression"
        );
    }

    #[tokio::test]
    async fn compress_chain_caches_by_tip_and_recompresses_on_new_tip() {
        use crate::history::NODE_BYTES;
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut ctx = ctx_with(crate::history::HistoryCompressor::new(
            Box::new(CountingSummarizer {
                calls: calls.clone(),
            }),
            128_000,
        ));

        // A message over NODE_BYTES forces exactly one summarizer call.
        let long = "x".repeat(NODE_BYTES * 2);
        let msgs = vec![llung_core::network::message::ChatMessage {
            id: "m1".into(),
            topic: "t".into(),
            sender_id: "p".into(),
            sender_name: "alice".into(),
            parent_id: None,
            kind: llung_core::network::message::MessageKind::Human,
            content: long.clone(),
            timestamp: 0,
        }];
        let tip = msgs[0].id.clone();

        let v1 = compress_chain(&mut ctx, &tip, &msgs).await;
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        let v2 = compress_chain(&mut ctx, &tip, &msgs).await;
        assert_eq!(v1, v2);
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "same tip must reuse the cached view"
        );

        compress_chain(&mut ctx, "a-different-tip", &msgs).await;
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "a new tip recompresses"
        );
    }

    /// Backend that replays canned replies in order, recording every
    /// prompt it was asked. `chat_stream` uses the trait default
    /// (non-streaming `chat` + one delta), which is what the zoom test
    /// wants.
    #[derive(Clone)]
    struct ScriptBackend {
        state: std::sync::Arc<ScriptState>,
    }
    struct ScriptState {
        replies: std::sync::Mutex<std::collections::VecDeque<String>>,
        prompts: std::sync::Mutex<Vec<Vec<ChatMessage>>>,
    }
    impl ScriptBackend {
        fn new(replies: Vec<String>) -> Self {
            Self {
                state: std::sync::Arc::new(ScriptState {
                    replies: std::sync::Mutex::new(replies.into()),
                    prompts: std::sync::Mutex::new(Vec::new()),
                }),
            }
        }
        fn prompts(&self) -> Vec<Vec<ChatMessage>> {
            self.state.prompts.lock().unwrap().clone()
        }
    }
    impl Backend for ScriptBackend {
        fn chat(
            &self,
            messages: &[ChatMessage],
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = llung_core::network::message::AgentResult<String>> + Send + '_>,
        > {
            self.state.prompts.lock().unwrap().push(messages.to_vec());
            let next = self
                .state
                .replies
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_else(|| "(script exhausted)".to_string());
            Box::pin(async move { Ok(next) })
        }
    }

    #[tokio::test]
    async fn plot_device_path_is_resolved_to_csv_by_the_host() {
        let dir = std::env::temp_dir();
        let csv = dir.join(format!("llung-resolve-{}.csv", std::process::id()));
        std::fs::write(&csv, "a,b\n1,2\n3,4\n").unwrap();

        let call = crate::tools::ToolCall {
            name: "plot_device".into(),
            arguments: format!(
                "{{\"path\":\"{}\",\"kind\":\"scatter\"}}",
                csv.display()
            ),
        };
        let resolved = crate::tools::resolve_data_file(call).await.unwrap();
        // The path stays (panels show it); the csv content is injected.
        assert!(resolved.arguments.contains("\"path\""));
        assert!(resolved.arguments.contains("a,b\\n1,2"), "{}", resolved.arguments);

        // Other tools are untouched; existing csv wins.
        let other = crate::tools::ToolCall {
            name: "landscape".into(),
            arguments: format!("{{\"path\":\"{}\"}}", csv.display()),
        };
        assert_eq!(crate::tools::resolve_data_file(other).await.unwrap().arguments, format!("{{\"path\":\"{}\"}}", csv.display()));
        let has_csv = crate::tools::ToolCall {
            name: "plot_device".into(),
            arguments: "{\"csv\":\"x\",\"path\":\"/nope\"}".into(),
        };
        assert!(crate::tools::resolve_data_file(has_csv).await.unwrap().arguments.contains("/nope"));

        // A missing file is a clean, model-readable error.
        let missing = crate::tools::ToolCall {
            name: "plot_device".into(),
            arguments: "{\"path\":\"/no/such.csv\"}".into(),
        };
        let err = crate::tools::resolve_data_file(missing).await.unwrap_err();
        assert!(err.contains("cannot read"), "{err}");

        std::fs::remove_file(&csv).ok();
    }

    #[tokio::test]
    async fn zoom_failure_is_fed_back_and_the_model_self_corrects() {
        use crate::history::NODE_BYTES;
        let backend = ScriptBackend::new(vec![
            // 1. malformed: missing 'end'.
            r#"TOOL {"name":"zoom","arguments":{"start":0}}"#.to_string(),
            // 2. corrected after feedback.
            r#"TOOL {"name":"zoom","arguments":{"start":0,"end":2}}"#.to_string(),
            // 3. the actual answer.
            "zoomed in and answered".to_string(),
        ]);
        let (cmd_tx, mut cmd_rx) = mpsc::channel(8);
        let (stream_tx, _stream_rx) = mpsc::channel(32);
        let (perm_tx, _perm_rx) = mpsc::channel(8);
        let being = libp2p::identity::Keypair::generate_ed25519()
            .public()
            .to_peer_id()
            .to_base58();
        let mut ctx = AgentLoopContext::new(
            AgentConfig::default(),
            Box::new(backend.clone()),
            crate::history::HistoryCompressor::new(
                Box::new(CountingSummarizer {
                    calls: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                }),
                128_000,
            ),
            test_db(),
            cmd_tx,
            "t".into(),
            being,
            "tester".into(),
            stream_tx,
            perm_tx,
        );

        let long = "x".repeat(NODE_BYTES * 2);
        let m1 = human_msg("t", "alice", &long);
        let mut m2 = human_msg("t", "alice", &long);
        m2.parent_id = Some(m1.id.clone());
        ctx.db.save_message(&m1).unwrap();
        ctx.db.save_message(&m2).unwrap();

        generate_reply(&mut ctx, "alice: @tester what did I say?", Some(&m2.id))
            .await
            .unwrap();

        let prompts = backend.prompts();
        assert_eq!(prompts.len(), 3, "bad zoom + correction + answer");
        // The second prompt feeds the failure back.
        assert!(
            prompts[1].iter().any(|m| m.content.contains("were rejected")
                && m.content.contains("missing 'end'")),
            "feedback must name the error: {:?}",
            prompts[1].iter().map(|m| &m.content).collect::<Vec<_>>()
        );
        // The third prompt is the zoomed re-ask, full text inline.
        assert!(prompts[2][1].content.contains(&long));

        // One publish: the final answer.
        let cmd = cmd_rx.try_recv().expect("a published reply");
        let NetworkCommand::PublishMessage { contents, .. } = cmd else {
            panic!("expected a publish command");
        };
        let msg: ChatMessage = serde_json::from_slice(&contents).unwrap();
        assert_eq!(msg.content, "zoomed in and answered");
        assert!(cmd_rx.try_recv().is_err(), "exactly one publish");
    }

    #[tokio::test]
    async fn zoom_tool_expands_compressed_history_and_reasks() {
        use crate::history::NODE_BYTES;
        let backend = ScriptBackend::new(vec![
            r#"TOOL {"name":"zoom","arguments":{"start":0,"end":2}}"#.to_string(),
            "the full picture, now in focus".to_string(),
        ]);
        let (cmd_tx, mut cmd_rx) = mpsc::channel(8);
        let (stream_tx, _stream_rx) = mpsc::channel(32);
        let (perm_tx, _perm_rx) = mpsc::channel(8);
        // A valid base58 peer id: publishing parses the being id.
        let being = libp2p::identity::Keypair::generate_ed25519()
            .public()
            .to_peer_id()
            .to_base58();
        let mut ctx = AgentLoopContext::new(
            AgentConfig::default(),
            Box::new(backend.clone()),
            crate::history::HistoryCompressor::new(
                Box::new(CountingSummarizer {
                    calls: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                }),
                128_000,
            ),
            test_db(),
            cmd_tx,
            "t".into(),
            being,
            "tester".into(),
            stream_tx,
            perm_tx,
        );

        // Two long messages on one branch: long enough that compression
        // summarizes them, so full text exists only after zooming.
        let long = "x".repeat(NODE_BYTES * 2);
        let m1 = human_msg("t", "alice", &long);
        let mut m2 = human_msg("t", "alice", &long);
        m2.parent_id = Some(m1.id.clone());
        ctx.db.save_message(&m1).unwrap();
        ctx.db.save_message(&m2).unwrap();

        generate_reply(&mut ctx, "alice: @tester what did I say?", Some(&m2.id))
            .await
            .unwrap();

        // Two model calls: the zoom, then the re-ask.
        let prompts = backend.prompts();
        assert_eq!(prompts.len(), 2, "zoom must trigger a re-ask");
        // First view: compressed — the long text must NOT be there,
        // and the lines are index-addressed for zoom.
        assert!(!prompts[0][1].content.contains(&long));
        assert!(prompts[0][1].content.contains("[0]"), "indexed lines shown");
        // Second view: expanded — full text present.
        assert!(prompts[1][1].content.contains(&long));

        // Exactly one published message: the final reply, on the branch.
        let cmd = cmd_rx.try_recv().expect("a published reply");
        let NetworkCommand::PublishMessage { contents, .. } = cmd else {
            panic!("expected a publish command");
        };
        let msg: ChatMessage = serde_json::from_slice(&contents).unwrap();
        assert_eq!(msg.content, "the full picture, now in focus");
        assert!(matches!(msg.kind, MessageKind::Agent));
        assert_eq!(msg.parent_id.as_deref(), Some(m2.id.as_str()));
        assert!(cmd_rx.try_recv().is_err(), "exactly one publish");
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

    fn skip_without_bwrap() -> bool {
        if std::env::var("LLUNG_SANDBOX").as_deref() == Ok("off") {
            eprintln!("LLUNG_SANDBOX=off; skipping");
            return true;
        }
        if crate::sandbox::closure::which("bwrap").is_none() {
            eprintln!("bubblewrap not on PATH; skipping");
            return true;
        }
        false
    }

    fn ctx_with_perm() -> (AgentLoopContext, mpsc::Receiver<PermissionRequest>) {
        let (cmd_tx, _cmd_rx) = mpsc::channel(8);
        let (stream_tx, _stream_rx) = mpsc::channel(8);
        let (perm_tx, perm_rx) = mpsc::channel(8);
        let ctx = AgentLoopContext::new(
            AgentConfig::default(),
            Box::new(crate::interface::OpenAiCompatible::llamacpp(1)),
            crate::history::HistoryCompressor::new(
                Box::new(crate::history::HeadlineSummarizer::new()),
                128_000,
            ),
            test_db(),
            cmd_tx,
            "t".into(),
            "being".into(),
            "tester".into(),
            stream_tx,
            perm_tx,
        );
        (ctx, perm_rx)
    }

    /// Prompt-required tier with no standing grant asks the operator;
    /// answering "always" runs it and every LATER call runs unprompted.
    #[tokio::test]
    async fn permission_prompt_once_then_standing_covers() {
        if skip_without_bwrap() {
            return;
        }
        let (mut ctx, mut perm_rx) = ctx_with_perm();
        let answerer = tokio::spawn(async move {
            let req = perm_rx.recv().await.expect("a permission request");
            assert_eq!(req.effect, "bash");
        assert!(req.summary.contains("$ echo granted"), "got: {}", req.summary);
        assert!(req.allow_standing, "proc:bash must allow standing");
        let _ = req.respond.send(PermissionAnswer::Standing);
            perm_rx
        });

        let out = execute_secured_tool(
            &mut ctx,
            "bash".into(),
            br#"{"cmd":"echo granted","timeout_secs":10}"#.to_vec(),
        )
        .await
        .expect("execution after the operator answered");
        assert!(out.contains("granted"), "got: {out}");
        assert!(
            ctx.kernel.check_standing(&Manifest {
                agent_id: ctx.self_being_id.clone(),
                target_effect: "bash".into(),
                arguments: vec![],
                nonce: 0,
            }),
            "standing grant must be recorded in the kernel"
        );

        let mut perm_rx = answerer.await.unwrap();
        let out2 = execute_secured_tool(
            &mut ctx,
            "bash".into(),
            br#"{"cmd":"echo covered","timeout_secs":10}"#.to_vec(),
        )
        .await
        .expect("standing grant covers the second call");
        assert!(out2.contains("covered"), "got: {out2}");
        assert!(
            perm_rx.try_recv().is_err(),
            "a standing grant must suppress the prompt"
        );
    }

    /// Denial: the model gets a tool error, the merkle history records a
    /// Denied receipt, and nothing is consumed — the next attempt asks
    /// again and can still be allowed.
    #[tokio::test]
    async fn denial_records_receipt_and_stays_retryable() {
        if skip_without_bwrap() {
            return;
        }
        let (mut ctx, mut perm_rx) = ctx_with_perm();
        let root_before = ctx.kernel.current_history_root;

        // Two denials, then one approval.
        let answerer = tokio::spawn(async move {
            for i in 0..3 {
                if let Some(req) = perm_rx.recv().await {
                    let answer = if i < 2 {
                        PermissionAnswer::Deny
                    } else {
                        PermissionAnswer::Once
                    };
                    let _ = req.respond.send(answer);
                }
            }
            perm_rx
        });

        let err = execute_secured_tool(
            &mut ctx,
            "bash".into(),
            br#"{"cmd":"echo nope","timeout_secs":10}"#.to_vec(),
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("permission denied"), "got: {err}");
        assert_ne!(
            ctx.kernel.current_history_root, root_before,
            "the denial must be folded into the merkle history"
        );

        // Deny again — the ask path was not consumed.
        let err2 = execute_secured_tool(
            &mut ctx,
            "bash".into(),
            br#"{"cmd":"echo nope2","timeout_secs":10}"#.to_vec(),
        )
        .await
        .unwrap_err();
        assert!(err2.to_string().contains("permission denied"), "got: {err2}");

        // Then approve: execution proceeds.
        let out = execute_secured_tool(
            &mut ctx,
            "bash".into(),
            br#"{"cmd":"echo finally","timeout_secs":10}"#.to_vec(),
        )
        .await
        .expect("approval after denials");
        assert!(out.contains("finally"), "got: {out}");

        let mut perm_rx = answerer.await.unwrap();
        assert!(
            perm_rx.try_recv().is_err(),
            "exactly three answers were consumed"
        );
    }

    /// A pre-existing standing grant (e.g. from /grant) means the prompt
    /// never fires even on the first call.
    #[tokio::test]
    async fn preexisting_standing_grant_never_prompts() {
        if skip_without_bwrap() {
            return;
        }
        let (mut ctx, mut perm_rx) = ctx_with_perm();
        ctx.kernel.grant_standing(StandingGrant {
            agent_id: ctx.self_being_id.clone(),
            effect: "bash".into(),
            authorized_by: ctx.supervisor.public_pin(),
        });
        let out = execute_secured_tool(
            &mut ctx,
            "bash".into(),
            br#"{"cmd":"echo pre-granted","timeout_secs":10}"#.to_vec(),
        )
        .await
        .expect("standing grant covers the first call too");
        assert!(out.contains("pre-granted"), "got: {out}");
        assert!(
            perm_rx.try_recv().is_err(),
            "no prompt when a standing grant covers the effect"
        );
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

