// crates/agent/src/orchestrator.rs
use crate::aukora::kernel::{Grant, KernelState, Manifest, StandingGrant};
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
        message::{Attachment, AttachmentKind, ChatMessage, MessageKind},
    },
    storage::db::Database,
    utils::generate_cid_from_bytes,
};
use rand::Rng;
use tokio::sync::mpsc;

/// Incremental reply text from a local agent turn, for UI streaming.
/// Local-only: the complete reply still travels as one gossipsub
/// message; remote peers see the block.
pub enum AgentStream {
    /// The model started working on reply `message_id`: no output yet.
    /// Lets the UI show a spinner during the thinking phase instead of
    /// silence before the first delta.
    Working {
        topic: String,
        message_id: String,
        being_id: String,
        sender_name: String,
    },
    /// Append reasoning scratchpad text to the in-progress reply. The
    /// UI counts it as activity (spinner, then an elided trace) — it is
    /// never the answer itself.
    Thought {
        topic: String,
        message_id: String,
        delta: String,
    },
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
    /// The ACTIVE topic: the one the current (or last) turn is on —
    /// replies, streams, and the background compactor use it. Starts as
    /// the spawn topic; switches per inbound message among `topics`.
    pub topic: String,
    /// Every topic this agent answers on: the spawn topic plus any it
    /// was invited to later (`AgentCommand::JoinTopic`).
    pub topics: Vec<String>,
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
    /// Where the agent works (workspace, runs, data dirs): the places
    /// snapshot this agent last saw. Refreshed from
    /// [`crate::places::current`] at the start of every turn, so the
    /// operator can hot-swap directories; every tool in a turn uses the
    /// same snapshot.
    pub places: std::sync::Arc<crate::places::Places>,
    /// Place changes the agent hasn't been told about yet (shown as a
    /// NOTICE in the next turn's prompt; cleared once a turn succeeds).
    pub pending_place_changes: Vec<String>,
    /// The tool-call format this agent's prompt teaches, resolved on the
    /// first turn (see [`resolve_tool_protocol`]). `None` = not yet.
    pub tool_protocol: Option<crate::tools::ToolProtocol>,
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
    /// Answer on `topic` too (`/invite <agent> #topic`). The UI also
    /// subscribes the agent's NETWORK to the topic; this tells the
    /// agent LOOP to stop filtering it out.
    JoinTopic { topic: String },
}

/// How often the background compactor recompresses the agent's home
/// branch between turns (OptChat's background compactor — the tree stays
/// warm instead of being rebuilt synchronously per turn).
const COMPACT_INTERVAL: std::time::Duration = std::time::Duration::from_secs(300);

/// Upper bound on `zoom` re-asks per turn — the model must not be able
/// to ping-pong context expansions forever.
const MAX_ZOOM_ROUNDS: u32 = 3;
/// Argument-error feedback rounds for ordinary tool calls: the model
/// gets this many chances to correct a rejected tool call's JSON
/// before the failure is published as the turn's result.
const MAX_TOOL_ROUNDS: u32 = 3;
/// Backend attempts per ask. Thinking-model endpoints intermittently
/// return empty or truncated replies (finish_reason=length, dropped
/// streams); a retry is nearly always enough, and publishing a
/// one-sentence non-answer as the turn's result (or an error panel for
/// a transient blip) is worse than a few extra seconds.
const MAX_ASK_ATTEMPTS: u32 = 3;
/// Tool executions per turn: after each tool result the model is
/// re-asked, so it can work iteratively (grep → read → edit → run in
/// the REPL, correcting as it goes) before producing its final answer.
const MAX_TOOL_STEPS: u32 = 8;
/// Cap on a tool result fed back into the prompt (chars), so a chatty
/// REPL or a huge grep can't blow the turn's context.
const TOOL_FEEDBACK_CHARS: usize = 4_000;

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
            topics: vec![topic.clone()],
            topic,
            self_being_id,
            self_name,
            places: crate::places::current(),
            pending_place_changes: Vec::new(),
            tool_protocol: None,
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
    // Resolve the sandbox runtime now, in the background, so the first
    // tool call doesn't pay for it (seconds on a cold Nix host).
    tokio::task::spawn_blocking(crate::sandbox::warm_up);

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
        AgentCommand::JoinTopic { topic } => {
            if !ctx.topics.contains(&topic) {
                tracing::info!(agent = %ctx.self_name, %topic, "joined topic");
                ctx.topics.push(topic);
            }
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

    if !ctx.topics.contains(&topic) || sender.to_base58() == ctx.self_being_id {
        return;
    }

    let Incoming::Message {
        text,
        kind,
        structured,
    } = decode_incoming(&data, &topic)
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

    // Conversational content earns a turn — media included, so an
    // addressed image (@agent what is this plot?) reaches a multimodal
    // backend. Tool traffic, notes, and system announcements stay
    // silent; the @mention gate below keeps media from chattering.
    if !kind_responds(&kind) {
        return;
    }

    // Gate: only spend a turn when addressed. Whether the turn is a
    // plain reply or a tool call is decided by the model's output.
    if !is_addressed(&text, &ctx.config.name) {
        return;
    }

    // Turns run one at a time per agent, so the active topic can simply
    // follow the turn: replies, streams and tool traffic land on the
    // topic the trigger came from (and the compactor follows along).
    ctx.topic = topic;

    let trigger_id = id_of(&structured);
    let trigger = structured.clone();
    refresh_places(ctx);
    resolve_tool_protocol(ctx).await;
    let result = generate_reply(&mut *ctx, &text, trigger_id.as_deref(), trigger.as_ref()).await;
    if result.is_ok() {
        // The model saw the NOTICE; don't repeat it.
        ctx.pending_place_changes.clear();
    }
    if let Err(e) = result {
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

/// Pick the tool-call format to teach: `LLUNG_TOOL_PROTOCOL` if set,
/// else whatever the model's own chat template uses (llama-server
/// `/props`), else the compact TOOL line. Resolved once per agent; an
/// unreachable server is retried next turn.
async fn resolve_tool_protocol(ctx: &mut AgentLoopContext) {
    if ctx.tool_protocol.is_some() {
        return;
    }
    if let Some(p) = std::env::var("LLUNG_TOOL_PROTOCOL")
        .ok()
        .and_then(|v| crate::tools::ToolProtocol::parse(&v))
    {
        ctx.tool_protocol = Some(p);
        return;
    }
    match ctx.backend.chat_template().await {
        Some(template) => {
            let p = crate::tools::ToolProtocol::from_chat_template(&template);
            tracing::info!(agent = %ctx.self_name, protocol = ?p, "tool protocol from chat template");
            ctx.tool_protocol = Some(p);
        }
        None => {} // Line for now; ask again next turn
    }
}

/// Adopt the live places snapshot if the operator changed it since this
/// agent's last turn, remembering what changed so the next prompt can
/// tell the model (paths in its history may now be stale).
fn refresh_places(ctx: &mut AgentLoopContext) {
    let now = crate::places::current();
    if now.generation == ctx.places.generation {
        return;
    }
    let changes = ctx.places.diff(&now);
    if !changes.is_empty() {
        tracing::info!(agent = %ctx.self_name, ?changes, "places changed");
        ctx.pending_place_changes.extend(changes);
    }
    ctx.places = now;
}

/// Periodic self-compaction (OptChat's background compactor): compress
/// the active topic's branch (the one the agent last spoke on) and warm
/// the view cache. Errors never
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
    let spinner = CompressorSpinner::new(ctx);
    if let Some(spinner) = &spinner {
        spinner.show().await;
    }
    let tree = ctx.compressor.compress(messages).await;
    if let Some(spinner) = spinner {
        spinner.done().await;
    }
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

/// A transient `working` line attributed to the history compressor
/// (e.g. "HistoryCompressor (SmolVLM) is working…"), shown while it
/// builds the summary tree. Compression can call a model for minutes on
/// a slow local backend; without its own line that time would show up
/// as the AGENT working. Only model-backed compressors get one — the
/// rule-based fallback is instant. Holds owned clones (no `&ctx`
/// across awaits: the turn future must stay Send).
struct CompressorSpinner {
    stream_tx: mpsc::Sender<AgentStream>,
    topic: String,
    message_id: String,
    sender_name: String,
}

impl CompressorSpinner {
    /// Sync on purpose: clone what the line needs out of `ctx` before
    /// any await, then [`Self::show`] it.
    fn new(ctx: &AgentLoopContext) -> Option<Self> {
        if !ctx.compressor.is_model_backed() {
            return None;
        }
        Some(Self {
            stream_tx: ctx.stream_tx.clone(),
            topic: ctx.topic.clone(),
            message_id: format!("{}-compress-{:016x}", ctx.self_being_id, {
                let mut rng = rand::rand_core::UnwrapErr(rand::rngs::SysRng);
                rng.next_u64()
            }),
            sender_name: ctx.compressor.display_name(),
        })
    }

    /// Put the spinner line on screen.
    async fn show(&self) {
        let _ = self
            .stream_tx
            .send(AgentStream::Working {
                topic: self.topic.clone(),
                message_id: self.message_id.clone(),
                // Not the agent's being: no avatar, not the agent's line.
                being_id: String::new(),
                sender_name: self.sender_name.clone(),
            })
            .await;
    }

    /// Compression finished: remove the line.
    async fn done(self) {
        let _ = self
            .stream_tx
            .send(AgentStream::Retract {
                topic: self.topic,
                message_id: self.message_id,
            })
            .await;
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

/// Which message kinds the agent will respond to. Media qualifies:
/// with the @mention gate, an addressed image is exactly the
/// multimodal-input case (see `ChatMessage::attachment`).
fn kind_responds(kind: &MessageKind) -> bool {
    matches!(
        kind,
        MessageKind::Human | MessageKind::Agent | MessageKind::Media(_)
    )
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

    // Tier::Host effects (show_image) execute HERE, not in the wasm
    // worker or the pure host dispatcher: publishing needs the agent's
    // own context (network stack, media cache). The receipt still
    // commits to the produced output through the same kernel gate.
    if effect.tier == crate::effects::Tier::Host {
        let output = match effect.name {
            "show_image" => show_image(ctx, &arguments).await?,
            other => {
                return Err(format!("host:{other} has no orchestrator executor").into());
            }
        };
        let grant = create_self_grant(&manifest, &ctx.supervisor);
        crate::aukora::security::commit_host_effect(&mut ctx.kernel, manifest, grant, &output)
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
        crate::aukora::persist::save_kernel_state(&ctx.db, &ctx.self_being_id, &ctx.kernel)?;
        return Ok(output);
    }

    // Self-grant, signed by the supervisor key pinned in the kernel.
    let grant = create_self_grant(&manifest, &ctx.supervisor);

    // Build receipt via Aukora kernel + wasm runtime
    let places = ctx.places.clone();
    let (_receipt, output) = crate::aukora::security::secure_execute_agent_tool_in(
        &mut ctx.kernel,
        &ctx.wasm_tx, // agent-rt wasm worker handle
        manifest,
        grant,
        &places,
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

/// Cap a tool result that is fed back into the prompt.
fn truncate_for_feedback(output: &str) -> String {
    let total = output.chars().count();
    if total <= TOOL_FEEDBACK_CHARS {
        return output.to_string();
    }
    let kept: String = output.chars().take(TOOL_FEEDBACK_CHARS).collect();
    format!("{kept}\n… [{} chars truncated]", total - TOOL_FEEDBACK_CHARS)
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

/// Publish a media message (image attachment + caption) on behalf of
/// the agent — the chat side of the `show_image` tool. Attachments
/// travel as `MessageKind::Media`; the TUI renders them like user
/// attachments and relay fetch works for peers either way.
async fn publish_agent_media_on(
    ctx: &mut AgentLoopContext,
    parent_id: Option<String>,
    attachment: Attachment,
    caption: String,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    publish(ctx, MessageKind::Media(attachment), parent_id, caption, None)
        .await
        .map(|_| ())
}

#[derive(serde::Deserialize)]
struct ShowImageArgs {
    path: String,
    #[serde(default)]
    caption: Option<String>,
}

/// Resolve a tool-supplied image path: relative paths resolve against
/// the workspace; the file must lie in the workspace or the runs
/// directory (where plots are saved). `..` segments and absolute paths
/// elsewhere canonicalize outside both roots and are rejected.
fn resolve_image_path(places: &crate::places::Places, arg: &str) -> Result<std::path::PathBuf, String> {
    crate::paths::agent_readable_in(arg, places.workspace(), &places.writable_roots()).map_err(
        |e| {
            if e.starts_with("outside:") {
                format!("'{arg}' escapes the workspace and the runs directory")
            } else {
                e
            }
        },
    )
}

/// `show_image` (plan Phase 2): publish a workspace image — a
/// matplotlib PNG the torch REPL saved — to the chat as a media
/// message from the agent's own being. Runs in the orchestrator
/// (needs `AgentLoopContext`); returns the confirmation string the
/// aukora receipt commits to. Mirrors the user-side `send_attachment`
/// in tui/app.rs: media cache -> relay publish -> media message.
async fn show_image(
    ctx: &mut AgentLoopContext,
    arguments: &[u8],
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let args: ShowImageArgs = serde_json::from_slice(arguments)
        .map_err(|e| format!("show_image: arguments must be JSON with 'path': {e}"))?;
    let path = resolve_image_path(&ctx.places, &args.path)
        .map_err(|e| format!("show_image: {e}"))?;
    let meta = std::fs::metadata(&path)
        .map_err(|e| format!("show_image: cannot stat '{}': {e}", args.path))?;
    if !meta.is_file() {
        return Err(format!("show_image: '{}' is not a file", args.path).into());
    }
    if meta.len() as usize > llung_core::media::image::MAX_IMAGE_BYTES {
        return Err(format!(
            "show_image: '{}' is {} bytes; the cap is {} — save a smaller figure",
            args.path,
            meta.len(),
            llung_core::media::image::MAX_IMAGE_BYTES
        )
        .into());
    }
    let bytes = std::fs::read(&path)
        .map_err(|e| format!("show_image: cannot read '{}': {e}", args.path))?;
    let cid = generate_cid_from_bytes(&bytes)
        .map_err(|e| format!("show_image: content id failed: {e}"))?;
    ctx.db.save_avatar_cache(&cid, &bytes)?;
    // from_path opens the image for width/height — non-image files are
    // rejected here.
    let attachment = Attachment::from_path(path.to_str().unwrap_or("unknown"), cid.clone())
        .map_err(|e| format!("show_image: '{}' is not a usable image: {e}", args.path))?;
    if attachment.kind != AttachmentKind::Image {
        return Err(format!("show_image: '{}' is not an image", args.path).into());
    }
    ctx.cmd_tx
        .send(NetworkCommand::ProvideMediaCid { cid: cid.clone() })
        .await?;
    // Images ≤ MAX_IMAGE_BYTES also go into the relay caches, so peers
    // behind NATs and the operator's other devices can fetch them.
    ctx.cmd_tx
        .send(NetworkCommand::PublishImage {
            cid: cid.clone(),
            mime_type: attachment.mime_type.clone(),
            data: bytes.clone(),
        })
        .await?;
    let filename = attachment.filename.clone();
    let size_kb = attachment.size_bytes / 1024;
    let caption = args.caption.unwrap_or_default();
    publish_agent_media_on(ctx, None, attachment, caption).await?;
    Ok(format!("published image {filename} ({size_kb} KB) as cid {cid}"))
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
fn turn_prompt(
    ctx: &AgentLoopContext,
    context: &str,
    trigger_text: &str,
    trigger_attachment: Option<&llung_core::network::message::Attachment>,
) -> Vec<ChatMessage> {
    turn_prompt_inner(ctx, context, trigger_text, None, trigger_attachment)
}

/// Like [`turn_prompt`], with tool-error feedback appended as the last
/// (most-weighted) turn — small models read recent context best.
fn turn_prompt_feedback(
    ctx: &AgentLoopContext,
    context: &str,
    trigger_text: &str,
    feedback: &str,
    trigger_attachment: Option<&llung_core::network::message::Attachment>,
) -> Vec<ChatMessage> {
    turn_prompt_inner(ctx, context, trigger_text, Some(feedback), trigger_attachment)
}

fn turn_prompt_inner(
    ctx: &AgentLoopContext,
    context: &str,
    trigger_text: &str,
    feedback: Option<&str>,
    trigger_attachment: Option<&llung_core::network::message::Attachment>,
) -> Vec<ChatMessage> {
    assemble_turn_prompt(
        &ctx.self_name,
        &ctx.places,
        ctx.tool_protocol.unwrap_or_default(),
        &ctx.pending_place_changes,
        context,
        trigger_text,
        feedback,
        trigger_attachment,
    )
}

/// The turn prompt from its parts — public so diagnostics
/// (`examples/replay_turn.rs`) rebuild EXACTLY what an agent sent.
#[allow(clippy::too_many_arguments)]
pub fn assemble_turn_prompt(
    self_name: &str,
    places: &crate::places::Places,
    protocol: crate::tools::ToolProtocol,
    pending_place_changes: &[String],
    context: &str,
    trigger_text: &str,
    feedback: Option<&str>,
    trigger_attachment: Option<&llung_core::network::message::Attachment>,
) -> Vec<ChatMessage> {
    // The triggering message carries the tool protocol AND the
    // optional image: a multimodal backend (SenseNova + mmproj) sees
    // the attachment here; a text-only one just reads the caption.
    // A pending places change leads the message: it is the most
    // important (and most recent) fact about the agent's environment.
    let notice = if pending_place_changes.is_empty() {
        String::new()
    } else {
        format!("{}\n\n", crate::tools::places_changed_notice(&pending_place_changes))
    };
    let mut trigger_msg = ChatMessage::user(format!(
        "{}New message: {}\n\n{}\n\n{}",
        notice,
        trigger_text,
        crate::tools::tool_instructions(places, protocol),
        crate::tools::environment_note(places),
    ));
    if let Some(att) = trigger_attachment {
        trigger_msg = trigger_msg.with_attachment(att.clone());
    }
    let mut v = vec![
        ChatMessage::system(format!(
            "You are {}, a helpful agent in a peer-to-peer chat. \
             History is compressed; lines are numbered like [3] or [2-5] \
             — use the zoom tool to expand a range to its full text. \
             Respond concisely.",
            self_name
        )),
        ChatMessage::user(context.to_string()),
        trigger_msg,
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
    let mut attempt = 0u32;
    loop {
        // Activity signal BEFORE the first token: the UI shows a spinner
        // instead of looking idle while the model thinks (re-sent on
        // every attempt and re-ask — it also resets the preview line).
        let _ = stream_tx
            .send(AgentStream::Working {
                topic: topic.to_string(),
                message_id: reply_id.to_string(),
                being_id: being_id.to_string(),
                sender_name: sender_name.to_string(),
            })
            .await;
        let (delta_tx, mut delta_rx) = mpsc::channel::<crate::interface::StreamChunk>(256);
        let forward_deltas = async {
            while let Some(chunk) = delta_rx.recv().await {
                let ev = match chunk {
                    crate::interface::StreamChunk::Text(delta) => AgentStream::Delta {
                        topic: topic.to_string(),
                        message_id: reply_id.to_string(),
                        being_id: being_id.to_string(),
                        sender_name: sender_name.to_string(),
                        delta,
                    },
                    crate::interface::StreamChunk::Thought(delta) => AgentStream::Thought {
                        topic: topic.to_string(),
                        message_id: reply_id.to_string(),
                        delta,
                    },
                };
                let _ = stream_tx.send(ev).await;
            }
        };
        let (reply, ()) = tokio::join!(backend.chat_stream(prompt, delta_tx), forward_deltas);
        match reply {
            Ok(reply) => return Ok(reply),
            Err(e) => {
                attempt += 1;
                if attempt >= MAX_ASK_ATTEMPTS {
                    return Err(e);
                }
                tracing::warn!(
                    "backend ask failed ({e}); retrying ({attempt}/{MAX_ASK_ATTEMPTS})"
                );
            }
        }
    }
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
    // `trigger`: the structured trigger message, when the peer speaks
    // the ChatMessage protocol. Its image attachment (if any) rides the
    // final prompt message into the backend — that is the
    // multimodal-input path.
    trigger: Option<&ChatMessage>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // load_dag boxes a non-Send error; re-box it so the future stays Send.
    let dag = ctx
        .db
        .load_dag()
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
            e.to_string().into()
        })?;
    // The reply streams to the local UI under a precomputed id while the
    // backend request runs; the final publish (below) reuses the id.
    // No agent `Working` signal yet: history compression runs first and
    // shows its OWN spinner line (`CompressorSpinner`) — that time is
    // the compressor model working, not the agent. `ask_streamed`
    // raises the agent's spinner right before the agent's model call.
    let reply_id = format!("{}-stream-{:016x}", ctx.self_being_id, {
        let mut rng = rand::rand_core::UnwrapErr(rand::rngs::SysRng);
        rng.next_u64()
    });

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

    let trigger_attachment = trigger.and_then(|m| m.image_attachment().cloned());
    let mut prompt = turn_prompt(ctx, &context, trigger_text, trigger_attachment.as_ref());
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
                         one corrected tool call, or just answer normally."
                    );
                    prompt = turn_prompt_feedback(
                        ctx,
                        &context,
                        trigger_text,
                        &feedback,
                        trigger_attachment.as_ref(),
                    );
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
        let spinner = CompressorSpinner::new(ctx);
        if let Some(spinner) = &spinner {
            spinner.show().await;
        }
        let mut tree = ctx.compressor.compress(&messages).await;
        if let Some(spinner) = spinner {
            spinner.done().await;
        }
        zoom_range(&mut tree, &messages, start, end);
        // Same budgeted cut; the zoomed range stands even past budget.
        context = ctx
            .compressor
            .flatten_budgeted(&tree, ctx.compressor.context_budget);
        prompt = turn_prompt(ctx, &context, trigger_text, trigger_attachment.as_ref());
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

    // The model may work iteratively with tools: each tool result is
    // fed back and it may call the next tool (REPL until it works,
    // grep → edit → bash, ...) or answer. Bad arguments are fed back
    // too, so the model can correct its JSON — same contract as the
    // zoom loop: bounded rounds, every call/response pair published.
    let mut steps = 0u32;
    let mut step_log: Vec<String> = Vec::new();
    // Unparseable tool markup gets fed back (bounded) instead of being
    // published: in the chat it would become history the model then
    // imitates. A corrected call re-enters the tool loop.
    let protocol = ctx.tool_protocol.unwrap_or_default();
    let mut malformed_rounds = 0u32;
    loop {
    'tool: while let Some(call) = crate::tools::parse_tool_call(&reply) {
        // The streamed preview was raw tool-call text: withdraw it.
        let _ = ctx
            .stream_tx
            .send(AgentStream::Retract {
                topic: ctx.topic.clone(),
                message_id: reply_id.clone(),
            })
            .await;
        tracing::info!(agent = %ctx.self_name, tool = %call.name, "model requested tool");
        let mut call = call;
        let mut attempts = 0u32;
        // Host-side data files: wasm has no I/O, so a `path` argument
        // is read HERE and injected as `csv` before the sandboxed call.
        let (content, error) = loop {
            // Confined: the host reads this file outside any sandbox,
            // so the agent may only name workspace / LLUNG_RO_BINDS paths.
            let places = ctx.places.clone();
            let resolved = crate::tools::resolve_data_file(call.clone(), Some(&places)).await;
            let result: Result<String, (String, String)> = match resolved {
                Ok(resolved) => {
                    match execute_secured_tool(
                        &mut *ctx,
                        resolved.name.clone(),
                        resolved.arguments.into_bytes(),
                    )
                    .await
                    {
                        Ok(output) => Ok(output),
                        Err(e) => Err((resolved.name.clone(), e.to_string())),
                    }
                }
                Err(e) => Err((call.name.clone(), format!("data file resolution failed: {e}"))),
            };
            match result {
                Ok(output) => break (output, false),
                Err((name, e)) => {
                    attempts += 1;
                    if attempts > MAX_TOOL_ROUNDS {
                        // Never go silent: rounds exhausted, the error
                        // panel shows it where it happened.
                        tracing::error!(tool = %name, "tool failed: {e}");
                        break (format!("tool '{name}' failed: {e}"), true);
                    }
                    tracing::warn!(tool = %name, "tool failed, feeding error back: {e}");
                    let feedback = format!(
                        "Your last reply was a tool call to '{name}', but it \
                         failed: {e}. Correct the arguments and reply with \
                         exactly one corrected tool call, or just answer \
                         normally."
                    );
                    prompt = turn_prompt_feedback(
                        ctx,
                        &context,
                        trigger_text,
                        &feedback,
                        trigger_attachment.as_ref(),
                    );
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
                        // Corrected call: hide the preview and run it.
                        Some(c) => {
                            let _ = ctx
                                .stream_tx
                                .send(AgentStream::Retract {
                                    topic: ctx.topic.clone(),
                                    message_id: reply_id.clone(),
                                })
                                .await;
                            call = c;
                        }
                        // The model moved on (prose or no tool): let the
                        // outer loop re-judge the new reply.
                        None => continue 'tool,
                    }
                }
            }
        };
        // Announce the call first: viewers show a pending tool panel, then
        // swap in the response (whose parent_id points back here). The
        // call itself hangs off the triggering message, so the tool
        // panel lands on the same branch.
        let tool_name = call.name.clone();
        let summary = summarize_output(&content);
        let call_id = publish(
            ctx,
            MessageKind::ToolCall {
                tool_name: tool_name.clone(),
                arguments: call.arguments.clone(),
            },
            trigger_id.map(str::to_string),
            call.arguments.clone(),
            None,
        )
        .await?;
        let response = MessageKind::ToolResponse {
            tool_name: tool_name.clone(),
            result_summary: summary.clone(),
            error,
        };
        publish(ctx, response, Some(call_id), content.clone(), None).await?;
        step_log.push(format!(
            "{tool_name}: {summary}{}",
            if error { " (FAILED)" } else { "" }
        ));
        steps += 1;
        if steps >= MAX_TOOL_STEPS {
            // The results are all published; hand the turn back with
            // a pointer instead of looping forever.
            return publish_agent_text_on(
                ctx,
                trigger_id.map(str::to_string),
                format!(
                    "⚠ I hit my per-turn tool budget ({MAX_TOOL_STEPS} calls). \
                     The results above are in the chat — ask me to continue \
                     and I'll pick up from them."
                ),
                None,
            )
            .await;
        }
        // Feed the result back as the most-recent (most-weighted)
        // turn: the model continues with the next tool call or the
        // final answer. The loop re-judges the reply at 'tool.
        let mut feedback = String::new();
        if step_log.len() > 1 {
            feedback.push_str("Tool calls so far this turn:\n");
            for line in &step_log[..step_log.len() - 1] {
                feedback.push_str("- ");
                feedback.push_str(line);
                feedback.push('\n');
            }
        }
        feedback.push_str(&format!(
            "Tool '{tool_name}' {}returned:\n```\n{}\n```\n\
             Continue working toward the request: reply with exactly one \
             tool call (in the format shown above) to run another tool, \
             or answer normally.",
            if error { "FAILED and " } else { "" },
            truncate_for_feedback(&content),
        ));
        prompt = turn_prompt_feedback(
            ctx,
            &context,
            trigger_text,
            &feedback,
            trigger_attachment.as_ref(),
        );
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

        if !crate::tools::looks_like_tool_markup(&reply) || malformed_rounds >= MAX_TOOL_ROUNDS {
            break;
        }
        malformed_rounds += 1;
        tracing::warn!(agent = %ctx.self_name, "unparseable tool markup; feeding back the format");
        let feedback = format!(
            "Your last reply looks like a tool call, but it could not be \
             parsed, so nothing ran. Reply with exactly one tool call in \
             exactly this format:\n{}\nor answer normally, without any tool \
             markup.",
            protocol.format()
        );
        prompt = turn_prompt_feedback(
            ctx,
            &context,
            trigger_text,
            &feedback,
            trigger_attachment.as_ref(),
        );
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
    if crate::tools::looks_like_tool_markup(&reply) {
        // Still malformed after the feedback rounds: say so in prose
        // rather than publishing markup into the history.
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
            "⚠ I tried to call a tool, but couldn't get the call format right, \
             so nothing ran. Please try asking again."
                .to_string(),
            None,
        )
        .await;
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
    use crate::aukora::security::secure_execute_agent_tool;
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
    async fn agent_answers_only_on_joined_topics() {
        // Spawned on "t"; `/invite` to another topic sends JoinTopic.
        // Before it, that topic's traffic is ignored; after it, a turn
        // there runs ON that topic (replies/streams follow the trigger).
        let mut ctx = ctx_with(crate::history::HistoryCompressor::new(
            Box::new(crate::history::HeadlineSummarizer::new()),
            128_000,
        ));
        let name = ctx.config.name.clone();
        let event = |topic: &str| {
            let msg = ChatMessage {
                id: format!("m-{topic}"),
                topic: topic.into(),
                sender_id: "p".into(),
                sender_name: "alice".into(),
                parent_id: None,
                kind: MessageKind::Human,
                content: format!("@{name} hi"),
                attachment: None,
                timestamp: 0,
            };
            NetworkEvent::MessageReceived {
                topic: topic.into(),
                sender: libp2p::PeerId::random(),
                data: serde_json::to_vec(&msg).unwrap(),
            }
        };

        assert_eq!(ctx.topics, vec!["t".to_string()]);
        handle_event(&mut ctx, event("other")).await;
        assert_eq!(ctx.topic, "t", "not invited yet: ignored");

        handle_agent_command(&mut ctx, AgentCommand::JoinTopic { topic: "other".into() }).await;
        handle_agent_command(&mut ctx, AgentCommand::JoinTopic { topic: "other".into() }).await;
        assert_eq!(ctx.topics, vec!["t".to_string(), "other".to_string()], "idempotent");

        // The turn itself fails (no model server on port 1) — what
        // matters is that it ran, on the joined topic.
        handle_event(&mut ctx, event("other")).await;
        assert_eq!(ctx.topic, "other");
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
            attachment: None,
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
        // `None`: unconfined (the local user's /plot path).
        let resolved = crate::tools::resolve_data_file(call, None).await.unwrap();
        // The path stays (panels show it); the csv content is injected.
        assert!(resolved.arguments.contains("\"path\""));
        assert!(resolved.arguments.contains("a,b\\n1,2"), "{}", resolved.arguments);

        // Other tools are untouched; existing csv wins.
        let other = crate::tools::ToolCall {
            name: "landscape".into(),
            arguments: format!("{{\"path\":\"{}\"}}", csv.display()),
        };
        assert_eq!(crate::tools::resolve_data_file(other, None).await.unwrap().arguments, format!("{{\"path\":\"{}\"}}", csv.display()));
        let has_csv = crate::tools::ToolCall {
            name: "plot_device".into(),
            arguments: "{\"csv\":\"x\",\"path\":\"/nope\"}".into(),
        };
        assert!(crate::tools::resolve_data_file(has_csv, None).await.unwrap().arguments.contains("/nope"));

        // A missing file is a clean, model-readable error.
        let missing = crate::tools::ToolCall {
            name: "plot_device".into(),
            arguments: "{\"path\":\"/no/such.csv\"}".into(),
        };
        let err = crate::tools::resolve_data_file(missing, None).await.unwrap_err();
        assert!(err.contains("cannot read"), "{err}");

        std::fs::remove_file(&csv).ok();
    }

    #[tokio::test]
    async fn agent_plot_device_reads_are_confined_to_the_workspace() {
        // The host reads plot_device's `path` outside any sandbox: an
        // AGENT may only name files the sandboxes would show it.
        let base = std::env::temp_dir().join(format!("llung-confine-{}", std::process::id()));
        let ws = base.join("ws");
        let outside = base.join("outside");
        std::fs::create_dir_all(&ws).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(ws.join("in.csv"), "a\n1\n").unwrap();
        std::fs::write(outside.join("secret.csv"), "s\n9\n").unwrap();
        let call = |path: &std::path::Path| crate::tools::ToolCall {
            name: "plot_device".into(),
            arguments: format!("{{\"path\":{:?}}}", path.display().to_string()),
        };

        let places = crate::places::Places::fixed(&ws, base.join("runs"), vec![]);
        let ok = crate::tools::resolve_data_file(call(&ws.join("in.csv")), Some(&places)).await;
        assert!(ok.unwrap().arguments.contains("a\\n1"));
        let err = crate::tools::resolve_data_file(call(&outside.join("secret.csv")), Some(&places))
            .await
            .unwrap_err();
        assert!(err.contains("outside what this agent may read"), "{err}");
        // The operator's own /plot is not confined.
        assert!(crate::tools::resolve_data_file(call(&outside.join("secret.csv")), None).await.is_ok());

        std::fs::remove_dir_all(&base).ok();
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

        generate_reply(&mut ctx, "alice: @tester what did I say?", Some(&m2.id), None)
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

    /// Backend that emits a reasoning chunk then the answer, to exercise
    /// the Working/Thought/Delta event mapping in `ask_streamed`.
    struct ThinkingBackend;
    impl crate::interface::Backend for ThinkingBackend {
        fn chat(
            &self,
            _messages: &[ChatMessage],
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = llung_core::network::message::AgentResult<String>> + Send + '_>,
        > {
            Box::pin(async { Ok("answer".to_string()) })
        }
        fn chat_stream<'a>(
            &'a self,
            _messages: &'a [ChatMessage],
            deltas: tokio::sync::mpsc::Sender<crate::interface::StreamChunk>,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = llung_core::network::message::AgentResult<String>> + Send + 'a>,
        > {
            Box::pin(async move {
                let _ = deltas
                    .send(crate::interface::StreamChunk::Thought("let me think…".into()))
                    .await;
                let _ = deltas
                    .send(crate::interface::StreamChunk::Text("answer".into()))
                    .await;
                Ok("answer".to_string())
            })
        }
    }

    #[tokio::test]
    async fn turn_streams_working_thought_then_delta_events() {
        let (cmd_tx, _cmd_rx) = mpsc::channel(8);
        let (stream_tx, mut stream_rx) = mpsc::channel(8);
        let (perm_tx, _perm_rx) = mpsc::channel(8);
        let being = libp2p::identity::Keypair::generate_ed25519()
            .public()
            .to_peer_id()
            .to_base58();
        let mut ctx = AgentLoopContext::new(
            AgentConfig::default(),
            Box::new(ThinkingBackend),
            crate::history::HistoryCompressor::new(
                Box::new(CountingSummarizer {
                    calls: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                }),
                128_000,
            )
            .with_label("SmolVLM"),
            test_db(),
            cmd_tx,
            "t".into(),
            being,
            "tester".into(),
            stream_tx,
            perm_tx,
        );

        generate_reply(&mut ctx, "alice: @tester hi", None, None).await.unwrap();

        let mut events = Vec::new();
        let mut compress_id = None;
        while let Ok(ev) = stream_rx.try_recv() {
            events.push(match ev {
                AgentStream::Working {
                    sender_name,
                    message_id,
                    ..
                } => {
                    if sender_name.starts_with("HistoryCompressor") {
                        compress_id = Some(message_id);
                    }
                    format!("working:{sender_name}")
                }
                AgentStream::Thought { .. } => "thought".into(),
                AgentStream::Delta { sender_name, .. } => format!("delta:{sender_name}"),
                AgentStream::Retract { message_id, .. } => {
                    assert_eq!(Some(&message_id), compress_id.as_ref(), "only the compressor line is retracted");
                    "retract".into()
                }
            });
        }
        assert_eq!(
            events,
            // History compression runs first under its OWN line (the
            // compressor model is working, not the agent), which is
            // retracted when done; then the agent's ask raises the
            // agent's spinner before any output.
            vec![
                "working:HistoryCompressor (SmolVLM)",
                "retract",
                "working:tester",
                "thought",
                "delta:tester",
            ],
            "compressor and agent activity attributed separately: {events:?}"
        );
    }

    /// Backend that fails like a flaky thinking-model endpoint (empty
    /// replies) a fixed number of times before succeeding.
    struct FlakyBackend {
        failures_left: std::sync::Arc<std::sync::Mutex<u32>>,
    }
    impl crate::interface::Backend for FlakyBackend {
        fn chat(
            &self,
            _messages: &[ChatMessage],
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = llung_core::network::message::AgentResult<String>> + Send + '_>,
        > {
            Box::pin(async move {
                let mut left = self.failures_left.lock().unwrap();
                if *left > 0 {
                    *left -= 1;
                    Err("model backend returned no content".into())
                } else {
                    Ok("recovered".to_string())
                }
            })
        }
    }

    #[tokio::test]
    async fn empty_backend_replies_are_retried_before_the_turn_fails() {
        let backend = FlakyBackend {
            failures_left: std::sync::Arc::new(std::sync::Mutex::new(2)),
        };
        let (cmd_tx, mut cmd_rx) = mpsc::channel(8);
        let (stream_tx, _stream_rx) = mpsc::channel(16);
        let (perm_tx, _perm_rx) = mpsc::channel(8);
        let being = libp2p::identity::Keypair::generate_ed25519()
            .public()
            .to_peer_id()
            .to_base58();
        let mut ctx = AgentLoopContext::new(
            AgentConfig::default(),
            Box::new(backend),
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

        generate_reply(&mut ctx, "alice: @tester hi", None, None).await.unwrap();

        // Two silent failures, then the third ask succeeds — the turn
        // publishes the recovered answer, not an error panel.
        let cmd = cmd_rx.try_recv().expect("a published reply");
        let NetworkCommand::PublishMessage { contents, .. } = cmd else {
            panic!("expected a publish command");
        };
        let msg: ChatMessage = serde_json::from_slice(&contents).unwrap();
        assert_eq!(msg.content, "recovered");
    }

    #[tokio::test]
    async fn tool_result_is_fed_back_so_the_model_can_continue() {
        let backend = ScriptBackend::new(vec![
            // 1. inspect the workspace...
            r#"TOOL {"name":"glob","arguments":{"pattern":"Cargo.toml"}}"#.to_string(),
            // 2. ...then answer with what it found.
            "the manifest is right there".to_string(),
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

        generate_reply(&mut ctx, "alice: @tester where is the manifest?", None, None)
            .await
            .unwrap();

        let prompts = backend.prompts();
        assert_eq!(prompts.len(), 2, "tool call + continuation");
        // The second prompt feeds the glob RESULT back (not an error).
        assert!(
            prompts[1]
                .iter()
                .any(|m| m.content.contains("Tool 'glob'") && m.content.contains("Cargo.toml")),
            "result feedback must name the tool and show output: {:?}",
            prompts[1].iter().map(|m| &m.content).collect::<Vec<_>>()
        );

        // Three publishes: ToolCall, ToolResponse, final answer.
        let mut kinds = Vec::new();
        while let Ok(cmd) = cmd_rx.try_recv() {
            let NetworkCommand::PublishMessage { contents, .. } = cmd else {
                continue;
            };
            let msg: ChatMessage = serde_json::from_slice(&contents).unwrap();
            kinds.push(msg.kind);
        }
        assert_eq!(kinds.len(), 3, "call, response, answer: {kinds:?}");
        assert!(matches!(kinds[0], MessageKind::ToolCall { .. }));
        assert!(matches!(kinds[1], MessageKind::ToolResponse { .. }));
        assert!(matches!(kinds[2], MessageKind::Agent));
    }

    fn scripted_ctx(replies: Vec<String>) -> (AgentLoopContext, ScriptBackend, mpsc::Receiver<NetworkCommand>) {
        let backend = ScriptBackend::new(replies);
        let (cmd_tx, cmd_rx) = mpsc::channel(16);
        let (stream_tx, _stream_rx) = mpsc::channel(64);
        let (perm_tx, _perm_rx) = mpsc::channel(8);
        let being = libp2p::identity::Keypair::generate_ed25519()
            .public()
            .to_peer_id()
            .to_base58();
        let ctx = AgentLoopContext::new(
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
        (ctx, backend, cmd_rx)
    }

    fn published(cmd_rx: &mut mpsc::Receiver<NetworkCommand>) -> Vec<ChatMessage> {
        let mut out = Vec::new();
        while let Ok(cmd) = cmd_rx.try_recv() {
            if let NetworkCommand::PublishMessage { contents, .. } = cmd {
                out.push(serde_json::from_slice(&contents).unwrap());
            }
        }
        out
    }

    /// Unparseable tool markup is fed back with the exact format (never
    /// published), and the corrected call runs.
    #[tokio::test]
    async fn malformed_tool_markup_is_fed_back_not_published() {
        let (mut ctx, backend, mut cmd_rx) = scripted_ctx(vec![
            // A call in no syntax we parse (unknown wrapper, no name).
            "<tool_call>\n<call>glob **/*.toml</call>\n</tool_call>".to_string(),
            r#"TOOL {"name":"glob","arguments":{"pattern":"Cargo.toml"}}"#.to_string(),
            "found it".to_string(),
        ]);
        generate_reply(&mut ctx, "alice: @tester find the manifest", None, None)
            .await
            .unwrap();
        let prompts = backend.prompts();
        assert_eq!(prompts.len(), 3);
        let fb = &prompts[1].last().unwrap().content;
        assert!(fb.contains("could not be parsed") && fb.contains("TOOL {"), "{fb}");
        let msgs = published(&mut cmd_rx);
        assert!(msgs.iter().all(|m| !m.content.contains("<tool_call>")), "markup published");
        assert!(matches!(msgs[0].kind, MessageKind::ToolCall { .. }));
        assert_eq!(msgs.last().unwrap().content, "found it");
    }

    /// A model that never gets the format right: a prose notice, not
    /// markup, lands in the chat.
    #[tokio::test]
    async fn persistently_malformed_markup_becomes_a_prose_notice() {
        let bad = "<tool_call><call>?</call></tool_call>".to_string();
        let (mut ctx, backend, mut cmd_rx) = scripted_ctx(vec![bad; 1 + MAX_TOOL_ROUNDS as usize]);
        generate_reply(&mut ctx, "alice: @tester do it", None, None).await.unwrap();
        assert_eq!(backend.prompts().len(), 1 + MAX_TOOL_ROUNDS as usize);
        let msgs = published(&mut cmd_rx);
        assert_eq!(msgs.len(), 1, "{msgs:?}");
        assert!(msgs[0].content.contains("couldn't get the call format right"));
    }

    /// The taught format follows the protocol: the native XML for
    /// Qwen-family templates, and the feedback repeats it.
    #[tokio::test]
    async fn native_protocol_is_taught_and_parsed() {
        let (mut ctx, backend, mut cmd_rx) = scripted_ctx(vec![
            "<tool_call>\n<function=glob>\n<parameter=pattern>\nCargo.toml\n</parameter>\n</function>\n</tool_call>".to_string(),
            "done".to_string(),
        ]);
        ctx.tool_protocol = Some(crate::tools::ToolProtocol::QwenXml);
        generate_reply(&mut ctx, "alice: @tester find it", None, None).await.unwrap();
        let first = &backend.prompts()[0];
        let taught = &first[2].content;
        assert!(taught.contains("<function=TOOL_NAME>") && !taught.contains("TOOL {"), "{taught}");
        let msgs = published(&mut cmd_rx);
        assert!(matches!(&msgs[0].kind, MessageKind::ToolCall { tool_name, .. } if tool_name == "glob"));
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

        generate_reply(&mut ctx, "alice: @tester what did I say?", Some(&m2.id), None)
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
            attachment: None,
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
        // Media DOES respond: the @mention gate above keeps it from
        // chattering, and an addressed image is the multimodal-input
        // case (attachment rides the trigger into the backend).
        assert!(kind_responds(&MessageKind::Media(llung_core::network::message::Attachment {
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

    /// Multimodal input end-to-end at the prompt level: a Media-kind
    /// trigger (@mention + image) passes its attachment into the final
    /// prompt message, where the backend converts it to an image part.
    #[tokio::test]
    async fn media_trigger_rides_its_attachment_into_the_prompt() {
        let backend = ScriptBackend::new(vec!["a plain answer".into()]);
        let (cmd_tx, _cmd_rx) = mpsc::channel(8);
        let (stream_tx, _stream_rx) = mpsc::channel(8);
        let (perm_tx, _perm_rx) = mpsc::channel(8);
        let being = libp2p::PeerId::random().to_string();
        let mut ctx = AgentLoopContext::new(
            AgentConfig::default(),
            Box::new(backend.clone()),
            crate::history::HistoryCompressor::new(
                Box::new(crate::history::HeadlineSummarizer::new()),
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

        let att = llung_core::network::message::Attachment {
            cid: "cid-plot".into(),
            filename: "loss.png".into(),
            mime_type: "image/png".into(),
            size_bytes: 100,
            kind: llung_core::network::message::AttachmentKind::Image,
            width: Some(640),
            height: Some(480),
        };
        // The user's image post: Media kind + top-level field (exactly
        // what tui::send_attachment gossips).
        let trigger = ChatMessage::user("@tester what trend is this?")
            .with_attachment(att.clone());
        let mut media_trigger = trigger.clone();
        media_trigger.kind = MessageKind::Media(att.clone());

        generate_reply(&mut ctx, "alice: @tester what trend is this?", None, Some(&media_trigger))
            .await
            .unwrap();

        let prompts = backend.prompts();
        assert_eq!(prompts.len(), 1, "no tool call -> a single ask");
        let trigger_msg = prompts[0].last().expect("a trigger message");
        assert_eq!(
            trigger_msg.image_attachment().map(|a| a.cid.as_str()),
            Some("cid-plot"),
            "the attachment rides the prompt's trigger message"
        );
    }

    /// A 1x1 red PNG — enough for Attachment::from_path to parse
    /// width/height and mime-detect as image/png.
    fn tiny_png_bytes() -> &'static [u8] {
        &[
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48,
            0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00,
            0x00, 0x90, 0x77, 0x53, 0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, 0x54, 0x78,
            0x9C, 0x63, 0xF8, 0xCF, 0xC0, 0x00, 0x00, 0x03, 0x01, 0x01, 0x00, 0xC9, 0xFE, 0x92,
            0xEF, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
        ]
    }

    fn ctx_with_rx() -> (AgentLoopContext, mpsc::Receiver<NetworkCommand>) {
        let (cmd_tx, cmd_rx) = mpsc::channel(8);
        let (stream_tx, _stream_rx) = mpsc::channel(8);
        let (perm_tx, _perm_rx) = mpsc::channel(8);
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
            // A REAL peer id: publish() parses self_being_id, and
            // show_image's whole point is publishing.
            libp2p::PeerId::random().to_string(),
            "tester".into(),
            stream_tx,
            perm_tx,
        );
        (ctx, cmd_rx)
    }

    /// A relative `..`-only path from `from` to `to` (both absolute),
    /// for escape-rejection tests.
    fn relative_escape(from: &std::path::Path, to: &std::path::Path) -> std::path::PathBuf {
        let mut out = std::path::PathBuf::new();
        for _ in 0..from.components().count().saturating_sub(1) {
            out.push("..");
        }
        for c in to.components().skip(1) {
            out.push(c.as_os_str());
        }
        out
    }

    /// A fixture dir under target/ (gitignored). Test contexts resolve
    /// their workspace to the process cwd (the crate dir during cargo
    /// test), so fixtures live at `target/llung-showimg-*/` and are
    /// addressed by paths RELATIVE to the workspace — no env mutation,
    /// safe under parallel tests.
    fn workspace_fixture(tag: &str) -> std::path::PathBuf {
        let dir = std::env::current_dir()
            .unwrap()
            .join("target")
            .join(format!(
                "llung-showimg-{tag}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// When the operator changes the places, the agent adopts the new
    /// snapshot at its next turn and the prompt leads with a NOTICE
    /// listing what changed; the Files section shows the new paths.
    #[tokio::test]
    async fn places_change_is_announced_in_the_next_prompt() {
        let (mut ctx, _cmd_rx) = ctx_with_rx();
        // Pretend the agent last saw an older snapshot.
        ctx.places = std::sync::Arc::new(crate::places::Places::fixed(
            "/old/workspace",
            "/old/runs",
            vec![],
        ));
        refresh_places(&mut ctx);
        let live = crate::places::current();
        assert_eq!(ctx.places.generation, live.generation, "adopted the live snapshot");
        assert!(
            ctx.pending_place_changes
                .iter()
                .any(|c| c.contains("/old/workspace") && c.contains(&live.workspace().display().to_string())),
            "{:?}",
            ctx.pending_place_changes
        );
        let prompt = turn_prompt(&ctx, "history", "@tester hi", None);
        let last = format!("{:?}", prompt.last().unwrap());
        assert!(last.contains("NOTICE"), "{last}");
        assert!(last.contains("/old/workspace"), "{last}");
        assert!(last.contains(&format!("your workspace is {}", live.workspace().display())), "{last}");

        // Once told (a successful turn clears it), no repeat — and no
        // notice while nothing changes.
        ctx.pending_place_changes.clear();
        refresh_places(&mut ctx);
        assert!(ctx.pending_place_changes.is_empty());
        let prompt = turn_prompt(&ctx, "history", "@tester hi", None);
        assert!(!format!("{:?}", prompt.last().unwrap()).contains("NOTICE"));
    }

    /// The plan's Phase-2 acceptance: an orchestrator-level show_image
    /// publishes a Media message AND returns the confirmation string
    /// (which the aukora receipt commits to).
    #[tokio::test]
    async fn show_image_publishes_media_message_and_confirms() {
        let ws = workspace_fixture("ok");
        let rel = std::path::Path::new("target").join(ws.file_name().unwrap());
        let plots = ws.join("experiments/runs/exp07");
        std::fs::create_dir_all(&plots).unwrap();
        std::fs::write(plots.join("loss.png"), tiny_png_bytes()).unwrap();
        let (mut ctx, mut cmd_rx) = ctx_with_rx();

        let root = ctx.kernel.current_history_root;
        let out = execute_secured_tool(
            &mut ctx,
            "show_image".into(),
            serde_json::json!({
                "path": rel.join("experiments/runs/exp07/loss.png").to_string_lossy(),
                "caption": "fitness over generations"
            })
            .to_string()
            .into_bytes(),
        )
        .await
        .expect("show_image executes");
        assert!(
            out.starts_with("published image loss.png"),
            "confirmation string: {out}"
        );
        assert!(out.contains("cid "), "the receipt commits to the cid: {out}");
        // The kernel gate ran exactly like any other tier.
        assert_ne!(ctx.kernel.current_history_root, root, "receipt committed");

        // Media cache holds the bytes (relay fetch path reads it).
        let cid = out.rsplit("cid ").next().unwrap().to_string();
        assert!(
            ctx.db.get_avatar_cache(&cid).ok().flatten().is_some(),
            "image bytes cached under {cid}"
        );

        // Command sequence: ProvideMediaCid, PublishImage, PublishMessage.
        let cmd = cmd_rx.try_recv().expect("ProvideMediaCid");
        let NetworkCommand::ProvideMediaCid { cid: c } = cmd else {
            panic!("expected ProvideMediaCid, got a different command");
        };
        assert_eq!(c, cid);
        let cmd = cmd_rx.try_recv().expect("PublishImage");
        let NetworkCommand::PublishImage { cid: c, mime_type, data } = cmd else {
            panic!("expected PublishImage, got a different command");
        };
        assert_eq!(c, cid);
        assert_eq!(mime_type, "image/png");
        assert_eq!(data, tiny_png_bytes());
        let cmd = cmd_rx.try_recv().expect("PublishMessage");
        let NetworkCommand::PublishMessage { contents, .. } = cmd else {
            panic!("expected PublishMessage");
        };
        let msg: ChatMessage = serde_json::from_slice(&contents).unwrap();
        let MessageKind::Media(att) = &msg.kind else {
            panic!("a media message, got {:?}", msg.kind);
        };
        assert_eq!(att.filename, "loss.png");
        assert_eq!(att.cid, cid);
        assert_eq!(msg.content, "fitness over generations", "caption travels as content");
        assert_eq!(msg.sender_name, "tester", "published from the agent's own being");
        assert!(cmd_rx.try_recv().is_err(), "exactly three commands");
        std::fs::remove_dir_all(&ws).ok();
    }

    /// Path-escape rejection: absolute paths outside the workspace and
    /// `..` climbs never resolve, even when the file exists.
    #[tokio::test]
    async fn show_image_rejects_paths_outside_the_workspace() {
        let ws = workspace_fixture("escape");
        // A real file genuinely OUTSIDE the workspace (target/ fixture
        // has `..` climbs of its own, so use the OS temp dir).
        let outside = std::env::temp_dir().join(format!("llung-showimg-outside-{}", std::process::id()));
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("secret.png"), tiny_png_bytes()).unwrap();
        let (mut ctx, _cmd_rx) = ctx_with_rx();

        let abs = outside.join("secret.png");
        let err = execute_secured_tool(
            &mut ctx,
            "show_image".into(),
            serde_json::json!({"path": abs.to_string_lossy()}).to_string().into_bytes(),
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string().contains("escapes the workspace"),
            "absolute escape must be refused: {err}"
        );

        // A `..` climb that really lands on the outside file.
        let cwd = std::env::current_dir().unwrap();
        let climb = relative_escape(&cwd, &abs);
        assert_eq!(
            cwd.join(&climb).canonicalize().unwrap(),
            abs.canonicalize().unwrap(),
            "test climb really reaches the file"
        );
        let err = execute_secured_tool(
            &mut ctx,
            "show_image".into(),
            serde_json::json!({"path": climb.to_string_lossy()}).to_string().into_bytes(),
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string().contains("escapes the workspace"),
            "dot-dot escape must be refused: {err}"
        );
        std::fs::remove_dir_all(&ws).ok();
        std::fs::remove_dir_all(&outside).ok();
    }

    /// Non-image files and over-cap images are refused before anything
    /// touches the network.
    #[tokio::test]
    async fn show_image_rejects_non_images_and_over_cap_files() {
        let ws = workspace_fixture("bad");
        let rel = std::path::Path::new("target").join(ws.file_name().unwrap());
        std::fs::write(ws.join("notes.txt"), b"not an image").unwrap();
        std::fs::write(
            ws.join("big.png"),
            vec![0u8; llung_core::media::image::MAX_IMAGE_BYTES + 1],
        )
        .unwrap();
        let (mut ctx, mut cmd_rx) = ctx_with_rx();

        let err = execute_secured_tool(
            &mut ctx,
            "show_image".into(),
            serde_json::json!({"path": rel.join("notes.txt").to_string_lossy()})
                .to_string()
                .into_bytes(),
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string().contains("not a usable image")
                || err.to_string().contains("not an image"),
            "non-image refused: {err}"
        );

        let err = execute_secured_tool(
            &mut ctx,
            "show_image".into(),
            serde_json::json!({"path": rel.join("big.png").to_string_lossy()})
                .to_string()
                .into_bytes(),
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("cap"), "over-cap refused: {err}");
        // Nothing was published for either refusal.
        assert!(cmd_rx.try_recv().is_err(), "no network commands on refusal");
        std::fs::remove_dir_all(&ws).ok();
    }
}

