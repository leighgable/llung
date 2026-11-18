use crate::avatar::{decode_attachment, decode_avatar_hex, fallback_avatar, render_avatar};
use crate::prefix_text::PrefixText;
use crate::registration_panel::RegistrationPanel;
use crate::{chat_panel::ChatPanel, taffy_ui::TaffyUi};
use agent::config::AgentConfig;
use agent::history::{HistoryCompressor, ModelSummarizer};
use agent::interface::OpenAiCompatible;
use agent::{AgentLoopContext, run_agent_loop};
use core::option::Option::None;
use crossterm::{
    ExecutableCommand,
    event::{Event, KeyCode, KeyEventKind, KeyModifiers},
};
use libp2p::{PeerId, gossipsub::IdentTopic};
use llung_core::protocol::invite::InvitePayload;
use llung_core::utils::generate_cid_from_bytes;
use llung_core::{
    app::LlungApp,
    config::CoreConfig,
    identity::{
        being::{Being, BeingKind, PresenceMessage, create_local_being},
        crypto::encrypt_to_public_key,
        machine::MachineIdentity,
    },
    network::{
        NetworkCommand, NetworkEvent,
        message::{Attachment, AttachmentKind, ChatMessage, MessageKind},
    },
    storage::db::Database,
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::Rect,
};
use serde_json;

use crate::settings::Settings;
use crate::theme::watch::ThemeWatcher;
use crate::theme::{Role, Theme};
use crate::tool_panel::ToolBlock;

/// Theme state resolved in `main` before the TUI starts.
pub struct ThemeStartup {
    pub theme: Theme,
    /// The selection that produced `theme` (name, pair or path).
    pub selection: String,
    pub settings: Settings,
    /// Messages to show once the chat is up (e.g. a bad saved theme).
    pub notices: Vec<String>,
}
use std::collections::HashMap;
use std::error::Error;
use std::io;
use std::result::Result;
// use taffy::{Dimension, Display, FlexDirection, NodeId, Size, TaffyTree};
use tokio::{sync::mpsc, time::Duration};

#[derive(Clone, Copy, PartialEq)]
pub enum AppScreen {
    Splash,
    Registration,
    Chat,
    // ProfilePicker,
}

#[derive(Clone, Copy, PartialEq)]
pub enum SidebarMode {
    Hidden,
    Peers,
    Chats,
    Media,
}

#[derive(Clone)]
pub struct PeerInfo {
    pub name: String,
    pub avatar_cid: Option<String>,
    pub avatar: Option<crate::avatar::AvatarThumbnail>,
    /// X25519 key learned from presence; required to encrypt DMs to them.
    pub enc_public_key: Option<[u8; 32]>,
}

pub struct App {
    terminal: Terminal<CrosstermBackend<io::Stdout>>,
    config: CoreConfig,
    machine: MachineIdentity,
    // Created lazily, once an account exists and the network starts.
    cmd_tx: Option<mpsc::Sender<NetworkCommand>>,
    my_name: String,

    taffy_ui: TaffyUi,
    theme: Theme,
    /// Selection behind `theme`, as the user wrote it (`/theme` shows it,
    /// `/theme reload` re-resolves it).
    theme_selection: String,
    theme_watch: ThemeWatcher,
    settings: Settings,
    /// System messages queued until the chat screen exists.
    pending_notices: Vec<String>,

    // Content
    screen: AppScreen,
    chat_panel: ChatPanel,
    registration: RegistrationPanel,
    input_text: String,
    /// The chat input line: cursor-based editing + history recall.
    input: crate::input::InputLine,
    last_size: (u16, u16),

    sidebar_mode: SidebarMode,
    peers: Vec<String>,
    chats: Vec<String>,
    media: Vec<String>,
    current_topic: String,

    peer_id: String,
    known_peers: HashMap<String, PeerInfo>,
    /// Local agents spawned via `/agent spawn`: name -> the command
    /// channel of the agent's own network stack. Everything else about
    /// an agent (being_id, avatar, presence) lives in `known_peers`,
    /// same as any remote peer — this is the only peer-specific extra.
    agent_cmd_txs: HashMap<String, mpsc::Sender<NetworkCommand>>,
    /// `/grant` commands into each local agent's orchestrator loop.
    agent_grant_txs: HashMap<String, mpsc::Sender<agent::AgentCommand>>,
    /// A permission prompt awaiting the operator's y/Y/n (see
    /// `agent::permissions::PermissionRequest`). While set, keys are
    /// intercepted by `handle_chat_key` before any editing.
    pending_perm: Option<agent::permissions::PermissionRequest>,
    /// Prompts waiting behind `pending_perm` (concurrent agents).
    perm_queue: std::collections::VecDeque<agent::permissions::PermissionRequest>,
    /// Our account's being_id — presence from the SAME being on a
    /// different machine is one of our other devices; dial it so the
    /// history-exchange protocol can sync the account across devices.
    self_being_id: String,
    /// Device addresses already dialed (dedup for presence redials).
    dialed_self_addrs: std::collections::HashSet<String>,
    /// Machines already mesh-dialed (general peers, not own devices).
    mesh_dialed: std::collections::HashSet<String>,
    /// Image cids already requested from the relays (dedup set).
    pending_image_fetches: std::collections::HashSet<String>,
    /// Relay-fetch completions: (cid, Some((mime, bytes)) | None).
    image_result_tx: mpsc::Sender<(String, Option<(String, Vec<u8>)>)>,
    /// Receiver half of `image_result_tx`; drained in `run`.
    image_result_rx: Option<mpsc::Receiver<(String, Option<(String, Vec<u8>)>)>>,
    /// Per-topic focus in the chat DAG: the message whose ancestor chain
    /// fills the panel (the "current branch"). Follows live messages
    /// that extend the chain; `/branch` moves it; the entry is created
    /// lazily by the first view render (following the topic's tip).
    branch_cursors: HashMap<String, String>,
    /// Sender half of the local agents' streaming channel; created in
    /// `run`, cloned into every spawned agent.
    stream_tx: Option<mpsc::Sender<agent::AgentStream>>,
    /// Sender half of the app-wide permission channel; cloned into
    /// every spawned agent. Requests arrive on the receiver in `run`.
    perm_tx: Option<mpsc::Sender<agent::permissions::PermissionRequest>>,
    /// The status row under the input (topic, agents, context length),
    /// recomputed when something relevant changes. `draw` has no DB
    /// access, so it is computed here, not in the render path.
    status_line: String,
}

impl App {
    pub fn new(
        terminal: Terminal<CrosstermBackend<io::Stdout>>,
        config: CoreConfig,
        machine: MachineIdentity,
        theme: ThemeStartup,
    ) -> Self {
        let size = terminal
            .size()
            .unwrap_or(ratatui::layout::Size::new(80, 24));
        let sidebar_mode = SidebarMode::Hidden;
        let (image_tx, image_rx) = mpsc::channel::<(String, Option<(String, Vec<u8>)>)>(32);

        Self {
            terminal,
            config,
            machine: machine.clone(),
            cmd_tx: None,
            my_name: String::new(),
            taffy_ui: TaffyUi::new_chat_layout(size.width, size.height, false),
            theme_watch: ThemeWatcher::new(theme.theme.source.as_deref()),
            theme: theme.theme,
            theme_selection: theme.selection,
            settings: theme.settings,
            pending_notices: theme.notices,
            screen: AppScreen::Splash,
            chat_panel: ChatPanel::new(String::new()),
            registration: RegistrationPanel::new(),
            input_text: String::new(),
            input: crate::input::InputLine::new(),
            last_size: (0, 0),
            sidebar_mode: sidebar_mode,
            peers: Vec::new(),
            chats: vec!["introductions".to_string()],
            media: Vec::new(),
            current_topic: "introductions".to_string(),
            peer_id: machine.peer_id.to_base58(),
            known_peers: HashMap::new(),
            agent_cmd_txs: HashMap::new(),
            agent_grant_txs: HashMap::new(),
            pending_perm: None,
            perm_queue: std::collections::VecDeque::new(),
            self_being_id: String::new(),
            dialed_self_addrs: std::collections::HashSet::new(),
            mesh_dialed: std::collections::HashSet::new(),
            pending_image_fetches: std::collections::HashSet::new(),
            image_result_tx: image_tx,
            image_result_rx: Some(image_rx),
            branch_cursors: HashMap::new(),
            stream_tx: None,
            perm_tx: None,
            status_line: String::new(),
        }
    }

    /// Recompute the status row: current topic, local agents, and the
    /// context an agent would read on the focused branch — the
    /// budgeted summary-tree view, not the raw chat bytes. The size is
    /// a deterministic dry-run with the rule-based summarizer (no model
    /// calls); the agent's own summaries differ in wording, not in
    /// shape or budget.
    fn update_status_line(&mut self, db: &Database) {
        const CONTEXT_BUDGET: usize = 64_000;
        let ctx = db.load_dag().ok().and_then(|dag| {
            let topic = self.current_topic.clone();
            let focus = self
                .branch_cursors
                .get(&topic)
                .cloned()
                .or_else(|| dag.tip_of(&topic))?;
            let view = dag.context_chain(&focus);
            let messages: Vec<llung_core::network::message::ChatMessage> =
                view.iter().map(|m| (*m).clone()).collect();
            Some((view.len(), compressed_view_bytes(&messages, CONTEXT_BUDGET)))
        });
        let mut left = format!("#{}", self.current_topic);
        if !self.agent_cmd_txs.is_empty() {
            let mut agents: Vec<&str> = self.agent_cmd_txs.keys().map(String::as_str).collect();
            agents.sort();
            left.push_str(&format!(" · agents: {}", agents.join(", ")));
        }
        left.push_str(&match ctx {
            Some((0, _)) | None => " · ctx empty".to_string(),
            Some((n, Some(bytes))) => {
                format!(" · ctx {n} msgs · view {}/{}", human_len(bytes), human_len(CONTEXT_BUDGET))
            }
            Some((n, None)) => format!(" · ctx {n} msgs"),
        });
        self.status_line = left;
    }

    pub async fn run(&mut self, db: &mut Database) -> Result<(), Box<dyn std::error::Error>> {
        let (term_tx, mut term_rx) = mpsc::channel::<Event>(32);
        crate::events::spawn_terminal_reader(term_tx);

        // The network only exists once an account does.
        let mut event_rx: Option<mpsc::Receiver<NetworkEvent>> = None;
        let mut tick = tokio::time::interval(Duration::from_millis(50));

        // Local agents stream their in-progress replies here; the
        // complete message still arrives over gossipsub afterwards.
        let (stream_tx, mut stream_rx) = mpsc::channel::<agent::AgentStream>(256);
        self.stream_tx = Some(stream_tx);

        // App-wide permission prompt channel: every spawned agent sends
        // `PermissionRequest`s here; the loop below queues them into
        // `pending_perm` (one prompt at a time on screen).
        let (perm_tx, mut perm_rx) = mpsc::channel::<agent::permissions::PermissionRequest>(32);
        self.perm_tx = Some(perm_tx);

        // ── Splash: show a frame, then resolve the account ──
        self.draw()?;
        match db.load_being()? {
            Some(being) => {
                self.setup_identity(&being);
                event_rx = Some(self.start_network(being).await?);
                self.screen = AppScreen::Chat;
                self.render_topic_view(db);
            }
            None => {
                self.screen = AppScreen::Registration;
            }
        }

        loop {
            tokio::select! {
                Some(ev) = term_rx.recv() => {
                    if let Event::Key(key) = ev {
                        if key.kind != KeyEventKind::Press {
                            continue;
                        }
                        match self.screen {
                            AppScreen::Splash => {}
                            AppScreen::Registration => {
                                if let Some(result) = self.registration.handle_key(key.code) {
                                    let avatar = if result.avatar.is_empty() {
                                        None
                                    } else {
                                        Some(result.avatar.as_str())
                                    };
                                    match create_local_being(db, result.name, avatar, BeingKind::Human) {
                                        Ok(being) => {
                                            self.setup_identity(&being);
                                            match self.start_network(being).await {
                                                Ok(rx) => {
                                                    event_rx = Some(rx);
                                                    self.screen = AppScreen::Chat;
                                                    self.render_topic_view(db);
                                                }
                                                Err(e) => {
                                                    self.registration.error_message =
                                                        Some(format!("Network failed to start: {e}"));
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            self.registration.error_message =
                                                Some(format!("Registration failed: {e}"));
                                        }
                                    }
                                }
                            }
                            AppScreen::Chat => {
                                self.handle_chat_key(key, db).await?;
                            }
                        }
                    }
                }
                // Disabled until the network registration complete
                Some(net_ev) = async {
                    match event_rx.as_mut() {
                        Some(rx) => rx.recv().await,
                        None => std::future::pending().await,
                    }
                } => {
                    self.handle_network_event(net_ev, db);
                }
                Some(stream_ev) = stream_rx.recv() => {
                    self.handle_agent_stream(stream_ev);
                }
                Some(req) = perm_rx.recv() => {
                    self.queue_permission(req);
                }
                Some((cid, result)) = async {
                    match self.image_result_rx.as_mut() {
                        Some(rx) => rx.recv().await,
                        None => std::future::pending().await,
                    }
                } => {
                    self.handle_image_result(db, cid, result);
                }
                _ = tick.tick() => {
                    // Animate braille spinners on running tool panels.
                    // The frame counter always advances (it's free); the
                    // layout cache is only dropped while something is
                    // actually spinning, so idle ticks cost nothing.
                    crate::spinner::tick();
                    if self.chat_panel.has_running_tools() {
                        self.chat_panel.invalidate();
                    }
                }
            };
            self.poll_theme_file();
            if self.screen == AppScreen::Chat && !self.pending_notices.is_empty() {
                for notice in std::mem::take(&mut self.pending_notices) {
                    self.system(notice);
                }
            }
            self.load_missing_avatars(db);
            self.draw()?;
        }
    }

    /// Adopt an account into the UI (chat panel title, own peer entry).
    fn setup_identity(&mut self, being: &Being) {
        self.my_name = being.human_name.clone();
        self.self_being_id = being.being_id.clone();
        self.chat_panel = ChatPanel::new(being.human_name.clone());
        self.known_peers.insert(
            self.peer_id.clone(),
            PeerInfo {
                name: being.human_name.clone(),
                avatar_cid: being.avatar_cid.clone(),
                avatar: None, // lazy loaded
                enc_public_key: None,
            },
        );
    }

    /// Resolve a peer query (exact name, case-insensitive, or a peer-id
    /// prefix of at least 4 characters) to a PeerId. Self is excluded.
    fn resolve_peer(&self, query: &str) -> std::result::Result<PeerId, String> {
        let q = query.trim();
        if q.is_empty() {
            return Err("empty peer name".into());
        }

        let mut matches: Vec<&String> = self
            .known_peers
            .iter()
            .filter(|(id, info)| *id != &self.peer_id && info.name.eq_ignore_ascii_case(q))
            .map(|(id, _)| id)
            .collect();

        if matches.is_empty() && q.len() >= 4 {
            matches = self
                .known_peers
                .iter()
                .filter(|(id, _)| *id != &self.peer_id && id.starts_with(q))
                .map(|(id, _)| id)
                .collect();
        }

        match matches.len() {
            0 => Err(format!("unknown peer '{q}' — see /peers for who is online")),
            1 => matches[0]
                .parse::<PeerId>()
                .map_err(|e| format!("'{q}' is not a valid peer id: {e}")),
            _ => Err(format!(
                "'{q}' is ambiguous — it matches: {}",
                matches
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
        }
    }

    /// Start the network core now that a Being exists.
    async fn start_network(
        &mut self,
        being: Being,
    ) -> Result<mpsc::Receiver<NetworkEvent>, Box<dyn std::error::Error>> {
        let (llung_app, event_rx) =
            LlungApp::init(self.config.clone(), self.machine.clone(), being).await?;
        self.cmd_tx = Some(llung_app.command_tx());
        if !llung_app.dns_available {
            self.pending_notices.push(
                "No DNS (offline?): local peers only. Restart llung after reconnecting \
                 to reach peers by DNS name."
                    .into(),
            );
        }
        Ok(event_rx)
    }

    /// Join a topic: track it in the chats sidebar, switch to it, and
    /// subscribe the network. Idempotent — safe to call on every invite.
    fn join_topic(&mut self, topic_id: &str) {
        if !self.chats.iter().any(|t| t == topic_id) {
            self.chats.push(topic_id.to_string());
        }
        self.current_topic = topic_id.to_string();
        if let Some(tx) = &self.cmd_tx {
            let _ = tx.try_send(NetworkCommand::SubscribeTopic {
                topic: IdentTopic::new(topic_id),
            });
        }
    }

    /// Rebuild the chat panel from the DAG: the ancestor chain of the
    /// focused message IS the current branch. Called at startup, on
    /// topic switches, and after every `/branch` move. When the focus
    /// is not the topic's tip, an indicator line marks the fork point.
    fn render_topic_view(&mut self, db: &Database) {
        let Ok(dag) = db.load_dag() else {
            return;
        };
        let topic = self.current_topic.clone();
        let Some(focus) = self
            .branch_cursors
            .get(&topic)
            .cloned()
            .or_else(|| dag.tip_of(&topic))
        else {
            self.chat_panel.clear();
            return;
        };
        // Resolve (and remember) the effective focus.
        self.branch_cursors.insert(topic.clone(), focus.clone());
        // The view is the same context chain agents read: the branch,
        // with parallel agent/tool chatter folded in; explicit forks
        // (human children off the chain) stay out.
        let view = dag.context_chain(&focus);
        let off_tip = dag.tip_of(&topic).as_deref() != Some(focus.as_str());
        let options = dag.branch_options(&focus);
        let focus_pos = view.iter().position(|m| m.id == focus);

        self.chat_panel.clear();
        for (i, msg) in view.iter().enumerate() {
            if off_tip && focus_pos == Some(i) {
                let sib = if options.len() > 1 {
                    format!(" · {} sibling branches", options.len() - 1)
                } else {
                    String::new()
                };
                self.chat_panel.push(
                    "system".into(),
                    "system".into(),
                    format!(
                        "⎇ on a branch{sib} — Enter forks here · \
                         /branch up/down/next/prev · /branch tip for the newest"
                    ),
                    None,
                    None,
                );
            }
            self.render_message_line(msg, db);
        }
        self.chat_panel.jump_to_bottom();
        self.update_status_line(db);
    }

    /// `/branch` — navigate and fork the chat tree. The tree itself is
    /// the per-topic subgraph of the message DAG (already persisted);
    /// this command only moves the view focus:
    ///
    ///   /branch            where am I?
    ///   /branch <n>        focus the message n-back on this branch;
    ///                      Enter then forks at it
    ///   /branch up         focus the parent message
    ///   /branch down       focus the greatest child (toward a tip)
    ///   /branch next|prev  switch to a sibling branch at the same fork
    ///   /branch tip        release the focus; follow the newest branch
    fn handle_branch_command(&mut self, args: &str, db: &Database) {
        let Ok(dag) = db.load_dag() else {
            self.chat_panel.push(
                "system".into(), "system".into(),
                "branch: failed to load chat history".into(),
                None, None,
            );
            return;
        };
        let topic = self.current_topic.clone();
        let focus = self
            .branch_cursors
            .get(&topic)
            .cloned()
            .or_else(|| dag.tip_of(&topic));

        match args.trim() {
            "" => {
                let Some(f) = focus else {
                    self.chat_panel.push(
                        "system".into(), "system".into(),
                        format!("branch: no history in #{topic}"),
                        None, None,
                    );
                    return;
                };
                let view = dag.context_chain(&f);
                let n = view.len();
                let options = dag.branch_options(&f);
                let sib = if options.len() > 1 {
                    format!(" · {} branches at this fork", options.len())
                } else {
                    String::new()
                };
                let newest = if dag.tip_of(&topic).as_deref() == Some(f.as_str()) {
                    ""
                } else {
                    " · NOT the newest branch"
                };
                let preview: String = dag.nodes.get(&f).map(|node| {
                    let c: String = node
                        .message
                        .content
                        .chars()
                        .filter(|c| *c != '\n')
                        .take(40)
                        .collect();
                    format!(" — at: \"{c}\"")
                })
                .unwrap_or_default();
                self.chat_panel.push(
                    "system".into(), "system".into(),
                    format!("⎇ {n} message{s} up from the root{sib}{newest}{preview}",
                            s = if n == 1 { "" } else { "s" }),
                    None, None,
                );
            }
            "tip" => {
                self.branch_cursors.remove(&topic);
                self.render_topic_view(db);
            }
            "up" => {
                let parent = focus
                    .as_ref()
                    .and_then(|f| dag.nodes.get(f))
                    .and_then(|n| n.message.parent_id.clone());
                self.branch_move(db, parent, "already at the root message");
            }
            "down" => {
                // The most recent child (timestamp, then hash as the
                // deterministic tie-break): drifts back toward the live
                // end of the conversation after an `up`.
                let child = focus.as_ref().and_then(|f| {
                    dag.nodes.get(f).map(|n| {
                        n.children
                            .iter()
                            .filter(|c| dag.nodes[*c].message.topic == topic)
                            .max_by_key(|c| {
                                (
                                    dag.nodes[*c].message.timestamp,
                                    dag.nodes[*c].hash,
                                    (*c).clone(),
                                )
                            })
                            .cloned()
                    })
                });
                self.branch_move(db, child.flatten(), "already at the end of this branch");
            }
            "next" | "prev" => {
                let Some(f) = focus else {
                    self.chat_panel.push(
                        "system".into(), "system".into(),
                        "branch: no history in this topic".into(),
                        None, None,
                    );
                    return;
                };
                let options = dag.branch_options(&f);
                let Some(idx) = options.iter().position(|o| o == &f) else {
                    return;
                };
                let delta = if args.trim() == "next" { 1isize } else { -1 };
                let new_idx = idx as isize + delta;
                if new_idx < 0 || new_idx as usize >= options.len() {
                    self.chat_panel.push(
                        "system".into(), "system".into(),
                        format!(
                            "branch: no {} sibling ({} branches at this fork)",
                            args.trim(),
                            options.len()
                        ),
                        None, None,
                    );
                    return;
                }
                let target = options[new_idx as usize].clone();
                self.branch_move(db, Some(target), "unreachable");
            }
            other => match other.parse::<usize>() {
                Ok(n) => {
                    let Some(f) = focus else {
                        self.chat_panel.push(
                            "system".into(), "system".into(),
                            "branch: no history in this topic".into(),
                            None, None,
                        );
                        return;
                    };
                    // Count back over the rendered view (what you see),
                    // not the bare ancestor chain.
                    let view = dag.context_chain(&f);
                    if n >= view.len() {
                        self.chat_panel.push(
                            "system".into(), "system".into(),
                            format!("branch: only {} message{s} in this view",
                                    view.len(),
                                    s = if view.len() == 1 { "" } else { "s" }),
                            None, None,
                        );
                        return;
                    }
                    let target = view[view.len() - 1 - n].id.clone();
                    self.branch_move(db, Some(target), "unreachable");
                }
                Err(_) => {
                    self.chat_panel.push(
                        "system".into(), "system".into(),
                        "Usage: /branch [n|up|down|next|prev|tip]".into(),
                        None, None,
                    );
                }
            },
        }
    }

    /// Move the branch focus and re-render; `None` means the move was
    /// invalid (`err` tells the user why).
    fn branch_move(&mut self, db: &Database, new_focus: Option<String>, err: &str) {
        if let Some(f) = new_focus {
            self.branch_cursors
                .insert(self.current_topic.clone(), f);
            self.render_topic_view(db);
        } else {
            self.chat_panel.push(
                "system".into(), "system".into(),
                format!("branch: {err}"),
                None, None,
            );
        }
    }

    fn load_missing_avatars(&mut self, db: &Database) {
        let mut loaded: Vec<(String, crate::avatar::AvatarThumbnail)> = Vec::new();
        let mut missing: Vec<String> = Vec::new();
        for (id, info) in self.known_peers.iter_mut() {
            if info.avatar.is_some() || info.avatar_cid.is_none() {
                continue;
            }
            let cid = info.avatar_cid.as_ref().unwrap();
            if let Ok(Some(bytes)) = db.get_avatar_cache(cid) {
                if let Some(thumb) = decode_avatar_hex(&bytes, 4) {
                    info.avatar = Some(thumb.clone());
                    loaded.push((id.clone(), thumb));
                }
            } else {
                // Not local: ask the relay cache (after the borrow ends).
                missing.push(cid.clone());
            }
        }
        for cid in missing {
            self.fetch_image_from_relays(&cid);
        }
        // Avatars often arrive after the messages did; retro-attach them.
        for (id, thumb) in loaded {
            self.chat_panel.update_peer_avatar(&id, &thumb);
        }
    }

    /// Fire a `GetImage` at the relays unless one is already inflight
    /// for this cid. Results are handled where `image_result_rx` is
    /// drained (see `run`).
    fn fetch_image_from_relays(&mut self, cid: &str) {
        if !self.pending_image_fetches.insert(cid.to_string()) {
            return;
        }
        let Some(tx) = &self.cmd_tx else {
            return;
        };
        let (respond, rx) = tokio::sync::oneshot::channel();
        let cid_owned = cid.to_string();
        let result_tx = self.image_result_tx.clone();
        // Send the command, then forward the result (or timeout) onto the
        // app-wide result channel so the main loop can cache + redraw.
        let tx = tx.clone();
        tokio::spawn(async move {
            let _ = tx
                .send(llung_core::network::command::NetworkCommand::GetImage {
                    cid: cid_owned.clone(),
                    responder: respond,
                })
                .await;
            let result = tokio::time::timeout(std::time::Duration::from_secs(35), rx)
                .await
                .ok()
                .and_then(|r| r.ok())
                .flatten();
            let _ = result_tx.send((cid_owned, result)).await;
        });
    }

    /// Cache a relay-fetched image locally; `load_missing_avatars`
    /// re-attaches thumbnails to the right peers by cid, and the panel
    /// redraw picks up message attachments from the same cache.
    fn on_image_fetched(&mut self, db: &Database, cid: &str, data: &[u8]) {
        if db.save_avatar_cache(cid, data).is_err() {
            return;
        }
        self.load_missing_avatars(db);
        self.chat_panel.invalidate();
    }

    async fn handle_chat_key(
        &mut self,
        key: crossterm::event::KeyEvent,
        db: &Database,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        // A pending permission prompt owns the keyboard until it is
        // answered: no typing into the chat while an agent waits.
        if self.pending_perm.is_some() {
            use agent::permissions::PermissionAnswer;
            match key.code {
                KeyCode::Char('y') => self.answer_permission(PermissionAnswer::Once),
                KeyCode::Char('Y') => {
                    let allow = self
                        .pending_perm
                        .as_ref()
                        .map(|r| r.allow_standing)
                        .unwrap_or(false);
                    if allow {
                        self.answer_permission(PermissionAnswer::Standing);
                    } else {
                        self.system(
                            "gpu:* never gets an implicit 'always' — use `/grant <effect> perpetual`",
                        );
                    }
                }
                KeyCode::Char('n') | KeyCode::Esc => {
                    self.answer_permission(PermissionAnswer::Deny)
                }
                _ => {}
            }
            return Ok(());
        }
        match key.code {
            // Typing inserts at the cursor.
            KeyCode::Char(c) => {
                self.input.insert(c);
            }

            // Backspace deletes before the cursor; Delete deletes under it.
            KeyCode::Backspace => {
                self.input.backspace();
            }
            KeyCode::Delete => {
                self.input.delete();
            }

            // Cursor movement.
            KeyCode::Left => self.input.move_left(),
            KeyCode::Right => self.input.move_right(),
            KeyCode::Home => self.input.home(),
            KeyCode::End if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.chat_panel.jump_to_bottom();
            }
            KeyCode::End => self.input.end(),

            // Up/Down recall input history; at the ends of the history
            // they fall back to scrolling the chat, as before.
            KeyCode::Up => {
                if !self.input.recall_older() {
                    self.chat_panel.scroll_up(1);
                }
            }
            KeyCode::Down => {
                if !self.input.recall_newer() {
                    self.chat_panel.scroll_down(1);
                }
            }

            // Send message
            KeyCode::Enter => {
                if key.modifiers.contains(KeyModifiers::SHIFT) {
                    self.input.insert('\n');
                } else {
                    let is_command = self.input.text.trim().starts_with('/');
                    if is_command {
                        let cmd = std::mem::take(&mut self.input.text);
                        self.input.submit();
                        self.handle_command(cmd.trim(), db).await?;
                    } else if !self.input.text.trim().is_empty() {
                        let raw = std::mem::take(&mut self.input.text);
                        self.input.submit();
                        let text = expand_file_mentions(raw.trim()).trim().to_string();
                        let timestamp = chrono::Utc::now().timestamp();
                        // The new message extends the focused node: the
                        // topic tip normally, or the `/branch` focus —
                        // which is how a message forks the chat.
                        let parent_id = self
                            .branch_cursors
                            .get(&self.current_topic)
                            .cloned()
                            .or_else(|| {
                                db.load_dag()
                                    .ok()
                                    .and_then(|dag| dag.tip_of(&self.current_topic))
                            });
                        let msg = ChatMessage {
                            id: generate_cid_from_bytes(
                                &(self.peer_id.clone() + &text + &timestamp.to_string()).as_bytes(),
                            )
                            .unwrap_or_else(|_| uuid::Uuid::new_v4().to_string()),
                            topic: self.current_topic.clone(),
                            parent_id,
                            sender_id: self.peer_id.clone(),
                            sender_name: self.my_name.clone(),
                            content: text.clone(),
                            timestamp: timestamp as u64,
                            kind: MessageKind::Human,
                        };

                        if let Err(e) = db.save_message(&msg) {
                            self.report_error("Failed to save message", e);
                        }
                        let my_avatar = self
                            .known_peers
                            .get(&self.peer_id)
                            .and_then(|i| i.avatar.clone());
                        let my_id = self.peer_id.clone();
                        self.chat_panel.push(
                            my_id,
                            self.my_name.clone(),
                            text.clone(),
                            my_avatar,
                            None,
                        );

                        let cmd = NetworkCommand::PublishMessage {
                            topic: IdentTopic::new(&self.current_topic),
                            contents: serde_json::to_vec(&msg)?,
                        };
                        if let Some(tx) = &self.cmd_tx {
                            tx.send(cmd).await?;
                        }
                        // The view follows the message just sent (the
                        // new tip of this branch).
                        self.branch_cursors
                            .insert(self.current_topic.clone(), msg.id.clone());
                    }
                }
            }
            // Chat scroll (Up/Down recall input history first — see above).
            KeyCode::PageUp => self.chat_panel.scroll_up(10),
            KeyCode::PageDown => self.chat_panel.scroll_down(10),

            // Quit
            KeyCode::Esc => std::process::exit(0),

            _ => {}
        }

        Ok(())
    }

    fn handle_network_event(&mut self, ev: NetworkEvent, db: &Database) {
        match ev {
            NetworkEvent::MessageReceived {
                sender,
                data,
                topic,
            } => {
                if let Ok(msg) = serde_json::from_slice::<ChatMessage>(&data) {
                    self.handle_chat_message(msg, db);
                    return;
                }

                if let Ok(presence) = serde_json::from_slice::<PresenceMessage>(&data) {
                    self.handle_presence(sender, presence);
                    return;
                }

                let text = String::from_utf8_lossy(&data).into_owned();
                self.handle_raw_text(sender, topic, text, db);
            }

            NetworkEvent::DirectMessageReceived { sender, payload } => {
                self.handle_direct_message(sender, payload, db);
            }

            NetworkEvent::HistoryReceived { from, messages } => {
                // Idempotent merge (INSERT OR IGNORE per message):
                // devices of one account converge without conflicts.
                let mut new_count = 0usize;
                for msg in &messages {
                    let had_it = db
                        .load_dag()
                        .ok()
                        .is_some_and(|dag| dag.nodes.contains_key(&msg.id));
                    if db.save_message(msg).is_ok() && !had_it {
                        new_count += 1;
                    }
                }
                tracing::info!(%from, total = messages.len(), new_count, "history synced");
                if new_count > 0 {
                    self.chat_panel.invalidate();
                    self.update_status_line(db);
                }
            }

            NetworkEvent::ImagePushed {
                from,
                cid,
                data,
                ..
            } => {
                // A relay (or peer) cached one of our images back to us
                // — usually our own avatar/attachment arriving on a
                // second device of this account.
                tracing::debug!(%from, %cid, bytes = data.len(), "image pushed; caching");
                self.on_image_fetched(db, &cid, &data);
            }
            _ => {}
        }
    }

    /// Queue a permission request: shown one at a time; concurrent
    /// asks from other agents wait their turn.
    fn queue_permission(&mut self, req: agent::permissions::PermissionRequest) {
        self.perm_queue.push_back(req);
        self.promote_permission();
    }

    /// Move the next queued request onto the screen (if idle).
    fn promote_permission(&mut self) {
        if self.pending_perm.is_some() {
            return;
        }
        if let Some(req) = self.perm_queue.pop_front() {
            let standing = if req.allow_standing { " · Y always" } else { "" };
            self.system(format!(
                "🔐 {} asks for `{}`{}: {} — [y once{} · n deny]",
                req.agent_name, req.effect, tier_label(req.tier), req.summary, standing,
            ));
            self.update_status_line_pending();
            self.pending_perm = Some(req);
        }
    }

    fn update_status_line_pending(&mut self) {
        if let Some(req) = &self.pending_perm {
            let always = if req.allow_standing { " · Y=always" } else { "" };
            self.status_line = format!(
                "⏳ permission: {} → `{}` — y=once{} · n=deny",
                req.agent_name, req.effect, always
            );
        }
    }

    /// Answer the on-screen permission prompt (y/Y/n).
    fn answer_permission(&mut self, answer: agent::permissions::PermissionAnswer) {
        if let Some(req) = self.pending_perm.take() {
            let verb = match answer {
                agent::permissions::PermissionAnswer::Once => "allowed (once)",
                agent::permissions::PermissionAnswer::Standing => "allowed (always)",
                agent::permissions::PermissionAnswer::Deny => "denied",
            };
            self.system(format!("🔐 `{}` for {} {}", req.effect, req.agent_name, verb));
            let _ = req.respond.send(answer);
        }
        self.promote_permission();
        // Restore the normal status row on the next relevant update;
        // cheap to redo now only when idle.
        if self.pending_perm.is_none() {
            self.status_line.clear();
        }
    }

    /// Show a system message in the chat panel.
    fn system(&mut self, text: impl Into<String>) {
        self.chat_panel
            .push("system".into(), "system".into(), text.into(), None, None);
    }

    /// Make `theme` active and watch its file (if any) for edits.
    fn apply_theme(&mut self, theme: Theme, selection: String) {
        self.theme_watch = ThemeWatcher::new(theme.source.as_deref());
        self.theme = theme;
        self.theme_selection = selection;
        // Kitty plot images bake in theme colors: lay out again.
        self.chat_panel.invalidate();
    }

    /// Hot reload: re-read the active theme file after it changes. An
    /// invalid edit keeps the current theme and reports the error.
    /// Successful reloads are silent, so saving repeatedly while tweaking
    /// colors doesn't flood the chat.
    fn poll_theme_file(&mut self) {
        if !self.theme_watch.poll() {
            return;
        }
        let Some(path) = self.theme_watch.path().map(|p| p.to_path_buf()) else {
            return;
        };
        match Theme::from_path(&path, self.theme.mode) {
            Ok(theme) => {
                tracing::info!("theme reloaded from {}", path.display());
                self.theme = theme;
                self.chat_panel.invalidate();
            }
            Err(e) => self.report_error("Theme reload failed (keeping current theme)", e),
        }
    }

    /// `/theme [reload | <name> | <light>/<dark> | <path.json>]`
    fn handle_theme_command(&mut self, args: &str) {
        match args {
            "" => {
                let source = match &self.theme.source {
                    Some(p) => p.display().to_string(),
                    None => "built-in".to_string(),
                };
                let available: Vec<String> = Theme::available()
                    .into_iter()
                    .map(|n| if n == self.theme.name { format!("{n}*") } else { n })
                    .collect();
                self.system(format!(
                    "Theme: {} ({source}), selection '{}', {:?} color",
                    self.theme.name, self.theme_selection, self.theme.mode
                ));
                self.system(format!("Available: {}", available.join(", ")));
                self.system(
                    "Usage: /theme <name | light/dark | path.json> (saved) · /theme reload",
                );
            }
            "reload" => match Theme::select(&self.theme_selection, self.theme.mode) {
                Ok(theme) => {
                    let name = theme.name.clone();
                    let selection = self.theme_selection.clone();
                    self.apply_theme(theme, selection);
                    self.system(format!("Theme '{name}' reloaded"));
                }
                Err(e) => self.report_error("Theme reload failed (keeping current theme)", e),
            },
            sel => match Theme::select(sel, self.theme.mode) {
                Ok(theme) => {
                    let name = theme.name.clone();
                    self.apply_theme(theme, sel.to_string());
                    match self.settings.set_theme(sel) {
                        Ok(()) => {
                            let at = self
                                .settings
                                .path()
                                .map(|p| format!(" (saved to {})", p.display()))
                                .unwrap_or_default();
                            self.system(format!("Theme set to '{name}'{at}"));
                        }
                        Err(e) => self.report_error(
                            &format!("Theme set to '{name}' but not saved"),
                            e,
                        ),
                    }
                }
                Err(e) => self.report_error("Theme not changed", e),
            },
        }
    }

    /// Surface an error in the chat panel as a system message.
    fn report_error(&mut self, context: &str, e: impl std::fmt::Display) {
        self.chat_panel.push(
            "system".into(),
            "system".into(),
            format!("{context}: {e}"),
            None,
            None,
        );
    }

    fn handle_chat_message(&mut self, msg: ChatMessage, db: &Database) {
        // Save to local history
        if let Err(e) = db.save_message(&msg) {
            self.report_error("Failed to save message", e);
        }

        // Learn new peers by their stable being_id
        if !msg.sender_name.is_empty() && !self.known_peers.contains_key(&msg.sender_id) {
            self.known_peers.insert(
                msg.sender_id.clone(),
                PeerInfo {
                    name: msg.sender_name.clone(),
                    avatar_cid: None,
                    avatar: None,
                    enc_public_key: None,
                },
            );
        }

        // A message image we don't have yet: ask the relay cache. The
        // render path reads the avatar/media cache; when the bytes land
        // (handle_image_result) the panel invalidates and shows it.
        if let MessageKind::Media(attachment) = &msg.kind {
            if attachment.kind == AttachmentKind::Image
                && db.get_avatar_cache(&attachment.cid).ok().flatten().is_none()
            {
                self.fetch_image_from_relays(&attachment.cid);
            }
        }

        // Only render if it's the current topic
        if msg.topic != self.current_topic {
            return;
        }

        // Branch views: humans fork explicitly via /branch, so their
        // messages render only if they extend the focused node (or are
        // unparented legacy traffic). Conversational traffic — agent
        // replies, tool calls/responses, system, media — renders if it
        // is attached ANYWHERE along the viewed branch (several agents
        // answering the same message stay in view, and in each other's
        // context), but chatter on explicit forks stays out.
        let conversational = matches!(
            msg.kind,
            MessageKind::Agent
                | MessageKind::System
                | MessageKind::ToolCall { .. }
                | MessageKind::ToolResponse { .. }
                | MessageKind::Media(_)
        );
        let extends_focus = match msg.parent_id {
            None => true,
            Some(ref p) if self.branch_cursors.get(&msg.topic).is_some_and(|f| f == p) => true,
            Some(ref p) => {
                conversational
                    && db.load_dag().ok().is_some_and(|dag| {
                        self.branch_cursors.get(&msg.topic).is_some_and(|f| {
                            dag.ancestors_of(f).iter().any(|m| m.id == *p)
                        })
                    })
            }
        };
        if !extends_focus {
            self.chat_panel.push(
                "system".into(),
                "system".into(),
                "new message on another branch — /branch tip for the newest".into(),
                None,
                None,
            );
            return;
        }

        // Drop the streaming preview of this message, if any — the
        // authoritative copy renders next (same id, gossipsub-signed).
        self.chat_panel.remove_stream_line(&msg.id);

        self.render_message_line(&msg, db);
        // The view follows messages that extend the current branch.
        self.branch_cursors.insert(msg.topic.clone(), msg.id.clone());
        self.update_status_line(db);
    }

    /// Local agents stream reply deltas while they generate; the final
    /// gossipsub message (same id) replaces the preview when it lands.
    fn handle_agent_stream(&mut self, ev: agent::AgentStream) {
        match ev {
            agent::AgentStream::Delta {
                topic,
                message_id,
                being_id,
                sender_name,
                delta,
            } => {
                if topic != self.current_topic {
                    return;
                }
                let avatar = self.known_peers.get(&being_id).and_then(|i| i.avatar.clone());
                self.chat_panel
                    .push_stream_delta(being_id, sender_name, avatar, &message_id, &delta);
            }
            agent::AgentStream::Retract { topic, message_id } => {
                if topic != self.current_topic {
                    return;
                }
                self.chat_panel.remove_stream_line(&message_id);
            }
        }
    }

    /// Render one chat message into the panel, by kind. Shared by live
    /// delivery and by branch-view rebuilds (render_topic_view).
    fn render_message_line(&mut self, msg: &ChatMessage, db: &Database) {
        let avatar = self
            .known_peers
            .get(&msg.sender_id)
            .and_then(|i| i.avatar.clone());

        match msg.kind {
            MessageKind::Human | MessageKind::Agent => {
                self.chat_panel.push(
                    msg.sender_id.clone(),
                    msg.sender_name.clone(),
                    msg.content.clone(),
                    avatar,
                    None,
                );
            }

            MessageKind::Media(ref attachment) => {
                // Decode thumbnail if we have the bytes locally
                if attachment.kind == AttachmentKind::Image {
                    if let Ok(Some(bytes)) = db.get_avatar_cache(&attachment.cid) {
                        if let Some(thumb) = decode_attachment(&bytes) {
                            self.chat_panel
                                .thumb_cache
                                .insert(attachment.cid.clone(), thumb);
                        }
                    }
                }
                self.chat_panel.push(
                    msg.sender_id.clone(),
                    msg.sender_name.clone(),
                    msg.content.clone(), // caption
                    avatar,
                    Some(attachment.clone()),
                );
            }

            MessageKind::System => {
                self.chat_panel
                    .push("system".into(), "system".into(), msg.content.clone(), None, None);
            }

            // pi-style tool panels: pending on the call, then replaced in
            // place by the response (which points back via parent_id).
            MessageKind::ToolCall { ref tool_name, ref arguments } => {
                let block = ToolBlock::pending(msg.id.clone(), tool_name.clone(), arguments);
                self.chat_panel
                    .push_tool(msg.sender_id.clone(), msg.sender_name.clone(), avatar, block);
            }
            MessageKind::ToolResponse { ref tool_name, error, .. } => {
                let block = ToolBlock::response(
                    msg.parent_id.clone(),
                    tool_name.clone(),
                    &msg.content,
                    error,
                );
                self.chat_panel.push_tool_response(
                    msg.sender_id.clone(),
                    msg.sender_name.clone(),
                    avatar,
                    msg.parent_id.as_deref(),
                    block,
                );
            }
            MessageKind::Notes => {
                // Agent-internal; not shown.
            }
        }
    }

    fn handle_presence(&mut self, sender: PeerId, presence: PresenceMessage) {
        // Key by being_id if your PresenceMessage has it, otherwise by peer_id
        let key = if presence.being_id.is_empty() {
            sender.to_base58()
        } else {
            presence.being_id.clone()
        };
        self.known_peers.insert(
            key.clone(),
            PeerInfo {
                name: presence.human_name.clone(),
                avatar_cid: presence.avatar_cid,
                avatar: None,
                enc_public_key: presence.enc_public_key,
            },
        );
        // Presence may arrive after the peer's messages did.
        self.chat_panel.update_peer_name(&key, &presence.human_name);

        // Another device of OUR account: dial its advertised addresses
        // so the history-exchange protocol can sync the account across
        // devices (over direct, hole-punched, or relay-circuit paths).
        // Own devices always get dialed, mesh rules don't apply.
        let their_machine = presence.machine_id.to_base58();
        let is_own_device = !self.self_being_id.is_empty()
            && presence.being_id == self.self_being_id
            && their_machine != self.peer_id;
        if is_own_device {
            for addr in &presence.listen_addrs {
                if self.dialed_self_addrs.insert(addr.clone()) {
                    tracing::info!(%addr, "dialing own device");
                    self.dial_addr(addr);
                }
            }
            return;
        }

        // Grow the p2p mesh: as the group scales, members should sync
        // history (and images) peer-to-peer instead of leaning on the
        // relay. Deterministic direction keeps exactly ONE side dialing
        // (halving connections on a full mesh); the cap bounds outbound
        // connections on big groups.
        if should_dial_for_mesh(
            &self.peer_id,
            &their_machine,
            self.mesh_dialed.len(),
            self.mesh_dialed.contains(&their_machine),
            MESH_TARGET,
        ) {
            self.mesh_dialed.insert(their_machine.clone());
            for addr in &presence.listen_addrs {
                tracing::info!(%addr, machine = %their_machine, "mesh dialing peer");
                self.dial_addr(addr);
            }
        }
    }

    /// Send one Dial command; unparsable addresses are skipped (logged).
    fn dial_addr(&self, addr: &str) {
        if let Some(tx) = &self.cmd_tx {
            let parsed = match addr.parse() {
                Ok(ma) => ma,
                Err(_) => {
                    tracing::warn!(%addr, "unparsable presence address");
                    return;
                }
            };
            let _ = tx
                .clone()
                .try_send(llung_core::network::command::NetworkCommand::Dial {
                    multiaddr: parsed,
                });
        }
    }

    fn handle_raw_text(&mut self, sender: PeerId, topic: String, text: String, db: &Database) {
        let sender_id = sender.to_base58();
        let sender_name = self
            .known_peers
            .get(&sender_id)
            .map(|i| i.name.clone())
            .filter(|n| !n.is_empty())
            .unwrap_or_else(|| {
                // Raw-text peer we haven't identified: short id beats "unknown".
                let mut short = sender_id.clone();
                short.truncate(8);
                short
            });

        let msg = ChatMessage {
            id: generate_cid_from_bytes(&(sender_id.clone() + &text).as_bytes())
                .unwrap_or_else(|_| uuid::Uuid::new_v4().to_string()),
            topic: topic.clone(),
            parent_id: None,
            sender_id: sender_id.clone(),
            sender_name: sender_name.clone(),
            content: text.clone(),
            timestamp: chrono::Utc::now().timestamp() as u64,
            kind: MessageKind::Human,
        };

        if let Err(e) = db.save_message(&msg) {
            self.report_error("Failed to save message", e);
        }

        if topic == self.current_topic {
            let avatar = self
                .known_peers
                .get(&sender_id)
                .and_then(|i| i.avatar.clone());
            self.chat_panel
                .push(sender_id, sender_name, text, avatar, None);
        }
    }

    pub fn draw(&mut self) -> io::Result<()> {
        let taffy_ui = &mut self.taffy_ui;
        let chat_panel = &mut self.chat_panel;
        let registration = &mut self.registration;
        let input_text = &self.input.text;
        let input_cursor = self.input.cursor;
        let status_line = &self.status_line;
        let screen = self.screen;
        let sidebar_mode = self.sidebar_mode;
        let peers = &self.known_peers;
        let chats = &self.chats;
        let current_topic = &self.current_topic;
        let theme = &self.theme;

        // The editing cursor's cell (set at the end of the draw, once
        // the buffer borrow ends).
        let mut cursor_pos: Option<(u16, u16)> = None;
        // Upload newly decoded Kitty images before the frame that shows
        // their placeholders. Order doesn't strictly matter (the terminal
        // renders placeholders once the image arrives), but this avoids a
        // blank flash. APC sequences don't move the cursor, so ratatui's
        // cursor tracking stays valid.
        let pending = crate::terminal::kitty::take_pending();
        if !pending.is_empty() {
            use std::io::Write;
            let backend = self.terminal.backend_mut();
            backend.write_all(&pending)?;
            backend.flush()?;
        }

        self.terminal.draw(|frame| {
            let area = frame.area();
            taffy_ui.resize(area.width, area.height);

            let chat_rect = taffy_ui.node_rect(taffy_ui.chat);
            let input_rect = taffy_ui.node_rect(taffy_ui.input);
            let sidebar_rect = taffy_ui.node_rect(taffy_ui.sidebar);
            let buf = frame.buffer_mut();

            match screen {
                AppScreen::Splash => {
                    Self::render_splash(buf, area, theme);
                }
                AppScreen::Registration => {
                    registration.render(buf, area, theme);
                }
                AppScreen::Chat => {
                    chat_panel.render(buf, chat_rect, theme);
                    // Cursor position is applied at the end of the
                    // closure (the buffer borrow must end first).
                    cursor_pos = Self::render_input_buf(
                        buf,
                        input_rect,
                        input_text,
                        input_cursor,
                        status_line,
                        theme,
                    );
                    Self::render_sidebar_buf(
                        buf,
                        sidebar_rect,
                        sidebar_mode,
                        peers,
                        chats,
                        current_topic,
                        theme,
                    );
                }
            }
            // Position the editing cursor last, once the buffer borrow
            // has ended. (Only meaningful on the Chat screen; harmless
            // elsewhere since other screens don't read it.)
            if screen == AppScreen::Chat {
                if let Some((cx, cy)) = cursor_pos {
                    frame.set_cursor_position(ratatui::layout::Position::new(cx, cy));
                }
            }
        })?;

        Ok(())
    }

    fn render_splash(buf: &mut ratatui::buffer::Buffer, area: Rect, theme: &Theme) {
        let canvas_bg = theme.bg(Role::Canvas);
        for y in area.y..(area.y + area.height) {
            for x in area.x..(area.x + area.width) {
                if let Some(cell) = buf.cell_mut((x, y)) {
                    cell.reset();
                    cell.set_bg(canvas_bg);
                }
            }
        }

        let center =
            |text: &str| -> u16 { area.x + area.width.saturating_sub(text.len() as u16) / 2 };
        let mid = area.y + area.height / 2;

        let logo = "llung";
        buf.set_string(
            center(logo),
            mid.saturating_sub(2),
            logo,
            theme.style(Role::Logo),
        );
        let status = "loading account…";
        buf.set_string(
            center(status),
            mid,
            status,
            theme.style(Role::Muted),
        );
    }

    fn handle_direct_message(&mut self, sender: PeerId, payload: Vec<u8>, db: &Database) {
        let plaintext = match db.decrypt_with_identity_key(&payload) {
            Ok(bytes) => bytes,
            Err(e) => {
                tracing::warn!("Failed to decrypt DM from {sender}: {e}");
                return;
            }
        };

        if let Ok(invite) = serde_json::from_slice::<InvitePayload>(&plaintext) {
            self.handle_invite(sender, invite);
            return;
        }

        let text = String::from_utf8_lossy(&plaintext).into_owned();
        let dm_id = sender.to_base58();
        let dm_name = format!("[DM] {}", self.resolve_name(&sender));
        self.chat_panel.push(dm_id, dm_name, text, None, None);
    }

    pub fn handle_invite(&mut self, sender: PeerId, invite: InvitePayload) {
        let topic_id = invite.topic_id;

        // Follow the invite into the new topic.
        self.join_topic(&topic_id);

        let from = self.resolve_name(&sender);
        self.chat_panel.push(
            "system".to_string(),
            "system".to_string(),
            format!(
                "{} invited you to '{}' — switched to #{}",
                from, invite.display_name, topic_id
            ),
            None,
            None,
        );
    }

    fn resolve_name(&self, sender: &PeerId) -> String {
        let id = sender.to_base58();
        self.known_peers
            .get(&id)
            .map(|info| info.name.clone())
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| id[..8.min(id.len())].to_string())
    }

    /// The input box: top border, wrapped text, bottom border, then the
    /// status row (pi-style). With fewer than 3 rows the bottom border
    /// and status row are dropped and only the top border is drawn.
    /// Returns the cell the editing cursor sits on (byte index into
    /// `text`), so the caller can position the terminal cursor.
    fn render_input_buf(
        buf: &mut ratatui::buffer::Buffer,
        area: Rect,
        text: &str,
        cursor: usize,
        status: &str,
        theme: &Theme,
    ) -> Option<(u16, u16)> {
        let canvas_bg = theme.bg(Role::Canvas);
        let border = theme.style(Role::BorderMuted);
        let buf_area = buf.area().clone();

        let x0 = area.x.min(buf_area.width);
        let y0 = area.y.min(buf_area.height);
        let x1 = (area.x + area.width).min(buf_area.width);
        let y1 = (area.y + area.height).min(buf_area.height);

        for y in y0..y1 {
            for x in x0..x1 {
                if let Some(cell) = buf.cell_mut((x, y)) {
                    cell.reset();
                    cell.set_bg(canvas_bg);
                }
            }
        }
        // Draw top border
        for x in x0..x1 {
            if let Some(cell) = buf.cell_mut((x, y0)) {
                cell.set_symbol("─");
                cell.set_style(border);
            }
        }

        let has_status_row = y1.saturating_sub(y0) >= 3;
        let text_end = if has_status_row { y1 - 2 } else { y1 };
        let max_lines = (text_end.saturating_sub(y0 + 1)) as usize;

        // Which wrapped line holds the cursor (byte offset within it).
        let pt = PrefixText::new(text.to_string());
        let lines: Vec<&str> = pt.wrap_lines((x1 - x0).saturating_sub(2)).collect();
        let mut cursor_line = 0usize;
        let mut cursor_col = 0usize; // byte offset within the line
        if !lines.is_empty() {
            let mut acc = 0usize;
            for (i, l) in lines.iter().enumerate() {
                if cursor <= acc + l.len() {
                    cursor_line = i;
                    cursor_col = cursor.saturating_sub(acc);
                    break;
                }
                acc += l.len() + 1; // implied '\n'
            }
        }
        // Keep the cursor visible: if it scrolled out of the tail window,
        // shift the window to show it.
        let start = lines
            .len()
            .saturating_sub(max_lines)
            .min(cursor_line);

        for (i, line) in lines[start..].iter().enumerate() {
            let y = y0 + 1 + i as u16;
            if y >= text_end {
                break;
            }
            buf.set_string(x0 + 1, y, line, theme.style(Role::Text));
        }

        // Cursor cell: column = display width of the line before the
        // cursor; row = the cursor's wrapped line (clamped visible).
        let (cursor_x, cursor_y) = {
            use unicode_width::UnicodeWidthStr;
            let line = lines.get(cursor_line).copied().unwrap_or("");
            let col = line[..cursor_col.min(line.len())].width() as u16;
            (
                (x0 + 1).saturating_add(col),
                (y0 + 1).saturating_add(cursor_line.saturating_sub(start) as u16),
            )
        };

        if has_status_row {
            // Bottom border demarcating the input, status row under it.
            for x in x0..x1 {
                if let Some(cell) = buf.cell_mut((x, y1 - 2)) {
                    cell.set_symbol("─");
                    cell.set_style(border);
                }
            }
            let width = (x1 - x0) as usize;
            let shown: String = status.chars().take(width.saturating_sub(2)).collect();
            buf.set_string(x0 + 1, y1 - 1, &shown, theme.style(Role::Dim));
        }

        if cursor_y < text_end {
            Some((cursor_x, cursor_y))
        } else {
            None
        }
    }

    fn render_sidebar_buf(
        buf: &mut ratatui::buffer::Buffer,
        area: Rect,
        mode: SidebarMode,
        peers: &HashMap<String, PeerInfo>,
        chats: &[String],
        current_topic: &str,
        theme: &Theme,
    ) {
        let panel_bg = theme.bg(Role::Panel);
        let buf_area = buf.area().clone();
        let x0 = area.x.min(buf_area.width);
        let y0 = area.y.min(buf_area.height);
        let x1 = (area.x + area.width).min(buf_area.width);
        let y1 = (area.y + area.height).min(buf_area.height);

        // Fill background
        for y in y0..y1 {
            for x in x0..x1 {
                if let Some(cell) = buf.cell_mut((x, y)) {
                    cell.reset();
                    cell.set_bg(panel_bg);
                }
            }
        }

        match mode {
            SidebarMode::Hidden => {}
            SidebarMode::Peers => {
                buf.set_string(x0 + 1, y0, " Peers ", theme.style(Role::PanelTitle));
                let mut y = y0 + 2;
                for (peer_id, info) in peers.iter() {
                    if y + 2 >= y1 {
                        break;
                    }

                    // Draw avatar (4x2 cells)
                    if let Some(ref thumb) = info.avatar {
                        render_avatar(buf, x0 + 1, y, thumb);
                    } else {
                        let initial = info.name.chars().next().unwrap_or('?');
                        fallback_avatar(
                            buf,
                            x0 + 1,
                            y,
                            initial,
                            theme.avatar_color(peer_id),
                            theme.avatar_text(),
                        );
                    }

                    let name = if info.name.is_empty() {
                        &peer_id[..8.min(peer_id.len())]
                    } else {
                        &info.name
                    };
                    let avail = (x1.saturating_sub(x0 + 1)) as usize;
                    buf.set_stringn(x0 + 1, y + 2, name, avail, theme.style(Role::PeerName));

                    // Avatar (2 rows) + name row + 1 row of padding.
                    y += 4;
                }
            }
            SidebarMode::Chats => {
                buf.set_string(x0 + 1, y0, " Chats ", theme.style(Role::PanelTitle));
                if chats.is_empty() {
                    buf.set_string(
                        x0 + 1,
                        y0 + 2,
                        "(no chats yet)",
                        theme.style(Role::Dim),
                    );
                } else {
                    let avail = (x1.saturating_sub(x0 + 1)) as usize;
                    let mut y = y0 + 2;
                    for topic in chats {
                        if y >= y1 {
                            break;
                        }
                        let is_current = topic == current_topic;
                        let style = if is_current {
                            theme.style(Role::ActiveItem)
                        } else {
                            theme.style(Role::InactiveItem)
                        };
                        buf.set_stringn(x0 + 1, y, format!("#{topic}"), avail, style);
                        y += 1;
                    }
                }
            }
            SidebarMode::Media => {
                buf.set_string(x0 + 1, y0, " Media ", theme.style(Role::PanelTitle));
                buf.set_string(
                    x0 + 1,
                    y0 + 2,
                    "(empty)",
                    theme.style(Role::Dim),
                );
            }
        }
    }

    async fn handle_command(
        &mut self,
        cmd: &str,
        db: &Database,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (name, args) = match cmd.find(' ') {
            Some(i) => (&cmd[..i], cmd[i + 1..].trim()),
            None => (cmd, ""),
        };

        match name {
            "/quit" | "/q" => {
                {
                    use std::io::Write;
                    let mut out = std::io::stdout();
                    let _ = out.write_all(&crate::terminal::cleanup_sequence());
                    let _ = out.flush();
                }
                crossterm::terminal::disable_raw_mode().ok();
                let _ = std::io::stdout().execute(crossterm::terminal::LeaveAlternateScreen);
                std::process::exit(0);
            }
            "/peers" => self.toggle_sidebar(SidebarMode::Peers),
            "/chats" => self.toggle_sidebar(SidebarMode::Chats),
            "/media" => self.toggle_sidebar(SidebarMode::Media),
            "/theme" => self.handle_theme_command(args),
            "/invite" => {
                if let Err(e) = self.handle_invite_command(args, db).await {
                    self.chat_panel
                        .push("system".to_string(), "system".to_string(), e, None, None);
                }
            }
            "/join" => {
                let topic_id = args
                    .split_whitespace()
                    .next()
                    .unwrap_or("")
                    .trim_start_matches('#');
                if topic_id.is_empty() {
                    self.chat_panel.push(
                        "system".to_string(),
                        "system".to_string(),
                        "usage: /join #<topic>".to_string(),
                        None,
                        None,
                    );
                } else {
                    self.join_topic(topic_id);
                    self.render_topic_view(db);
                    self.chat_panel.push(
                        "system".to_string(),
                        "system".to_string(),
                        format!("Switched to #{topic_id}"),
                        None,
                        None,
                    );
                }
            }

            "/branch" => self.handle_branch_command(args, db),

            "/agent" => {
                let mut parts = args.split_whitespace();
                let sub = parts.next().unwrap_or("");
                match sub {
                    "spawn" => {
                        let name = parts.next().unwrap_or("").to_string();
                        if name.is_empty() {
                            self.chat_panel.push(
                                "system".into(), "system".into(),
                                "Usage: /agent spawn <name> [backend=llamacpp|lemonade|kimi|kimi-coding] [model=<id>] [thinking=low|medium|high]".into(),
                                None, None,
                            );
                            return Ok(());
                        }

                        // A name collision leaves two agent loops answering
                        // the same @mention (e.g. one with a dead backend,
                        // one healthy) — confusing at best. Require a fresh
                        // name per spawn.
                        if self.agent_cmd_txs.contains_key(&name) {
                            self.chat_panel.push(
                                "system".into(), "system".into(),
                                format!("Agent '{name}' already exists. Pick another name (or restart to respawn)."),
                                None, None,
                            );
                            return Ok(());
                        }

                        // Optional terms after the name, e.g.
                        // `/agent spawn helper backend=lemonade` or
                        // `/agent spawn scout backend=kimi model=kimi-k2-thinking thinking=high`.
                        const SPAWN_USAGE: &str = "Usage: /agent spawn <name> [backend=llamacpp|lemonade|kimi|kimi-coding] [model=<id>] [thinking=low|medium|high]";
                        let mut backend_kind = agent::config::BackendKind::default();
                        let mut model: Option<String> = None;
                        let mut thinking: Option<agent::config::ThinkingLevel> = None;
                        for term in parts {
                            if let Some(value) = term.strip_prefix("backend=") {
                                match agent::config::BackendKind::parse(value) {
                                    Some(kind) => backend_kind = kind,
                                    None => {
                                        self.chat_panel.push(
                                            "system".into(), "system".into(),
                                            format!("Unknown backend '{value}'. Try: llamacpp, lemonade, kimi, kimi-coding"),
                                            None, None,
                                        );
                                        return Ok(());
                                    }
                                }
                            } else if let Some(value) = term.strip_prefix("model=") {
                                model = Some(value.to_string());
                            } else if let Some(value) = term.strip_prefix("thinking=") {
                                match agent::config::ThinkingLevel::parse(value) {
                                    Some(level) => thinking = Some(level),
                                    None => {
                                        self.chat_panel.push(
                                            "system".into(), "system".into(),
                                            format!("Unknown thinking level '{value}'. Try: low, medium, high"),
                                            None, None,
                                        );
                                        return Ok(());
                                    }
                                }
                            } else {
                                self.chat_panel.push(
                                    "system".into(), "system".into(),
                                    format!("Unknown term '{term}'. {SPAWN_USAGE}"),
                                    None, None,
                                );
                                return Ok(());
                            }
                        }

                        // Cloud backends need an API key BEFORE we
                        // provision anything — resolving it late would
                        // leave an orphan Being in the DB.
                        let api_key = match backend_kind.key_provider() {
                            Some(provider) => match llung_core::secrets::load_api_key(provider).await {
                                Ok(key) => Some(key),
                                Err(e) => {
                                    self.chat_panel.push(
                                        "system".into(), "system".into(),
                                        format!("Kimi backend needs an API key: {e}"),
                                        None, None,
                                    );
                                    return Ok(());
                                }
                            },
                            None => None,
                        };

                        // Provision in DB: new Being + new Machine
                        let agent_id = agent::provision_agent(db, &name)?;
            
                        // Start the agent's own network stack
                        let (agent_app, agent_event_rx) = LlungApp::init(
                            CoreConfig {
                                db_path: self.config.db_path.clone(),
                                chat_topic: self.current_topic.clone(),
                                relays: CoreConfig::relays_from_env(),
                            },
                            agent_id.machine,
                            agent_id.being.clone(),
                        ).await?;

                        // Spawn the agent loop. The Aukora kernel and the
                        // wasm tool worker are wired up inside `new`; the
                        // history compressor shares the agent's model
                        // backend (with a compressor system prompt).
                        let agent_cmd_tx = agent_app.command_tx();

                        // Permission prompts arrive on the app-wide
                        // `perm_tx` (created in `run`); our answer goes
                        // back on the request's oneshot. `/grant`
                        // commands travel the parallel cmd channel.
                        let (grant_tx, grant_cmd_rx) = mpsc::channel::<agent::AgentCommand>(8);

                        // Same-process wiring: our node and the agent's
                        // node never discover each other when the host
                        // filters multicast (broken mDNS), so hand the
                        // agent our listen addresses and have it dial us
                        // directly. Without this every topic publish dies
                        // with NoPeersSubscribedToTopic.
                        if let Some(our_tx) = &self.cmd_tx {
                            let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
                            if our_tx
                                .send(NetworkCommand::GetListenAddrs { responder: resp_tx })
                                .await
                                .is_ok()
                            {
                                if let Ok(addrs) = resp_rx.await {
                                    for addr in addrs {
                                        let _ = agent_cmd_tx
                                            .send(NetworkCommand::Dial { multiaddr: addr })
                                            .await;
                                    }
                                }
                            }
                        }
                        // Model default is backend-specific: a Kimi
                        // spawn without model= gets the Kimi default.
                        let agent_config = AgentConfig {
                            name: name.clone(),
                            model: model.clone().unwrap_or_else(|| match backend_kind {
                                agent::config::BackendKind::Kimi => "kimi-k2-0905-preview".to_string(),
                                agent::config::BackendKind::KimiCoding => "kimi-for-coding".to_string(),
                                _ => AgentConfig::default().model,
                            }),
                            backend: backend_kind,
                            thinking,
                            ..Default::default()
                        };
                        let mut backend = OpenAiCompatible::from_config(&agent_config);
                        if let Some(key) = api_key {
                            backend = backend.with_api_key(key);
                        }

                        // Warn early when the model server isn't there —
                        // otherwise the agent simply never answers and it
                        // looks like a hang.
                        if let Err(e) = backend.probe().await {
                            self.chat_panel.push(
                                "system".into(), "system".into(),
                                format!(
                                    "Warning: model backend {backend_kind:?} unreachable ({e}). \
                                     The agent will fail until it is up — or respawn with backend=..."
                                ),
                                None, None,
                            );
                        }
                        let stream_tx = self
                            .stream_tx
                            .clone()
                            .ok_or("streaming channel not initialized")?;
                        let perm_tx = self
                            .perm_tx
                            .clone()
                            .ok_or("permission channel not initialized")?;
                        let ctx = AgentLoopContext::new(
                            agent_config.clone(),
                            Box::new(backend.clone()),
                            HistoryCompressor::new(
                                Box::new(ModelSummarizer::new(Box::new(backend))),
                                64_000,
                            ),
                            Database::open(
                                self.config.db_path.to_str().ok_or("non-UTF8 db path")?,
                            )?,
                            agent_cmd_tx.clone(),
                            self.current_topic.clone(),
                            agent_id.being.being_id.clone(),
                            name.clone(),
                            stream_tx,
                            perm_tx,
                        );
                        tokio::spawn(run_agent_loop(ctx, agent_event_rx, grant_cmd_rx));

                        // The agent's Being is already a peer (its own
                        // machine + being in the DB, listed in known_peers
                        // below). Keep only what /invite needs: a handle
                        // to tell its network stack to join a topic.
                        self.agent_cmd_txs.insert(name.clone(), agent_cmd_tx);

                        // Also add to known_peers so presence rendering works
                        self.agent_grant_txs.insert(name.clone(), grant_tx);
                        self.known_peers.insert(
                            agent_id.being.being_id.clone(),
                            PeerInfo {
                                name: name.clone(),
                                avatar_cid: None,
                                avatar: None,
                                enc_public_key: None,
                            }
                        );

                        self.chat_panel.push(
                            "system".into(), "system".into(),
                            format!("Agent '{name}' is ready (backend: {backend_kind:?}). Invite it with: /invite {name} #{}", self.current_topic),
                            None, None,
                        );
                        self.update_status_line(db);
                    }
                    _ => {
                        self.chat_panel.push(
                            "system".into(), "system".into(),
                            "Unknown /agent subcommand. Try: /agent spawn <name> [backend=llamacpp|lemonade]".into(),
                            None, None,
                        );
                    }
                }
            }

            "/attach" => {
                let rest = args;
                let (path_str, caption) = if rest.starts_with('"') {
                    if let Some(end_quote) = rest[1..].find('"') {
                        let path = &rest[1..=end_quote];
                        let cap = rest[end_quote + 2..].trim();
                        (
                            path,
                            if cap.is_empty() {
                                None
                            } else {
                                Some(cap.to_string())
                            },
                        )
                    } else {
                        self.chat_panel.push(
                            "system".into(),
                            String::new(),
                            "Usage: /attach <path> [caption]".into(),
                            None,
                            None,
                        );
                        return Ok(());
                    }
                } else {
                    let mut parts = rest.splitn(2, ' ');
                    let path = parts.next().unwrap_or("");
                    let cap = parts.next().map(|s| s.to_string());
                    (path, cap)
                };

                if path_str.is_empty() {
                    self.chat_panel.push(
                        "system".into(),
                        String::new(),
                        "Usage: /attach <path> [caption]".into(),
                        None,
                        None,
                    );
                    return Ok(());
                }

                let expanded = shellexpand::tilde(path_str);
                let path = std::path::Path::new(expanded.as_ref());

                if !path.exists() {
                    self.chat_panel.push(
                        "system".into(),
                        String::new(),
                        format!("File not found: {}", path.display()),
                        None,
                        None,
                    );
                    return Ok(());
                }

                if let Err(e) = self.send_attachment(path, caption, db).await {
                    self.chat_panel.push(
                        "system".into(),
                        String::new(),
                        format!("Failed to attach: {e}"),
                        None,
                        None,
                    );
                }
            }
            "/plot" => {
                // /plot <path> [kind=scatter] [x=col] [y=col] [z=col] [bins=N]
                // Run plot_device on a CSV file directly — the pending
                // panel spins while the wasm tool runs.
                let mut parts = args.split_whitespace();
                let Some(path) = parts.next() else {
                    self.chat_panel.push(
                        "system".into(), "system".into(),
                        "Usage: /plot <csv-path> [kind=scatter|histogram|cdf] [x=col] [y=col] [z=col] [bins=N]".into(),
                        None, None,
                    );
                    return Ok(());
                };
                let mut map = serde_json::json!({
                    "path": normalize_plot_path(path),
                });
                for kv in parts {
                    let Some((k, v)) = kv.split_once('=') else {
                        self.chat_panel.push(
                            "system".into(), "system".into(),
                            format!("Usage: /plot <csv-path> [kind=…] [x=col] [y=col] [z=col] [bins=N] — '{kv}' is not key=value"),
                            None, None,
                        );
                        return Ok(());
                    };
                    map[k] = serde_json::Value::String(v.to_string());
                }
                let arguments = map.to_string();
                let call_id = generate_cid_from_bytes(
                    &(self.peer_id.clone() + &arguments).as_bytes(),
                )
                .unwrap_or_else(|_| uuid::Uuid::new_v4().to_string());
                self.chat_panel.push_tool(
                    self.peer_id.clone(),
                    self.my_name.clone(),
                    None,
                    ToolBlock::pending(call_id.clone(), "plot_device".into(), &arguments),
                );
                match agent::tools::run_tool_locally("plot_device", &arguments).await {
                    Ok(output) => {
                        let block = ToolBlock::response(
                            Some(call_id.clone()),
                            "plot_device".into(),
                            &output,
                            false,
                        );
                        self.chat_panel.push_tool_response(
                            self.peer_id.clone(),
                            self.my_name.clone(),
                            None,
                            Some(&call_id),
                            block,
                        );
                    }
                    Err(e) => {
                        let block = ToolBlock::response(
                            Some(call_id.clone()),
                            "plot_device".into(),
                            &e,
                            true,
                        );
                        self.chat_panel.push_tool_response(
                            self.peer_id.clone(),
                            self.my_name.clone(),
                            None,
                            Some(&call_id),
                            block,
                        );
                    }
                }
            }
            "/tools" => {
                for line in Self::format_tool_listing(args.trim()) {
                    self.chat_panel
                        .push("system".into(), "system".into(), line, None, None);
                }
            }
            "/grant" => {
                match parse_grant_command(args) {
                    Err(usage) => self.system(usage),
                    Ok(GrantCommand::Pending) => {
                        if let Some(req) = &self.pending_perm {
                            self.system(format!(
                                "pending: {} asks `{}`{} — answer with y/Y/n",
                                req.agent_name,
                                req.effect,
                                tier_label(req.tier)
                            ));
                        } else if !self.perm_queue.is_empty() {
                            self.system(format!("{} permission request(s) queued", self.perm_queue.len()));
                        } else {
                            self.system("no pending permission requests");
                        }
                    }
                    Ok(GrantCommand::List) => self.grant_list().await,
                    Ok(GrantCommand::Grant { effect, agent }) => {
                        self.grant_change(effect, agent, true).await;
                    }
                    Ok(GrantCommand::Revoke { effect, agent }) => {
                        self.grant_change(effect, agent, false).await;
                    }
                }
            }
            "/help" | "/h" | "/?" => {
                for line in Self::help_text() {
                    self.chat_panel
                        .push("system".into(), "system".into(), line, None, None);
                }
            }
            _ => {
                self.chat_panel.push(
                    "system".into(),
                    "system".into(),
                    format!("Unknown command '{name}' — try /help"),
                    None,
                    None,
                );
            }
        }
        Ok(())
    }

    /// Drain one relay-fetched image result: cache it, drop the inflight
    /// marker, redraw.
    fn handle_image_result(
        &mut self,
        db: &Database,
        cid: String,
        result: Option<(String, Vec<u8>)>,
    ) {
        self.pending_image_fetches.remove(&cid);
        match result {
            Some((_mime, data)) => self.on_image_fetched(db, &cid, &data),
            None => tracing::debug!(%cid, "image not on any relay (yet)"),
        }
    }

    /// `/grant list`: standing grants per local agent (round-trips into
    /// each agent's orchestrator loop).
    async fn grant_list(&mut self) {
        if self.agent_grant_txs.is_empty() {
            self.system("no local agents — spawn one with /agent spawn");
            return;
        }
        let mut names: Vec<String> = self.agent_grant_txs.keys().cloned().collect();
        names.sort();
        for name in names {
            let tx = self.agent_grant_txs.get(&name).expect("key from keys");
            let (respond, rx) = tokio::sync::oneshot::channel();
            if tx.send(agent::AgentCommand::ListGrants { respond }).await.is_err() {
                self.system(format!("{name}: agent loop unreachable"));
                continue;
            }
            match tokio::time::timeout(std::time::Duration::from_secs(2), rx).await {
                Ok(Ok(effects)) if effects.is_empty() => {
                    self.system(format!("{name}: (no standing grants)"));
                }
                Ok(Ok(effects)) => {
                    self.system(format!("{name}: {}", effects.join(", ")));
                }
                Ok(Err(_)) => self.system(format!("{name}: grant list dropped")),
                Err(_) => self.system(format!("{name}: grant list timed out")),
            }
        }
    }

    /// `/grant <effect> perpetual [agent]` / `revoke`: add or remove a
    /// standing grant. Without `agent`, applies to every local agent.
    async fn grant_change(&mut self, effect: String, agent: Option<String>, grant: bool) {
        let targets: Vec<String> = match &agent {
            Some(a) => vec![a.clone()],
            None => {
                let mut all: Vec<String> = self.agent_grant_txs.keys().cloned().collect();
                all.sort();
                all
            }
        };
        if targets.is_empty() {
            self.system("no local agents — spawn one with /agent spawn");
            return;
        }
        for name in targets {
            let Some(tx) = self.agent_grant_txs.get(&name) else {
                self.system(format!("no local agent named '{name}'"));
                continue;
            };
            let cmd = if grant {
                agent::AgentCommand::GrantStanding { effect: effect.clone() }
            } else {
                agent::AgentCommand::Revoke { effect: effect.clone() }
            };
            if tx.send(cmd).await.is_err() {
                self.system(format!("{name}: agent loop unreachable"));
            } else {
                let verb = if grant { "granted (perpetual)" } else { "revoked" };
                self.system(format!("{name}: `{effect}` {verb}"));
            }
        }
    }

    /// `/tools` output: every tool (no arg), or one tool's description,
    /// usage, and argument schema (`/tools <name>`).
    fn format_tool_listing(name: &str) -> Vec<String> {
        let catalog = agent::tools::tool_catalog();
        if name.is_empty() {
            let mut lines = vec!["Tools:".to_string()];
            for spec in catalog {
                lines.push(format!("  {} — {}", spec.name, spec.description));
            }
            lines.push("Usage: /tools <name> — full description and arguments".into());
            return lines;
        }
        let Some(spec) = catalog.iter().find(|t| t.name == name) else {
            return vec![format!(
                "Unknown tool '{name}' — try /tools for the list"
            )];
        };
        let mut lines = vec![
            format!("{} — {}", spec.name, spec.description),
            format!("usage: {}", spec.usage),
        ];
        if let Ok(schema) = serde_json::from_str::<serde_json::Value>(spec.schema) {
            if let Ok(pretty) = serde_json::to_string_pretty(&schema) {
                lines.push("arguments (JSON schema):".into());
                lines.extend(pretty.lines().map(str::to_string));
            }
        }
        lines
    }

    /// Every slash command, one per line, for `/help`.
    fn help_text() -> Vec<String> {
        vec![
            "Commands:".into(),
            "  /join #<topic> — switch to a topic (and subscribe)".into(),
            "  /invite <peer> [peer...] [#<topic>] [display name] — invite peers to a topic (default: current chat)".into(),
            "  /agent spawn <name> [backend=llamacpp|lemonade|kimi|kimi-coding] [model=<id>] [thinking=low|medium|high] — spawn a local agent, then invite it like any peer".into(),
            "  kimi needs an API key once: `llung-cli key set kimi` (login keyring) or LLUNG_KIMI_API_KEY".into(),
            "  /branch — where am I in the chat tree?".into(),
            "  /branch <n> — focus the message n-back; your next message forks the chat there".into(),
            "  /branch up|down — walk to the parent / most recent child message".into(),
            "  /branch next|prev — switch to a sibling branch at the same fork".into(),
            "  /branch tip — return to the newest branch".into(),
            "  /tools [name] — list tools, or show one tool's description and arguments".into(),
            "  /grant list — standing tool grants per local agent".into(),
            "  /grant <effect> perpetual [agent] — pre-authorize a tool tier (e.g. bash, repl); agent defaults to all".into(),
            "  /grant revoke <effect> [agent] — remove a standing grant".into(),
            "  /grant pending — show the permission prompt queue".into(),
            "  /plot <csv-path> [kind=scatter|histogram|cdf] [x=col] [y=col] [z=col] — run plot_device on a file yourself".into(),
            "  /attach <path> [caption] — send an image".into(),
            "  /peers · /chats · /media — toggle the sidebars".into(),
            "  /theme <name | light/dark | path.json> · /theme reload — theme picker".into(),
            "  /quit — exit".into(),
            "Example — spawn an agent and ask it to plot:".into(),
            "  /agent spawn plotter backend=lemonade".into(),
            "  /invite plotter #introductions".into(),
            "  @plotter plot cos(x)*sin(y)".into(),
            "Example — fork the conversation:".into(),
            "  /branch 3          (next message replies 3 messages back)".into(),
            "  actually, let's talk about Y".into(),
            "  /branch tip        (back to the newest branch)".into(),
        ]
    }

    async fn send_attachment(
        &mut self,
        path: &std::path::Path,
        caption: Option<String>,
        db: &Database,
    ) -> Result<(), Box<dyn Error>> {
        let bytes = std::fs::read(path)?;

        let cid = generate_cid_from_bytes(&bytes)?;
        db.save_avatar_cache(&cid, &bytes)?;

        let attachment = Attachment::from_path(path.to_str().unwrap_or("unknown"), cid.clone())?;

        if let Some(tx) = &self.cmd_tx {
            tx.send(NetworkCommand::ProvideMediaCid { cid: cid.clone() })
                .await?;
            // Lightweight images (avatars, message images ≤512KB) also
            // go into the relay caches, so peers behind NATs and your
            // other devices can fetch them later.
            let mime = attachment.mime_type.clone();
            if attachment.kind == AttachmentKind::Image
                && bytes.len() <= llung_core::media::image::MAX_IMAGE_BYTES
            {
                tx.send(NetworkCommand::PublishImage {
                    cid: cid.clone(),
                    mime_type: mime,
                    data: bytes.clone(),
                })
                .await?;
            }
        }

        let my_avatar = self
            .known_peers
            .get(&self.peer_id)
            .and_then(|info| info.avatar.clone());

        if attachment.kind == AttachmentKind::Image {
            if let Some(thumb) = decode_attachment(&bytes) {
                self.chat_panel
                    .thumb_cache
                    .insert(attachment.cid.clone(), thumb);
            }
        }

        let timestamp = chrono::Utc::now().timestamp();
        let msg = ChatMessage {
            id: generate_cid_from_bytes(
                &(self.peer_id.clone() + &cid + &timestamp.to_string()).as_bytes(),
            )
            .unwrap_or_else(|_| uuid::Uuid::new_v4().to_string()),
            topic: self.current_topic.clone(),
            parent_id: None,
            sender_id: self.peer_id.clone(),
            sender_name: self.my_name.clone(),
            content: caption.clone().unwrap_or_default(),
            timestamp: timestamp as u64,
            kind: MessageKind::Media(attachment.clone()),
        };

        db.save_message(&msg)?;

        self.chat_panel.push(
            self.peer_id.clone(),
            self.my_name.clone(),
            msg.content.clone(),
            my_avatar,
            Some(attachment),
        );

        if let Some(tx) = &self.cmd_tx {
            tx.send(NetworkCommand::PublishMessage {
                topic: IdentTopic::new(&self.current_topic),
                contents: serde_json::to_vec(&msg)?,
            })
            .await?;
        }

        Ok(())
    }

    async fn handle_invite_command(
        &mut self,
        args: &str,
        db: &Database,
    ) -> std::result::Result<(), String> {
        let (peer_queries, topic_id, display_name) =
            parse_invite_args(args, &self.current_topic)?;

        let mut remote_targets: Vec<PeerId> = Vec::new();

        for q in &peer_queries {
            // LOCAL AGENT: by name. It is a peer like any other — we just
            // also hold its command channel, so tell it to join directly
            // instead of sending an encrypted DM to itself.
            if let Some(tx) = self.agent_cmd_txs.get(q) {
                let _ = tx.send(NetworkCommand::SubscribeTopic {
                    topic: IdentTopic::new(&topic_id),
                }).await;
                continue;
            }
            // REMOTE PEER: resolve via known_peers / DHT
            remote_targets.push(self.resolve_peer(q)?);
        }

        // ── Remote peers: encrypted DM invite ──
        for target in &remote_targets {
            let id = target.to_base58();
            let info = self.known_peers.get(&id);
            let name = info.map(|i| i.name.as_str()).unwrap_or(&id);
            let Some(pk) = info.and_then(|i| i.enc_public_key) else {
                return Err(format!("no encryption key for '{name}'"));
            };

            let payload = InvitePayload {
                topic_id: topic_id.clone(),
                encryption_key: random_hex_key(),
                display_name: display_name.clone(),
                invited_by: self.my_name.clone(),
                timestamp: chrono::Utc::now().timestamp(),
            };
            let json = serde_json::to_vec(&payload).map_err(|e| e.to_string())?;
            let encrypted = encrypt_to_public_key(&pk, &json).map_err(|e| e.to_string())?;

            let tx = self.cmd_tx.as_ref().ok_or("network not connected")?;
            tx.send(NetworkCommand::SendDirectMessage {
                target: *target,
                payload: encrypted,
            }).await.map_err(|e| e.to_string())?;
        }

        // Join the topic ourselves
        self.join_topic(&topic_id);
        self.render_topic_view(db);

        let invited_names = peer_queries.join(", ");
        self.chat_panel.push(
            "system".into(), "system".into(),
            format!("Invited {invited_names} to '{display_name}' — now in #{topic_id}"),
            None, None,
        );
        Ok(())
    }

    fn toggle_sidebar(&mut self, mode: SidebarMode) {
        if self.sidebar_mode == mode {
            self.sidebar_mode = SidebarMode::Hidden;
            self.taffy_ui.set_sidebar_visible(false);
        } else {
            self.sidebar_mode = mode;
            self.taffy_ui.set_sidebar_visible(true);
        }
    }
}

/// `/grant` subcommands, parsed for `handle_command`.
#[derive(Debug, PartialEq)]
enum GrantCommand {
    List,
    Pending,
    Grant { effect: String, agent: Option<String> },
    Revoke { effect: String, agent: Option<String> },
}

const GRANT_USAGE: &str = "Usage: /grant list | /grant <effect> perpetual [agent] | /grant revoke <effect> [agent] | /grant pending";

/// Effects are keyed by the bare tool name used in manifests
/// (`bash`, `repl`, `gpu:futhark`). `agent` defaults to all local agents.
fn parse_grant_command(args: &str) -> Result<GrantCommand, String> {
    let parts: Vec<&str> = args.split_whitespace().collect();
    match parts.as_slice() {
        ["list"] => Ok(GrantCommand::List),
        ["pending"] => Ok(GrantCommand::Pending),
        ["revoke", effect] => Ok(GrantCommand::Revoke {
            effect: effect.to_string(),
            agent: None,
        }),
        ["revoke", effect, agent] => Ok(GrantCommand::Revoke {
            effect: effect.to_string(),
            agent: Some(agent.to_string()),
        }),
        [effect, "perpetual"] => Ok(GrantCommand::Grant {
            effect: effect.to_string(),
            agent: None,
        }),
        [effect, "perpetual", agent] => Ok(GrantCommand::Grant {
            effect: effect.to_string(),
            agent: Some(agent.to_string()),
        }),
        _ => Err(GRANT_USAGE.into()),
    }
}

/// Target number of mesh-dialed peers (outbound p2p connections). With
/// deterministic single-direction dialing, a group of N members forms
/// ~N/2 outbound + N/2 inbound links per node; gossipsub's own mesh
/// targets ~6 peers, so this sits comfortably above it.
const MESH_TARGET: usize = 8;

/// Should we dial this presence-advertising peer to grow the p2p mesh?
/// Pure decision — unit tested.
///
/// - Deterministic direction (`own_machine < their_machine`): exactly
///   one side dials, halving full-mesh connections.
/// - Cap: bounds outbound connections as the group scales.
/// - Dedup: never re-dial a machine (presence repeats every 15s).
fn should_dial_for_mesh(
    own_machine: &str,
    their_machine: &str,
    mesh_dialed_count: usize,
    already_dialed: bool,
    target: usize,
) -> bool {
    !already_dialed && own_machine < their_machine && mesh_dialed_count < target
}

/// Human description of an effect tier for permission prompts.
fn tier_label(tier: agent::effects::Tier) -> &'static str {
    use agent::effects::Tier;
    match tier {
        Tier::Wasm => "",
        Tier::FsRead => " (filesystem read)",
        Tier::Edit => " (file edit)",
        Tier::ProcBash => " (sandboxed shell)",
        Tier::Repl => " (interpreter)",
        Tier::Gpu => " (GPU)",
    }
}

/// `/plot` paths accept the mention form too: strip a leading `@`,
/// expand `~`.
fn normalize_plot_path(token: &str) -> String {
    shellexpand::tilde(token.strip_prefix('@').unwrap_or(token)).to_string()
}

/// pi-style file mentions: tokens like `@/path/to.csv` or `@~/data.csv`
/// expand to the plain absolute path plus a small descriptor (CSV
/// shape, or byte size) so the model can call file-reading tools
/// (`plot_device`) with the path instead of transcribing the data.
fn expand_file_mentions(text: &str) -> String {
    text.split(' ').map(expand_one_mention).collect::<Vec<_>>().join(" ")
}

fn expand_one_mention(token: &str) -> String {
    let Some(rest) = token
        .strip_prefix('@')
        .filter(|r| r.starts_with('/') || r.starts_with('~'))
    else {
        return token.to_string();
    };
    let expanded = shellexpand::tilde(rest).to_string();
    let Ok(meta) = std::fs::metadata(&expanded) else {
        return token.to_string();
    };
    if !meta.is_file() {
        return token.to_string();
    }
    let desc = if rest.ends_with(".csv") {
        csv_file_desc(&expanded)
    } else {
        format!("{} bytes", meta.len())
    };
    format!("{expanded} (file: {desc})")
}

fn csv_file_desc(path: &str) -> String {
    let Ok(content) = std::fs::read_to_string(path) else {
        return "unreadable".into();
    };
    let lines: Vec<&str> = content.lines().filter(|l| !l.trim().is_empty()).collect();
    let Some(header) = lines.first() else {
        return "empty csv".into();
    };
    let cols = header.split(',').count();
    let rows = lines.len().saturating_sub(1);
    format!("csv, {cols} columns, {rows} rows")
}

/// Bytes of the budgeted summary-tree view an agent would read for
/// `messages` — a dry-run with the rule-based summarizer (no model
/// calls, so the status row stays instant). Returns None when the
/// compressor cannot run (shouldn't happen; the caller falls back).
fn compressed_view_bytes(messages: &[llung_core::network::message::ChatMessage], budget: usize) -> Option<usize> {
    use agent::history::{HeadlineSummarizer, HistoryCompressor};
    let compressor = HistoryCompressor::new(Box::new(HeadlineSummarizer::new()), budget);
    let tree = futures::executor::block_on(compressor.compress(messages));
    let view = compressor.flatten_budgeted(&tree, budget);
    Some(view.len())
}

/// 950 → "950", 3456 → "3.5k", 1_234_567 → "1.2M".
fn human_len(n: usize) -> String {
    if n < 1_000 {
        n.to_string()
    } else if n < 1_000_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    }
}

/// Parse `/invite` arguments into `(peer queries, topic id, display name)`.
/// The first `#`-prefixed token splits peers from the topic; any remaining
/// tokens become the display name (falling back to the topic id).
/// Split `/invite` args into `(peers, topic id, display name)`. The
/// `#<topic>` token is optional: without it the invite targets
/// `default_topic` (the current chat).
fn parse_invite_args(
    args: &str,
    default_topic: &str,
) -> std::result::Result<(Vec<String>, String, String), String> {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    let hash_pos = tokens.iter().position(|t| t.starts_with('#'));
    let (peers, topic_id, display_name) = match hash_pos {
        Some(pos) => {
            if pos == 0 {
                return Err("name at least one peer to invite (see /peers)".into());
            }
            let topic_id = tokens[pos].trim_start_matches('#').to_string();
            if topic_id.is_empty() {
                return Err("topic name cannot be empty".into());
            }
            let display_name = if tokens.len() > pos + 1 {
                tokens[pos + 1..].join(" ")
            } else {
                topic_id.clone()
            };
            (
                tokens[..pos].iter().map(|s| s.to_string()).collect(),
                topic_id,
                display_name,
            )
        }
        None => {
            if tokens.is_empty() {
                return Err("usage: /invite <peer> [peer...] [#<topic>] [display name]".into());
            }
            // No #topic: invite to the current chat. (A custom display
            // name requires an explicit topic, since everything here is
            // peer names.)
            let topic_id = default_topic.trim_start_matches('#').to_string();
            if topic_id.is_empty() {
                return Err("no current topic — use /invite <peer> #<topic>".into());
            }
            let display_name = topic_id.clone();
            (
                tokens.iter().map(|s| s.to_string()).collect(),
                topic_id,
                display_name,
            )
        }
    };
    Ok((peers, topic_id, display_name))
}

/// Random 32-byte hex string for `InvitePayload::encryption_key`
/// (reserved for future per-topic encryption).
fn random_hex_key() -> String {
    use rand::Rng;
    let mut rng = rand_core::UnwrapErr(rand::rngs::SysRng);
    let mut bytes = [0u8; 32];
    rng.fill_bytes(&mut bytes);
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grant_command_parsing() {
        assert_eq!(parse_grant_command("list"), Ok(GrantCommand::List));
        assert_eq!(parse_grant_command("pending"), Ok(GrantCommand::Pending));
        assert_eq!(
            parse_grant_command("bash perpetual"),
            Ok(GrantCommand::Grant { effect: "bash".into(), agent: None })
        );
        assert_eq!(
            parse_grant_command("repl perpetual plotter"),
            Ok(GrantCommand::Grant { effect: "repl".into(), agent: Some("plotter".into()) })
        );
        assert_eq!(
            parse_grant_command("revoke bash"),
            Ok(GrantCommand::Revoke { effect: "bash".into(), agent: None })
        );
        assert_eq!(
            parse_grant_command("revoke gpu:futhark plotter"),
            Ok(GrantCommand::Revoke {
                effect: "gpu:futhark".into(),
                agent: Some("plotter".into())
            })
        );
        assert!(parse_grant_command("").unwrap_err().contains("Usage"));
        assert!(parse_grant_command("bash").unwrap_err().contains("Usage"));
        assert!(parse_grant_command("bash forever").unwrap_err().contains("Usage"));
    }

    #[test]
    fn mesh_dialing_is_directional_capped_and_deduped() {
        // Direction: exactly one side dials.
        assert!(should_dial_for_mesh("aaa", "bbb", 0, false, 8));
        assert!(!should_dial_for_mesh("bbb", "aaa", 0, false, 8));
        assert!(!should_dial_for_mesh("aaa", "aaa", 0, false, 8), "never dial self");
        // Dedup.
        assert!(!should_dial_for_mesh("aaa", "bbb", 1, true, 8));
        // Cap.
        assert!(!should_dial_for_mesh("aaa", "zzz", 8, false, 8));
        assert!(should_dial_for_mesh("aaa", "zzz", 7, false, 8));
    }

    #[test]
    fn tier_labels_describe_effect_classes() {
        assert_eq!(tier_label(agent::effects::Tier::Wasm), "");
        assert_eq!(tier_label(agent::effects::Tier::ProcBash), " (sandboxed shell)");
        assert_eq!(tier_label(agent::effects::Tier::Gpu), " (GPU)");
    }

    #[test]
    fn parse_invite_basic() {
        let (peers, topic, display) = parse_invite_args("alice bob #book-club", "default").unwrap();
        assert_eq!(peers, vec!["alice", "bob"]);
        assert_eq!(topic, "book-club");
        assert_eq!(display, "book-club");
    }

    #[test]
    fn parse_invite_with_display_name() {
        let (peers, topic, display) =
            parse_invite_args("agent1 #book-club The Book Club", "default").unwrap();
        assert_eq!(peers, vec!["agent1"]);
        assert_eq!(topic, "book-club");
        assert_eq!(display, "The Book Club");
    }

    #[test]
    fn parse_invite_defaults_to_current_topic() {
        let (peers, topic, display) = parse_invite_args("plotter", "introductions").unwrap();
        assert_eq!(peers, vec!["plotter"]);
        assert_eq!(topic, "introductions");
        assert_eq!(display, "introductions");

        // A stray '#' on the default topic must not break anything.
        let (_, topic, _) = parse_invite_args("plotter helper", "#introductions").unwrap();
        assert_eq!(topic, "introductions");
    }

    #[test]
    fn parse_invite_requires_args() {
        assert!(
            parse_invite_args("", "introductions")
                .unwrap_err()
                .contains("usage")
        );
        // No topic given and no current topic to fall back on.
        assert!(
            parse_invite_args("alice", "")
                .unwrap_err()
                .contains("no current topic")
        );
    }

    #[test]
    fn parse_invite_requires_peers_and_topic() {
        assert!(
            parse_invite_args("#topic", "default")
                .unwrap_err()
                .contains("peer")
        );
        assert!(
            parse_invite_args("alice #", "default")
                .unwrap_err()
                .contains("empty")
        );
    }

    #[test]
    fn random_key_is_64_hex_chars() {
        let key = random_hex_key();
        assert_eq!(key.len(), 64);
        assert!(key.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn input_box_has_two_borders_and_a_status_row() {
        let theme = crate::theme::Theme::default();
        let mut buf = ratatui::buffer::Buffer::empty(ratatui::layout::Rect::new(0, 0, 40, 4));
        App::render_input_buf(&mut buf, ratatui::layout::Rect::new(0, 0, 40, 4), "hi", 2, "#t · ctx 1 msgs / 2", &theme);
        // Row 0 and row 2 are the demarcating lines; row 3 is the status.
        assert!(buf[(0, 0)].symbol() == "─");
        assert!(buf[(39, 0)].symbol() == "─");
        assert_eq!(buf[(1, 1)].symbol(), "h");
        assert!(buf[(0, 2)].symbol() == "─");
        assert!(buf[(39, 2)].symbol() == "─");
        assert!(buf[(1, 3)].symbol() == "#");
    }

    #[test]
    fn plot_paths_accept_mentions_and_tilde() {
        assert_eq!(normalize_plot_path("@~/Documents/flights.csv"), shellexpand::tilde("~/Documents/flights.csv").to_string());
        assert_eq!(normalize_plot_path("~/Documents/flights.csv"), shellexpand::tilde("~/Documents/flights.csv").to_string());
        assert_eq!(normalize_plot_path("/tmp/x.csv"), "/tmp/x.csv");
    }

    #[test]
    fn file_mentions_expand_with_csv_shape() {
        let dir = std::env::temp_dir();
        let csv = dir.join("llung-mention-test.csv");
        std::fs::write(&csv, "age,height\n30,170\n40,180\n").unwrap();

        let out = expand_file_mentions(&format!("plot @{} please", csv.display()));
        assert!(out.contains(csv.to_str().unwrap()), "{out}");
        assert!(out.contains("csv, 2 columns, 2 rows"), "{out}");

        // Agent mentions (@name) and non-files pass through untouched.
        assert_eq!(expand_file_mentions("@plotter hi"), "@plotter hi");
        assert_eq!(expand_file_mentions("@/no/such/file.csv ok"), "@/no/such/file.csv ok");

        std::fs::remove_file(&csv).ok();
    }

    #[test]
    fn tools_listing_shows_every_tool_then_details() {
        let lines = App::format_tool_listing("");
        assert!(lines[0].starts_with("Tools:"));
        for spec in agent::tools::tool_catalog() {
            assert!(
                lines.iter().any(|l| l.contains(spec.name)),
                "listing must include {}",
                spec.name
            );
        }

        // Detail view: description, usage, and the argument schema.
        let lines = App::format_tool_listing("artist");
        assert!(lines[0].starts_with("artist — "));
        assert!(lines.iter().any(|l| l.starts_with("usage: ")));
        let schema = lines
            .iter()
            .position(|l| l.starts_with("arguments (JSON schema):"))
            .expect("schema section");
        assert!(lines[schema + 1..].iter().any(|l| l.contains("\"seed\"")));
        assert!(lines[schema + 1..].iter().any(|l| l.contains("\"style\"")));

        assert!(App::format_tool_listing("nope")[0].contains("Unknown tool"));
    }

    #[test]
    fn human_len_scales() {
        assert_eq!(human_len(950), "950");
        assert_eq!(human_len(3_456), "3.5k");
        assert_eq!(human_len(1_234_567), "1.2M");
    }
}
