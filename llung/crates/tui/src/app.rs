use crate::avatar::{decode_avatar_hex, fallback_avatar, render_avatar};
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

use crate::theme::{Role, Theme};
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

    // Content
    screen: AppScreen,
    chat_panel: ChatPanel,
    registration: RegistrationPanel,
    input_text: String,
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
}

impl App {
    pub fn new(
        terminal: Terminal<CrosstermBackend<io::Stdout>>,
        config: CoreConfig,
        machine: MachineIdentity,
    ) -> Self {
        let size = terminal
            .size()
            .unwrap_or(ratatui::layout::Size::new(80, 24));
        let sidebar_mode = SidebarMode::Hidden;

        Self {
            terminal,
            config,
            machine: machine.clone(),
            cmd_tx: None,
            my_name: String::new(),
            taffy_ui: TaffyUi::new_chat_layout(size.width, size.height, false),
            theme: Theme::from_env(),
            screen: AppScreen::Splash,
            chat_panel: ChatPanel::new(String::new()),
            registration: RegistrationPanel::new(),
            input_text: String::new(),
            last_size: (0, 0),
            sidebar_mode: sidebar_mode,
            peers: Vec::new(),
            chats: vec!["introductions".to_string()],
            media: Vec::new(),
            current_topic: "introductions".to_string(),
            peer_id: machine.peer_id.to_base58(),
            known_peers: HashMap::new(),
            agent_cmd_txs: HashMap::new(),
        }
    }

    pub async fn run(&mut self, db: &mut Database) -> Result<(), Box<dyn std::error::Error>> {
        let (term_tx, mut term_rx) = mpsc::channel::<Event>(32);
        crate::events::spawn_terminal_reader(term_tx);

        // The network only exists once an account does.
        let mut event_rx: Option<mpsc::Receiver<NetworkEvent>> = None;
        let mut tick = tokio::time::interval(Duration::from_millis(50));

        // ── Splash: show a frame, then resolve the account ──
        self.draw()?;
        match db.load_being()? {
            Some(being) => {
                self.setup_identity(&being);
                event_rx = Some(self.start_network(being).await?);
                self.screen = AppScreen::Chat;
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
                _ = tick.tick() => {}
            };
            self.load_missing_avatars(db);
            self.draw()?;
        }
    }

    /// Adopt an account into the UI (chat panel title, own peer entry).
    fn setup_identity(&mut self, being: &Being) {
        self.my_name = being.human_name.clone();
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

    fn load_missing_avatars(&mut self, db: &Database) {
        let mut loaded: Vec<(String, crate::avatar::AvatarThumbnail)> = Vec::new();
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
            }
        }
        // Avatars often arrive after the messages did; retro-attach them.
        for (id, thumb) in loaded {
            self.chat_panel.update_peer_avatar(&id, &thumb);
        }
    }

    async fn handle_chat_key(
        &mut self,
        key: crossterm::event::KeyEvent,
        db: &Database,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        match key.code {
            // Typing into the input buffer
            KeyCode::Char(c) => {
                self.input_text.push(c);
            }

            // Backspace
            KeyCode::Backspace => {
                self.input_text.pop();
            }

            // Send message
            KeyCode::Enter => {
                if key.modifiers.contains(KeyModifiers::SHIFT) {
                    self.input_text.push('\n');
                } else {
                    let is_command = self.input_text.trim().starts_with('/');
                    if is_command {
                        let cmd = std::mem::take(&mut self.input_text);
                        self.handle_command(cmd.trim(), db).await?;
                    } else if !self.input_text.trim().is_empty() {
                        let text = std::mem::take(&mut self.input_text);
                        let text = text.trim().to_string();
                        let timestamp = chrono::Utc::now().timestamp();
                        let msg = ChatMessage {
                            id: generate_cid_from_bytes(
                                &(self.peer_id.clone() + &text + &timestamp.to_string()).as_bytes(),
                            )
                            .unwrap_or_else(|_| uuid::Uuid::new_v4().to_string()),
                            topic: self.current_topic.clone(),
                            parent_id: None,
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
                    }
                }
            }
            // Scroll history
            KeyCode::Up => self.chat_panel.scroll_up(1),
            KeyCode::Down => self.chat_panel.scroll_down(1),
            KeyCode::PageUp => self.chat_panel.scroll_up(10),
            KeyCode::PageDown => self.chat_panel.scroll_down(10),
            KeyCode::End => self.chat_panel.jump_to_bottom(),

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
            _ => {}
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

        // Only render if it's the current topic
        if msg.topic != self.current_topic {
            return;
        }

        let avatar = self
            .known_peers
            .get(&msg.sender_id)
            .and_then(|i| i.avatar.clone());

        match msg.kind {
            MessageKind::Human | MessageKind::Agent => {
                self.chat_panel
                    .push(msg.sender_id, msg.sender_name, msg.content, avatar, None);
            }

            MessageKind::Media(ref attachment) => {
                // Decode thumbnail if we have the bytes locally
                if attachment.kind == AttachmentKind::Image {
                    if let Ok(Some(bytes)) = db.get_avatar_cache(&attachment.cid) {
                        if let Some(thumb) = decode_avatar_hex(&bytes, 6) {
                            self.chat_panel
                                .thumb_cache
                                .insert(attachment.cid.clone(), thumb);
                        }
                    }
                }
                self.chat_panel.push(
                    msg.sender_id,
                    msg.sender_name,
                    msg.content, // caption
                    avatar,
                    Some(attachment.clone()),
                );
            }

            MessageKind::System => {
                self.chat_panel
                    .push("system".into(), "system".into(), msg.content, None, None);
            }

            MessageKind::ToolCall { .. }
            | MessageKind::ToolResponse { .. }
            | MessageKind::Notes => {
                // Agents render these; humans might ignore or show collapsed
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
        let input_text = &self.input_text;
        let screen = self.screen;
        let sidebar_mode = self.sidebar_mode;
        let peers = &self.known_peers;
        let chats = &self.chats;
        let current_topic = &self.current_topic;
        let theme = &self.theme;

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
                    Self::render_input_buf(buf, input_rect, input_text, theme);
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

    fn render_input_buf(
        buf: &mut ratatui::buffer::Buffer,
        area: Rect,
        text: &str,
        theme: &Theme,
    ) {
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

        if text.is_empty() || y0 + 1 >= y1 {
            return;
        }

        let inner_width = (x1 - x0).saturating_sub(2);
        let prefix = PrefixText::new(text.to_string());

        let max_lines = (y1 - y0 - 1) as usize;
        let lines: Vec<&str> = prefix.wrap_lines(inner_width).collect();
        let start = lines.len().saturating_sub(max_lines);
        let visible = &lines[start..];

        for (i, line) in visible.iter().enumerate() {
            let y = y0 + 1 + i as u16;
            if y >= y1 {
                break;
            }
            buf.set_string(x0 + 1, y, line, theme.style(Role::Text));
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
                crossterm::terminal::disable_raw_mode().ok();
                let _ = std::io::stdout().execute(crossterm::terminal::LeaveAlternateScreen);
                std::process::exit(0);
            }
            "/peers" => self.toggle_sidebar(SidebarMode::Peers),
            "/chats" => self.toggle_sidebar(SidebarMode::Chats),
            "/media" => self.toggle_sidebar(SidebarMode::Media),
            "/invite" => {
                if let Err(e) = self.handle_invite_command(args).await {
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
                    self.chat_panel.push(
                        "system".to_string(),
                        "system".to_string(),
                        format!("Switched to #{topic_id}"),
                        None,
                        None,
                    );
                }
            }

            "/agent" => {
                let mut parts = args.split_whitespace();
                let sub = parts.next().unwrap_or("");
                match sub {
                    "spawn" => {
                        let name = parts.next().unwrap_or("").to_string();
                        if name.is_empty() {
                            self.chat_panel.push(
                                "system".into(), "system".into(),
                                "Usage: /agent spawn <name>".into(),
                                None, None,
                            );
                            return Ok(());
                        }

                        // Provision in DB: new Being + new Machine
                        let agent_id = agent::provision_agent(db, &name)?;
            
                        // Start the agent's own network stack
                        let (agent_app, agent_event_rx) = LlungApp::init(
                            CoreConfig {
                                db_path: self.config.db_path.clone(),
                                chat_topic: self.current_topic.clone(),
                            },
                            agent_id.machine,
                            agent_id.being.clone(),
                        ).await?;

                        // Spawn the agent loop. The Aukora kernel and the
                        // wasm tool worker are wired up inside `new`; the
                        // history compressor shares the agent's model
                        // backend (with a compressor system prompt).
                        let agent_cmd_tx = agent_app.command_tx();
                        let agent_config =
                            AgentConfig { name: name.clone(), ..Default::default() };
                        let backend = OpenAiCompatible::from_config(&agent_config);
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
                        );
                        tokio::spawn(run_agent_loop(ctx, agent_event_rx));

                        // The agent's Being is already a peer (its own
                        // machine + being in the DB, listed in known_peers
                        // below). Keep only what /invite needs: a handle
                        // to tell its network stack to join a topic.
                        self.agent_cmd_txs.insert(name.clone(), agent_cmd_tx);

                        // Also add to known_peers so presence rendering works
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
                            format!("Agent '{name}' is ready. Invite it with: /invite {name} #{}", self.current_topic),
                            None, None,
                        );
                    }
                    _ => {
                        self.chat_panel.push(
                            "system".into(), "system".into(),
                            "Unknown /agent subcommand. Try: /agent spawn <name>".into(),
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
            _ => {}
        }
        Ok(())
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
        }

        let my_avatar = self
            .known_peers
            .get(&self.peer_id)
            .and_then(|info| info.avatar.clone());

        if attachment.kind == AttachmentKind::Image {
            if let Some(thumb) = decode_avatar_hex(&bytes, 6) {
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

    async fn handle_invite_command(&mut self, args: &str) -> std::result::Result<(), String> {
        let (peer_queries, topic_id, display_name) = parse_invite_args(args)?;

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

/// Parse `/invite` arguments into `(peer queries, topic id, display name)`.
/// The first `#`-prefixed token splits peers from the topic; any remaining
/// tokens become the display name (falling back to the topic id).
fn parse_invite_args(args: &str) -> std::result::Result<(Vec<String>, String, String), String> {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    let Some(hash_pos) = tokens.iter().position(|t| t.starts_with('#')) else {
        return Err("usage: /invite <peer> [peer...] #<topic> [display name]".into());
    };
    if hash_pos == 0 {
        return Err("name at least one peer to invite (see /peers)".into());
    }

    let topic_id = tokens[hash_pos].trim_start_matches('#').to_string();
    if topic_id.is_empty() {
        return Err("topic name cannot be empty".into());
    }

    let display_name = if tokens.len() > hash_pos + 1 {
        tokens[hash_pos + 1..].join(" ")
    } else {
        topic_id.clone()
    };

    let peers = tokens[..hash_pos].iter().map(|s| s.to_string()).collect();
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
    fn parse_invite_basic() {
        let (peers, topic, display) = parse_invite_args("alice bob #book-club").unwrap();
        assert_eq!(peers, vec!["alice", "bob"]);
        assert_eq!(topic, "book-club");
        assert_eq!(display, "book-club");
    }

    #[test]
    fn parse_invite_with_display_name() {
        let (peers, topic, display) = parse_invite_args("agent1 #book-club The Book Club").unwrap();
        assert_eq!(peers, vec!["agent1"]);
        assert_eq!(topic, "book-club");
        assert_eq!(display, "The Book Club");
    }

    #[test]
    fn parse_invite_requires_sigil() {
        assert!(
            parse_invite_args("alice bob")
                .unwrap_err()
                .contains("usage")
        );
    }

    #[test]
    fn parse_invite_requires_peers_and_topic() {
        assert!(parse_invite_args("#topic").unwrap_err().contains("peer"));
        assert!(parse_invite_args("alice #").unwrap_err().contains("empty"));
    }

    #[test]
    fn random_key_is_64_hex_chars() {
        let key = random_hex_key();
        assert_eq!(key.len(), 64);
        assert!(key.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
