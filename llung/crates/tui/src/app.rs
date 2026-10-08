use crate::avatar::{fallback_avatar, render_avatar};
use crate::prefix_text::PrefixText;
use crate::registration_panel::RegistrationPanel;
use crate::{chat_panel::ChatPanel, taffy_ui::TaffyUi};
use crossterm::{
    ExecutableCommand,
    event::{Event, KeyCode, KeyEventKind, KeyModifiers},
};
use libp2p::{PeerId, gossipsub::IdentTopic};
use llung_core::utils::generate_cid_from_bytes;
use llung_core::protocol::invite::InvitePayload;
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
        message::{ChatMessage, MessageKind},
    },
    storage::db::Database,
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::Rect,
    style::{Color, Modifier, Style},
};
use std::collections::HashMap;
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
        }
    }

    pub async fn run(
        &mut self,
        db: &mut Database,
    ) -> Result<(), Box<dyn std::error::Error>> {
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
                // Disabled until the network comes up (registration complete).
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
                matches.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
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
        for info in self.known_peers.values_mut() {
            if info.avatar.is_some() || info.avatar_cid.is_none() {
                continue;
            }
            let cid = info.avatar_cid.as_ref().unwrap();
            if let Ok(Some(bytes)) = db.get_avatar_cache(cid) {
                info.avatar = crate::avatar::decode_avatar_hex(&bytes, 4);
            }
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
                        self.handle_command(cmd.trim()).await?;
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

                        let _ = db.save_message(&msg);
                        let my_avatar = self
                            .known_peers
                            .get(&self.peer_id)
                            .and_then(|i| i.avatar.clone());
                        let my_id = self.peer_id.clone();
                        self.chat_panel
                            .push(my_id, self.my_name.clone(), text.clone(), my_avatar);
                        let cmd = NetworkCommand::PublishMessage {
                            topic: IdentTopic::new(&self.current_topic),
                            contents: text.into_bytes(),
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
                let sender_id = sender.to_base58();

                if let Ok(presence) = serde_json::from_slice::<PresenceMessage>(&data) {
                    let human_name = presence.human_name;
                    self.known_peers.insert(
                        sender_id.clone(),
                        PeerInfo {
                            name: human_name.clone(),
                            avatar_cid: presence.avatar_cid,
                            avatar: None, // lazy loaded
                            enc_public_key: presence.enc_public_key,
                        },
                    );
                    // Messages received before presence get their names now.
                    self.chat_panel.update_peer_name(&sender_id, &human_name);
                    return;
                }

                let text = String::from_utf8_lossy(&data).into_owned();

                let sender_name = self
                    .known_peers
                    .get(&sender_id)
                    .map(|info| info.name.clone())
                    .unwrap_or_else(|| "Unknown".to_string());

                let timestamp = chrono::Utc::now().timestamp();
                let msg = ChatMessage {
                    id: generate_cid_from_bytes(
                        &(sender.to_base58() + &text + &timestamp.to_string()).into_bytes(),
                    )
                    .unwrap_or_else(|_| uuid::Uuid::new_v4().to_string()),
                    topic: topic.clone(),
                    parent_id: None, // Todo reply threading
                    sender_id: sender.to_base58(),
                    sender_name: sender_name.clone(),
                    content: text.clone(),
                    timestamp: timestamp as u64,
                    kind: MessageKind::Human,
                };

                let _ = db.save_message(&msg);

                if topic == self.current_topic {
                    let avatar = self
                        .known_peers
                        .get(&sender_id)
                        .and_then(|info| info.avatar.clone());

                    self.chat_panel.push(sender_id, msg.sender_name, text, avatar);
                }
            }

            NetworkEvent::DirectMessageReceived { sender, payload } => {
                self.handle_direct_message(sender, payload, db);
            }
            _ => {}
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

        self.terminal.draw(|frame| {
            let area = frame.area();
            taffy_ui.resize(area.width, area.height);

            let chat_rect = taffy_ui.node_rect(taffy_ui.chat);
            let input_rect = taffy_ui.node_rect(taffy_ui.input);
            let sidebar_rect = taffy_ui.node_rect(taffy_ui.sidebar);
            let buf = frame.buffer_mut();

            match screen {
                AppScreen::Splash => {
                    Self::render_splash(buf, area);
                }
                AppScreen::Registration => {
                    registration.render(buf, area);
                }
                AppScreen::Chat => {
                    chat_panel.render(buf, chat_rect);
                    Self::render_input_buf(buf, input_rect, input_text);
                    Self::render_sidebar_buf(
                        buf,
                        sidebar_rect,
                        sidebar_mode,
                        peers,
                        chats,
                        current_topic,
                    );
                }
            }
        })?;

        Ok(())
    }

    fn render_splash(buf: &mut ratatui::buffer::Buffer, area: Rect) {
        for y in area.y..(area.y + area.height) {
            for x in area.x..(area.x + area.width) {
                if let Some(cell) = buf.cell_mut((x, y)) {
                    cell.reset();
                    cell.set_bg(Color::Black);
                }
            }
        }

        let center = |text: &str| -> u16 {
            area.x + area.width.saturating_sub(text.len() as u16) / 2
        };
        let mid = area.y + area.height / 2;

        let logo = "llung";
        buf.set_string(
            center(logo),
            mid.saturating_sub(2),
            logo,
            Style::default().fg(Color::Cyan),
        );
        let status = "loading account…";
        buf.set_string(
            center(status),
            mid,
            status,
            Style::default().fg(Color::DarkGray),
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
        self.chat_panel.push(dm_id, dm_name, text, None);
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

    fn render_input_buf(buf: &mut ratatui::buffer::Buffer, area: Rect, text: &str) {
        let buf_area = buf.area().clone();

        let x0 = area.x.min(buf_area.width);
        let y0 = area.y.min(buf_area.height);
        let x1 = (area.x + area.width).min(buf_area.width);
        let y1 = (area.y + area.height).min(buf_area.height);

        for y in y0..y1 {
            for x in x0..x1 {
                if let Some(cell) = buf.cell_mut((x, y)) {
                    cell.reset();
                    cell.set_bg(Color::Black);
                }
            }
        }
        // Draw top border
        for x in x0..x1 {
            if let Some(cell) = buf.cell_mut((x, y0)) {
                cell.set_symbol("─");
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
            buf.set_string(x0 + 1, y, line, Style::default());
        }
    }

    fn render_sidebar_buf(
        buf: &mut ratatui::buffer::Buffer,
        area: Rect,
        mode: SidebarMode,
        peers: &HashMap<String, PeerInfo>,
        chats: &[String],
        current_topic: &str,
    ) {
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
                    cell.set_bg(Color::DarkGray);
                }
            }
        }

        match mode {
            SidebarMode::Hidden => {}
            SidebarMode::Peers => {
                buf.set_string(x0 + 1, y0, " Peers ", Style::default().fg(Color::White));
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
                        let hash = peer_id.bytes().fold(0u8, |a, b| a.wrapping_mul(b));
                        let bg = Color::Rgb(hash, hash.wrapping_mul(7), hash.wrapping_mul(13));
                        fallback_avatar(buf, x0 + 1, y, initial, bg);
                    }

                    let name = if info.name.is_empty() {
                        &peer_id[..8.min(peer_id.len())]
                    } else {
                        &info.name
                    };
                    let avail = (x1.saturating_sub(x0 + 1)) as usize;
                    buf.set_stringn(x0 + 1, y + 2, name, avail, Style::default().fg(Color::Cyan));
                }
            }
            SidebarMode::Chats => {
                buf.set_string(x0 + 1, y0, " Chats ", Style::default().fg(Color::White));
                if chats.is_empty() {
                    buf.set_string(
                        x0 + 1,
                        y0 + 2,
                        "(no chats yet)",
                        Style::default().fg(Color::DarkGray),
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
                            Style::default()
                                .fg(Color::Green)
                                .add_modifier(Modifier::BOLD)
                        } else {
                            Style::default().fg(Color::DarkGray)
                        };
                        buf.set_stringn(x0 + 1, y, format!("#{topic}"), avail, style);
                        y += 1;
                    }
                }
            }
            SidebarMode::Media => {
                buf.set_string(x0 + 1, y0, " Media ", Style::default().fg(Color::White));
                buf.set_string(
                    x0 + 1,
                    y0 + 2,
                    "(empty)",
                    Style::default().fg(Color::DarkGray),
                );
            }
        }
    }

    async fn handle_command(&mut self, cmd: &str) -> Result<(), Box<dyn std::error::Error>> {
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
                        .push("system".to_string(), "system".to_string(), e, None);
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
                    );
                } else {
                    self.join_topic(topic_id);
                    self.chat_panel.push(
                        "system".to_string(),
                        "system".to_string(),
                        format!("Switched to #{topic_id}"),
                        None,
                    );
                }
            }
            _ => {
                self.chat_panel.push(
                    "system".to_string(),
                    "system".to_string(),
                    format!("Unknown command: {cmd} (try /invite <peer>... #<topic>)"),
                    None,
                );
            }
        }
        Ok(())
    }

    /// /invite <peer> [peer...] #<topic> [display name]
    ///
    /// The `#` sigil delimits peers from the topic: everything before it is a
    /// peer list, the token itself is the topic, and anything after is an
    /// optional human-friendly name for the topic.
    async fn handle_invite_command(&mut self, args: &str) -> std::result::Result<(), String> {
        let (peer_queries, topic_id, display_name) = parse_invite_args(args)?;

        // Resolve everyone before sending anything.
        let mut targets = Vec::new();
        for q in &peer_queries {
            targets.push(self.resolve_peer(q)?);
        }

        let timestamp = chrono::Utc::now().timestamp();

        for target in &targets {
            let id = target.to_base58();
            let info = self.known_peers.get(&id);
            let name = info.map(|i| i.name.as_str()).unwrap_or(&id);
            let Some(pk) = info.and_then(|i| i.enc_public_key) else {
                return Err(format!(
                    "no encryption key for '{name}' yet — wait a few seconds for their presence"
                ));
            };

            let payload = InvitePayload {
                topic_id: topic_id.clone(),
                encryption_key: random_hex_key(),
                display_name: display_name.clone(),
                invited_by: self.my_name.clone(),
                timestamp,
            };
            let json = serde_json::to_vec(&payload).map_err(|e| e.to_string())?;
            let encrypted = encrypt_to_public_key(&pk, &json).map_err(|e| e.to_string())?;

            let tx = self.cmd_tx.as_ref().ok_or("network not connected")?;
            tx.send(NetworkCommand::SendDirectMessage {
                target: *target,
                payload: encrypted,
            })
            .await
            .map_err(|e| e.to_string())?;
        }

        // Join the new topic ourselves so we see the conversation.
        self.join_topic(&topic_id);

        self.chat_panel.push(
            "system".to_string(),
            "system".to_string(),
            format!(
                "Invited {} to '{}' — you are now in #{}",
                peer_queries.join(", "),
                display_name,
                topic_id
            ),
            None,
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
        assert!(parse_invite_args("alice bob").unwrap_err().contains("usage"));
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
