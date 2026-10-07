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
    identity::being::{BeingKind, PresenceMessage, create_local_being},
    network::{NetworkCommand, NetworkEvent, message::ChatMessage},
    storage::db::Database,
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::Rect,
    style::{Color, Style},
};
use std::collections::HashMap;
use std::io;
use std::result::Result;
// use taffy::{Dimension, Display, FlexDirection, NodeId, Size, TaffyTree};
use tokio::{sync::mpsc, time::Duration};

#[derive(Clone, Copy, PartialEq)]
pub enum AppScreen {
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
}

pub struct App {
    terminal: Terminal<CrosstermBackend<io::Stdout>>,
    cmd_tx: mpsc::Sender<NetworkCommand>,
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
        cmd_tx: mpsc::Sender<NetworkCommand>,
        my_name: String,
        peer_id: String,
    ) -> Self {
        let size = terminal
            .size()
            .unwrap_or(ratatui::layout::Size::new(80, 24));
        let mut known_peers = HashMap::new();
        known_peers.insert(
            peer_id.clone(),
            PeerInfo {
                name: my_name.clone(),
                avatar_cid: None,
                avatar: None,
            },
        );
        let sidebar_mode = SidebarMode::Hidden;

        Self {
            terminal,
            cmd_tx,
            my_name: my_name.clone(),
            taffy_ui: TaffyUi::new_chat_layout(size.width, size.height, false),
            screen: AppScreen::Chat,
            chat_panel: ChatPanel::new(my_name),
            registration: RegistrationPanel::new(),
            input_text: String::new(),
            last_size: (0, 0),
            sidebar_mode: sidebar_mode,
            peers: Vec::new(),
            chats: Vec::new(),
            media: Vec::new(),
            current_topic: "introductions".to_string(),
            peer_id: peer_id,
            known_peers: known_peers,
        }
    }

    pub async fn run(
        &mut self,
        mut term_rx: tokio::sync::mpsc::Receiver<Event>,
        mut event_rx: tokio::sync::mpsc::Receiver<NetworkEvent>,
        db: &mut Database,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut tick = tokio::time::interval(Duration::from_millis(50));
        self.draw()?;
        loop {
            tokio::select! {
                Some(ev) = term_rx.recv() => {
                    if let Event::Key(key) = ev {
                        if key.kind != KeyEventKind::Press {
                            continue;
                        }
                        match self.screen {
                            AppScreen::Registration => {
                                if let Some(
                                    result
                                ) = self.registration.handle_key(
                                    key.code
                                ) {
                                    let avatar = if result.avatar.is_empty() {
                                        None
                                    } else {
                                        Some(result.avatar.as_str())
                                    };
                                create_local_being(
                                    db,
                                    result.name,
                                    avatar,
                                    BeingKind::Human,
                                )?;
                                self.screen = AppScreen::Chat;
                                }
                            }
                            AppScreen::Chat => {
                                self.handle_chat_key(key, db).await?;
                            }
                        }
                    }
                }
                Some(net_ev) = event_rx.recv() => {
                    self.handle_network_event(net_ev, db);
                }
                _ = tick.tick() => {}
            };
            self.load_missing_avatars(db);
            self.draw()?;
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
                        };

                        let _ = db.save_message(&msg);
                        let my_avatar = self
                            .known_peers
                            .get(&self.peer_id)
                            .and_then(|i| i.avatar.clone());
                        self.chat_panel
                            .push(self.my_name.clone(), text.clone(), my_avatar);
                        let cmd = NetworkCommand::PublishMessage {
                            topic: IdentTopic::new("introductions"),
                            contents: text.into_bytes(),
                        };
                        self.cmd_tx.send(cmd).await?;
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

    async fn handle_terminal_event(&mut self, ev: Event) -> Result<(), Box<dyn std::error::Error>> {
        if let Event::Key(key) = ev {
            if key.kind != KeyEventKind::Press {
                return Ok(());
            }
            match key.code {
                KeyCode::Char(c) => self.input_text.push(c),
                KeyCode::Backspace => {
                    self.input_text.pop();
                }
                KeyCode::Enter => {
                    let cmd = NetworkCommand::PublishMessage {
                        topic: IdentTopic::new("introductions"),
                        contents: std::mem::take(&mut self.input_text).into_bytes(),
                    };
                    self.cmd_tx.send(cmd).await;
                    self.input_text.clear();
                }
                KeyCode::Up => self.chat_panel.scroll_up(3),
                KeyCode::Down => self.chat_panel.scroll_down(3),
                KeyCode::Esc => std::process::exit(0),
                _ => {}
            }
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
                    self.known_peers.insert(
                        sender_id,
                        PeerInfo {
                            name: presence.human_name,
                            avatar_cid: presence.avatar_cid,
                            avatar: None, // lazy loaded
                        },
                    );
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
                };

                let _ = db.save_message(&msg);

                if topic == self.current_topic {
                    let avatar = self
                        .known_peers
                        .get(&sender_id)
                        .and_then(|info| info.avatar.clone());

                    self.chat_panel.push(msg.sender_name, text, avatar);
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

        self.terminal.draw(|frame| {
            let area = frame.area();
            taffy_ui.resize(area.width, area.height);

            let chat_rect = taffy_ui.node_rect(taffy_ui.chat);
            let input_rect = taffy_ui.node_rect(taffy_ui.input);
            let sidebar_rect = taffy_ui.node_rect(taffy_ui.sidebar);
            let buf = frame.buffer_mut();

            match screen {
                AppScreen::Registration => {
                    registration.render(buf, area);
                }
                AppScreen::Chat => {
                    chat_panel.render(buf, chat_rect);
                    Self::render_input_buf(buf, input_rect, input_text);
                    Self::render_sidebar_buf(buf, sidebar_rect, sidebar_mode, peers);
                }
            }
        })?;

        Ok(())
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
        self.chat_panel
            .push(format!("[DM] {}", self.resolve_name(&sender)), text, None);
    }

    pub fn handle_invite(&mut self, sender: PeerId, invite: InvitePayload) {
        // TODO: persist invite to the database once a topic-invite store exists
        let _ = sender;

        let topic = IdentTopic::new(&invite.topic_id);
        let _ = self.cmd_tx.send(NetworkCommand::SubscribeTopic { topic });

        self.chat_panel.push(
            "system".to_string(),
            format!(
                "{} invited you to '{}'",
                invite.invited_by, invite.display_name
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
                buf.set_string(
                    x0 + 1,
                    y0 + 2,
                    "#introductions",
                    Style::default().fg(Color::Green),
                );
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
        match cmd {
            "/quit" | "/q" => {
                crossterm::terminal::disable_raw_mode().ok();
                let _ = std::io::stdout().execute(crossterm::terminal::LeaveAlternateScreen);
                std::process::exit(0);
            }
            "/peers" => self.toggle_sidebar(SidebarMode::Peers),
            "/chats" => self.toggle_sidebar(SidebarMode::Chats),
            "/media" => self.toggle_sidebar(SidebarMode::Media),
            _ => {
                self.chat_panel.push(
                    "system".to_string(),
                    format!("Unknown command: {}", cmd),
                    None,
                );
            }
        }
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
