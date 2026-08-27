use crate::prefix_text::PrefixText;
use crate::registration_panel::RegistrationPanel;
use crate::{chat_panel::ChatPanel, taffy_ui::TaffyUi};
use crossterm::{
    ExecutableCommand,
    event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
    terminal::{LeaveAlternateScreen, disable_raw_mode},
};
use libp2p::gossipsub::IdentTopic;
use llung_core::{
    identity::being::{BeingKind, create_local_being},
    network::{NetworkCommand, NetworkEvent},
    storage::db::Database,
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::Rect,
    style::{Color, Style},
};
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
}

impl App {
    pub fn new(
        terminal: Terminal<CrosstermBackend<io::Stdout>>,
        cmd_tx: mpsc::Sender<NetworkCommand>,
        my_name: String,
    ) -> Self {
        let size = terminal
            .size()
            .unwrap_or(ratatui::layout::Size::new(80, 24));

        Self {
            terminal,
            cmd_tx,
            my_name: my_name.clone(),
            taffy_ui: TaffyUi::new_chat_layout(size.width, size.height),
            screen: AppScreen::Chat,
            chat_panel: ChatPanel::new(my_name),
            registration: RegistrationPanel::new(),
            input_text: String::new(),
            last_size: (0, 0),
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
                                self.handle_chat_key(key).await?;
                            }
                        }
                    }
                }
                Some(net_ev) = event_rx.recv() => {
                    self.handle_network_event(net_ev);
                }
                _ = tick.tick() => {}
            }
            self.draw()?;
        }
    }

    async fn handle_chat_key(
        &mut self,
        key: crossterm::event::KeyEvent,
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
                    let trimmed = self.input_text.trim();

                    if trimmed == "/quit" || trimmed == "/q" {
                        disable_raw_mode().ok();
                        let _ = std::io::stdout().execute(LeaveAlternateScreen);
                        std::process::exit(0);
                    }

                    if !trimmed.is_empty() {
                        let text = std::mem::take(&mut self.input_text);
                        self.chat_panel.push(self.my_name.clone(), text.clone());
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

    fn handle_network_event(&mut self, ev: NetworkEvent) {
        match ev {
            NetworkEvent::MessageReceived {
                sender,
                data,
                topic,
            } => {
                let text = String::from_utf8_lossy(&data).into_owned();
                self.chat_panel.push(sender.to_string(), text);
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
                    Self::render_sidebar_buf(buf, sidebar_rect);
                }
            }
        })?;

        Ok(())
    }

    fn render_input_buf(buf: &mut ratatui::buffer::Buffer, area: Rect, text: &str) {
        let buf_area = buf.area().clone();
        let max_y = buf_area.height;
        let max_x = buf_area.width;

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

    fn render_sidebar_buf(buf: &mut ratatui::buffer::Buffer, area: Rect) {
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
                    cell.set_bg(ratatui::style::Color::DarkGray);
                }
            }
        }
        // Draw title
        buf.set_string(x0 + 1, y0, " Peers ", Style::default().fg(Color::White));
    }
}
