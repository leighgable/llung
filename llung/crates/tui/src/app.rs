use llung_core::{app::LlungApp, network::NetworkEvent};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph};
use tokio::sync::mpsc;

pub enum AppScreen {
    SessionPicker, // list profiles / create new
    Chat,
}

pub struct TuiApp {
    pub screen: AppScreen,
    pub messages: Vec<String>,
    pub input: String,
    pub event_rx: mpsc::Receiver<NetworkEvent>,
    pub cmd_tx: mpsc::Sender<llung_core::network::NetworkCommand>,
}

impl TuiApp {
    pub async fn run(
        &mut self,
        mut term_rx: mpsc::Receiver<crossterm::event::Event>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut tick = tokio::time::interval(tokio::time::Duration::from_millis(250));

        loop {
            tokio::select! {
                Some(term_ev) = term_rx.recv() => {
                    self.handle_terminal_event(term_ev).await?;
                }
                Some(net_ev) = self.event_rx.recv() => {
                    self.handle_network_event(net_ev);
                }
                _ = tick.tick() => { /* redraw on next frame */ }
            }

            // Draw frame
            // terminal.draw(|f| ui::draw(f, self))?;
        }
    }

    fn handle_network_event(&mut self, ev: NetworkEvent) {
        match ev {
            NetworkEvent::MessageReceived { sender, data, .. } => {
                let text = String::from_utf8_lossy(&data);
                self.messages.push(format!("[{}]: {}", sender, text));
            }
            _ => {}
        }
    }

    async fn handle_terminal_event(
        &mut self,
        ev: crossterm::event::Event,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use crossterm::event::{Event::Key, KeyCode, KeyEventKind};
        if let Key(key) = ev {
            if key.kind != KeyEventKind::Press {
                return Ok(());
            }
            match key.code {
                KeyCode::Char(c) => self.input.push(c),
                KeyCode::Backspace => {
                    self.input.pop();
                }
                KeyCode::Enter => {
                    let line = std::mem::take(&mut self.input);
                    let cmd = llung_core::network::NetworkCommand::PublishMessage {
                        topic: libp2p::gossipsub::IdentTopic::new("introductions"),
                        contents: line.into_bytes(),
                    };
                    self.cmd_tx.send(cmd).await?;
                }
                KeyCode::Esc => std::process::exit(0),
                _ => {}
            }
        }
        Ok(())
    }
}
