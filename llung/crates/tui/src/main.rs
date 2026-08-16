mod app;
mod events;

use llung_core::{
    app::{CoreConfig, LlungApp},
    identity::session,
};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. TUI profile picker (renders a List widget, not stdin prompts)
    let db_path = tui_session_picker().await?; // you implement this with ratatui

    // 2. Init core
    let (core, event_rx) = LlungApp::init(CoreConfig {
        db_path,
        agent: None,
        chat_topic: "introductions".into(),
    })
    .await?;

    // 3. Setup terminal
    // crossterm::terminal::enable_raw_mode()?;
    // let mut terminal = ratatui::init();

    // 4. Bridge crossterm events into tokio
    let (term_tx, term_rx) = mpsc::channel(32);
    events::spawn_terminal_reader(term_tx);

    // 5. Run TUI app
    let mut app = TuiApp {
        screen: AppScreen::Chat,
        messages: Vec::new(),
        input: String::new(),
        event_rx,
        cmd_tx: core.command_tx(),
    };
    app.run(term_rx).await?;

    Ok(())
}
