// tui/src/main.rs
use crossterm::{
    ExecutableCommand,
    event::{Event, KeyEventKind},
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use llung_core::{
    app::LlungApp,
    config::CoreConfig,
    identity::{
        being::{BeingKind, create_local_being},
        session::{ensure_profile_dir, list_identities},
    },
    network::NetworkEvent,
    storage::db::Database,
};
use ratatui::{Terminal, backend::CrosstermBackend};
use std::io::{self, Write, stdout};
use tokio::sync::mpsc;

mod app;
mod avatar;
mod chat_panel;
mod events;
mod prefix_text;
mod registration_panel;
mod taffy_ui;

use app::App;
use registration_panel::RegistrationPanel;
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = tracing_subscriber::fmt()
        .with_writer(std::fs::File::create("llung.log").unwrap())
        .with_ansi(false)
        .try_init();

    // ── 1. Profile selection in NORMAL terminal mode ──
    let db_path = run_tui_session().await?;
    let (mut db, machine) = Database::new(db_path.to_str().unwrap())?;

    // ── 2. Setup terminal ONCE (now we're in TUI mode) ──
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout());
    let mut terminal = Terminal::new(backend)?;

    // ── 3. Load or create Being (TUI registration if needed) ──
    let being = match db.load_being()? {
        Some(b) => b,
        None => {
            let mut reg = RegistrationPanel::new();
            loop {
                terminal.draw(|frame| {
                    let area = frame.area();
                    reg.render(frame.buffer_mut(), area);
                })?;

                // Blocking read is fine here; registration is short-lived
                if let Event::Key(key) = crossterm::event::read()? {
                    if key.kind == KeyEventKind::Press {
                        if let Some(result) = reg.handle_key(key.code) {
                            let avatar = if result.avatar.is_empty() {
                                None
                            } else {
                                Some(result.avatar.as_str())
                            };
                            break create_local_being(&db, result.name, avatar, BeingKind::Human)?;
                        }
                    }
                }
            }
        }
    };

    // ── 4. Start network core ──
    let my_name = being.human_name.clone();
    let my_peer_id = machine.peer_id.to_base58();
    let (llung_app, event_rx) = LlungApp::init(
        CoreConfig {
            db_path,
            chat_topic: "introductions".into(),
        },
        machine,
        being,
    )
    .await?;

    let cmd_tx = llung_app.command_tx();

    // ── 5. Hand terminal to App and run chat ──
    let mut app = App::new(terminal, cmd_tx, my_name, my_peer_id.clone());

    let (term_tx, term_rx) = mpsc::channel::<crossterm::event::Event>(32);
    events::spawn_terminal_reader(term_tx);

    let result = app.run(term_rx, event_rx, &mut db).await;

    // ── 6. Cleanup ──
    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    result
}

// ── Stdin profile picker (runs before raw mode) ──
async fn run_tui_session() -> io::Result<std::path::PathBuf> {
    let identities = list_identities();

    if identities.is_empty() {
        println!("No existing profiles found.");
        return run_tui_registration().await;
    }

    println!("\n=== Llung TUI ===\n");
    for (i, id) in identities.iter().enumerate() {
        println!("  {}. {}", i + 1, id.profile_name);
    }
    let next = identities.len() + 1;
    println!("  {}. Create new profile", next);
    println!("  {}. Exit", next + 1);

    loop {
        print!("\nSelect: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        match input.trim().parse::<usize>() {
            Ok(n) if n >= 1 && n <= identities.len() => {
                return Ok(identities[n - 1].db_path.clone());
            }
            Ok(n) if n == next => return run_tui_registration().await,
            Ok(n) if n == next + 1 => std::process::exit(0),
            _ => println!("Invalid option, try again."),
        }
    }
}

async fn run_tui_registration() -> io::Result<std::path::PathBuf> {
    print!("Choose a profile name: ");
    io::stdout().flush()?;

    let mut name = String::new();
    io::stdin().read_line(&mut name)?;
    let name = name.trim().to_lowercase();

    if name.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "empty name"));
    }

    let db_path = ensure_profile_dir(&name)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

    println!("Created profile '{}'.", name);
    Ok(db_path)
}
