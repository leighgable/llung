// tui/src/main.rs
use crossterm::{
    ExecutableCommand,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use llung_core::{
    config::CoreConfig,
    identity::session::{ensure_profile_dir, list_identities},
    storage::db::Database,
};
use ratatui::{Terminal, backend::CrosstermBackend};
use std::io::{self, Write, stdout};
use tracing_subscriber;

mod app;
mod avatar;
mod chat_panel;
mod events;
mod prefix_text;
mod registration_panel;
mod taffy_ui;

use app::App;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = tracing_subscriber::fmt()
        .with_writer(std::fs::File::create("llung.log").unwrap())
        .with_ansi(false)
        .try_init();

    // ── 1. Profile selection in NORMAL terminal mode ──
    let db_path = run_profile_picker().await?;
    let (mut db, machine) = Database::new(db_path.to_str().unwrap())?;

    // ── 2. Enter the TUI ──
    // The App owns everything from here: splash screen -> account load ->
    // registration (if needed) -> network start -> chat.
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    let config = CoreConfig::new(db_path, "introductions".into());
    let mut app = App::new(terminal, config, machine);
    let result = app.run(&mut db).await;

    // ── 3. Cleanup ──
    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    result
}

// ── Stdin profile picker (runs before raw mode) ──
async fn run_profile_picker() -> io::Result<std::path::PathBuf> {
    let identities = list_identities();

    if identities.is_empty() {
        println!("No existing profiles found.");
        return register_new_profile().await;
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
            Ok(n) if n == next => return register_new_profile().await,
            Ok(n) if n == next + 1 => std::process::exit(0),
            _ => println!("Invalid option, try again."),
        }
    }
}

async fn register_new_profile() -> io::Result<std::path::PathBuf> {
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
