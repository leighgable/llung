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
mod settings;
mod taffy_ui;
mod terminal;
mod theme;

use app::{App, ThemeStartup};
use settings::Settings;
use theme::{ColorMode, Theme};

const USAGE: &str = "\
Usage: tui [OPTIONS]

Options:
  --theme <THEME>   Theme for this session (not saved): a name, a light/dark
                    pair, or a path to a .json theme file
  --list-themes     List available themes and exit
  -h, --help        Show this help

Environment:
  LLUNG_THEME       Theme selection, overridden by --theme
  LLUNG_COLOR_MODE  Force color mode: truecolor | 256
  LLUNG_GRAPHICS    Force image backend: kitty | halfblocks

Themes are pi-compatible JSON files, searched in ~/.config/llung/themes
and ~/.pi/agent/themes. Use /theme in the chat to switch and save.";

struct Cli {
    theme: Option<String>,
}

fn parse_args() -> Result<Cli, String> {
    let mut cli = Cli { theme: None };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            "--list-themes" => {
                for name in Theme::available() {
                    println!("{name}");
                }
                std::process::exit(0);
            }
            "--theme" => {
                cli.theme = Some(args.next().ok_or("--theme needs a value")?);
            }
            _ => match arg.strip_prefix("--theme=") {
                Some(v) => cli.theme = Some(v.to_string()),
                None => return Err(format!("unknown argument: {arg}\n\n{USAGE}")),
            },
        }
    }
    Ok(cli)
}

/// Resolve the startup theme. Precedence: `--theme` (this session only)
/// > `LLUNG_THEME` > saved setting > `system`.
///
/// A bad `--theme` is a hard error, reported before the TUI starts. A bad
/// env or saved selection falls back to `system`, with a notice shown in
/// the chat.
fn initial_theme(cli_theme: Option<String>) -> Result<ThemeStartup, String> {
    let mode = ColorMode::detect();
    let (settings, settings_err) = Settings::load();
    let mut notices: Vec<String> = settings_err
        .into_iter()
        .map(|e| format!("Settings ignored: {e}"))
        .collect();

    if let Some(sel) = cli_theme {
        let theme = Theme::select(&sel, mode).map_err(|e| format!("--theme: {e}"))?;
        return Ok(ThemeStartup { theme, selection: sel, settings, notices });
    }

    let env = std::env::var("LLUNG_THEME").ok().filter(|s| !s.trim().is_empty());
    let (source, sel) = match (&env, settings.theme()) {
        (Some(sel), _) => ("LLUNG_THEME", sel.clone()),
        (None, Some(sel)) => ("saved theme", sel.to_string()),
        (None, None) => ("default", "system".to_string()),
    };
    let (theme, selection) = match Theme::select(&sel, mode) {
        Ok(t) => (t, sel),
        Err(e) => {
            tracing::warn!("{source} '{sel}': {e}");
            notices.push(format!("{source} '{sel}' failed ({e}); using system"));
            (Theme::system(mode), "system".to_string())
        }
    };
    Ok(ThemeStartup { theme, selection, settings, notices })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = tracing_subscriber::fmt()
        .with_writer(std::fs::File::create("llung.log").unwrap())
        .with_ansi(false)
        .try_init();

    let cli = match parse_args() {
        Ok(cli) => cli,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    // Probe the terminal (colors for the `system` theme and `light/dark`,
    // Kitty graphics for inline images) before anything else reads input.
    terminal::init();
    // Resolve the theme before the profile picker, so a bad --theme fails fast.
    let theme = match initial_theme(cli.theme) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };

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
    let mut app = App::new(terminal, config, machine, theme);
    let result = app.run(&mut db).await;

    // ── 3. Cleanup ──
    let _ = stdout().write_all(&terminal::cleanup_sequence());
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
        if io::stdin().read_line(&mut input)? == 0 {
            // EOF (stdin closed/piped): don't spin on "Invalid option".
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "no profile selected"));
        }

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
