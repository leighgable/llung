// tui/src/main.rs
use crossterm::{
    ExecutableCommand,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use llung_core::{
    config::CoreConfig,
    identity::session::resolve_account_path,
    storage::db::Database,
};
use ratatui::{Terminal, backend::CrosstermBackend};
use std::io::{Write, stdout};
use tracing_subscriber;

mod app;
mod avatar;
mod chat_panel;
mod completion;
mod events;
mod input;
mod md;
mod prefix_text;
mod registration_panel;
mod settings;
mod spinner;
mod taffy_ui;
mod terminal;
mod theme;
mod tool_panel;

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
            _ => {
                if let Some(v) = arg.strip_prefix("--theme=") {
                    cli.theme = Some(v.to_string());
                } else {
                    return Err(format!("unknown argument: {arg}\n\n{USAGE}"));
                }
            }
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

/// Where the log goes — a per-user location, never the launch
/// directory (which may be any project, or read-only like `/`):
///
/// * `LLUNG_LOG_FILE` overrides.
/// * Linux/BSD: `$XDG_STATE_HOME/llung/llung.log` (~/.local/state/…).
/// * macOS: `~/Library/Logs/llung/llung.log`.
/// * Elsewhere: the local data dir, else the temp dir.
fn log_file_path() -> std::path::PathBuf {
    if let Some(p) = std::env::var_os("LLUNG_LOG_FILE").filter(|p| !p.is_empty()) {
        return std::path::PathBuf::from(p);
    }
    let dir = if cfg!(target_os = "macos") {
        dirs::home_dir().map(|h| h.join("Library/Logs"))
    } else {
        dirs::state_dir().or_else(dirs::data_local_dir)
    }
    .unwrap_or_else(std::env::temp_dir);
    dir.join("llung").join("llung.log")
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Logging is best-effort: an unwritable log location must never
    // stop the app from starting (it used to `unwrap()` a file in the
    // launch directory).
    let log_path = log_file_path();
    if let Some(dir) = log_path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Ok(file) = std::fs::File::create(&log_path) {
        let _ = tracing_subscriber::fmt()
            .with_writer(file)
            .with_ansi(false)
            .try_init();
    }
    // Where the agents' wasm tools resolve (env, next to the exe, or the
    // source tree): the first thing to check when a tool won't load.
    tracing::info!(tools_dir = %agent::paths::tools_dir().display(), "wasm tools");

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

    // Single account: load it, adopt a legacy profile, or register one.
    // Runs before the TUI because a fresh account prints to the plain
    // terminal; afterwards everything is styled.
    let db_path = match resolve_account_path() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("account setup failed: {e}");
            std::process::exit(2);
        }
    };

    // ── Enter the TUI; everything from here is styled ──
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    // CSI u key reporting where supported: makes Shift+Enter distinct
    // from Enter (multi-line messages); cleanup_sequence pops it.
    terminal::push_keyboard_enhancement();
    let result = run_tui(db_path, theme).await;

    // ── Cleanup (also after errors) ──
    let _ = stdout().write_all(&terminal::cleanup_sequence());
    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    result
}

/// Open the single account database and hand the terminal to the App:
/// splash screen -> account load -> registration (only when the account
/// has no being yet) -> network start -> chat.
async fn run_tui(
    db_path: std::path::PathBuf,
    theme: ThemeStartup,
) -> Result<(), Box<dyn std::error::Error>> {
    let term = Terminal::new(CrosstermBackend::new(stdout()))?;

    tracing::info!("opening account at {}", db_path.display());
    let (mut db, machine) = Database::new(
        db_path.to_str().ok_or("account path is not valid UTF-8")?,
    )?;

    // The App owns everything from here: splash screen -> account load ->
    // registration (if needed) -> network start -> chat.
    let config = CoreConfig::new(db_path, "introductions".into())
        .with_relays(CoreConfig::relays_from_env());
    let mut app = App::new(term, config, machine, theme);
    app.run(&mut db).await
}
