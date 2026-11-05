// tui/src/main.rs
use crossterm::{
    ExecutableCommand,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use llung_core::{
    config::CoreConfig,
    identity::session::{list_identities, profile_db_path},
    storage::db::Database,
};
use ratatui::{Terminal, backend::CrosstermBackend};
use std::io::{Write, stdout};
use tracing_subscriber;

mod app;
mod avatar;
mod chat_panel;
mod events;
mod prefix_text;
mod profile_picker;
mod registration_panel;
mod settings;
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
  --profile <NAME>  Open this profile directly, skipping the picker
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
    profile: Option<String>,
}

fn parse_args() -> Result<Cli, String> {
    let mut cli = Cli { theme: None, profile: None };
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
            "--profile" => {
                cli.profile = Some(args.next().ok_or("--profile needs a value")?);
            }
            _ => {
                if let Some(v) = arg.strip_prefix("--theme=") {
                    cli.theme = Some(v.to_string());
                } else if let Some(v) = arg.strip_prefix("--profile=") {
                    cli.profile = Some(v.to_string());
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

    // `--profile` must name an existing profile; checked before the TUI
    // starts so the error lands in the normal terminal.
    let direct_profile = match cli.profile {
        Some(name) => {
            if !list_identities().iter().any(|p| p.profile_name == name) {
                eprintln!("no profile named '{name}' (start without --profile to create one)");
                std::process::exit(2);
            }
            Some((profile_db_path(&name), name))
        }
        None => None,
    };

    // ── Enter the TUI; everything from here is styled ──
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let result = run_tui(direct_profile, theme).await;

    // ── Cleanup (also after errors) ──
    let _ = stdout().write_all(&terminal::cleanup_sequence());
    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    result
}

/// Profile picker (unless `--profile` chose already), then the app, on
/// one terminal: the picker hands it over without leaving the TUI.
async fn run_tui(
    direct_profile: Option<(std::path::PathBuf, String)>,
    mut theme: ThemeStartup,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut term = Terminal::new(CrosstermBackend::new(stdout()))?;

    let (db_path, profile) = match direct_profile {
        Some(p) => p,
        None => {
            let last = theme.settings.last_profile().map(String::from);
            match profile_picker::run(&mut term, &theme.theme, last.as_deref())? {
                Some(p) => p,
                None => return Ok(()), // quit from the picker
            }
        }
    };
    if let Err(e) = theme.settings.set_last_profile(&profile) {
        tracing::warn!("couldn't remember last profile: {e}");
    }
    tracing::info!("opening profile '{profile}' at {}", db_path.display());

    let (mut db, machine) = Database::new(
        db_path.to_str().ok_or("profile path is not valid UTF-8")?,
    )?;

    // The App owns everything from here: splash screen -> account load ->
    // registration (if needed) -> network start -> chat.
    let config = CoreConfig::new(db_path, "introductions".into());
    let mut app = App::new(term, config, machine, theme);
    app.run(&mut db).await
}
