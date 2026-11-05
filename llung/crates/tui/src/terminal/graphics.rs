//! Which image backend to use, decided once at startup.
//!
//! The decision is about the *terminal*, not the machine or the build:
//! the same binary shows real images in Ghostty (also over SSH from it)
//! and falls back to half-block cells in a plain ANSI terminal.

use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Graphics {
    /// Kitty graphics protocol with Unicode placeholders (Ghostty, Kitty).
    Kitty,
    /// `▀` cells with fg/bg colors: works in any truecolor/256 terminal.
    HalfBlocks,
}

static GRAPHICS: OnceLock<Graphics> = OnceLock::new();

/// The active backend; half-blocks until [`set`] is called.
pub fn graphics() -> Graphics {
    GRAPHICS.get().copied().unwrap_or(Graphics::HalfBlocks)
}

pub fn set(g: Graphics) {
    let _ = GRAPHICS.set(g);
}

/// The environment variables the decision reads, injectable for tests.
#[derive(Default)]
pub struct Env {
    pub llung_graphics: Option<String>,
    pub term: Option<String>,
    pub term_program: Option<String>,
    pub kitty_window_id: Option<String>,
    pub tmux: Option<String>,
}

impl Env {
    pub fn from_process() -> Self {
        let var = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        Self {
            llung_graphics: var("LLUNG_GRAPHICS"),
            term: var("TERM"),
            term_program: var("TERM_PROGRAM"),
            kitty_window_id: var("KITTY_WINDOW_ID"),
            tmux: var("TMUX"),
        }
    }
}

/// Choose the backend. `kitty_ok`: the terminal answered the graphics
/// query. Returns the choice and a human-readable reason (logged).
///
/// The query proves the protocol exists, but not that the terminal
/// supports *Unicode placeholders*, which we rely on (several terminals
/// implement the protocol partially). So Kitty mode also requires a
/// terminal known to support them. `LLUNG_GRAPHICS=kitty|halfblocks`
/// overrides everything.
pub fn decide(env: &Env, kitty_ok: bool) -> (Graphics, String) {
    match env.llung_graphics.as_deref().map(str::to_ascii_lowercase).as_deref() {
        Some("kitty") => return (Graphics::Kitty, "forced by LLUNG_GRAPHICS".into()),
        Some("halfblocks") => return (Graphics::HalfBlocks, "forced by LLUNG_GRAPHICS".into()),
        Some(other) => tracing::warn!("LLUNG_GRAPHICS={other} unknown (kitty|halfblocks)"),
        None => {}
    }

    let term = env.term.as_deref().unwrap_or("");
    // tmux/screen need DCS passthrough and re-anchoring of placeholders;
    // not supported (yet).
    if env.tmux.is_some() || term.starts_with("tmux") || term.starts_with("screen") {
        return (Graphics::HalfBlocks, "inside tmux/screen".into());
    }
    if !kitty_ok {
        return (Graphics::HalfBlocks, "terminal did not answer the Kitty graphics query".into());
    }

    // TERM survives SSH (when the terminfo entry is installed remotely);
    // TERM_PROGRAM and KITTY_WINDOW_ID usually only exist locally.
    let known = term == "xterm-ghostty"
        || term == "xterm-kitty"
        || env.term_program.as_deref() == Some("ghostty")
        || env.kitty_window_id.is_some();
    if known {
        (Graphics::Kitty, format!("Kitty graphics with placeholders (TERM={term})"))
    } else {
        (
            Graphics::HalfBlocks,
            "Kitty graphics answered, but Unicode placeholder support is unverified \
             for this terminal; set LLUNG_GRAPHICS=kitty to try"
                .into(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(term: &str) -> Env {
        Env { term: Some(term.into()), ..Default::default() }
    }

    #[test]
    fn ghostty_and_kitty() {
        assert_eq!(decide(&env("xterm-ghostty"), true).0, Graphics::Kitty);
        assert_eq!(decide(&env("xterm-kitty"), true).0, Graphics::Kitty);
        let local = Env {
            term: Some("xterm-256color".into()),
            term_program: Some("ghostty".into()),
            ..Default::default()
        };
        assert_eq!(decide(&local, true).0, Graphics::Kitty);
    }

    #[test]
    fn needs_query_answer() {
        assert_eq!(decide(&env("xterm-ghostty"), false).0, Graphics::HalfBlocks);
    }

    #[test]
    fn unknown_terminal_is_conservative() {
        assert_eq!(decide(&env("xterm-256color"), true).0, Graphics::HalfBlocks);
    }

    #[test]
    fn tmux_falls_back() {
        let e = Env { tmux: Some("/tmp/tmux-1000/default,1,0".into()), ..env("xterm-ghostty") };
        assert_eq!(decide(&e, true).0, Graphics::HalfBlocks);
        assert_eq!(decide(&env("tmux-256color"), true).0, Graphics::HalfBlocks);
    }

    #[test]
    fn overrides() {
        let force = |v: &str, term: &str| Env { llung_graphics: Some(v.into()), ..env(term) };
        assert_eq!(decide(&force("kitty", "xterm"), false).0, Graphics::Kitty);
        assert_eq!(decide(&force("HalfBlocks", "xterm-ghostty"), true).0, Graphics::HalfBlocks);
    }
}
