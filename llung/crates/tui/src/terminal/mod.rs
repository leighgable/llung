//! The terminal llung runs *in*: capability detection and the graphics
//! protocols used to draw into it.
//!
//! - [`query`]: one startup round-trip (colors, Kitty graphics support).
//! - [`graphics`]: picks the image backend from that report + environment.
//! - [`kitty`]: Kitty graphics protocol via Unicode placeholders.
//!
//! # Extending: embedded terminal panes (libghostty-vt)
//!
//! This module is about the *host* terminal. Embedding terminals *inside*
//! llung (e.g. a pane showing an agent's live shell) is the opposite
//! direction: llung becomes the terminal emulator for a child process.
//! Planned shape, so the pieces here stay compatible:
//!
//! - **Build:** an opt-in cargo feature (`ghostty = ["dep:libghostty-vt-sys"]`
//!   in `crates/tui/Cargo.toml`), with a `-sys` crate linking a Nix-built
//!   libghostty-vt (Zig). Only the fast machine's `devShells.full` provides
//!   it; the default build never needs Zig. Gate one module:
//!   `#[cfg(feature = "ghostty")] mod pane;` next to the modules below.
//! - **Runtime:** a PTY (e.g. `portable-pty`) running the child. Its output
//!   bytes go into libghostty-vt, which maintains the screen state (grid,
//!   styles, scrollback, cursor).
//! - **Rendering:** each frame, copy the visible grid into the pane's
//!   `Rect` of the ratatui `Buffer`, mapping the child's colors through the
//!   active `Theme` (ANSI 0-15 → the theme's palette, like the `system`
//!   theme does) so panes match the UI.
//! - **Images in panes:** when libghostty-vt reports Kitty images from the
//!   child, re-upload them through [`kitty::upload`] and emit placeholders
//!   at the pane-translated cell positions. Children then get inline
//!   images "for free" whenever the host is Kitty-capable, and half-blocks
//!   otherwise (decode the RGBA like `avatar::decode_halfblocks`).
//! - **Input:** while a pane is focused, forward key events (re-encoded as
//!   VT sequences) to the PTY instead of the chat input; resize the PTY
//!   when Taffy resizes the pane (see `taffy_ui::NodeContent`).
//! - **Without the feature:** a `TerminalPane` trait with a stub
//!   implementation that renders "embedded terminals need the `ghostty`
//!   build", so call sites need no `cfg`s.

pub mod graphics;
pub mod kitty;
pub mod query;

use std::sync::OnceLock;

/// Whether the Kitty keyboard protocol flags were pushed at startup;
/// cleanup pops them only when they were.
static KEYBOARD_ENHANCED: OnceLock<bool> = OnceLock::new();

/// Terminals that speak the Kitty keyboard protocol (progressive
/// enhancement / CSI u), which is what lets the app SEE modified keys
/// like Shift+Enter as distinct key events — without it, Shift+Enter
/// arrives byte-identical to Enter and both submit the chat message.
fn keyboard_enhancement_supported(env: &graphics::Env) -> bool {
    matches!(
        env.term_program.as_deref(),
        Some("kitty" | "ghostty" | "WezTerm" | "foot")
    ) || env.kitty_window_id.is_some()
        || env.term.as_deref().is_some_and(|t| t.contains("kitty"))
}

/// Ask the terminal to report keys with their modifiers (CSI u), so
/// Shift+Enter is distinguishable from Enter. No-op on terminals that
/// don't support the protocol. Call after entering the alternate
/// screen; [`cleanup_sequence`] pops the flags. Returns whether the
/// push happened.
pub fn push_keyboard_enhancement() -> bool {
    let supported = keyboard_enhancement_supported(&graphics::Env::from_process());
    // Even on failure (terminal ignored the sequence) the answer for
    // cleanup purposes is "don't pop what wasn't accepted" — but the
    // sequence is also how we find out, so only pop on supported
    // terminals regardless of the write result.
    let _ = KEYBOARD_ENHANCED.set(supported);
    if !supported {
        return false;
    }
    use crossterm::event::KeyboardEnhancementFlags as Flags;
    use crossterm::event::PushKeyboardEnhancementFlags;
    let flags = Flags::DISAMBIGUATE_ESCAPE_CODES | Flags::REPORT_ALL_KEYS_AS_ESCAPE_CODES;
    match crossterm::execute!(std::io::stdout(), PushKeyboardEnhancementFlags(flags)) {
        Ok(()) => true,
        Err(e) => {
            tracing::warn!("keyboard enhancement flags rejected: {e}");
            false
        }
    }
}

/// Probe the terminal once at startup and record the results for the
/// theme (`system` colors) and image rendering. Call before the TUI
/// starts reading input; briefly enters raw mode itself. Without a
/// terminal, everything falls back to defaults.
pub fn init() {
    let report = if crossterm::terminal::enable_raw_mode().is_ok() {
        let report = query::query();
        let _ = crossterm::terminal::disable_raw_mode();
        report
    } else {
        query::TerminalReport::default()
    };

    let (choice, reason) = graphics::decide(&graphics::Env::from_process(), report.kitty_graphics);
    tracing::info!("graphics: {choice:?} ({reason})");
    graphics::set(choice);
    crate::theme::system::set_terminal_colors(report.colors);
}

/// Escape sequences to write before leaving the alternate screen: pops
/// the keyboard enhancement flags and frees uploaded images so they
/// don't linger in terminal memory.
pub fn cleanup_sequence() -> Vec<u8> {
    let mut out = Vec::new();
    if KEYBOARD_ENHANCED.get().copied().unwrap_or(false) {
        out.extend_from_slice(b"\x1b[<1u"); // PopKeyboardEnhancementFlags
    }
    match graphics::graphics() {
        graphics::Graphics::Kitty => out.extend_from_slice(&kitty::cleanup_sequence()),
        graphics::Graphics::HalfBlocks => {}
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keyboard_enhancement_support_detection() {
        let env = |tp: &str, term: &str| graphics::Env {
            term_program: Some(tp.into()),
            term: Some(term.into()),
            ..Default::default()
        };
        assert!(keyboard_enhancement_supported(&env("ghostty", "xterm-ghostty")));
        assert!(keyboard_enhancement_supported(&env("kitty", "xterm-kitty")));
        assert!(keyboard_enhancement_supported(&env("WezTerm", "xterm-256color")));
        assert!(keyboard_enhancement_supported(&env("", "xterm-kitty")));
        let kitty_win = graphics::Env {
            term_program: None,
            term: Some("xterm-256color".into()),
            kitty_window_id: Some("1".into()),
            ..Default::default()
        };
        assert!(keyboard_enhancement_supported(&kitty_win));
        // Plain xterm/alacritty etc. send Shift+Enter identical to Enter.
        assert!(!keyboard_enhancement_supported(&env("Alacritty", "xterm-256color")));
        assert!(!keyboard_enhancement_supported(&env("", "xterm-256color")));
    }
}
