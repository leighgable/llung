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

/// Escape sequences to write before leaving the alternate screen: frees
/// uploaded images so they don't linger in terminal memory.
pub fn cleanup_sequence() -> Vec<u8> {
    match graphics::graphics() {
        graphics::Graphics::Kitty => kitty::cleanup_sequence(),
        graphics::Graphics::HalfBlocks => Vec::new(),
    }
}
