//! A braille pixel spinner for in-progress work (pending tool calls).
//!
//! Pure text — braille patterns are ordinary Unicode, so this animates
//! identically in Ghostty, Zellij, and plain image-free terminals. One
//! global frame counter advanced by the app's tick; layouts read the
//! current frame. The counter lives here (not in App state) so
//! `ToolBlock::layout` — which has no access to the app — can read it,
//! and so tests can step frames deterministically.

use std::sync::atomic::{AtomicU64, Ordering};

/// cli-spinners' classic "braille" cycle: a dot orbits the 2x4 cell.
pub const FRAMES: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

static FRAME: AtomicU64 = AtomicU64::new(0);

/// Advance the animation one step. Called from the app's tick.
pub fn tick() {
    FRAME.fetch_add(1, Ordering::Relaxed);
}

/// The frame to draw right now.
pub fn frame() -> char {
    FRAMES[(FRAME.load(Ordering::Relaxed) as usize) % FRAMES.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cycles_through_frames() {
        let start = FRAME.load(Ordering::Relaxed);
        assert!(FRAMES.contains(&frame()));
        tick();
        assert_eq!(
            frame(),
            FRAMES[((start + 1) as usize) % FRAMES.len()]
        );
        // Full cycle returns to the same frame.
        for _ in 0..FRAMES.len() - 1 {
            tick();
        }
        assert_eq!(frame(), FRAMES[(start as usize) % FRAMES.len()]);
    }
}
