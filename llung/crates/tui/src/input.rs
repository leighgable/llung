//! The chat input line: cursor-based editing over a single (possibly
//! multi-line) string, plus a bounded history with shell-style recall.
//!
//! Up/Down recall older/newer entries; when there is nothing older/newer
//! the keypress falls through to the caller (the chat scrolls). Editing
//! a recalled entry resets the recall position, like bash.

/// Bound for the input history — "some previous entries", not a log.
const HISTORY_CAP: usize = 100;

#[derive(Default)]
pub struct InputLine {
    pub text: String,
    /// Byte index into `text`, always on a char boundary.
    pub cursor: usize,
    history: Vec<String>,
    /// Recall position into `history`; None = not navigating.
    pos: Option<usize>,
    /// What was in the buffer when recall started (restored on the way
    /// back down past the newest entry).
    stash: String,
}

impl InputLine {
    pub fn new() -> Self {
        Self::default()
    }

    fn set(&mut self, s: &str) {
        self.text.clear();
        self.text.push_str(s);
        self.cursor = self.text.len();
    }

    pub fn insert(&mut self, c: char) {
        self.text.insert(self.cursor, c);
        self.cursor += c.len_utf8();
        // Editing leaves recall mode (bash-style).
        self.pos = None;
    }

    /// Delete the char before the cursor. False when at the start.
    pub fn backspace(&mut self) -> bool {
        let Some(prev) = self.prev_boundary() else {
            return false;
        };
        self.text.drain(prev..self.cursor);
        self.cursor = prev;
        self.pos = None;
        true
    }

    /// Delete the char under the cursor. False when at the end.
    pub fn delete(&mut self) -> bool {
        let Some(next) = self.next_boundary() else {
            return false;
        };
        self.text.drain(self.cursor..next);
        self.pos = None;
        true
    }

    fn prev_boundary(&self) -> Option<usize> {
        if self.cursor == 0 {
            return None;
        }
        Some(self.text[..self.cursor].char_indices().last().map(|(i, _)| i).unwrap_or(0))
    }

    fn next_boundary(&self) -> Option<usize> {
        if self.cursor >= self.text.len() {
            return None;
        }
        Some(self.text[self.cursor..]
            .char_indices()
            .nth(1)
            .map(|(i, _)| self.cursor + i)
            .unwrap_or(self.text.len()))
    }

    pub fn move_left(&mut self) {
        if let Some(prev) = self.prev_boundary() {
            self.cursor = prev;
        }
    }

    pub fn move_right(&mut self) {
        if let Some(next) = self.next_boundary() {
            self.cursor = next;
        }
    }

    pub fn home(&mut self) {
        self.cursor = 0;
    }

    pub fn end(&mut self) {
        self.cursor = self.text.len();
    }

    /// Recall the previous history entry. False when already at the
    /// oldest entry (or history is empty) — the caller may use the
    /// keypress for something else (chat scroll).
    pub fn recall_older(&mut self) -> bool {
        if self.history.is_empty() {
            return false;
        }
        match self.pos {
            None => {
                self.stash = self.text.clone();
                let i = self.history.len() - 1;
                self.pos = Some(i);
                let entry = self.history[i].clone();
                self.set(&entry);
                true
            }
            Some(0) => false,
            Some(i) => {
                let i = i - 1;
                self.pos = Some(i);
                let entry = self.history[i].clone();
                self.set(&entry);
                true
            }
        }
    }

    /// Recall the next history entry; past the newest, the stashed
    /// in-progress buffer is restored. False when not navigating.
    pub fn recall_newer(&mut self) -> bool {
        match self.pos {
            None => false,
            Some(i) if i + 1 >= self.history.len() => {
                let stash = std::mem::take(&mut self.stash);
                self.set(&stash);
                self.pos = None;
                true
            }
            Some(i) => {
                let i = i + 1;
                self.pos = Some(i);
                let entry = self.history[i].clone();
                self.set(&entry);
                true
            }
        }
    }

    /// Record a submitted buffer: trimmed into the history (consecutive
    /// duplicates collapsed, bounded), then cleared.
    pub fn submit(&mut self) {
        let line = self.text.trim().to_string();
        if !line.is_empty() && self.history.last() != Some(&line) {
            self.history.push(line);
            if self.history.len() > HISTORY_CAP {
                self.history.remove(0);
            }
        }
        self.text.clear();
        self.cursor = 0;
        self.pos = None;
        self.stash.clear();
    }

    #[cfg(test)]
    pub fn history_len(&self) -> usize {
        self.history.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cursor_insert_and_delete_mid_line() {
        let mut i = InputLine::new();
        i.insert('a');
        i.insert('b');
        i.insert('c');
        i.move_left();
        i.insert('X');
        assert_eq!(i.text, "abXc");
        assert!(i.backspace());
        assert_eq!(i.text, "abc");
        i.move_left();
        i.delete();
        assert_eq!(i.text, "ac");
        // At the start: backspace is a no-op.
        i.home();
        assert!(!i.backspace());
        assert_eq!(i.text, "ac");
        i.end();
        assert!(!i.delete());
    }

    #[test]
    fn history_recall_and_stash_restore() {
        let mut i = InputLine::new();
        i.set("first");
        i.submit();
        i.set("second");
        i.submit();

        // Type something, then recall: it is stashed, not lost.
        i.set("in progress");
        assert!(i.recall_older());
        assert_eq!(i.text, "second");
        assert!(i.recall_older());
        assert_eq!(i.text, "first");
        // Oldest: nothing more.
        assert!(!i.recall_older());
        // Walk back down past the newest: the stash reappears.
        assert!(i.recall_newer());
        assert_eq!(i.text, "second");
        assert!(i.recall_newer());
        assert_eq!(i.text, "in progress");
        // Not navigating anymore.
        assert!(!i.recall_newer());
    }

    #[test]
    fn empty_history_falls_through() {
        let mut i = InputLine::new();
        assert!(!i.recall_older());
        assert!(!i.recall_newer());
    }

    #[test]
    fn submit_dedupes_and_bounds() {
        let mut i = InputLine::new();
        i.set("same");
        i.submit();
        i.set("same");
        i.submit();
        assert_eq!(i.history_len(), 1);
        for n in 0..150 {
            i.set(&n.to_string());
            i.submit();
        }
        assert_eq!(i.history_len(), HISTORY_CAP);
    }

    #[test]
    fn editing_resets_recall() {
        let mut i = InputLine::new();
        i.set("one");
        i.submit();
        assert!(i.recall_older());
        assert_eq!(i.text, "one");
        i.insert('!');
        assert_eq!(i.text, "one!");
        // Back to live editing: recall starts fresh from the stash.
        assert!(i.recall_older());
        assert_eq!(i.text, "one");
    }
}
