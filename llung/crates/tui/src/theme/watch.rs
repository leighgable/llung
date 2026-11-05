//! Hot reload for the active theme file.
//!
//! Polls the file's modification time and size, at most every
//! [`POLL_INTERVAL`]. For a single file this costs one `stat` per
//! interval. Unlike inotify-style watches it follows editors that save by
//! renaming a temp file over the original, and a save that writes in
//! several steps reloads once instead of once per write.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

pub const POLL_INTERVAL: Duration = Duration::from_millis(500);

/// File fingerprint; `None` when the file is missing.
type Stamp = Option<(SystemTime, u64)>;

fn stamp(path: &Path) -> Stamp {
    let meta = std::fs::metadata(path).ok()?;
    Some((meta.modified().ok()?, meta.len()))
}

#[derive(Debug)]
pub struct ThemeWatcher {
    path: Option<PathBuf>,
    last: Stamp,
    next_check: Instant,
}

impl ThemeWatcher {
    /// Watch `path`; `None` (built-in themes) never reports changes.
    pub fn new(path: Option<&Path>) -> Self {
        Self::new_at(path, Instant::now())
    }

    /// Clock-injected constructor, so tests don't depend on timing.
    fn new_at(path: Option<&Path>, now: Instant) -> Self {
        Self {
            last: path.and_then(stamp),
            path: path.map(Path::to_path_buf),
            next_check: now + POLL_INTERVAL,
        }
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// True when the file changed (modified, replaced, deleted or
    /// recreated) since the last report. Cheap to call often: it only
    /// checks the file every [`POLL_INTERVAL`].
    pub fn poll(&mut self) -> bool {
        self.poll_at(Instant::now())
    }

    fn poll_at(&mut self, now: Instant) -> bool {
        let Some(path) = &self.path else { return false };
        if now < self.next_check {
            return false;
        }
        self.next_check = now + POLL_INTERVAL;
        let current = stamp(path);
        if current == self.last {
            return false;
        }
        self.last = current;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_never_changes() {
        let mut w = ThemeWatcher::new(None);
        assert!(!w.poll_at(Instant::now() + POLL_INTERVAL * 2));
    }

    #[test]
    fn detects_change_delete_and_recreate() {
        let dir = std::env::temp_dir().join(format!("llung-watch-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.json");
        std::fs::write(&path, "a").unwrap();

        let mut t = Instant::now();
        let mut w = ThemeWatcher::new_at(Some(&path), t);
        let mut tick = || {
            t += POLL_INTERVAL;
            t
        };

        assert!(!w.poll_at(tick()), "unchanged");
        std::fs::write(&path, "abc").unwrap(); // size change: robust to coarse mtime
        assert!(w.poll_at(tick()), "modified");
        assert!(!w.poll_at(tick()), "reported once");
        std::fs::remove_file(&path).unwrap();
        assert!(w.poll_at(tick()), "deleted");
        std::fs::write(&path, "abcd").unwrap();
        assert!(w.poll_at(tick()), "recreated");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rate_limited() {
        let dir = std::env::temp_dir().join(format!("llung-watch-rl-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.json");
        std::fs::write(&path, "a").unwrap();
        let now = Instant::now();
        let mut w = ThemeWatcher::new_at(Some(&path), now);
        std::fs::write(&path, "ab").unwrap();
        assert!(!w.poll_at(now), "too soon");
        assert!(w.poll_at(now + POLL_INTERVAL));
        std::fs::remove_dir_all(&dir).ok();
    }
}
