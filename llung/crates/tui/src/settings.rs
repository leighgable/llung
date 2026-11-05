//! Persistent tui settings: `$XDG_CONFIG_HOME/llung/settings.json`
//! (default `~/.config/llung/settings.json`), modelled on pi's settings
//! file.
//!
//! Keys this version doesn't know are preserved on save, so older and
//! newer builds can share one file.

use serde_json::{Map, Value};
use std::io;
use std::path::{Path, PathBuf};

/// `$XDG_CONFIG_HOME/llung` (default `~/.config/llung`).
pub fn config_dir() -> Option<PathBuf> {
    let env = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    env("XDG_CONFIG_HOME")
        .or_else(|| env("HOME").map(|h| h.join(".config")))
        .map(|c| c.join("llung"))
}

#[derive(Debug, Default)]
pub struct Settings {
    path: Option<PathBuf>,
    data: Map<String, Value>,
}

impl Settings {
    /// Load from the default location. A missing file is an empty
    /// settings set. An unreadable or invalid file is also loaded empty,
    /// with the error returned so it can be shown; saving then overwrites
    /// the bad file.
    pub fn load() -> (Self, Option<String>) {
        match config_dir() {
            Some(dir) => Self::load_from(&dir.join("settings.json")),
            None => (Self::default(), None),
        }
    }

    pub fn load_from(path: &Path) -> (Self, Option<String>) {
        let mut settings = Self {
            path: Some(path.to_path_buf()),
            data: Map::new(),
        };
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return (settings, None),
            Err(e) => return (settings, Some(format!("{}: {e}", path.display()))),
        };
        match serde_json::from_str::<Value>(&text) {
            Ok(Value::Object(map)) => settings.data = map,
            Ok(_) => return (settings, Some(format!("{}: expected a JSON object", path.display()))),
            Err(e) => return (settings, Some(format!("{}: {e}", path.display()))),
        }
        (settings, None)
    }

    /// Theme selection: a name, a `light/dark` pair, or a path to a
    /// `.json` file.
    pub fn theme(&self) -> Option<&str> {
        self.data.get("theme")?.as_str().filter(|s| !s.trim().is_empty())
    }

    pub fn set_theme(&mut self, theme: &str) -> io::Result<()> {
        self.data.insert("theme".into(), Value::String(theme.to_string()));
        self.save()
    }

    /// Write atomically (temp file + rename) so a crash never leaves a
    /// truncated settings file.
    fn save(&self) -> io::Result<()> {
        let Some(path) = &self.path else {
            return Err(io::Error::new(io::ErrorKind::NotFound, "no config directory ($HOME unset)"));
        };
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut json = serde_json::to_string_pretty(&self.data).map_err(io::Error::other)?;
        json.push('\n');
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, json)?;
        std::fs::rename(&tmp, path)
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("llung-settings-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("settings.json")
    }

    #[test]
    fn missing_file_is_empty() {
        let path = temp("missing");
        std::fs::remove_file(&path).ok();
        let (s, err) = Settings::load_from(&path);
        assert!(err.is_none());
        assert_eq!(s.theme(), None);
    }

    #[test]
    fn round_trip_preserves_unknown_keys() {
        let path = temp("roundtrip");
        std::fs::write(&path, r#"{"future": {"x": 1}, "theme": "dark"}"#).unwrap();
        let (mut s, err) = Settings::load_from(&path);
        assert!(err.is_none());
        assert_eq!(s.theme(), Some("dark"));
        s.set_theme("light/dark").unwrap();

        let (s2, _) = Settings::load_from(&path);
        assert_eq!(s2.theme(), Some("light/dark"));
        assert_eq!(s2.data["future"]["x"], 1);
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn invalid_file_reports_error() {
        let path = temp("invalid");
        std::fs::write(&path, "{ nope").unwrap();
        let (s, err) = Settings::load_from(&path);
        assert!(err.is_some());
        assert_eq!(s.theme(), None);
        std::fs::write(&path, "[1]").unwrap();
        assert!(Settings::load_from(&path).1.unwrap().contains("object"));
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }
}
