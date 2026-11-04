//! Semantic theming, modelled on pi's themes.
//!
//! Renderers never name concrete colors. They ask for a [`Role`], and the
//! active [`Theme`] resolves it to a ratatui [`Style`]. Resolution is a
//! field lookup, so it costs nothing on the hot render path, and caches
//! (e.g. the chat wrap cache) can store roles and stay valid across theme
//! switches.
//!
//! Themes are pi theme JSON files (see [`file`]): the built-in `dark` and
//! `light` are pi's own, and any pi theme loads unchanged.

mod color;
pub mod file;
mod oklab;

use std::path::{Path, PathBuf};

pub use color::ColorMode;
pub use file::{Appearance, LlungColors, Palette, ThemeError};
use ratatui::style::{Color, Modifier, Style};

const BUILTIN_DARK: &str = include_str!("../../themes/dark.json");
const BUILTIN_LIGHT: &str = include_str!("../../themes/light.json");

/// A resolved theme, ready for rendering.
#[derive(Clone, Debug)]
pub struct Theme {
    pub name: String,
    pub appearance: Appearance,
    pub mode: ColorMode,
    /// pi's color tokens.
    pub colors: Palette,
    /// llung-specific colors (`llung.*` vars).
    pub llung: LlungColors,
    /// File the theme was loaded from (`None` for built-ins).
    pub source: Option<PathBuf>,
}

/// What a piece of UI *is*, independent of how it looks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)] // full role set; not every role has a consumer yet
pub enum Role {
    /// Unstyled: inherit whatever is already in the cell.
    Plain,
    Text,
    Muted,
    Dim,
    Accent,
    Success,
    Error,
    Warning,

    /// Background fills.
    Canvas,
    Panel,

    Border,
    BorderAccent,
    BorderMuted,

    /// Title text of a panel / box ("Peers", "Chats", …).
    PanelTitle,
    /// Focused input field.
    Selected,
    /// Current item in a list (e.g. active topic).
    ActiveItem,
    /// Other items in a list.
    InactiveItem,

    MyMessage,
    PeerMessage,
    PeerName,

    AttachmentPending,
    AttachmentAudio,
    AttachmentVideo,
    AttachmentDocument,
    AttachmentMeta,

    Button,
    ButtonFocused,
    Logo,
}

impl Theme {
    /// Resolve a role to a concrete style.
    pub fn style(&self, role: Role) -> Style {
        let c = &self.colors;
        let x = &self.llung;
        let s = Style::default();
        match role {
            Role::Plain => s,
            Role::Text => s.fg(c.text),
            Role::Muted => s.fg(c.muted),
            Role::Dim => s.fg(c.dim),
            Role::Accent => s.fg(c.accent),
            Role::Success => s.fg(c.success),
            Role::Error => s.fg(c.error),
            Role::Warning => s.fg(c.warning),

            Role::Canvas => s.fg(c.text).bg(x.bg),
            Role::Panel => s.fg(c.text).bg(x.panel_bg),

            Role::Border => s.fg(c.border),
            Role::BorderAccent => s.fg(c.border_accent),
            Role::BorderMuted => s.fg(c.border_muted),

            Role::PanelTitle => s.fg(c.text).add_modifier(Modifier::BOLD),
            Role::Selected => s.fg(c.text).bg(c.selected_bg),
            Role::ActiveItem => s.fg(c.accent).add_modifier(Modifier::BOLD),
            Role::InactiveItem => s.fg(c.muted),

            Role::MyMessage => s.fg(c.user_message_text).bg(c.user_message_bg),
            Role::PeerMessage => s.fg(x.peer_message_text),
            Role::PeerName => s.fg(x.peer_name).add_modifier(Modifier::BOLD),

            Role::AttachmentPending => s.fg(c.warning),
            Role::AttachmentAudio => s.fg(c.success),
            Role::AttachmentVideo => s.fg(c.accent),
            Role::AttachmentDocument => s.fg(c.md_link),
            Role::AttachmentMeta => s.fg(c.muted),

            Role::Button => s.fg(c.success),
            Role::ButtonFocused => s
                .fg(self.concrete_bg())
                .bg(c.success)
                .add_modifier(Modifier::BOLD),
            Role::Logo => s.fg(c.accent).add_modifier(Modifier::BOLD),
        }
    }

    /// Convenience: background color of a fill role.
    pub fn bg(&self, role: Role) -> Color {
        self.style(role).bg.unwrap_or(Color::Reset)
    }

    /// Canvas background, or black/white by appearance when it's the
    /// terminal default (where a concrete color is needed, e.g. text on a
    /// colored button).
    fn concrete_bg(&self) -> Color {
        match (self.llung.bg, self.appearance) {
            (Color::Reset, Appearance::Dark) => self.mode.adapt(Color::Rgb(0, 0, 0)),
            (Color::Reset, Appearance::Light) => self.mode.adapt(Color::Rgb(255, 255, 255)),
            (c, _) => c,
        }
    }

    /// Fallback avatar background for a peer. The hue comes from `seed`,
    /// OKHSL saturation and lightness from the theme (`llung.avatar`), so
    /// generated avatars look equally vivid across hues and fit the palette.
    pub fn avatar_color(&self, seed: &str) -> Color {
        // FNV-1a: stable, well-distributed, no deps.
        let hash = seed.bytes().fold(0x811c_9dc5u32, |h, b| {
            (h ^ b as u32).wrapping_mul(0x0100_0193)
        });
        let (s, l) = self.llung.avatar_sl;
        self.mode
            .adapt(color::rgb(oklab::okhsl_to_rgb((hash % 360) as f64, s, l)))
    }

    /// Foreground for text drawn on top of [`Self::avatar_color`].
    pub fn avatar_text(&self) -> Color {
        if self.llung.avatar_sl.1 > 0.6 {
            Color::Black
        } else {
            Color::White
        }
    }

    // ── Loading ──

    /// Parse a theme from JSON.
    pub fn from_json(json: &str, mode: ColorMode) -> Result<Self, ThemeError> {
        let r = file::parse(json, mode)?;
        Ok(Self {
            name: r.name,
            appearance: r.appearance,
            mode,
            colors: r.colors,
            llung: r.llung,
            source: None,
        })
    }

    /// Load a theme file.
    pub fn from_path(path: &Path, mode: ColorMode) -> Result<Self, ThemeError> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| ThemeError(format!("{}: {e}", path.display())))?;
        let mut theme = Self::from_json(&json, mode)
            .map_err(|e| ThemeError(format!("{}: {e}", path.display())))?;
        theme.source = Some(path.to_path_buf());
        Ok(theme)
    }

    pub fn dark() -> Self {
        Self::builtin("dark", ColorMode::detect()).expect("built-in dark theme")
    }

    #[allow(dead_code)] // used by the upcoming `/theme` command
    pub fn light() -> Self {
        Self::builtin("light", ColorMode::detect()).expect("built-in light theme")
    }

    fn builtin(name: &str, mode: ColorMode) -> Option<Self> {
        let json = match name {
            "dark" => BUILTIN_DARK,
            "light" => BUILTIN_LIGHT,
            _ => return None,
        };
        Some(Self::from_json(json, mode).expect("built-in themes are valid"))
    }

    /// Look up a theme by name: built-ins first, then `<name>.json` in each
    /// of [`theme_dirs`].
    pub fn load(name: &str, mode: ColorMode) -> Result<Self, ThemeError> {
        if let Some(t) = Self::builtin(name, mode) {
            return Ok(t);
        }
        if name.contains('/') || name.contains('\\') || name.starts_with('.') {
            return Err(ThemeError(format!("invalid theme name: {name}")));
        }
        for dir in theme_dirs() {
            let path = dir.join(format!("{name}.json"));
            if path.is_file() {
                let theme = Self::from_path(&path, mode)?;
                if theme.name != name {
                    tracing::warn!(
                        "{}: theme name \"{}\" doesn't match filename",
                        path.display(),
                        theme.name
                    );
                }
                return Ok(theme);
            }
        }
        Err(ThemeError(format!("theme not found: {name}")))
    }

    /// Theme selected via `LLUNG_THEME` (a name, or a path to a `.json`
    /// file), falling back to `dark` with a logged warning on any error.
    /// Interim until the theme becomes a persisted config setting.
    pub fn from_env() -> Self {
        let mode = ColorMode::detect();
        let Ok(sel) = std::env::var("LLUNG_THEME") else {
            return Self::dark();
        };
        let result = if sel.ends_with(".json") {
            let path = shellexpand::tilde(&sel).into_owned();
            Self::from_path(Path::new(&path), mode)
        } else {
            Self::load(&sel, mode)
        };
        result.unwrap_or_else(|e| {
            tracing::warn!("theme: {e}; using dark");
            Self::dark()
        })
    }

    /// Names of all available themes (built-ins first, then user themes).
    #[allow(dead_code)] // used by the upcoming `/theme` command
    pub fn available() -> Vec<String> {
        let mut names = vec!["dark".to_string(), "light".to_string()];
        for dir in theme_dirs() {
            let Ok(entries) = std::fs::read_dir(&dir) else {
                continue;
            };
            let mut found: Vec<String> = entries
                .filter_map(|e| e.ok())
                .filter_map(|e| {
                    let p = e.path();
                    (p.extension()? == "json").then(|| p.file_stem()?.to_str().map(String::from))?
                })
                .filter(|n| !names.contains(n))
                .collect();
            found.sort();
            names.extend(found);
        }
        names
    }
}

impl Default for Theme {
    fn default() -> Self {
        Self::dark()
    }
}

/// Theme search path, highest priority first:
/// 1. `$XDG_CONFIG_HOME/llung/themes` (default `~/.config/llung/themes`)
/// 2. pi's user themes: `$PI_CODING_AGENT_DIR/themes` (default
///    `~/.pi/agent/themes`), so existing pi themes just work.
pub fn theme_dirs() -> Vec<PathBuf> {
    let env = |k: &str| {
        std::env::var_os(k)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    let home = env("HOME");
    let mut dirs = Vec::new();
    if let Some(cfg) = env("XDG_CONFIG_HOME").or_else(|| home.as_ref().map(|h| h.join(".config"))) {
        dirs.push(cfg.join("llung/themes"));
    }
    if let Some(pi) =
        env("PI_CODING_AGENT_DIR").or_else(|| home.as_ref().map(|h| h.join(".pi/agent")))
    {
        dirs.push(pi.join("themes"));
    }
    dirs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtins_resolve() {
        let d = Theme::builtin("dark", ColorMode::TrueColor).unwrap();
        assert_eq!(d.name, "dark");
        assert_eq!(d.appearance, Appearance::Dark);
        let l = Theme::builtin("light", ColorMode::TrueColor).unwrap();
        assert_eq!(l.appearance, Appearance::Light);
        assert!(Theme::builtin("nope", ColorMode::TrueColor).is_none());
    }

    #[test]
    fn load_rejects_paths() {
        assert!(Theme::load("../etc/passwd", ColorMode::TrueColor).is_err());
        assert!(Theme::load("definitely-not-a-theme-xyz", ColorMode::TrueColor).is_err());
    }

    #[test]
    fn from_path_reads_file() {
        let dir = std::env::temp_dir().join(format!("llung-theme-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mine.json");
        std::fs::write(&path, BUILTIN_DARK.replacen("\"dark\"", "\"mine\"", 1)).unwrap();
        let t = Theme::from_path(&path, ColorMode::TrueColor).unwrap();
        assert_eq!(t.name, "mine");
        assert_eq!(t.source.as_deref(), Some(path.as_path()));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn plain_is_unstyled() {
        let t = Theme::builtin("dark", ColorMode::TrueColor).unwrap();
        assert_eq!(t.style(Role::Plain), Style::default());
    }

    #[test]
    fn roles_use_palette() {
        let t = Theme::builtin("dark", ColorMode::TrueColor).unwrap();
        assert_eq!(t.style(Role::Accent).fg, Some(t.colors.accent));
        assert_eq!(t.bg(Role::MyMessage), t.colors.user_message_bg);
        assert_eq!(t.bg(Role::Panel), t.llung.panel_bg);
        assert_eq!(t.bg(Role::Canvas), Color::Reset);
        // Focused button needs a concrete fg even with a terminal-default bg.
        assert_eq!(t.style(Role::ButtonFocused).fg, Some(Color::Rgb(0, 0, 0)));
    }

    #[test]
    fn avatar_color_is_stable_and_varied() {
        let t = Theme::builtin("dark", ColorMode::TrueColor).unwrap();
        assert_eq!(t.avatar_color("alice"), t.avatar_color("alice"));
        assert_ne!(t.avatar_color("alice"), t.avatar_color("bob"));
        assert_eq!(t.avatar_text(), Color::White);
        let l = Theme::builtin("light", ColorMode::TrueColor).unwrap();
        assert_eq!(l.avatar_text(), Color::Black);
    }
}
