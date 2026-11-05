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
pub mod system;
pub mod watch;

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
    /// `system` theme on a terminal that reported no colors: secondary
    /// neutral text uses the terminal default plus SGR 2 (faint) instead
    /// of a concrete gray, as in pi.
    pub faint_neutrals: bool,
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
        // Faint secondary text (no-report `system` theme only).
        let f = |st: Style| {
            if self.faint_neutrals { st.add_modifier(Modifier::DIM) } else { st }
        };
        match role {
            Role::Plain => s,
            Role::Text => s.fg(c.text),
            Role::Muted => f(s.fg(c.muted)),
            Role::Dim => f(s.fg(c.dim)),
            Role::Accent => s.fg(c.accent),
            Role::Success => s.fg(c.success),
            Role::Error => s.fg(c.error),
            Role::Warning => s.fg(c.warning),

            Role::Canvas => s.fg(c.text).bg(x.bg),
            Role::Panel => s.fg(c.text).bg(x.panel_bg),

            Role::Border => s.fg(c.border),
            Role::BorderAccent => s.fg(c.border_accent),
            Role::BorderMuted => f(s.fg(c.border_muted)),

            Role::PanelTitle => s.fg(c.text).add_modifier(Modifier::BOLD),
            Role::Selected => s.fg(c.text).bg(c.selected_bg),
            Role::ActiveItem => s.fg(c.accent).add_modifier(Modifier::BOLD),
            Role::InactiveItem => f(s.fg(c.muted)),

            Role::MyMessage => s.fg(c.user_message_text).bg(c.user_message_bg),
            Role::PeerMessage => s.fg(x.peer_message_text),
            Role::PeerName => s.fg(x.peer_name).add_modifier(Modifier::BOLD),

            Role::AttachmentPending => s.fg(c.warning),
            Role::AttachmentAudio => s.fg(c.success),
            Role::AttachmentVideo => s.fg(c.accent),
            Role::AttachmentDocument => s.fg(c.md_link),
            Role::AttachmentMeta => f(s.fg(c.muted)),

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
        Ok(Self::from_resolved(file::parse(json, mode)?, mode))
    }

    fn from_resolved(r: file::ResolvedTheme, mode: ColorMode) -> Self {
        Self {
            name: r.name,
            appearance: r.appearance,
            mode,
            colors: r.colors,
            llung: r.llung,
            source: None,
            faint_neutrals: false,
        }
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

    /// The `system` theme, generated from the colors the terminal reported
    /// at startup (see [`query`]). Like pi, this is the default theme.
    pub fn system(mode: ColorMode) -> Self {
        let generated = system::generate(&system::terminal_colors(), colorfgbg_appearance());
        let resolved = file::parse_system(generated.theme, mode)
            .expect("generated system theme is valid");
        let mut theme = Self::from_resolved(resolved, mode);
        theme.faint_neutrals = generated.faint_neutrals;
        theme
    }

    fn builtin(name: &str, mode: ColorMode) -> Option<Self> {
        let json = match name {
            system::SYSTEM_THEME_NAME => return Some(Self::system(mode)),
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

    /// Resolve a theme *selection* as written in settings, `--theme` or
    /// `/theme`:
    /// - a path to a `.json` file (`~` expanded),
    /// - a `light/dark` pair, picked by [`terminal_appearance`],
    /// - a theme name (see [`Theme::load`]).
    pub fn select(selection: &str, mode: ColorMode) -> Result<Self, ThemeError> {
        let sel = selection.trim();
        if sel.is_empty() {
            return Err(ThemeError("empty theme selection".into()));
        }
        if sel.ends_with(".json") {
            let path = shellexpand::tilde(sel).into_owned();
            return Self::from_path(Path::new(&path), mode);
        }
        if let Some((light, dark)) = sel.split_once('/') {
            let name = match terminal_appearance() {
                Appearance::Light => light.trim(),
                Appearance::Dark => dark.trim(),
            };
            return Self::load(name, mode);
        }
        Self::load(sel, mode)
    }

    /// Names of all available themes (built-ins first, then user themes).
    pub fn available() -> Vec<String> {
        let mut names: Vec<String> = ["system", "dark", "light"].map(String::from).to_vec();
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
        Self::system(ColorMode::detect())
    }
}

/// Theme search path, highest priority first:
/// 1. `<config_dir>/themes` (default `~/.config/llung/themes`)
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
    if let Some(cfg) = crate::settings::config_dir() {
        dirs.push(cfg.join("themes"));
    }
    if let Some(pi) =
        env("PI_CODING_AGENT_DIR").or_else(|| home.as_ref().map(|h| h.join(".pi/agent")))
    {
        dirs.push(pi.join("themes"));
    }
    dirs
}

/// Whether the terminal has a light or dark background: from the colors
/// it reported at startup (OSC 10/11), else `COLORFGBG` (`"fg;bg"`, set by
/// rxvt, Konsole, iTerm2 and others), else dark. Picks the side of a
/// `light/dark` theme pair.
pub fn terminal_appearance() -> Appearance {
    let reported = system::terminal_colors();
    match reported.background {
        Some(bg) => system::terminal_appearance(bg, reported.foreground),
        None => colorfgbg_appearance(),
    }
}

fn colorfgbg_appearance() -> Appearance {
    appearance_from_colorfgbg(std::env::var("COLORFGBG").ok().as_deref())
}

fn appearance_from_colorfgbg(value: Option<&str>) -> Appearance {
    let bg = value
        .and_then(|v| v.rsplit(';').next())
        .and_then(|b| b.trim().parse::<u8>().ok());
    match bg {
        // ANSI white / bright colors except bright black (8).
        Some(7) | Some(9..=15) => Appearance::Light,
        _ => Appearance::Dark,
    }
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
    fn select_forms() {
        let m = ColorMode::TrueColor;
        assert_eq!(Theme::select("light", m).unwrap().name, "light");
        let pair = Theme::select("light/dark", m).unwrap();
        assert!(pair.name == "light" || pair.name == "dark");
        assert!(Theme::select("", m).is_err());
        assert!(Theme::select("/nonexistent/x.json", m).is_err());
        // Exactly one side of a pair is used; a bad name on that side fails.
        assert!(Theme::select("nope/nope", m).is_err());
    }

    #[test]
    fn colorfgbg() {
        assert_eq!(appearance_from_colorfgbg(None), Appearance::Dark);
        assert_eq!(appearance_from_colorfgbg(Some("15;0")), Appearance::Dark);
        assert_eq!(appearance_from_colorfgbg(Some("0;15")), Appearance::Light);
        assert_eq!(appearance_from_colorfgbg(Some("0;default;7")), Appearance::Light);
        assert_eq!(appearance_from_colorfgbg(Some("7;8")), Appearance::Dark);
        assert_eq!(appearance_from_colorfgbg(Some("garbage")), Appearance::Dark);
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
