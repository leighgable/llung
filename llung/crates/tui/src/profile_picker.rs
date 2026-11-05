//! Styled profile picker, shown in the TUI before anything else.
//!
//! Runs on the same terminal as the app, before the `App` exists: the
//! app needs an open profile database, which is what this screen
//! chooses. `main` then opens the database and hands the terminal over.
//!
//! # Future: real login
//! This screen is where authentication goes. Planned shape:
//! - `PickerAction::Open` becomes "profile chosen": switch to a
//!   `Mode::Unlock { profile, passphrase }` row (masked input) instead of
//!   returning immediately.
//! - On Enter, derive a key from the passphrase (e.g. Argon2id) and use
//!   it to unlock the profile's keys / open an encrypted database;
//!   failures stay on this screen as `self.error`.
//! - `Mode::Naming` (new profile) gains a "choose passphrase" step.
//! Everything after (`Database::new`, `App`) stays unchanged.

use crate::avatar::{AvatarThumbnail, decode_avatar_hex, fallback_avatar, render_avatar};
use crate::theme::{Role, Theme};
use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use llung_core::identity::session::{ensure_profile_dir, list_identities};
use llung_core::storage::db::Database;
use ratatui::backend::CrosstermBackend;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::widgets::{Block, BorderType, Borders, Widget};
use ratatui::Terminal;
use std::io::{self, Write};
use std::path::PathBuf;

/// A profile on disk, with whatever the picker could learn about it.
#[derive(Clone)]
pub struct ProfileEntry {
    pub name: String,
    pub db_path: PathBuf,
    /// The account's display name, once registration has happened.
    pub display_name: Option<String>,
    pub avatar: Option<AvatarThumbnail>,
}

/// Profiles from the data directory, sorted by name. Display names and
/// avatars are best-effort: an unreadable database just shows its name.
pub fn load_profiles() -> Vec<ProfileEntry> {
    let mut profiles: Vec<ProfileEntry> = list_identities()
        .into_iter()
        .map(|id| {
            let (display_name, avatar) = id
                .db_path
                .to_str()
                .and_then(|p| Database::open(p).ok())
                .and_then(|db| {
                    let being = db.load_being().ok().flatten()?;
                    let avatar = being
                        .avatar_cid
                        .as_deref()
                        .and_then(|cid| db.get_avatar_cache(cid).ok().flatten())
                        .and_then(|bytes| decode_avatar_hex(&bytes, 4));
                    Some((Some(being.human_name), avatar))
                })
                .unwrap_or((None, None));
            ProfileEntry { name: id.profile_name, db_path: id.db_path, display_name, avatar }
        })
        .collect();
    profiles.sort_by(|a, b| a.name.cmp(&b.name));
    profiles
}

#[derive(Debug, PartialEq)]
pub enum PickerAction {
    /// Open an existing profile: (database path, profile name).
    Open(PathBuf, String),
    /// Create a new profile with this (validated) name.
    Create(String),
    Quit,
}

#[derive(Debug, PartialEq)]
enum Mode {
    Browse,
    Naming { input: String },
}

pub struct ProfilePicker {
    profiles: Vec<ProfileEntry>,
    /// Index into `profiles`; `profiles.len()` is the "New profile" row.
    selected: usize,
    mode: Mode,
    pub error: Option<String>,
}

const MAX_NAME: usize = 32;

/// Profile names become directory names: keep them simple.
pub fn validate_name(raw: &str, existing: &[ProfileEntry]) -> Result<String, String> {
    let name = raw.trim().to_lowercase();
    if name.is_empty() {
        return Err("Type a name for the profile".into());
    }
    if name.chars().count() > MAX_NAME {
        return Err(format!("At most {MAX_NAME} characters"));
    }
    if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
        return Err("Use letters, digits, - and _".into());
    }
    if existing.iter().any(|p| p.name == name) {
        return Err(format!("A profile named '{name}' already exists"));
    }
    Ok(name)
}

impl ProfilePicker {
    /// `last`: profile to preselect (the one used last time). With no
    /// profiles at all, opens straight into naming a new one.
    pub fn new(profiles: Vec<ProfileEntry>, last: Option<&str>) -> Self {
        let selected = last
            .and_then(|l| profiles.iter().position(|p| p.name == l))
            .unwrap_or(0);
        let mode = if profiles.is_empty() {
            Mode::Naming { input: String::new() }
        } else {
            Mode::Browse
        };
        Self { profiles, selected, mode, error: None }
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> Option<PickerAction> {
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            return Some(PickerAction::Quit);
        }
        self.error = None;
        let new_row = self.profiles.len();
        match &mut self.mode {
            Mode::Browse => match key.code {
                KeyCode::Up | KeyCode::Char('k') => self.selected = self.selected.saturating_sub(1),
                KeyCode::Down | KeyCode::Char('j') => self.selected = (self.selected + 1).min(new_row),
                KeyCode::Home => self.selected = 0,
                KeyCode::End => self.selected = new_row,
                KeyCode::Char('n') => self.start_naming(),
                KeyCode::Enter if self.selected == new_row => self.start_naming(),
                KeyCode::Enter => {
                    // LOGIN: unlock step goes here (see module docs).
                    let p = &self.profiles[self.selected];
                    return Some(PickerAction::Open(p.db_path.clone(), p.name.clone()));
                }
                KeyCode::Esc | KeyCode::Char('q') => return Some(PickerAction::Quit),
                _ => {}
            },
            Mode::Naming { input } => match key.code {
                KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                    if input.chars().count() < MAX_NAME {
                        input.push(c);
                    }
                }
                KeyCode::Backspace => {
                    input.pop();
                }
                KeyCode::Enter => match validate_name(input, &self.profiles) {
                    Ok(name) => return Some(PickerAction::Create(name)),
                    Err(e) => self.error = Some(e),
                },
                KeyCode::Esc if self.profiles.is_empty() => return Some(PickerAction::Quit),
                KeyCode::Esc => self.mode = Mode::Browse,
                _ => {}
            },
        }
        None
    }

    fn start_naming(&mut self) {
        self.selected = self.profiles.len();
        self.mode = Mode::Naming { input: String::new() };
    }

    // ── Rendering ──

    pub fn render(&self, buf: &mut Buffer, area: Rect, theme: &Theme) {
        let canvas = theme.style(Role::Canvas);
        for y in area.top()..area.bottom() {
            for x in area.left()..area.right() {
                if let Some(cell) = buf.cell_mut((x, y)) {
                    cell.reset();
                    cell.set_style(canvas);
                }
            }
        }

        let card_w = area.width.saturating_sub(4).min(56);
        if card_w < 20 || area.height < 10 {
            buf.set_stringn(area.x, area.y, "window too small", area.width as usize, theme.style(Role::Warning));
            return;
        }
        // Rows: 3 lines per profile (2 content + gap), 1 for the new row.
        let wanted_inner = self.profiles.len() as u16 * 3 + 1;
        let max_inner = area.height.saturating_sub(9).max(4);
        let inner_h = wanted_inner.min(max_inner);
        let total_h = 3 + inner_h + 2 + 2; // header, card, error + hints
        let x0 = area.x + (area.width - card_w) / 2;
        let mut y = area.y + area.height.saturating_sub(total_h) / 2;

        let center = |text: &str| x0 + card_w.saturating_sub(text.chars().count() as u16) / 2;
        buf.set_string(center("llung"), y, "llung", theme.style(Role::Logo));
        let subtitle = match self.mode {
            Mode::Browse => "choose a profile",
            Mode::Naming { .. } => "create a profile",
        };
        buf.set_string(center(subtitle), y + 1, subtitle, theme.style(Role::Muted));
        y += 3;

        let card = Rect::new(x0, y, card_w, inner_h + 2);
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(theme.style(Role::BorderMuted))
            .render(card, buf);
        let inner = Rect::new(card.x + 1, card.y + 1, card.width - 2, inner_h);

        // Scroll so the selection stays visible; the new row is pinned.
        let visible = ((inner_h.saturating_sub(1)) / 3).max(1) as usize;
        let first = self.selected.min(self.profiles.len().saturating_sub(1)).saturating_sub(visible - 1);
        let mut row_y = inner.y;
        for (i, profile) in self.profiles.iter().enumerate().skip(first).take(visible) {
            self.render_profile(buf, inner, row_y, profile, i == self.selected, theme);
            row_y += 3;
        }
        self.render_new_row(buf, Rect::new(inner.x, inner.bottom() - 1, inner.width, 1), theme);

        y = card.bottom() + 1;
        if let Some(err) = &self.error {
            buf.set_stringn(x0 + 1, y, err, card_w as usize, theme.style(Role::Error));
        }
        let hints = match self.mode {
            Mode::Browse => "↑↓ select · enter open · n new · q quit",
            Mode::Naming { .. } if self.profiles.is_empty() => "enter create · esc quit",
            Mode::Naming { .. } => "enter create · esc back",
        };
        buf.set_string(center(hints), y + 1, hints, theme.style(Role::Dim));
    }

    fn render_profile(
        &self,
        buf: &mut Buffer,
        inner: Rect,
        y: u16,
        p: &ProfileEntry,
        selected: bool,
        theme: &Theme,
    ) {
        if selected {
            fill(buf, Rect::new(inner.x, y, inner.width, 2), theme.style(Role::Selected));
            buf.set_string(inner.x + 1, y, "›", theme.style(Role::Accent));
        }
        let ax = inner.x + 3;
        match &p.avatar {
            Some(thumb) => render_avatar(buf, ax, y, thumb),
            None => {
                let initial = p.display_name.as_deref().unwrap_or(&p.name).chars().next().unwrap_or('?');
                fallback_avatar(buf, ax + 1, y, initial.to_ascii_uppercase(), theme.avatar_color(&p.name), theme.avatar_text());
            }
        }
        let tx = ax + 6;
        let room = inner.right().saturating_sub(tx) as usize;
        let (title, detail) = match &p.display_name {
            Some(d) => (d.clone(), p.name.clone()),
            None => (p.name.clone(), "not set up yet".to_string()),
        };
        let title_role = if selected { Role::PanelTitle } else { Role::Text };
        buf.set_stringn(tx, y, &title, room, theme.style(title_role));
        buf.set_stringn(tx, y + 1, &detail, room, theme.style(Role::Muted));
    }

    fn render_new_row(&self, buf: &mut Buffer, row: Rect, theme: &Theme) {
        let selected = self.selected == self.profiles.len();
        if selected {
            fill(buf, row, theme.style(Role::Selected));
            buf.set_string(row.x + 1, row.y, "›", theme.style(Role::Accent));
        }
        let x = row.x + 3;
        let room = row.right().saturating_sub(x) as usize;
        match &self.mode {
            Mode::Naming { input } => {
                buf.set_stringn(x, row.y, "name ", room, theme.style(Role::Muted));
                let text = format!("{input}▏");
                buf.set_stringn(x + 5, row.y, &text, room.saturating_sub(5), theme.style(Role::Text));
            }
            Mode::Browse => {
                buf.set_stringn(x, row.y, "+ New profile", room, theme.style(Role::Accent));
            }
        }
    }
}

fn fill(buf: &mut Buffer, r: Rect, style: ratatui::style::Style) {
    for y in r.top()..r.bottom() {
        for x in r.left()..r.right() {
            if let Some(cell) = buf.cell_mut((x, y)) {
                cell.set_style(style);
            }
        }
    }
}

/// Run the picker until a profile is chosen (or created) or the user
/// quits (`None`). Creating makes the profile directory; the database
/// itself is created when `main` opens it.
pub fn run(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    theme: &Theme,
    last: Option<&str>,
) -> io::Result<Option<(PathBuf, String)>> {
    let mut picker = ProfilePicker::new(load_profiles(), last);
    loop {
        // Avatars may have queued Kitty uploads.
        let pending = crate::terminal::kitty::take_pending();
        if !pending.is_empty() {
            let backend = terminal.backend_mut();
            backend.write_all(&pending)?;
            backend.flush()?;
        }
        terminal.draw(|f| {
            let area = f.area();
            picker.render(f.buffer_mut(), area, theme);
        })?;

        // Resize and other events just redraw.
        if let Event::Key(key) = crossterm::event::read()? {
            if key.kind != KeyEventKind::Press {
                continue;
            }
            match picker.handle_key(key) {
                Some(PickerAction::Open(path, name)) => return Ok(Some((path, name))),
                Some(PickerAction::Create(name)) => match ensure_profile_dir(&name) {
                    Ok(path) => return Ok(Some((path, name))),
                    Err(e) => picker.error = Some(format!("Couldn't create profile: {e}")),
                },
                Some(PickerAction::Quit) => return Ok(None),
                None => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(name: &str, display: Option<&str>) -> ProfileEntry {
        ProfileEntry {
            name: name.into(),
            db_path: PathBuf::from(format!("/p/{name}/node.db")),
            display_name: display.map(String::from),
            avatar: None,
        }
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn picker() -> ProfilePicker {
        ProfilePicker::new(vec![entry("alice", Some("Alice")), entry("work", None)], Some("work"))
    }

    #[test]
    fn preselects_last_profile_and_opens_it() {
        let mut p = picker();
        assert_eq!(p.selected, 1);
        assert_eq!(
            p.handle_key(key(KeyCode::Enter)),
            Some(PickerAction::Open("/p/work/node.db".into(), "work".into()))
        );
    }

    #[test]
    fn navigation_is_clamped() {
        let mut p = picker();
        p.handle_key(key(KeyCode::Up));
        p.handle_key(key(KeyCode::Up));
        assert_eq!(p.selected, 0);
        for _ in 0..5 {
            p.handle_key(key(KeyCode::Char('j')));
        }
        assert_eq!(p.selected, 2, "stops on the New profile row");
    }

    #[test]
    fn create_flow_validates() {
        let mut p = picker();
        p.handle_key(key(KeyCode::Char('n')));
        for c in "Alice".chars() {
            p.handle_key(key(KeyCode::Char(c)));
        }
        assert_eq!(p.handle_key(key(KeyCode::Enter)), None);
        assert!(p.error.as_deref().unwrap().contains("already exists"));
        for _ in 0..5 {
            p.handle_key(key(KeyCode::Backspace));
        }
        for c in "Home-Lab".chars() {
            p.handle_key(key(KeyCode::Char(c)));
        }
        assert_eq!(p.handle_key(key(KeyCode::Enter)), Some(PickerAction::Create("home-lab".into())));
    }

    #[test]
    fn esc_backs_out_or_quits() {
        let mut p = picker();
        p.handle_key(key(KeyCode::Char('n')));
        assert_eq!(p.handle_key(key(KeyCode::Esc)), None);
        assert_eq!(p.mode, Mode::Browse);
        assert_eq!(p.handle_key(key(KeyCode::Esc)), Some(PickerAction::Quit));

        let mut empty = ProfilePicker::new(Vec::new(), None);
        assert!(matches!(empty.mode, Mode::Naming { .. }), "no profiles: name one");
        assert_eq!(empty.handle_key(key(KeyCode::Esc)), Some(PickerAction::Quit));
        let ctrl_c = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert_eq!(picker().handle_key(ctrl_c), Some(PickerAction::Quit));
    }

    #[test]
    fn name_rules() {
        assert_eq!(validate_name("  Work ", &[]), Ok("work".into()));
        assert!(validate_name("", &[]).is_err());
        assert!(validate_name("a/b", &[]).is_err());
        assert!(validate_name("../x", &[]).is_err());
        assert!(validate_name(&"x".repeat(33), &[]).is_err());
    }

    fn text_of(buf: &Buffer) -> String {
        let a = buf.area;
        (0..a.height)
            .map(|y| (0..a.width).map(|x| buf[(x, y)].symbol()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn renders_profiles_selection_and_hints() {
        let theme = Theme::default();
        let p = picker();
        let area = Rect::new(0, 0, 80, 24);
        let mut buf = Buffer::empty(area);
        p.render(&mut buf, area, &theme);
        let text = text_of(&buf);
        for needle in ["llung", "choose a profile", "Alice", "alice", "work", "not set up yet", "+ New profile", "enter open"] {
            assert!(text.contains(needle), "missing {needle:?} in:\n{text}");
        }
        // The selected row ("work") carries the selection background.
        let (x, y) = text
            .lines()
            .enumerate()
            .find_map(|(y, l)| l.find("not set up yet").map(|b| (l[..b].chars().count() as u16, y as u16)))
            .unwrap();
        assert_eq!(buf[(x, y)].bg, theme.style(Role::Selected).bg.unwrap());
    }

    #[test]
    fn renders_naming_and_errors_and_tiny_windows() {
        let theme = Theme::default();
        let mut p = picker();
        p.handle_key(key(KeyCode::Char('n')));
        p.handle_key(key(KeyCode::Char('x')));
        p.error = Some("Boom".into());
        let area = Rect::new(0, 0, 80, 24);
        let mut buf = Buffer::empty(area);
        p.render(&mut buf, area, &theme);
        let text = text_of(&buf);
        assert!(text.contains("name x▏") && text.contains("Boom") && text.contains("esc back"), "{text}");

        let tiny = Rect::new(0, 0, 18, 5);
        let mut buf = Buffer::empty(tiny);
        p.render(&mut buf, tiny, &theme);
        assert!(text_of(&buf).contains("too small"));
    }

    #[test]
    fn many_profiles_scroll_with_selection() {
        let theme = Theme::default();
        let many: Vec<_> = (0..20).map(|i| entry(&format!("p{i:02}"), None)).collect();
        let p = ProfilePicker::new(many, Some("p19"));
        let area = Rect::new(0, 0, 80, 24);
        let mut buf = Buffer::empty(area);
        p.render(&mut buf, area, &theme);
        let text = text_of(&buf);
        assert!(text.contains("p19"), "selected profile scrolled into view");
        assert!(!text.contains("p00"), "top scrolled away");
        assert!(text.contains("+ New profile"), "new row pinned");
    }
}
