use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use crate::completion::{self, Completion, Item};
use crate::theme::{Role, Theme};
use ratatui::text::Line;
use ratatui::widgets::{Block, Borders, Paragraph, Widget};
use std::path::PathBuf;

pub struct RegistrationPanel {
    pub display_name: String,
    pub avatar_path: String,
    pub focus: FieldFocus,
    pub error_message: Option<String>,
    /// File picker for the avatar field (pi-style): open whenever the
    /// avatar field is focused and its text is a completable path
    /// fragment. Derived from `avatar_path` after every edit, so there
    /// is no state to keep in sync with the field.
    completion: Option<Completion>,
    /// Set by Esc / by accepting a finished path: the picker stays
    /// hidden until the text is edited again or the field is re-focused.
    picker_dismissed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FieldFocus {
    Name,
    Avatar,
    Submit,
}

pub struct RegistrationResult {
    pub name: String,
    /// Raw (unexpanded) path as typed by the user; empty = no avatar.
    pub avatar: String,
}

impl RegistrationPanel {
    pub fn new() -> Self {
        Self {
            display_name: String::new(),
            avatar_path: String::new(),
            focus: FieldFocus::Name,
            error_message: None,
            completion: None,
            picker_dismissed: false,
        }
    }

    /// Re-derive the avatar picker from the field text. Selection
    /// survives a recompute while the same directory is still listed
    /// (typing a filter fragment must not reset the cursor).
    fn refresh_picker(&mut self) {
        if self.focus != FieldFocus::Avatar || self.picker_dismissed {
            self.completion = None;
            return;
        }
        let prev = self.completion.take();
        let mut next = completion::for_path_field(&self.avatar_path);
        if let (Some(p), Some(n)) = (&prev, &mut next)
            && p.title == n.title
            && p.replace_start == n.replace_start
        {
            n.selected = p.selected.min(n.items.len().saturating_sub(1));
        }
        self.completion = next;
    }


    /// Substitute the picked entry into the field. Directories keep the
    /// picker open to complete inside them; a finished file path closes
    /// it (the status line below the field takes over the feedback).
    fn accept_picker(&mut self, item: &Item) {
        self.avatar_path = item.replace.clone();
        if item.keep_open {
            self.refresh_picker();
        } else {
            self.completion = None;
            self.picker_dismissed = true;
        }
    }

    pub fn handle_key(&mut self, code: crossterm::event::KeyCode) -> Option<RegistrationResult> {
        use crossterm::event::KeyCode;

        // The avatar picker owns these keys while open (same behaviour
        // as the chat input's picker): arrows navigate, Enter/Tab pick,
        // Esc closes. Everything else edits the field normally.
        if self.completion.is_some() {
            match code {
                KeyCode::Up => {
                    if let Some(c) = &mut self.completion {
                        c.move_selection(-1);
                    }
                    return None;
                }
                KeyCode::Down => {
                    if let Some(c) = &mut self.completion {
                        c.move_selection(1);
                    }
                    return None;
                }
                KeyCode::Esc => {
                    self.picker_dismissed = true;
                    self.completion = None;
                    return None;
                }
                KeyCode::Enter | KeyCode::Tab => {
                    let item = self
                        .completion
                        .as_ref()
                        .map(|c| c.items[c.selected].clone());
                    if let Some(item) = item {
                        self.accept_picker(&item);
                    }
                    return None;
                }
                _ => {}
            }
        }

        match code {
            KeyCode::Tab => self.cycle_focus(),
            KeyCode::BackTab => self.cycle_focus_back(),
            KeyCode::Char(c) if self.focus == FieldFocus::Name => {
                self.error_message = None;
                self.display_name.push(c);
            }
            KeyCode::Char(c) if self.focus == FieldFocus::Avatar => {
                self.error_message = None;
                // Editing re-enables the picker after an Esc-dismissal.
                self.picker_dismissed = false;
                self.avatar_path.push(c);
            }
            KeyCode::Backspace if self.focus == FieldFocus::Name => {
                self.display_name.pop();
            }
            KeyCode::Backspace if self.focus == FieldFocus::Avatar => {
                self.picker_dismissed = false;
                self.avatar_path.pop();
            }
            KeyCode::Enter if self.focus == FieldFocus::Submit => match self.validate() {
                Ok(result) => return Some(result),
                Err(e) => self.error_message = Some(e),
            },
            _ => {}
        }
        self.refresh_picker();
        None
    }

    /// The avatar path with `~` expanded and relative paths made absolute
    /// (relative to the directory llung was launched from).
    pub fn resolved_avatar_path(&self) -> Option<PathBuf> {
        let raw = self.avatar_path.trim();
        if raw.is_empty() {
            return None;
        }

        let expanded = if raw == "~" {
            home_dir()?
        } else if let Some(rest) = raw.strip_prefix("~/") {
            home_dir()?.join(rest)
        } else {
            let path = PathBuf::from(raw);
            // Show (and later open) an absolute path so there is never
            // any doubt about where the file is being looked up.
            if path.is_absolute() {
                path
            } else {
                std::env::current_dir().ok()?.join(path)
            }
        };

        Some(expanded)
    }

    /// Live feedback for the avatar field: `(line, ok)`.
    pub fn avatar_status(&self) -> Option<(String, bool)> {
        let resolved = self.resolved_avatar_path()?;
        let display = resolved.display().to_string();

        if !resolved.exists() {
            Some((format!("✗ not found: {display}"), false))
        } else if !resolved.is_file() {
            Some((format!("✗ not a regular file: {display}"), false))
        } else {
            let kb = std::fs::metadata(&resolved).map(|m| m.len() / 1024).unwrap_or(0);
            Some((format!("✓ {display} ({kb} KB)"), true))
        }
    }

    fn validate(&self) -> std::result::Result<RegistrationResult, String> {
        if self.display_name.trim().is_empty() {
            return Err("Display name is required".into());
        }

        if let Some(resolved) = self.resolved_avatar_path() {
            if !resolved.is_file() {
                return Err(format!(
                    "Avatar file not found: {}",
                    resolved.display()
                ));
            }
            // Fail here, in the form, instead of deep inside account creation.
            image::open(&resolved).map_err(|e| {
                format!("'{}' is not a readable image: {e}", resolved.display())
            })?;
        }

        Ok(RegistrationResult {
            name: self.display_name.trim().to_string(),
            avatar: self.avatar_path.trim().to_string(),
        })
    }

    fn cycle_focus(&mut self) {
        self.focus = match self.focus {
            FieldFocus::Name => FieldFocus::Avatar,
            FieldFocus::Avatar => FieldFocus::Submit,
            FieldFocus::Submit => FieldFocus::Name,
        };
        self.on_focus_change();
    }

    fn cycle_focus_back(&mut self) {
        self.focus = match self.focus {
            FieldFocus::Name => FieldFocus::Submit,
            FieldFocus::Avatar => FieldFocus::Name,
            FieldFocus::Submit => FieldFocus::Avatar,
        };
        self.on_focus_change();
    }

    /// Re-entering the avatar field re-arms its picker (an Esc or a
    /// finished path dismisses it until the text is edited).
    fn on_focus_change(&mut self) {
        if self.focus == FieldFocus::Avatar {
            self.picker_dismissed = false;
        } else {
            self.completion = None;
        }
    }

    pub fn render(&self, buf: &mut Buffer, area: Rect, theme: &Theme) {
        let canvas_bg = theme.bg(Role::Canvas);
        // Clear area
        for y in area.y..(area.y + area.height) {
            for x in area.x..(area.x + area.width) {
                if let Some(cell) = buf.cell_mut((x, y)) {
                    cell.reset();
                    cell.set_bg(canvas_bg);
                }
            }
        }

        let block = Block::default()
            .title(Line::styled(" Create Your Identity ", theme.style(Role::PanelTitle)))
            .borders(Borders::ALL)
            .border_style(theme.style(Role::BorderAccent));

        let inner = block.inner(area);
        block.render(area, buf);

        let name_style = if self.focus == FieldFocus::Name {
            theme.style(Role::Selected)
        } else {
            theme.style(Role::Text)
        };
        let name_para = Paragraph::new(format!("Name: {}", self.display_name)).style(name_style);
        name_para.render(Rect::new(inner.x, inner.y + 1, inner.width, 1), buf);

        let avatar_style = if self.focus == FieldFocus::Avatar {
            theme.style(Role::Selected)
        } else {
            theme.style(Role::Text)
        };
        let avatar_para =
            Paragraph::new(format!("Avatar (optional): {}", self.avatar_path)).style(avatar_style);
        avatar_para.render(Rect::new(inner.x, inner.y + 3, inner.width, 1), buf);

        // Live resolution status, so the user can see *where* the app is looking.
        if let Some((line, ok)) = self.avatar_status() {
            let status_style = if ok {
                theme.style(Role::Success)
            } else {
                theme.style(Role::Error)
            };
            let status = Paragraph::new(line).style(status_style);
            status.render(Rect::new(inner.x + 2, inner.y + 4, inner.width.saturating_sub(2), 1), buf);
        }

        // The avatar picker grows downward from the field; the submit
        // button and error line yield the space (nothing else moves).
        let picker_rows = self
            .completion
            .as_ref()
            .map_or(0, |c| c.visible_rows() as u16);
        if let Some(comp) = &self.completion {
            completion::render_picker(
                buf,
                Rect::new(inner.x, inner.y + 5, inner.width, picker_rows),
                comp,
                theme,
            );
        }
        let submit_y = inner.y + 5 + picker_rows + 1; // one blank row of gap

        let submit_style = if self.focus == FieldFocus::Submit {
            theme.style(Role::ButtonFocused)
        } else {
            theme.style(Role::Button)
        };
        let submit = Paragraph::new(" [ Submit ] ").style(submit_style);
        submit.render(Rect::new(inner.x, submit_y, inner.width, 1), buf);

        if let Some(ref err) = self.error_message {
            let err_para = Paragraph::new(err.as_str()).style(theme.style(Role::Error));
            err_para.render(Rect::new(inner.x, submit_y + 2, inner.width, 1), buf);
        }

        let cwd = std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "<unknown>".to_string());
        let hint = Paragraph::new(format!(
            "Tab switches fields · picker: ↑↓ choose, Enter pick, Esc close · relative paths resolve from {cwd} · ~ is expanded"
        ))
        .style(theme.style(Role::Dim));
        hint.render(
            Rect::new(
                inner.x,
                inner.y + inner.height.saturating_sub(1),
                inner.width,
                1,
            ),
            buf,
        );
    }
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ../../data/1.jpg relative to crates/tui is the repo's sample image.
    const SAMPLE: &str = "../../data/1.jpg";

    fn panel_with_avatar(path: &str) -> RegistrationPanel {
        let mut panel = RegistrationPanel::new();
        panel.avatar_path = path.to_string();
        panel
    }

    #[test]
    fn relative_avatar_path_resolves_from_cwd() {
        // The repo's sample image is optional content: without it the
        // resolution half is still checkable, the rest is skipped.
        let sample_present = std::path::Path::new(SAMPLE).is_file();
        let panel = panel_with_avatar(SAMPLE);
        let resolved = panel.resolved_avatar_path().expect("should resolve");
        assert!(resolved.is_absolute(), "{resolved:?} should be absolute");
        assert!(resolved.ends_with("data/1.jpg"), "unexpected: {resolved:?}");
        if !sample_present {
            eprintln!("sample image {resolved:?} not present; skipping the read half");
            return;
        }

        let (line, ok) = panel.avatar_status().expect("status should exist");
        assert!(ok, "status should be ok: {line}");
        assert!(line.contains("✓"));
    }

    #[test]
    fn missing_avatar_path_reports_not_found() {
        let panel = panel_with_avatar("data/nope-does-not-exist.jpg");
        let resolved = panel.resolved_avatar_path().expect("should resolve");
        assert!(!resolved.exists());

        let (line, ok) = panel.avatar_status().expect("status should exist");
        assert!(!ok);
        assert!(line.contains("✗ not found"));
        assert!(line.contains(resolved.to_str().unwrap()), "status shows the looked-up path");

        // Submitting with a missing file must fail with the resolved path in the error.
        assert!(panel.validate().is_err());
    }

    #[test]
    fn empty_avatar_is_allowed() {
        let mut panel = RegistrationPanel::new();
        assert!(panel.resolved_avatar_path().is_none());
        assert!(panel.avatar_status().is_none());
        assert!(panel.validate().is_err()); // name still required

        panel.display_name = "Alice".to_string();
        let result = panel.validate().expect("empty avatar should validate");
        assert!(result.avatar.is_empty());
    }

    fn type_chars(panel: &mut RegistrationPanel, s: &str) {
        for c in s.chars() {
            panel.handle_key(crossterm::event::KeyCode::Char(c));
        }
    }

    #[test]
    fn focusing_avatar_opens_the_file_picker() {
        let mut panel = RegistrationPanel::new();
        assert!(panel.completion.is_none());

        // Tab into the avatar field: the picker lists the launch dir.
        panel.handle_key(crossterm::event::KeyCode::Tab);
        assert_eq!(panel.focus, FieldFocus::Avatar);
        let c = panel.completion.as_ref().expect("picker should open");
        assert!(c.items.iter().any(|i| i.label == "Cargo.toml"));
        assert!(c.items.first().unwrap().keep_open, "dirs sort first");
    }

    #[test]
    fn picker_filters_as_you_type_and_accepts_a_file() {
        let mut panel = RegistrationPanel::new();
        panel.handle_key(crossterm::event::KeyCode::Tab); // -> Avatar
        type_chars(&mut panel, "Carg");
        let c = panel.completion.as_ref().expect("picker stays open");
        assert_eq!(c.items.len(), 1);
        assert_eq!(c.items[0].label, "Cargo.toml");

        // Enter picks the file: field set, picker closed (the ✓ status
        // line takes over the feedback).
        panel.handle_key(crossterm::event::KeyCode::Enter);
        assert_eq!(panel.avatar_path, "Cargo.toml");
        assert!(panel.completion.is_none());
        let (line, ok) = panel.avatar_status().expect("status");
        assert!(ok, "{line}");
    }

    #[test]
    fn picking_a_directory_descends_and_esc_closes_until_edited() {
        let mut panel = RegistrationPanel::new();
        panel.handle_key(crossterm::event::KeyCode::Tab); // -> Avatar
        type_chars(&mut panel, "sr");
        panel.handle_key(crossterm::event::KeyCode::Enter); // pick src/
        assert_eq!(panel.avatar_path, "src/");
        let c = panel.completion.as_ref().expect("picker descends");
        assert!(c.title.contains("file · src/"), "{:?}", c.title);

        // Esc closes; a non-editing key does not reopen it.
        panel.handle_key(crossterm::event::KeyCode::Esc);
        assert!(panel.completion.is_none());
        panel.handle_key(crossterm::event::KeyCode::Left);
        assert!(panel.completion.is_none());
        // Editing re-enables it.
        panel.handle_key(crossterm::event::KeyCode::Backspace);
        assert!(panel.completion.is_some());
    }

    #[test]
    fn picker_keys_do_not_leak_into_other_fields() {
        let mut panel = RegistrationPanel::new();
        // Name field focused: no picker, arrows/enter do nothing special.
        type_chars(&mut panel, "Alice");
        assert!(panel.completion.is_none());
        // Tab from Name lands on Avatar (picker opens), another Tab
        // while the picker is open PICKS instead of moving focus.
        panel.handle_key(crossterm::event::KeyCode::Tab);
        assert!(panel.completion.is_some());
        panel.handle_key(crossterm::event::KeyCode::Tab); // pick first dir
        assert!(panel.avatar_path.ends_with('/'));
        // Now the picker is still open (descended); Esc closes, and the
        // next Tab moves focus to Submit.
        panel.handle_key(crossterm::event::KeyCode::Esc);
        panel.handle_key(crossterm::event::KeyCode::Tab);
        assert_eq!(panel.focus, FieldFocus::Submit);
    }

    #[test]
    fn submit_validates_image_is_readable() {
        // A real file that is not an image.
        let me = std::env::current_exe().unwrap();
        let mut panel = RegistrationPanel::new();
        panel.display_name = "Alice".to_string();
        panel.avatar_path = me.to_string_lossy().into_owned();
        assert!(panel.resolved_avatar_path().unwrap().is_file());
        let err = match panel.validate() {
            Ok(_) => panic!("non-image should not validate"),
            Err(e) => e,
        };
        assert!(err.contains("not a readable image"), "got: {err}");
    }
}
