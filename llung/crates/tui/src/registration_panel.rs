use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::widgets::{Block, Borders, Paragraph, Widget};
use std::path::PathBuf;

pub struct RegistrationPanel {
    pub display_name: String,
    pub avatar_path: String,
    pub focus: FieldFocus,
    pub error_message: Option<String>,
}

#[derive(Clone, Copy, PartialEq)]
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
        }
    }

    pub fn handle_key(&mut self, code: crossterm::event::KeyCode) -> Option<RegistrationResult> {
        use crossterm::event::KeyCode;
        match code {
            KeyCode::Tab => self.cycle_focus(),
            KeyCode::BackTab => self.cycle_focus_back(),
            KeyCode::Char(c) if self.focus == FieldFocus::Name => {
                self.error_message = None;
                self.display_name.push(c);
            }
            KeyCode::Char(c) if self.focus == FieldFocus::Avatar => {
                self.error_message = None;
                self.avatar_path.push(c);
            }
            KeyCode::Backspace if self.focus == FieldFocus::Name => {
                self.display_name.pop();
            }
            KeyCode::Backspace if self.focus == FieldFocus::Avatar => {
                self.avatar_path.pop();
            }
            KeyCode::Enter if self.focus == FieldFocus::Submit => match self.validate() {
                Ok(result) => return Some(result),
                Err(e) => self.error_message = Some(e),
            },
            _ => {}
        }
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
    }

    fn cycle_focus_back(&mut self) {
        self.focus = match self.focus {
            FieldFocus::Name => FieldFocus::Submit,
            FieldFocus::Avatar => FieldFocus::Name,
            FieldFocus::Submit => FieldFocus::Avatar,
        };
    }

    pub fn render(&self, buf: &mut Buffer, area: Rect) {
        // Clear area
        for y in area.y..(area.y + area.height) {
            for x in area.x..(area.x + area.width) {
                if let Some(cell) = buf.cell_mut((x, y)) {
                    cell.reset();
                    cell.set_bg(Color::Black);
                }
            }
        }

        let block = Block::default()
            .title(" Create Your Identity ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::White));

        let inner = block.inner(area);
        block.render(area, buf);

        let name_style = if self.focus == FieldFocus::Name {
            Style::default().fg(Color::Yellow).bg(Color::DarkGray)
        } else {
            Style::default()
        };
        let name_para = Paragraph::new(format!("Name: {}", self.display_name)).style(name_style);
        name_para.render(Rect::new(inner.x, inner.y + 1, inner.width, 1), buf);

        let avatar_style = if self.focus == FieldFocus::Avatar {
            Style::default().fg(Color::Yellow).bg(Color::DarkGray)
        } else {
            Style::default()
        };
        let avatar_para =
            Paragraph::new(format!("Avatar (optional): {}", self.avatar_path)).style(avatar_style);
        avatar_para.render(Rect::new(inner.x, inner.y + 3, inner.width, 1), buf);

        // Live resolution status, so the user can see *where* the app is looking.
        if let Some((line, ok)) = self.avatar_status() {
            let status_style = if ok {
                Style::default().fg(Color::Green)
            } else {
                Style::default().fg(Color::Red)
            };
            let status = Paragraph::new(line).style(status_style);
            status.render(Rect::new(inner.x + 2, inner.y + 4, inner.width.saturating_sub(2), 1), buf);
        }

        let submit_style = if self.focus == FieldFocus::Submit {
            Style::default().fg(Color::Black).bg(Color::Green)
        } else {
            Style::default().fg(Color::Green)
        };
        let submit = Paragraph::new(" [ Submit ] ").style(submit_style);
        submit.render(Rect::new(inner.x, inner.y + 6, inner.width, 1), buf);

        if let Some(ref err) = self.error_message {
            let err_para = Paragraph::new(err.as_str()).style(Style::default().fg(Color::Red));
            err_para.render(Rect::new(inner.x, inner.y + 8, inner.width, 1), buf);
        }

        let cwd = std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "<unknown>".to_string());
        let hint = Paragraph::new(format!(
            "Tab switches fields · relative paths resolve from {cwd} · ~ is expanded"
        ))
        .style(Style::default().fg(Color::DarkGray));
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
        let panel = panel_with_avatar(SAMPLE);
        let resolved = panel.resolved_avatar_path().expect("should resolve");
        assert!(resolved.is_absolute(), "{resolved:?} should be absolute");
        assert!(resolved.ends_with("data/1.jpg"), "unexpected: {resolved:?}");
        assert!(resolved.is_file(), "sample image missing: {resolved:?}");

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
