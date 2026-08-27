use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::widgets::{Block, Borders, Paragraph, Widget};

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
            KeyCode::Char(c) if self.focus == FieldFocus::Name => self.display_name.push(c),
            KeyCode::Char(c) if self.focus == FieldFocus::Avatar => self.avatar_path.push(c),
            KeyCode::Backspace if self.focus == FieldFocus::Name => {
                self.display_name.pop();
            }
            KeyCode::Backspace if self.focus == FieldFocus::Avatar => {
                self.avatar_path.pop();
            }
            KeyCode::Enter if self.focus == FieldFocus::Submit => {
                if self.display_name.trim().is_empty() {
                    self.error_message = Some("Display name is required".into());
                    return None;
                }
                return Some(RegistrationResult {
                    name: self.display_name.trim().to_string(),
                    avatar: self.avatar_path.trim().to_string(),
                });
            }
            _ => {}
        }
        None
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

        // Render with ratatui's built-in widgets for forms (simpler than raw buffer for this)
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

        let submit_style = if self.focus == FieldFocus::Submit {
            Style::default().fg(Color::Black).bg(Color::Green)
        } else {
            Style::default().fg(Color::Green)
        };
        let submit = Paragraph::new(" [ Submit ] ").style(submit_style);
        submit.render(Rect::new(inner.x, inner.y + 5, inner.width, 1), buf);

        if let Some(ref err) = self.error_message {
            let err_para = Paragraph::new(err.as_str()).style(Style::default().fg(Color::Red));
            err_para.render(Rect::new(inner.x, inner.y + 7, inner.width, 1), buf);
        }
    }
}

pub struct RegistrationResult {
    pub name: String,
    pub avatar: String,
}
