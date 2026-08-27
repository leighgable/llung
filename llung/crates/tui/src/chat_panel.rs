use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use unicode_width::UnicodeWidthStr;

use crate::prefix_text::PrefixText;

pub struct ChatMessage {
    pub sender: String,
    pub content: PrefixText,
    pub is_me: bool,
}

/// One rendered screen line.
struct VisualLine {
    text: String,
    x_offset: u16,
    style: Style,
}

pub struct ChatPanel {
    messages: Vec<ChatMessage>,
    my_name: String,
    scroll_offset: usize,
    auto_scroll: bool,
    cache: Vec<VisualLine>,
    cached_width: u16,
}

impl ChatPanel {
    pub fn new(my_name: String) -> Self {
        Self {
            messages: Vec::new(),
            my_name,
            scroll_offset: 0,
            auto_scroll: true,
            cache: Vec::new(),
            cached_width: 0,
        }
    }

    /// a network message arrives.
    pub fn push(&mut self, sender: String, text: String) {
        let is_me = sender == self.my_name;
        self.messages.push(ChatMessage {
            sender,
            content: PrefixText::new(text),
            is_me,
        });
        self.cache.clear(); // invalidate
        self.cached_width = 0;
    }

    fn rewrap(&mut self, width: u16) {
        if width == 0 || width == self.cached_width {
            return;
        }
        self.cache.clear();
        self.cached_width = width;

        for msg in &self.messages {
            if msg.is_me {
                let margin = 2u16.min(width);
                let content_width = width.saturating_sub(margin + 2);

                for line in msg.content.wrap_lines(content_width) {
                    self.cache.push(VisualLine {
                        text: line.to_string(),
                        x_offset: margin,
                        style: Style::default().fg(Color::Green),
                    });
                }
            } else {
                let header = format!("{}: ", msg.sender);
                let header_width = header.width() as u16;
                let right_margin = 2u16;

                if header_width >= width.saturating_sub(right_margin) {
                    // sender name is too wide; put it on its own line.
                    self.cache.push(VisualLine {
                        text: header,
                        x_offset: width.saturating_sub(header_width as u16 + right_margin),
                        style: Style::default().fg(Color::Cyan),
                    });
                    let indent = 2u16.min(width);
                    for line in msg.content.wrap_lines(width.saturating_sub(indent)) {
                        self.cache.push(VisualLine {
                            text: line.to_string(),
                            x_offset: width.saturating_sub(header_width as u16 + right_margin),
                            style: Style::default().fg(Color::Cyan),
                        });
                    }
                } else {
                    let content_width = width - header_width - right_margin;
                    let mut first = true;
                    for line in msg.content.wrap_lines(content_width) {
                        let line_text = if first {
                            format!("{}{}", header, line)
                        } else {
                            line.to_string()
                        };

                        let line_width = line_text.width() as u16;
                        let x_offset = width.saturating_sub(line_width + right_margin);

                        self.cache.push(VisualLine {
                            text: line_text,
                            x_offset,
                            style: Style::default().fg(Color::Cyan),
                        });

                        first = false;
                    }
                }
            }
        }
    }

    pub fn render(&mut self, buf: &mut Buffer, area: Rect) {
        self.rewrap(area.width);

        let total = self.cache.len();
        let height = area.height as usize;

        if self.auto_scroll {
            self.scroll_offset = total.saturating_sub(height);
        } else {
            let max = total.saturating_sub(height);
            self.scroll_offset = self.scroll_offset.min(max);

            if self.scroll_offset >= max {
                self.auto_scroll = true;
            }
        }

        let start = self.scroll_offset;
        let end = (start + height).min(total);

        // ── Defensive: clamp to actual buffer bounds ──
        let buf_area = buf.area();
        let max_y = buf_area.height;
        let max_x = buf_area.width;

        let area_x0 = area.x.min(max_x);
        let area_x1 = (area.x + area.width).min(max_x);
        let area_y0 = area.y.min(max_y);
        let area_y1 = (area.y + area.height).min(max_y);

        let mut y = area_y0;

        // Draw visible lines
        for i in start..end {
            if y >= area_y1 {
                break;
            }
            let line = &self.cache[i];

            // Clear row
            for col in area_x0..area_x1 {
                if let Some(cell) = buf.cell_mut((col, y)) {
                    cell.reset();
                    cell.set_bg(Color::Black);
                }
            }

            // Draw text (clamped to not overflow the row)
            let text_x = (area_x0 + line.x_offset).min(max_x);
            if text_x < max_x && y < max_y {
                let max_len = (area_x1.saturating_sub(text_x)) as usize;
                buf.set_stringn(text_x, y, &line.text, max_len, line.style);
            }

            y += 1;
        }

        // Clear remaining rows below last message
        while y < area_y1 {
            for col in area_x0..area_x1 {
                if let Some(cell) = buf.cell_mut((col, y)) {
                    cell.reset();
                    cell.set_bg(Color::Black);
                }
            }
            y += 1;
        }
    }

    pub fn scroll_up(&mut self, lines: usize) {
        self.auto_scroll = false;
        self.scroll_offset = self.scroll_offset.saturating_sub(lines);
    }

    pub fn scroll_down(&mut self, lines: usize) {
        self.scroll_offset += lines;
    }

    pub fn jump_to_bottom(&mut self) {
        self.auto_scroll = true;
    }
}
