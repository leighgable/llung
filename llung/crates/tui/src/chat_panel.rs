use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use unicode_width::UnicodeWidthStr;

use crate::prefix_text::PrefixText;

pub struct ChatLine {
    pub sender: String,
    pub content: PrefixText,
    pub is_me: bool,
    pub avatar: Option<crate::avatar::AvatarThumbnail>,
}

struct VisualLine {
    text: String,
    x_offset: u16,
    style: Style,
    is_gap: bool, // true for blank separator lines
}

pub struct ChatPanel {
    messages: Vec<ChatLine>,
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

    pub fn push(
        &mut self,
        sender: String,
        text: String,
        avatar: Option<crate::avatar::AvatarThumbnail>,
    ) {
        let is_me = sender.trim() == self.my_name.trim();
        self.messages.push(ChatLine {
            sender,
            content: PrefixText::new(text),
            is_me,
            avatar,
        });
        self.cache.clear();
        self.cached_width = 0;
    }

    fn rewrap(&mut self, width: u16) {
        if width == 0 || width == self.cached_width {
            return;
        }
        self.cache.clear();
        self.cached_width = width;

        let max_block_width = (width as u32 * 80 / 100) as u16;
        let mut last_was_me = false;

        for msg in &self.messages {
            // ── Insert a blank gap between speakers ──
            if !self.cache.is_empty() && msg.is_me != last_was_me {
                self.cache.push(VisualLine {
                    text: String::new(),
                    x_offset: 0,
                    style: Style::default(),
                    is_gap: true,
                });
            }
            last_was_me = msg.is_me;

            if msg.is_me {
                // ── MY MESSAGES: right-aligned, bright yellow, dark bg ──
                let right_margin = 2u16;
                let max_content = max_block_width.saturating_sub(right_margin + 2);

                for line in msg.content.wrap_lines(max_content) {
                    let line_width = line.width() as u16;
                    let x_offset = width.saturating_sub(line_width + right_margin);

                    self.cache.push(VisualLine {
                        text: line.to_string(),
                        x_offset,
                        style: Style::default()
                            .fg(Color::Yellow)
                            .bg(Color::Rgb(40, 40, 20)) // subtle dark yellow bg
                            .add_modifier(Modifier::BOLD),
                        is_gap: false,
                    });
                }
            } else {
                // ── OTHERS: left-aligned, cyan, no bg ──
                let avatar_width = if msg.avatar.is_some() { 3 } else { 0 };
                let left_margin = 2 + avatar_width;
                let header = format!("{} ", msg.sender); // sender + space
                let header_width = header.width() as u16;
                let max_content = max_block_width.saturating_sub(left_margin + 2);

                if header_width >= max_content {
                    // Sender name too long: put on its own line
                    self.cache.push(VisualLine {
                        text: header,
                        x_offset: left_margin,
                        style: Style::default().fg(Color::Cyan).add_modifier(Modifier::DIM),
                        is_gap: false,
                    });

                    for line in msg.content.wrap_lines(max_content) {
                        self.cache.push(VisualLine {
                            text: line.to_string(),
                            x_offset: left_margin + 2,
                            style: Style::default().fg(Color::Cyan),
                            is_gap: false,
                        });
                    }
                } else {
                    // First line includes sender prefix
                    let mut first = true;
                    for line in msg
                        .content
                        .wrap_lines(max_content.saturating_sub(header_width))
                    {
                        let text = if first {
                            format!("{}{}", header, line)
                        } else {
                            line.to_string()
                        };

                        self.cache.push(VisualLine {
                            text,
                            x_offset: left_margin,
                            style: Style::default().fg(Color::Cyan),
                            is_gap: false,
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

        let buf_area = buf.area();
        let max_y = buf_area.height;
        let max_x = buf_area.width;

        let area_x0 = area.x.min(max_x);
        let area_x1 = (area.x + area.width).min(max_x);
        let area_y0 = area.y.min(max_y);
        let area_y1 = (area.y + area.height).min(max_y);

        let mut y = area_y0;

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

            if !line.is_gap {
                let text_x = (area_x0 + line.x_offset).min(max_x);
                if text_x < max_x {
                    let max_len = (area_x1.saturating_sub(text_x)) as usize;
                    buf.set_stringn(text_x, y, &line.text, max_len, line.style);
                }
            }

            y += 1;
        }

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
