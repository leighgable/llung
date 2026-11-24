use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use crate::theme::{Role, Theme, ToolState};
use crate::tool_panel::{ImageRow, ToolBlock};
use unicode_width::UnicodeWidthStr;

use crate::avatar::{AvatarThumbnail, render_avatar};
use crate::md;
use crate::prefix_text::PrefixText;
use llung_core::network::message::{Attachment, AttachmentKind};
use std::collections::HashMap;

fn human_size(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit = 0;
    while size >= 1024.0 && unit < UNITS.len() - 1 {
        size /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{} {}", bytes, UNITS[0])
    } else {
        format!("{:.1} {}", size, UNITS[unit])
    }
}

pub struct ChatLine {
    /// Stable identity (base58 PeerId) used to re-resolve the display
    /// name when presence arrives after the message did.
    pub sender_id: String,
    pub sender: String,
    pub content: PrefixText,
    pub is_me: bool,
    pub avatar: Option<AvatarThumbnail>,
    pub attachment: Option<Attachment>,
    /// Tool run shown as a panel instead of `content`.
    pub tool: Option<ToolBlock>,
    /// Set while this line is a streaming preview: the id the final
    /// gossipsub message will carry, so it can replace this line.
    pub stream_id: Option<String>,
    /// Reasoning scratchpad chars accumulated on this stream (Kimi-style
    /// `reasoning_content`). Shown as activity and elided to a one-line
    /// trace, never as the answer.
    pub thought_chars: usize,
    /// True until the first answer-text delta arrives: the line renders
    /// as an animated `working`/`thinking` spinner.
    pub stream_working: bool,
}

/// Reasoning longer than this is collapsed to a `(thought N chars)`
/// trace instead of staying on screen (pi-style elision).
const THOUGHT_ELIDE_CHARS: usize = 160;

#[derive(Default)]
struct VisualLine {
    text: String,
    x_offset: u16,
    /// Semantic role, resolved against the theme at draw time so the
    /// wrap cache survives theme switches.
    role: Role,
    is_gap: bool, // true for blank separator lines
    /// Index of the ChatLine this row belongs to; lets the renderer draw
    /// the per-message avatar once at the block's first row.
    msg_index: usize,
    /// Attachment block starts on this row; rendered by `render_attachment`.
    attachment: Option<Attachment>,
    /// Continuation row of a multi-row attachment block (e.g. image
    /// thumbnails). Skipped by the renderer so the thumbnail above isn't
    /// cleared.
    attachment_cont: bool,
    /// Styled runs drawn after `text` (tool panels).
    segments: Vec<(String, Role)>,
    /// Background fill `(role, width)` from `x_offset` (tool panels).
    fill: Option<(Role, u16)>,
    /// Kitty image row drawn 1 column into the line (tool plots).
    image: Option<ImageRow>,
}

/// Rendered footprint of an attachment: (block width, block height in rows).
fn attachment_block_size(
    attachment: &Attachment,
    thumb: Option<&AvatarThumbnail>,
) -> (u16, u16) {
    match attachment.kind {
        AttachmentKind::Image => match thumb {
            Some(t) => (t.width(), t.height()),
            None => ("loading...".width() as u16, 1),
        },
        AttachmentKind::Audio => (
            ("Audio".width().max(attachment.filename.width())) as u16,
            2,
        ),
        AttachmentKind::Video => ("Video".width() as u16, 1),
        AttachmentKind::Document => {
            let label = format!(
                "{} ({})",
                attachment.filename,
                human_size(attachment.size_bytes)
            );
            (label.width() as u16, 1)
        }
        AttachmentKind::Unknown => ((attachment.filename.width() + 2) as u16, 1),
    }
}

pub struct ChatPanel {
    messages: Vec<ChatLine>,
    my_name: String,
    scroll_offset: usize,
    auto_scroll: bool,
    cache: Vec<VisualLine>,
    cached_width: u16,
    pub thumb_cache: HashMap<String, AvatarThumbnail>,
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
            thumb_cache: HashMap::new(),
        }
    }

    pub fn push(
        &mut self,
        sender_id: String,
        sender: String,
        text: String,
        avatar: Option<AvatarThumbnail>,
        attachment: Option<Attachment>,
    ) {
        let is_me = sender.trim() == self.my_name.trim();
        self.messages.push(ChatLine {
            sender_id,
            sender,
            content: PrefixText::new(text),
            is_me,
            avatar,
            attachment,
            tool: None,
            stream_id: None,
            thought_chars: 0,
            stream_working: false,
        });
        self.cache.clear();
        self.cached_width = 0;
    }

    /// Show a tool call as a pending panel. Idempotent per `call_id`:
    /// gossipsub does not order an agent's ToolCall/ToolResponse pair, so
    /// the response can arrive first — and a delivered call must never
    /// stack a second `running…` panel (or resurrect a finished one).
    pub fn push_tool(
        &mut self,
        sender_id: String,
        sender: String,
        avatar: Option<AvatarThumbnail>,
        block: ToolBlock,
    ) {
        self.upsert_tool(sender_id, sender, avatar, block);
    }

    /// Insert a tool panel, or update the existing panel with the same
    /// `call_id`. A finished panel (success/error) is never replaced by
    /// a pending one; anything else updates in place so the spinner
    /// gives way to the result exactly once.
    fn upsert_tool(
        &mut self,
        sender_id: String,
        sender: String,
        avatar: Option<AvatarThumbnail>,
        block: ToolBlock,
    ) {
        if let Some(cid) = block.call_id.as_deref() {
            if let Some(line) = self.messages.iter_mut().find(|m| {
                m.tool.as_ref().and_then(|t| t.call_id.as_deref()) == Some(cid)
            }) {
                let keep_finished = line
                    .tool
                    .as_ref()
                    .is_some_and(|t| t.state != ToolState::Pending)
                    && block.state == ToolState::Pending;
                if !keep_finished {
                    line.tool = Some(block);
                    self.invalidate();
                }
                return;
            }
        }
        let is_me = sender.trim() == self.my_name.trim();
        self.messages.push(ChatLine {
            sender_id,
            sender,
            content: PrefixText::new(String::new()),
            is_me,
            avatar,
            attachment: None,
            tool: Some(block),
            stream_id: None,
            thought_chars: 0,
            stream_working: false,
        });
        self.invalidate();
    }

    /// The model started a turn, an attempt, or a re-ask for `stream_id`:
    /// show the `working` spinner until the first answer-text delta
    /// arrives. Sent before EVERY backend call (so the spinner covers
    /// between-tool-steps thinking); when the line already exists — a
    /// retry after a truncated attempt — it resets to the spinner state.
    pub fn push_stream_start(
        &mut self,
        sender_id: String,
        sender: String,
        avatar: Option<AvatarThumbnail>,
        stream_id: &str,
    ) {
        if let Some(line) = self
            .messages
            .iter_mut()
            .rev()
            .find(|m| m.stream_id.as_deref() == Some(stream_id))
        {
            line.content = PrefixText::new(String::new());
            line.thought_chars = 0;
            line.stream_working = true;
            self.invalidate();
            return;
        }
        let is_me = sender.trim() == self.my_name.trim();
        self.messages.push(ChatLine {
            sender_id,
            sender,
            content: PrefixText::new(String::new()),
            is_me,
            avatar,
            attachment: None,
            tool: None,
            stream_id: Some(stream_id.to_string()),
            thought_chars: 0,
            stream_working: true,
        });
        self.invalidate();
    }

    /// Reasoning scratchpad text for `stream_id`: counted as activity
    /// (the spinner becomes `thinking (N chars)`), elided once long.
    pub fn push_stream_thought(&mut self, stream_id: &str, delta: &str) {
        if let Some(line) = self
            .messages
            .iter_mut()
            .rev()
            .find(|m| m.stream_id.as_deref() == Some(stream_id))
        {
            line.thought_chars += delta.chars().count();
            self.invalidate();
        }
    }

    /// Append a streaming delta to the in-progress reply `stream_id`,
    /// starting the line on first delta. The final gossipsub message
    /// carries the same id and replaces the preview via
    /// [`Self::remove_stream_line`].
    pub fn push_stream_delta(
        &mut self,
        sender_id: String,
        sender: String,
        avatar: Option<AvatarThumbnail>,
        stream_id: &str,
        delta: &str,
    ) {
        if let Some(line) = self
            .messages
            .iter_mut()
            .rev()
            .find(|m| m.stream_id.as_deref() == Some(stream_id))
        {
            line.stream_working = false;
            line.content.append(delta);
        } else {
            let is_me = sender.trim() == self.my_name.trim();
            self.messages.push(ChatLine {
                sender_id,
                sender,
                content: PrefixText::new(delta.to_string()),
                is_me,
                avatar,
                attachment: None,
                tool: None,
                stream_id: Some(stream_id.to_string()),
                thought_chars: 0,
                stream_working: false,
            });
        }
        self.invalidate();
    }

    /// Resolve a streaming preview: the reply became a tool call, or the
    /// final message with the same id is about to render. Long reasoning
    /// is kept as an elided `(thought N chars)` trace instead of
    /// vanishing. Returns true when a stream line was consumed.
    pub fn remove_stream_line(&mut self, stream_id: &str) -> bool {
        let Some(pos) = self
            .messages
            .iter()
            .rposition(|m| m.stream_id.as_deref() == Some(stream_id))
        else {
            return false;
        };
        let line = &mut self.messages[pos];
        if line.thought_chars >= THOUGHT_ELIDE_CHARS {
            // Detach the elided trace: it stays as a normal muted line.
            line.content = PrefixText::new(format!("(thought {} chars)", line.thought_chars));
            line.stream_id = None;
            line.stream_working = false;
            line.tool = None;
        } else {
            self.messages.remove(pos);
        }
        self.invalidate();
        true
    }

    /// True while any stream line is still in its spinner phase — the
    /// signal for the tick loop to keep invalidating so it animates.
    pub fn has_spinning_streams(&self) -> bool {
        self.messages
            .iter()
            .any(|m| m.stream_id.is_some() && m.stream_working)
    }

    /// A tool response: replaces the pending panel of the call it answers
    /// (matched by `parent_id`), or appears as a new panel if the call was
    /// never seen (e.g. joined mid-run).
    pub fn push_tool_response(
        &mut self,
        sender_id: String,
        sender: String,
        avatar: Option<AvatarThumbnail>,
        parent_id: Option<&str>,
        block: ToolBlock,
    ) {
        let pending = parent_id.and_then(|pid| {
            self.messages.iter_mut().rev().find(|m| {
                m.tool.as_ref().is_some_and(|t| t.call_id.as_deref() == Some(pid))
            })
        });
        match pending {
            Some(line) => {
                line.tool = Some(block);
                self.invalidate();
            }
            None => self.push_tool(sender_id, sender, avatar, block),
        }
    }

    /// True while any tool panel is still running — the signal for the
    /// tick loop to keep invalidating the layout so the spinner animates.
    pub fn has_running_tools(&self) -> bool {
        self.messages.iter().any(|m| {
            m.tool.as_ref().is_some_and(|t| t.state == ToolState::Pending)
        })
    }

    /// Remove every line (branch views rebuild the panel from the DAG).
    pub fn clear(&mut self) {
        self.messages.clear();
        self.invalidate();
        self.scroll_offset = 0;
        self.auto_scroll = true;
    }

    /// Drop the layout cache, e.g. after a theme change (Kitty plot images
    /// bake in theme colors).
    pub fn invalidate(&mut self) {
        self.cache.clear();
        self.cached_width = 0;
    }

    /// Update the display name on every line from `peer_id`. Called when a
    /// presence message arrives after that peer's chat messages did.
    pub fn update_peer_name(&mut self, peer_id: &str, name: &str) {
        let mut changed = false;
        for msg in &mut self.messages {
            if msg.sender_id == peer_id && msg.sender != name {
                msg.sender = name.to_string();
                changed = true;
            }
        }
        if changed {
            self.cache.clear();
            self.cached_width = 0;
        }
    }

    /// Attach an avatar to every line from `peer_id` that doesn't have
    /// one yet (the avatar bytes often arrive via presence after the
    /// messages did). Lines then render with the avatar prefix instead of
    /// the text name header.
    pub fn update_peer_avatar(&mut self, peer_id: &str, avatar: &AvatarThumbnail) {
        let mut changed = false;
        for msg in &mut self.messages {
            if msg.sender_id == peer_id && msg.avatar.is_none() {
                msg.avatar = Some(avatar.clone());
                changed = true;
            }
        }
        if changed {
            self.cache.clear();
            self.cached_width = 0;
        }
    }

    fn rewrap(&mut self, width: u16, theme: &Theme) {
        if width == 0 || width == self.cached_width {
            return;
        }
        self.cache.clear();
        self.cached_width = width;

        let max_block_width = (width as u32 * 80 / 100) as u16;
        let mut last_was_me = false;

        for (i, msg) in self.messages.iter().enumerate() {
            // ── Insert a blank gap between speakers ──
            if !self.cache.is_empty() && msg.is_me != last_was_me {
                self.cache.push(VisualLine {
                    text: String::new(),
                    x_offset: 0,
                    role: Role::Plain,
                    is_gap: true,
                    msg_index: i,
                    attachment: None,
                    attachment_cont: false,
                    ..Default::default()
                });
            }
            last_was_me = msg.is_me;

            // ── STREAMING PREVIEW: pi-style `working` spinner until the
            // first answer-text delta; long reasoning stays elided ──
            if msg.stream_id.is_some() {
                let avatar_pad = msg.avatar.as_ref().map(|a| a.width() + 1).unwrap_or(0);
                let left_margin = 2 + avatar_pad;
                if msg.content.as_str().is_empty() {
                    let label = if msg.thought_chars > 0 {
                        format!(
                            "{} thinking… ({} chars)",
                            crate::spinner::frame(),
                            msg.thought_chars
                        )
                    } else {
                        format!("{} {} is working…", crate::spinner::frame(), msg.sender)
                    };
                    self.cache.push(VisualLine {
                        text: label,
                        x_offset: left_margin,
                        role: Role::Muted,
                        msg_index: i,
                        ..Default::default()
                    });
                    continue;
                }
                // Answer text is arriving: long reasoning shows as an
                // elided trace above the stream, pi-style.
                if msg.thought_chars >= THOUGHT_ELIDE_CHARS {
                    self.cache.push(VisualLine {
                        text: format!("(thought {} chars)", msg.thought_chars),
                        x_offset: left_margin,
                        role: Role::Muted,
                        msg_index: i,
                        ..Default::default()
                    });
                }
                // Fall through to the normal markdown/plain rendering.
            }

            // ── TOOL PANELS: pi-style tinted block, near full width ──
            if let Some(ref tool) = msg.tool {
                let avatar_pad = msg.avatar.as_ref().map(|a| a.width() + 1).unwrap_or(0);
                let left_margin = 2 + avatar_pad;
                if msg.avatar.is_none() {
                    self.cache.push(VisualLine {
                        text: msg.sender.clone(),
                        x_offset: left_margin,
                        role: Role::PeerName,
                        msg_index: i,
                        ..Default::default()
                    });
                }
                let panel_w = width.saturating_sub(left_margin + 2).max(12);
                for line in tool.layout(panel_w, theme) {
                    self.cache.push(VisualLine {
                        x_offset: left_margin,
                        msg_index: i,
                        segments: line.segments,
                        fill: Some((Role::ToolPanel(tool.state), panel_w)),
                        image: line.image,
                        ..Default::default()
                    });
                }
                continue;
            }

            if md::has_markdown(msg.content.as_str()) {
                // ── MARKDOWN: blocks with structure and styles ──
                let lines = Self::markdown_lines(
                    i,
                    msg,
                    md::blocks(msg.content.as_str()),
                    width,
                    max_block_width,
                );
                self.cache.extend(lines);
                last_was_me = msg.is_me;
                continue;
            }

            if msg.is_me {
                // ── MY MESSAGES: right-aligned bubble ──
                let right_margin = 2u16;
                let max_content = max_block_width.saturating_sub(right_margin + 2);

                for line in msg.content.wrap_lines(max_content) {
                    let line_width = line.width() as u16;
                    let x_offset = width.saturating_sub(line_width + right_margin);

                    self.cache.push(VisualLine {
                        text: line.to_string(),
                        x_offset,
                        role: Role::MyMessage,
                        is_gap: false,
                        msg_index: i,
                        attachment: None,
                        attachment_cont: false,
                        ..Default::default()
                    });
                }
            } else {
                // ── OTHERS: left-aligned, no bg ──
                // With an avatar, the bubble is prefaced by the thumbnail
                // (drawn at render time) and the text name header is
                // omitted; without one, the name prefixes the first line.
                let avatar_pad = msg.avatar.as_ref().map(|a| a.width() + 1).unwrap_or(0);
                let left_margin = 2 + avatar_pad;
                let header = if msg.avatar.is_some() {
                    String::new()
                } else {
                    format!("{} ", msg.sender) // sender + space
                };
                let header_width = header.width() as u16;
                let max_content = max_block_width.saturating_sub(left_margin + 2);

                if header_width >= max_content {
                    // Sender name too long: put on its own line
                    self.cache.push(VisualLine {
                        text: header,
                        x_offset: left_margin,
                        role: Role::PeerName,
                        is_gap: false,
                        msg_index: i,
                        attachment: None,
                        attachment_cont: false,
                        ..Default::default()
                    });

                    for line in msg.content.wrap_lines(max_content) {
                        self.cache.push(VisualLine {
                            text: line.to_string(),
                            x_offset: left_margin + 2,
                            role: Role::PeerMessage,
                            is_gap: false,
                            msg_index: i,
                            attachment: None,
                            attachment_cont: false,
                            ..Default::default()
                        });
                    }
                } else {
                    // First line includes sender prefix (when there is one)
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
                            role: Role::PeerMessage,
                            is_gap: false,
                            msg_index: i,
                            attachment: None,
                            attachment_cont: false,
                            ..Default::default()
                        });
                        first = false;
                    }
                }
            }

            // ── Attachment block goes directly under the message text ──
            if let Some(ref att) = msg.attachment {
                let thumb = self.thumb_cache.get(&att.cid);
                let (block_width, block_rows) = attachment_block_size(att, thumb);
                let x_offset = if msg.is_me {
                    width.saturating_sub(block_width + 2)
                } else {
                    let avatar_pad = msg.avatar.as_ref().map(|a| a.width() + 1).unwrap_or(0);
                    2 + avatar_pad
                };
                for r in 0..block_rows.max(1) {
                    self.cache.push(VisualLine {
                        text: String::new(),
                        x_offset,
                        role: Role::Plain,
                        is_gap: false,
                        msg_index: i,
                        attachment: if r == 0 { Some(att.clone()) } else { None },
                        attachment_cont: r > 0,
                        ..Default::default()
                    });
                }
            }
        }
    }

    /// Markdown path: emit blocks (headings, code panels, quotes,
    /// lists, rules, inline styles) instead of the plain bubble.
    fn markdown_lines(
        i: usize,
        msg: &ChatLine,
        blocks: Vec<md::Block>,
        width: u16,
        max_block_width: u16,
    ) -> Vec<VisualLine> {
        use unicode_width::UnicodeWidthStr;

        let mut out: Vec<VisualLine> = Vec::new();

        let base = if msg.is_me { Role::MyMessage } else { Role::PeerMessage };
        let avatar_pad = msg.avatar.as_ref().map(|a| a.width() + 1).unwrap_or(0);
        let right_margin = 2u16;
        let left_margin = 2 + avatar_pad;
        let max_content = max_block_width.saturating_sub(left_margin.max(right_margin) + 2);

        // Sender header on its own line (plain path prefixes the first
        // wrapped line; blocks make that accounting not worth it).
        if !msg.is_me && msg.avatar.is_none() {
            out.push(VisualLine {
                text: msg.sender.clone(),
                x_offset: left_margin,
                role: Role::PeerName,
                msg_index: i,
                ..Default::default()
            });
        }

        // Right-alignment for my messages, per line width.
        let mut place = |line: VisualLine, plain_w: usize| {
            let mut line = line;
            if msg.is_me {
                line.x_offset = width.saturating_sub(plain_w as u16 + right_margin);
            }
            line.msg_index = i;
            out.push(line);
        };

        for block in blocks {
            match block {
                md::Block::Heading(_, text) => {
                    let pt = PrefixText::new(text);
                    for l in pt.wrap_lines(max_content) {
                        let w = l.width();
                        place(
                            VisualLine {
                                segments: md::inline(l, Role::MdHeading),
                                ..Default::default()
                            },
                            w,
                        );
                    }
                }
                md::Block::Para(text) => {
                    let pt = PrefixText::new(text);
                    for l in pt.wrap_lines(max_content) {
                        let w = l.width();
                        place(
                            VisualLine {
                                segments: md::inline(l, base),
                                ..Default::default()
                            },
                            w,
                        );
                    }
                }
                md::Block::Quote(text) => {
                    let pt = PrefixText::new(text);
                    let inner = max_content.saturating_sub(2);
                    for l in pt.wrap_lines(inner) {
                        let w = l.width() + 2;
                        let mut segments = vec![("▎ ".to_string(), Role::MdQuoteBorder)];
                        segments.extend(md::inline(l, Role::MdQuote));
                        place(
                            VisualLine {
                                segments,
                                ..Default::default()
                            },
                            w,
                        );
                    }
                }
                md::Block::List { marker, text } => {
                    let bullet = format!("{marker} ");
                    let bullet_w = bullet.width();
                    let pt = PrefixText::new(text);
                    let inner = max_content.saturating_sub(bullet_w as u16);
                    let mut first = true;
                    for l in pt.wrap_lines(inner) {
                        let w = l.width() + bullet_w;
                        let mut segments = Vec::new();
                        if first {
                            segments.push((bullet.clone(), Role::MdListBullet));
                        } else {
                            segments.push((" ".repeat(bullet_w), base));
                        }
                        segments.extend(md::inline(l, base));
                        place(
                            VisualLine {
                                segments,
                                ..Default::default()
                            },
                            w,
                        );
                        first = false;
                    }
                }
                md::Block::Hr => {
                    let bar: String = "─".repeat(max_content as usize);
                    place(
                        VisualLine {
                            text: bar,
                            role: Role::MdHr,
                            ..Default::default()
                        },
                        max_content as usize,
                    );
                }
                md::Block::Code(code) => {
                    let lines: Vec<&str> = code.lines().collect();
                    let longest = lines.iter().map(|l| l.width()).max().unwrap_or(0);
                    let w = (longest + 2).clamp(8, max_content as usize).max(8);
                    let border: String = "─".repeat(w.saturating_sub(2));
                    place(
                        VisualLine {
                            text: format!("╭{border}╮"),
                            role: Role::MdCodeBlockBorder,
                            ..Default::default()
                        },
                        w,
                    );
                    for l in lines {
                        let shown: String = l.chars().take(w - 2).collect();
                        let pad = (w - 2).saturating_sub(shown.chars().count());
                        place(
                            VisualLine {
                                text: format!(" {shown}{} ", " ".repeat(pad)),
                                role: Role::MdCodeBlock,
                                ..Default::default()
                            },
                            w,
                        );
                    }
                    place(
                        VisualLine {
                            text: format!("╰{border}╯"),
                            role: Role::MdCodeBlockBorder,
                            ..Default::default()
                        },
                        w,
                    );
                }
            }
        }
        out
    }

    pub fn render(&mut self, buf: &mut Buffer, area: Rect, theme: &Theme) {
        let canvas_bg = theme.bg(Role::Canvas);
        self.rewrap(area.width, theme);

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

        // Avatar prefix currently being drawn: (x0, x1, until_y). Its
        // columns are skipped by the row-clear pass on subsequent rows.
        let mut avatar_zone: Option<(u16, u16, u16)> = None;
        let mut last_msg_index: Option<usize> = None;
        let mut avatar_attempted = false;

        for i in start..end {
            if y >= area_y1 {
                break;
            }
            let line = &self.cache[i];

            // Continuation rows of a multi-row attachment block are owned by
            // the thumbnail rendered above; don't clear or draw over them.
            if line.attachment_cont {
                y += 1;
                continue;
            }

            // A new message block begins on this row: reset the avatar
            // state (a gap row carries the upcoming message's index, so
            // the previous peer's zone dies before the new bubble starts).
            if Some(line.msg_index) != last_msg_index {
                last_msg_index = Some(line.msg_index);
                avatar_zone = None;
                avatar_attempted = false;
            }

            // Clear row, leaving any active avatar zone untouched
            for col in area_x0..area_x1 {
                if let Some((zx0, zx1, zy)) = avatar_zone {
                    if y < zy && col >= zx0 && col < zx1 {
                        continue;
                    }
                }
                if let Some(cell) = buf.cell_mut((col, y)) {
                    cell.reset();
                    cell.set_bg(canvas_bg);
                }
            }

            // First row of a message from someone else: draw the avatar
            // prefix (the name-in-text fallback is handled at wrap time).
            if !avatar_attempted {
                avatar_attempted = true;
                if let Some(msg) = self.messages.get(line.msg_index) {
                    if !msg.is_me {
                        if let Some(ref thumb) = msg.avatar {
                            let ax = area_x0 + 2;
                            if ax + thumb.width() <= max_x && y + thumb.height() <= area_y1 {
                                render_avatar(buf, ax, y, thumb);
                                avatar_zone = Some((ax, ax + thumb.width(), y + thumb.height()));
                            }
                        }
                    }
                }
            }

            if let Some(ref attachment) = line.attachment {
                let x = (area_x0 + line.x_offset).min(max_x);
                self.render_attachment(buf, x, y, attachment, theme);
            }

            if !line.is_gap {
                let text_x = (area_x0 + line.x_offset).min(max_x);
                if let Some((fill_role, fill_w)) = line.fill {
                    let style = theme.style(fill_role);
                    for col in text_x..(text_x.saturating_add(fill_w)).min(area_x1) {
                        if let Some(cell) = buf.cell_mut((col, y)) {
                            cell.set_style(style);
                        }
                    }
                }
                if text_x < max_x {
                    let max_len = (area_x1.saturating_sub(text_x)) as usize;
                    let (mut sx, _) =
                        buf.set_stringn(text_x, y, &line.text, max_len, theme.style(line.role));
                    for (text, role) in &line.segments {
                        if sx >= area_x1 {
                            break;
                        }
                        let room = (area_x1 - sx) as usize;
                        sx = buf.set_stringn(sx, y, text, room, theme.style(*role)).0;
                    }
                }
                if let Some(img) = line.image {
                    let ix = text_x.saturating_add(1);
                    let cols = img.cols.min(area_x1.saturating_sub(ix));
                    crate::terminal::kitty::render_row(buf, ix, y, img.id, cols, img.row);
                }
            }

            y += 1;
        }

        while y < area_y1 {
            for col in area_x0..area_x1 {
                if let Some(cell) = buf.cell_mut((col, y)) {
                    cell.reset();
                    cell.set_bg(canvas_bg);
                }
            }
            y += 1;
        }
    }

    fn render_attachment(
        &self,
        buf: &mut ratatui::buffer::Buffer,
        x: u16,
        y: u16,
        attachment: &Attachment,
        theme: &Theme,
    ) {
        match attachment.kind {
            AttachmentKind::Image => {
                if let Some(ref thumb) = self.thumb_cache.get(&attachment.cid) {
                    render_avatar(buf, x, y, thumb);
                } else {
                    buf.set_string(x, y, "loading...", theme.style(Role::AttachmentPending));
                }
            }
            AttachmentKind::Audio => {
                buf.set_string(x, y, "Audio", theme.style(Role::AttachmentAudio));
                buf.set_string(
                    x,
                    y + 1,
                    &attachment.filename,
                    theme.style(Role::AttachmentMeta),
                );
            }
            AttachmentKind::Video => {
                buf.set_string(x, y, "Video", theme.style(Role::AttachmentVideo));
            }
            AttachmentKind::Document => {
                let label = format!(
                    "{} ({})",
                    attachment.filename,
                    human_size(attachment.size_bytes)
                );
                buf.set_string(x, y, &label, theme.style(Role::AttachmentDocument));
            }
            AttachmentKind::Unknown => {
                buf.set_string(
                    x,
                    y,
                    &format!(": {}", attachment.filename),
                    theme.style(Role::Plain),
                );
            }
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

#[cfg(test)]
mod tests {
    #[test]
    fn markdown_messages_emit_styled_blocks() {
        let mut panel = ChatPanel::new("me".into());
        panel.push(
            "p".into(),
            "plotter".into(),
            "## Result\n\nHere is `cos(x)*sin(y)`:\n\n```\nlet z = cos(x) * sin(y);\n```\n\n- one\n- two\n".into(),
            None,
            None,
        );
        panel.invalidate();
        let theme = crate::theme::Theme::default();
        panel.rewrap(80, &theme);

        // A code block: tinted body lines + border lines.
        assert!(panel.cache.iter().any(|l| l.role == Role::MdCodeBlock
            && l.text.contains("let z = cos(x)")));
        assert!(panel.cache.iter().any(|l| l.role == Role::MdCodeBlockBorder
            && l.text.starts_with('╭')));
        // A heading line (segments styled MdHeading).
        assert!(panel.cache.iter().any(|l| l.segments.iter().any(|(_, r)| *r == Role::MdHeading)));
        // Inline code segment in the paragraph.
        assert!(panel.cache.iter().any(|l| l.segments.iter().any(|(t, r)| *r == Role::MdCode && t.contains("cos(x)*sin(y)"))));
        // List bullet styling.
        assert!(panel.cache.iter().any(|l| l.segments.iter().any(|(_, r)| *r == Role::MdListBullet)));
        // Word-aware wrap: no mid-word "## " mangling, header on own line.
        assert!(panel.cache.iter().any(|l| l.role == Role::PeerName && l.text == "plotter"));
    }

    #[test]
    fn my_multiline_markdown_renders_as_styled_bubble() {
        let mut panel = ChatPanel::new("me".into());
        // What a typed multi-line markdown message looks like after the
        // Enter-handling keeps newlines: a list plus strong text.
        panel.push(
            "me".into(),
            "me".into(),
            "Shopping list:\n- eggs\n- **coffee**\n".into(),
            None,
            None,
        );
        panel.invalidate();
        let theme = crate::theme::Theme::default();
        panel.rewrap(80, &theme);

        // Block lines carry their text in styled segments (text stays
        // empty); newlines split blocks without losing content.
        let seg_text = |l: &VisualLine| -> String {
            l.segments.iter().map(|(t, _)| t.as_str()).collect()
        };
        assert!(panel.cache.iter().any(|l| l
            .segments
            .iter()
            .any(|(t, r)| t.contains("Shopping list:") && *r == Role::MyMessage)));
        // Both list items survive as styled list lines, strong included.
        let bullets: Vec<_> = panel
            .cache
            .iter()
            .filter(|l| l.segments.iter().any(|(_, r)| *r == Role::MdListBullet))
            .collect();
        assert_eq!(bullets.len(), 2, "two list items expected");
        assert!(bullets.iter().any(|l| seg_text(l).contains("eggs")));
        assert!(bullets.iter().any(|l| l.segments.iter().any(|(t, r)| *r == Role::MdStrong && t == "coffee")));
        // My bubble: right-aligned (x_offset past the middle on 80 cols).
        assert!(bullets.iter().all(|l| l.x_offset > 40), "right-aligned expected");
    }

    #[test]
    fn compressor_spinner_is_attributed_to_the_compressor() {
        // The agent loop streams history compression under its own
        // line (empty being id, compressor display name), retracted
        // before the agent's own `working` line appears.
        let mut panel = ChatPanel::new("me".into());
        let theme = crate::theme::Theme::default();
        panel.push_stream_start(String::new(), "HistoryCompressor (SmolVLM)".into(), None, "c1");
        panel.rewrap(80, &theme);
        let texts: Vec<&String> = panel.cache.iter().map(|l| &l.text).collect();
        assert!(
            texts.iter().any(|t| t.contains("HistoryCompressor (SmolVLM) is working")),
            "{texts:?}"
        );
        assert!(!texts.iter().any(|t| t.contains("sensei")), "{texts:?}");

        assert!(panel.remove_stream_line("c1"));
        panel.push_stream_start("p".into(), "sensei".into(), None, "s1");
        panel.rewrap(80, &theme);
        let texts: Vec<&String> = panel.cache.iter().map(|l| &l.text).collect();
        assert!(texts.iter().any(|t| t.contains("sensei is working")), "{texts:?}");
        assert!(!texts.iter().any(|t| t.contains("HistoryCompressor")), "{texts:?}");
    }

    #[test]
    fn stream_shows_spinner_then_elides_thoughts() {
        let mut panel = ChatPanel::new("me".into());
        let theme = crate::theme::Theme::default();

        // Turn start: no output yet -> `working` spinner line.
        panel.push_stream_start("p".into(), "kimi".into(), None, "s1");
        panel.rewrap(80, &theme);
        assert!(
            panel.cache.iter().any(|l| l.text.contains("working")),
            "spinner line: {:?}",
            panel.cache.iter().map(|l| &l.text).collect::<Vec<_>>()
        );

        // Reasoning arrives: spinner becomes `thinking (N chars)`.
        panel.push_stream_thought("s1", &"x".repeat(500));
        panel.rewrap(80, &theme);
        assert!(
            panel.cache.iter().any(|l| l.text.contains("thinking") && l.text.contains("500")),
            "thinking line: {:?}",
            panel.cache.iter().map(|l| &l.text).collect::<Vec<_>>()
        );

        // Answer text streams: elided thought trace above it, spinner gone.
        panel.push_stream_delta("p".into(), "kimi".into(), None, "s1", "the answer");
        panel.rewrap(80, &theme);
        assert!(
            panel.cache.iter().any(|l| l.text.contains("(thought 500 chars)")),
            "elided trace: {:?}",
            panel.cache.iter().map(|l| &l.text).collect::<Vec<_>>()
        );
        assert!(panel.cache.iter().any(|l| l.text.contains("the answer")));
        assert!(!panel.cache.iter().any(|l| l.text.contains("working")));

        // Resolution (reply was a tool call / final message landed): the
        // long thought survives as the elided trace instead of vanishing.
        assert!(panel.remove_stream_line("s1"));
        panel.rewrap(80, &theme);
        assert!(
            panel.cache.iter().any(|l| l.text.contains("(thought 500 chars)")),
            "trace survives retraction"
        );
        assert!(!panel.cache.iter().any(|l| l.text.contains("the answer")));
    }

    #[test]
    fn stream_start_resets_a_partial_attempt() {
        let mut panel = ChatPanel::new("me".into());
        let theme = crate::theme::Theme::default();
        panel.push_stream_start("p".into(), "kimi".into(), None, "s1");
        panel.push_stream_delta("p".into(), "kimi".into(), None, "s1", "partial plan…");
        // A retry after a truncated attempt: back to the spinner, the
        // partial text must not linger.
        panel.push_stream_start("p".into(), "kimi".into(), None, "s1");
        panel.rewrap(80, &theme);
        assert!(panel.cache.iter().any(|l| l.text.contains("working")));
        assert!(!panel.cache.iter().any(|l| l.text.contains("partial")));
    }

    #[test]
    fn short_thoughts_vanish_on_retract() {
        let mut panel = ChatPanel::new("me".into());
        let theme = crate::theme::Theme::default();
        panel.push_stream_start("p".into(), "kimi".into(), None, "s1");
        panel.push_stream_thought("s1", "hmm");
        panel.push_stream_delta("p".into(), "kimi".into(), None, "s1", "TOOL {\"name\":\"glob\"}");
        panel.rewrap(80, &theme);
        // Under the elide threshold: nothing worth keeping.
        assert!(panel.remove_stream_line("s1"));
        panel.rewrap(80, &theme);
        assert!(!panel.cache.iter().any(|l| l.text.contains("thought")));
    }

    #[test]
    fn stream_deltas_append_to_one_line_and_finalize() {
        let mut panel = ChatPanel::new("me".into());
        panel.push_stream_delta("p".into(), "plotter".into(), None, "m1", "Here ");
        panel.push_stream_delta("p".into(), "plotter".into(), None, "m1", "is the plot");
        assert_eq!(panel.messages.len(), 1);
        assert_eq!(panel.messages[0].stream_id.as_deref(), Some("m1"));

        // A different stream starts a new line.
        panel.push_stream_delta("p".into(), "plotter".into(), None, "m2", "…");
        assert_eq!(panel.messages.len(), 2);

        // The final message replaces the preview.
        assert!(panel.remove_stream_line("m1"));
        assert_eq!(panel.messages.len(), 1);
        assert!(!panel.remove_stream_line("m1"), "second removal is a no-op");
        assert!(!panel.remove_stream_line("missing"));
    }
    use super::*;
    use crate::avatar::decode_avatar_hex;

    fn tiny_png() -> Vec<u8> {
        let img = image::RgbaImage::from_pixel(8, 8, image::Rgba([200, 100, 50, 255]));
        let mut bytes = std::io::Cursor::new(Vec::new());
        image::write_buffer_with_format(
            &mut bytes,
            &img,
            8,
            8,
            image::ExtendedColorType::Rgba8,
            image::ImageFormat::Png,
        )
        .unwrap();
        bytes.into_inner()
    }

    fn first_text_line(panel: &mut ChatPanel) -> String {
        panel.rewrap(80, &crate::theme::Theme::default());
        panel
            .cache
            .iter()
            .find(|l| !l.is_gap && !l.text.is_empty())
            .map(|l| l.text.clone())
            .expect("a text line")
    }

    #[test]
    fn message_with_avatar_omits_name_header() {
        let mut panel = ChatPanel::new("me".to_string());
        let thumb = decode_avatar_hex(&tiny_png(), 4).expect("decodable avatar");
        panel.push(
            "peer1".into(),
            "alice".into(),
            "hello there".into(),
            Some(thumb),
            None,
        );

        let line = first_text_line(&mut panel);
        assert!(!line.contains("alice"), "avatar present, name must go: {line}");
    }

    #[test]
    fn message_without_avatar_keeps_name_header() {
        let mut panel = ChatPanel::new("me".to_string());
        panel.push("peer1".into(), "alice".into(), "hello there".into(), None, None);

        let line = first_text_line(&mut panel);
        assert!(line.starts_with("alice "), "name prefix expected: {line}");
    }

    #[test]
    fn avatar_message_reserves_margin_for_thumbnail() {
        let mut panel = ChatPanel::new("me".to_string());
        let thumb = decode_avatar_hex(&tiny_png(), 4).expect("decodable avatar");
        let thumb_w = thumb.width();
        panel.push(
            "peer1".into(),
            "alice".into(),
            "hello there".into(),
            Some(thumb),
            None,
        );

        panel.rewrap(80, &crate::theme::Theme::default());
        let line = panel
            .cache
            .iter()
            .find(|l| !l.is_gap && !l.text.is_empty())
            .unwrap();
        // left margin (2) + thumbnail + gap (1)
        assert!(line.x_offset >= 2 + thumb_w + 1, "x_offset {}", line.x_offset);
    }

    #[test]
    fn update_peer_avatar_rewrites_existing_lines() {
        let mut panel = ChatPanel::new("me".to_string());
        panel.push("peer1".into(), "???".into(), "early message".into(), None, None);
        assert!(first_text_line(&mut panel).starts_with("???"));

        let thumb = decode_avatar_hex(&tiny_png(), 4).unwrap();
        panel.update_peer_avatar("peer1", &thumb);

        let line = first_text_line(&mut panel);
        assert!(!line.contains("???"), "avatar should replace the name: {line}");
    }

    #[test]
    fn update_peer_name_rewrites_existing_lines() {
        let mut panel = ChatPanel::new("me".to_string());
        panel.push("peer1".into(), "???".into(), "early message".into(), None, None);

        panel.update_peer_name("peer1", "bob");
        assert!(first_text_line(&mut panel).starts_with("bob "));
    }

    /// The full path with real `landscape` output: pending panel, replaced
    /// in place by the response; rows stay rows; dots are colored by
    /// height and readable on the panel (the "black on black" report).
    #[test]
    fn landscape_tool_panel_end_to_end() {
        use crate::theme::Theme;

        let mode = crate::theme::ColorMode::TrueColor;
        for theme in [
            Theme::default(),
            Theme::select("dark", mode).unwrap(),
            Theme::select("light", mode).unwrap(),
        ] {
            check_landscape_panel(&theme);
        }
    }

    fn check_landscape_panel(theme: &crate::theme::Theme) {
        use crate::theme::{Role, ToolState, contrast::wcag_contrast};
        use crate::tool_panel::ToolBlock;

        let mut panel = ChatPanel::new("me".into());
        let args = r#"{"expr": "sin(x)*cos(y)", "width": 120, "height": 40}"#;
        panel.push_tool(
            "agent-id".into(),
            "plotter".into(),
            None,
            ToolBlock::pending("call-1".into(), "landscape".into(), args),
        );
        let output = landscape_tool::run(args).unwrap();
        panel.push_tool_response(
            "agent-id".into(),
            "plotter".into(),
            None,
            Some("call-1"),
            ToolBlock::response(Some("call-1".into()), "landscape".into(), &output, false),
        );
        assert_eq!(panel.messages.len(), 1, "response replaced the pending panel");
        assert_eq!(panel.messages[0].tool.as_ref().unwrap().state, ToolState::Success);

        let area = Rect::new(0, 0, 100, 60);
        let mut buf = Buffer::empty(area);
        panel.render(&mut buf, area, theme);

        let panel_bg = theme.bg(Role::ToolPanel(ToolState::Success));
        let mut fgs = std::collections::HashSet::new();
        let mut dots = 0;
        for y in 0..area.height {
            for x in 0..area.width {
                let cell = &buf[(x, y)];
                let ch = cell.symbol().chars().next().unwrap_or(' ');
                if ('\u{2801}'..='\u{28FF}').contains(&ch) {
                    dots += 1;
                    assert_eq!(cell.bg, panel_bg, "dot on the panel at ({x},{y})");
                    fgs.insert(format!("{:?}", cell.fg));
                    if let (Some(fg), Some(bg)) =
                        (crate::theme::color_rgb(cell.fg), crate::theme::color_rgb(cell.bg))
                    {
                        assert!(wcag_contrast(fg, bg) >= 2.9, "{}: readable dot at ({x},{y})", theme.name);
                    }
                }
            }
        }
        assert!(dots > 100, "{}: a real plot is visible ({dots} dot cells)", theme.name);
        assert!(fgs.len() >= 4, "colored by height: {} distinct colors", fgs.len());
        // The plot fits the panel: nothing drawn past the right edge.
        assert!((0..area.height).all(|y| buf[(area.width - 1, y)].symbol() == " "));
    }
}
