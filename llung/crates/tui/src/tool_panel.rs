//! pi-style tool panels in the chat: a tinted block per tool run
//! (pending → success / error), with plots fitted to the viewer's width
//! and colored by height.
//!
//! Tool output arrives as `ToolResponse` chat messages. `braille-v1`
//! render documents (see `crates/tools/landscape/src/lib.rs::run`) become
//! [`Plot`]s; anything else is shown as wrapped text. Legacy plain braille
//! (older tool builds) is recognized too, drawn in one color.
//!
//! Layout is theme-independent (roles only), except Kitty images, whose
//! pixels bake in the theme's plot colors: the chat panel re-lays out on
//! theme change.

use crate::prefix_text::PrefixText;
use crate::terminal::graphics::{Graphics, graphics};
use crate::terminal::kitty;
use crate::theme::{PLOT_LEVELS, Role, Theme, ToolState};
use std::hash::{Hash, Hasher};

/// One tool run, as shown in the chat.
#[derive(Clone, Debug)]
pub struct ToolBlock {
    /// Id of the `ToolCall` message, for matching its response.
    pub call_id: Option<String>,
    pub tool: String,
    pub state: ToolState,
    /// Short description next to the tool name (expression, arguments).
    pub subtitle: String,
    pub body: ToolBody,
}

#[derive(Clone, Debug)]
pub enum ToolBody {
    Running,
    Plot(Plot),
    /// A `sugi-v1` render document: node coordinates + edges, drawn as
    /// boxes with orthogonal connectors (see `SugiDiagram`).
    Sugi(SugiDiagram),
    Text(String),
}

impl ToolBlock {
    /// A call that hasn't returned yet.
    pub fn pending(call_id: String, tool: String, arguments: &str) -> Self {
        Self {
            call_id: Some(call_id),
            tool,
            state: ToolState::Pending,
            subtitle: summarize_args(arguments),
            body: ToolBody::Running,
        }
    }

    /// A finished call. `content` is the tool output or error text.
    pub fn response(call_id: Option<String>, tool: String, content: &str, error: bool) -> Self {
        let (state, subtitle, body) = if error {
            (ToolState::Error, String::new(), ToolBody::Text(content.to_string()))
        } else if let Some(plot) = Plot::parse(content) {
            (ToolState::Success, plot.title.clone(), ToolBody::Plot(plot))
        } else if let Some(sugi) = SugiDiagram::parse(content) {
            (ToolState::Success, sugi.title.clone(), ToolBody::Sugi(sugi))
        } else {
            (ToolState::Success, String::new(), ToolBody::Text(content.to_string()))
        };
        Self { call_id, tool, state, subtitle, body }
    }

    /// Lay out as panel lines `width` columns wide (including 1 column of
    /// padding on each side). The caller fills each line's background
    /// with [`Role::ToolPanel`].
    pub fn layout(&self, width: u16, theme: &Theme) -> Vec<PanelLine> {
        let inner = width.saturating_sub(2).max(1);
        let mut lines = Vec::new();

        let (icon, icon_role) = match self.state {
            ToolState::Pending => ("◌", Role::Muted),
            ToolState::Success => ("●", Role::Success),
            ToolState::Error => ("✗", Role::Error),
        };
        let mut title = vec![
            (format!(" {icon} "), icon_role),
            (self.tool.clone(), Role::ToolTitle),
        ];
        if !self.subtitle.is_empty() {
            let room = (inner as usize).saturating_sub(self.tool.chars().count() + 4);
            title.push((format!("  {}", truncate(&self.subtitle, room)), Role::ToolOutput));
        }
        lines.push(PanelLine::text(title));

        match &self.body {
            ToolBody::Running => {
                // The braille spinner is plain Unicode text: it animates
                // in any terminal, graphics protocol or not.
                lines.push(PanelLine::text(vec![
                    (format!(" {} running…", crate::spinner::frame()), Role::Muted),
                ]));
            }
            ToolBody::Text(text) => {
                let role = if self.state == ToolState::Error { Role::Error } else { Role::ToolOutput };
                for line in PrefixText::new(text.clone()).wrap_lines(inner) {
                    lines.push(PanelLine::text(vec![(format!(" {line}"), role)]));
                }
            }
            ToolBody::Plot(plot) => {
                let fitted = plot.fit(inner as usize);
                lines.extend(plot_lines(&fitted, theme));
                if fitted.has_levels() {
                    lines.push(legend(&fitted));
                }
            }
            ToolBody::Sugi(sugi) => {
                for row in sugi.render_rows(inner as usize) {
                    lines.push(PanelLine::text(vec![(format!(" {row}"), Role::ToolOutput)]));
                }
            }
        }
        // Bottom padding, like pi's tool boxes.
        lines.push(PanelLine::text(Vec::new()));
        lines
    }
}

/// One row of a tool panel.
#[derive(Clone, Debug, PartialEq)]
pub struct PanelLine {
    /// Styled runs, left to right, starting at the panel's left edge.
    pub segments: Vec<(String, Role)>,
    /// Kitty image row drawn after the segments (1 column inset).
    pub image: Option<ImageRow>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageRow {
    pub id: u32,
    pub cols: u16,
    pub row: u16,
}

impl PanelLine {
    fn text(segments: Vec<(String, Role)>) -> Self {
        Self { segments, image: None }
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
}

/// `{"expr": "sin(x)"}` → `sin(x)`. A `title` wins over `expr`
/// (diagram/artist pass one). Extra arguments (camera choices) are
/// appended: `{"expr": "sin(x)", "pitch": 0}` → `sin(x)  pitch=0` —
/// a model-picked bad view must be visible in the panel, not hidden.
fn summarize_args(arguments: &str) -> String {
    match serde_json::from_str::<serde_json::Value>(arguments) {
        Ok(serde_json::Value::Object(map)) => {
            let mut parts: Vec<String> = Vec::new();
            if let Some(title) = map.get("title").and_then(|v| v.as_str()) {
                parts.push(title.to_string());
            } else if let Some(expr) = map.get("expr").and_then(|v| v.as_str()) {
                parts.push(expr.to_string());
            }
            for (k, v) in map.iter() {
                if k == "expr" || k == "title" || k == "graph" || k == "csv" {
                    continue;
                }
                parts.push(format!(
                    "{k}={}",
                    v.as_str().map(String::from).unwrap_or(v.to_string())
                ));
            }
            parts.join("  ")
        }
        _ => arguments.trim().to_string(),
    }
}

// ── Sugi diagrams ──

/// One node of a `sugi-v1` render document (character units).
#[derive(Clone, Debug, PartialEq)]
struct SugiNode {
    id: String,
    label: String,
    x: f64,
    y: f64,
    w: f64,
    h: f64,
}

#[derive(Clone, Debug, PartialEq)]
struct SugiEdge {
    from: String,
    to: String,
    label: Option<String>,
}

/// A layered graph layout from the `sugi` tool: the tool does the
/// Sugiyama math and returns coordinates; THIS side places boxes on a
/// character grid and draws orthogonal connectors with
/// `ratatui::symbols::line` glyphs, so the diagram re-fits the viewer's
/// width at render time.
#[derive(Clone, Debug, PartialEq)]
pub struct SugiDiagram {
    pub title: String,
    nodes: Vec<SugiNode>,
    edges: Vec<SugiEdge>,
    width: f64,
    height: f64,
    max_width: Option<f64>,
}

impl SugiDiagram {
    pub fn parse(content: &str) -> Option<SugiDiagram> {
        let doc: serde_json::Value = serde_json::from_str(content).ok()?;
        if doc.get("render")?.as_str()? != "sugi-v1" {
            return None;
        }
        let num = |v: &serde_json::Value, k: &str| v.get(k).and_then(|n| n.as_f64());
        let nodes = doc
            .get("nodes")?
            .as_array()?
            .iter()
            .map(|n| {
                Some(SugiNode {
                    id: n.get("id")?.as_str()?.to_string(),
                    label: n.get("label")?.as_str()?.to_string(),
                    x: num(n, "x")?,
                    y: num(n, "y")?,
                    w: num(n, "w")?,
                    h: num(n, "h")?,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        let edges = doc
            .get("edges")?
            .as_array()?
            .iter()
            .map(|e| {
                Some(SugiEdge {
                    from: e.get("from")?.as_str()?.to_string(),
                    to: e.get("to")?.as_str()?.to_string(),
                    label: e.get("label").and_then(|l| l.as_str()).map(String::from),
                })
            })
            .collect::<Option<Vec<_>>>()?;
        if nodes.is_empty() {
            return None;
        }
        Some(SugiDiagram {
            title: doc.get("title").and_then(|t| t.as_str()).unwrap_or("").to_string(),
            nodes,
            edges,
            width: doc.get("width").and_then(|w| w.as_f64()).unwrap_or(40.0),
            height: doc.get("height").and_then(|h| h.as_f64()).unwrap_or(10.0),
            max_width: doc.get("max_width").and_then(|w| w.as_f64()),
        })
    }

    /// Draw the diagram at most `max_cols` columns wide; returns the
    /// rows (without panel padding — the caller adds it).
    fn render_rows(&self, max_cols: usize) -> Vec<String> {
        use ratatui::symbols::line as g;
        const MAX_ROWS: f64 = 40.0;
        let limit = self.max_width.unwrap_or(f64::INFINITY).min(max_cols as f64);
        let scale = (limit / self.width.max(1.0))
            .min(1.0)
            .min(MAX_ROWS / self.height.max(1.0));
        let rows = ((self.height * scale).ceil() as usize).max(1);
        let mut grid = vec![vec![' '; max_cols]; rows];

        let rect = |id: &str| -> Option<(usize, usize, usize, usize)> {
            let n = self.nodes.iter().find(|n| n.id == id)?;
            Some((
                (n.x * scale).round() as usize,
                (n.y * scale).round() as usize,
                ((n.w * scale).round() as usize).max(4),
                ((n.h * scale).round() as usize).max(3),
            ))
        };

        // Line-drawing chars merge to a cross on overlap; boxes redraw
        // their borders afterwards, so connectors visually pass BEHIND
        // node boxes.
        let mut put = |grid: &mut Vec<Vec<char>>, x: usize, y: usize, c: char| {
            if y >= grid.len() || x >= grid[y].len() {
                return;
            }
            let old = grid[y][x];
            // Cross only when two straight segments intersect; corners
            // and labels drawn over a straight line win outright.
            grid[y][x] = if old != ' ' && old != c && is_straight(old) && is_straight(c) {
                '┼'
            } else {
                c
            };
        };

        // 1) Edges, orthogonal: exit bottom-center, elbow at the
        // vertical midpoint, enter top-center with an arrow.
        for edge in &self.edges {
            let Some((sx, sy, sw, sh)) = rect(&edge.from) else { continue };
            let Some((tx, ty, tw, _th)) = rect(&edge.to) else { continue };
            let (scx, tcx) = (sx + sw / 2, tx + tw / 2);
            if ty >= sy + sh {
                // Target below: the standard layered case.
                let start_y = sy + sh; // just under the bottom border
                let arrow_y = ty - 1; // just above the top border
                if arrow_y <= start_y {
                    continue;
                }
                let mid_y = (start_y + arrow_y) / 2;
                for y in start_y..=mid_y.min(arrow_y - 1) {
                    put(&mut grid, scx, y, g::VERTICAL.chars().next().unwrap());
                }
                if scx != tcx {
                    let (lo, hi) = (scx.min(tcx), scx.max(tcx));
                    for x in lo..=hi {
                        put(&mut grid, x, mid_y, g::HORIZONTAL.chars().next().unwrap());
                    }
                    // Corners: down→horizontal, horizontal→down.
                    let (c1, c2) = if tcx > scx {
                        ('└', '┌')
                    } else {
                        ('┘', '┐')
                    };
                    put(&mut grid, scx, mid_y, c1);
                    put(&mut grid, tcx, mid_y, c2);
                    for y in (mid_y + 1)..arrow_y {
                        put(&mut grid, tcx, y, g::VERTICAL.chars().next().unwrap());
                    }
                    // Edge label sits on the horizontal segment.
                    if let Some(label) = &edge.label {
                        let label: String = label.chars().take(hi - lo).collect();
                        let lx = lo + (hi - lo).saturating_sub(label.chars().count()) / 2;
                        for (i, ch) in label.chars().enumerate() {
                            put(&mut grid, lx + i, mid_y, ch);
                        }
                    }
                } else {
                    // Straight drop, no elbow.
                    for y in start_y..arrow_y {
                        put(&mut grid, scx, y, g::VERTICAL.chars().next().unwrap());
                    }
                }
                put(&mut grid, tcx, arrow_y, '▼');
            } else {
                // Same layer / target above (non-layered leftovers):
                // straight horizontal connector with a side arrow.
                let y = sy + sh / 2;
                let (lo, hi) = (sx.min(tx) + sw, sx.max(tx));
                if hi <= lo {
                    continue;
                }
                for x in lo..hi {
                    put(&mut grid, x, y, g::HORIZONTAL.chars().next().unwrap());
                }
                put(&mut grid, if tx > sx { hi - 1 } else { lo }, y, if tx > sx { '▶' } else { '◀' });
            }
        }

        // 2) Boxes (over the connectors): rounded borders, label
        // centered on the middle row.
        for node in &self.nodes {
            let Some((x, y, w, h)) = rect(&node.id) else { continue };
            if x + w > max_cols || y + h > rows {
                continue;
            }
            for i in 0..w {
                put(&mut grid, x + i, y, if i == 0 { '╭' } else if i == w - 1 { '╮' } else { '─' });
                put(
                    &mut grid,
                    x + i,
                    y + h - 1,
                    if i == 0 { '╰' } else if i == w - 1 { '╯' } else { '─' },
                );
            }
            for row in y + 1..y + h - 1 {
                put(&mut grid, x, row, g::VERTICAL.chars().next().unwrap());
                put(&mut grid, x + w - 1, row, g::VERTICAL.chars().next().unwrap());
            }
            let label: String = node.label.chars().take(w.saturating_sub(2)).collect();
            let lx = x + 1 + (w.saturating_sub(2 + label.chars().count())) / 2;
            let ly = y + h / 2;
            for (i, ch) in label.chars().enumerate() {
                put(&mut grid, lx + i, ly, ch);
            }
        }

        grid.into_iter()
            .map(|row| {
                let s: String = row.into_iter().collect::<String>();
                s.trim_end().to_string()
            })
            .collect()
    }
}

fn is_straight(c: char) -> bool {
    matches!(c, '─' | '│')
}

// ── Plots ──

/// Braille dot bit for dot (dx, dy) within a cell (dots 1-8).
const DOT_BITS: [[u8; 4]; 2] = [[0x01, 0x02, 0x04, 0x80], [0x08, 0x10, 0x20, 0x40]];

/// A braille plot: per-cell dot masks plus optional height levels.
#[derive(Clone, Debug, PartialEq)]
pub struct Plot {
    pub cols: usize,
    pub rows: usize,
    /// Braille dot mask per cell, row-major.
    masks: Vec<u8>,
    /// Height level 0-9 per cell (`None`: empty, or no level data).
    levels: Vec<Option<u8>>,
    pub title: String,
    pub z_range: Option<(f64, f64)>,
}

impl Plot {
    /// A `braille-v1` render document, or legacy plain braille text.
    pub fn parse(content: &str) -> Option<Plot> {
        Self::parse_doc(content).or_else(|| Self::parse_legacy(content))
    }

    fn parse_doc(content: &str) -> Option<Plot> {
        let doc: serde_json::Value = serde_json::from_str(content).ok()?;
        if doc.get("render")?.as_str()? != "braille-v1" {
            return None;
        }
        let rows: Vec<&str> = doc.get("rows")?.as_array()?.iter().filter_map(|r| r.as_str()).collect();
        let levels: Vec<&str> = doc
            .get("levels")
            .and_then(|l| l.as_array())
            .map(|a| a.iter().filter_map(|r| r.as_str()).collect())
            .unwrap_or_default();
        let z_range = doc.get("z_range").and_then(|z| {
            let a = z.as_array()?;
            Some((a.first()?.as_f64()?, a.get(1)?.as_f64()?))
        });
        let title = doc.get("title").and_then(|t| t.as_str()).unwrap_or("").to_string();
        Self::from_rows(&rows, &levels, title, z_range)
    }

    /// Text made only of braille characters and newlines.
    fn parse_legacy(content: &str) -> Option<Plot> {
        let rows: Vec<&str> = content.lines().collect();
        let all_braille = rows
            .iter()
            .flat_map(|r| r.chars())
            .all(|c| ('\u{2800}'..='\u{28FF}').contains(&c));
        if rows.is_empty() || !all_braille {
            return None;
        }
        Self::from_rows(&rows, &[], String::new(), None)
    }

    fn from_rows(rows: &[&str], levels: &[&str], title: String, z_range: Option<(f64, f64)>) -> Option<Plot> {
        let cols = rows.iter().map(|r| r.chars().count()).max()?;
        if cols == 0 {
            return None;
        }
        let mut masks = vec![0u8; cols * rows.len()];
        let mut lv = vec![None; cols * rows.len()];
        for (y, row) in rows.iter().enumerate() {
            let level_row: Vec<char> = levels.get(y).map(|l| l.chars().collect()).unwrap_or_default();
            for (x, ch) in row.chars().enumerate() {
                let mask = (ch as u32).checked_sub(0x2800).filter(|m| *m <= 0xFF)? as u8;
                masks[y * cols + x] = mask;
                if mask != 0 {
                    lv[y * cols + x] = level_row
                        .get(x)
                        .and_then(|c| c.to_digit(10))
                        .map(|d| d as u8);
                }
            }
        }
        Some(Plot { cols, rows: rows.len(), masks, levels: lv, title, z_range })
    }

    fn has_levels(&self) -> bool {
        self.levels.iter().any(Option::is_some)
    }

    fn dot(&self, x: usize, y: usize) -> bool {
        let (cx, cy) = (x / 2, y / 4);
        self.masks[cy * self.cols + cx] & DOT_BITS[x % 2][y % 4] != 0
    }

    /// Scale down to at most `max_cols` columns, keeping the aspect ratio.
    /// Dots are OR-pooled (thin lines stay visible) and each target cell
    /// takes the highest level of the cells mapped onto it.
    pub fn fit(&self, max_cols: usize) -> Plot {
        if self.cols <= max_cols || max_cols == 0 {
            return self.clone();
        }
        let scale = max_cols as f64 / self.cols as f64;
        let (dw, dh) = (self.cols * 2, self.rows * 4);
        let tdw = max_cols * 2;
        let tdh = ((dh as f64 * scale).ceil() as usize).max(1);
        let (tcols, trows) = (max_cols, tdh.div_ceil(4));
        let mut masks = vec![0u8; tcols * trows];
        let mut levels: Vec<Option<u8>> = vec![None; tcols * trows];
        for y in 0..dh {
            for x in 0..dw {
                if !self.dot(x, y) {
                    continue;
                }
                let tx = ((x as f64 * scale) as usize).min(tdw - 1);
                let ty = ((y as f64 * scale) as usize).min(tdh - 1);
                let cell = (ty / 4) * tcols + tx / 2;
                masks[cell] |= DOT_BITS[tx % 2][ty % 4];
                let src = self.levels[(y / 4) * self.cols + x / 2];
                levels[cell] = levels[cell].max(src);
            }
        }
        Plot {
            cols: tcols,
            rows: trows,
            masks,
            levels,
            title: self.title.clone(),
            z_range: self.z_range,
        }
    }

    /// Role for a cell's dots.
    fn role(&self, cell: usize) -> Role {
        match self.levels[cell] {
            Some(l) => Role::PlotLevel(l),
            // Legacy output without heights: one readable accent.
            None => Role::Accent,
        }
    }

    /// Braille rows as styled runs (empty cells as spaces).
    fn braille_lines(&self) -> Vec<PanelLine> {
        (0..self.rows)
            .map(|y| {
                let mut segments: Vec<(String, Role)> = vec![(" ".into(), Role::Plain)];
                for x in 0..self.cols {
                    let cell = y * self.cols + x;
                    let mask = self.masks[cell];
                    let (ch, role) = if mask == 0 {
                        (' ', Role::Plain)
                    } else {
                        (char::from_u32(0x2800 + mask as u32).unwrap_or(' '), self.role(cell))
                    };
                    match segments.last_mut() {
                        Some((s, r)) if *r == role => s.push(ch),
                        _ => segments.push((ch.to_string(), role)),
                    }
                }
                PanelLine::text(segments)
            })
            .collect()
    }

    /// Render dots as an anti-aliased RGBA image for the Kitty backend.
    /// Each dot becomes a soft disc joined to its set neighbours (right,
    /// down, both diagonals) by capsules, so dotted braille lines come out
    /// as smooth strokes. Colors come from the theme's plot gradient;
    /// empty space is transparent (the panel shows through).
    fn to_png(&self, theme: &Theme) -> Option<Vec<u8>> {
        const S: f32 = 4.0; // pixels per dot
        let (dw, dh) = (self.cols * 2, self.rows * 4);
        let (w, h) = ((dw as f32 * S) as u32, (dh as f32 * S) as u32);
        let mut img = image::RgbaImage::new(w, h);
        let fallback = crate::theme::color_rgb(theme.colors.accent).unwrap_or([255, 255, 255]);
        let color_of = |cell: usize| match self.levels[cell] {
            Some(l) => crate::theme::color_rgb(theme.plot[(l as usize).min(PLOT_LEVELS - 1)])
                .unwrap_or(fallback),
            None => fallback,
        };
        let radius = 0.42 * S;
        let center = |x: usize, y: usize| ((x as f32 + 0.5) * S, (y as f32 + 0.5) * S);
        for y in 0..dh {
            for x in 0..dw {
                if !self.dot(x, y) {
                    continue;
                }
                let rgb = color_of((y / 4) * self.cols + x / 2);
                let a = center(x, y);
                stamp_capsule(&mut img, a, a, radius, rgb);
                for (nx, ny) in [(x + 1, y), (x, y + 1), (x + 1, y + 1), (x.wrapping_sub(1), y + 1)] {
                    if nx < dw && ny < dh && self.dot(nx, ny) {
                        stamp_capsule(&mut img, a, center(nx, ny), radius, rgb);
                    }
                }
            }
        }
        let mut out = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(img).write_to(&mut out, image::ImageFormat::Png).ok()?;
        Some(out.into_inner())
    }

    /// Upload as a Kitty image (deduplicated by content + colors).
    fn upload(&self, theme: &Theme) -> Option<u32> {
        if self.cols > kitty::MAX_CELLS as usize || self.rows > kitty::MAX_CELLS as usize {
            return None;
        }
        let mut h = std::collections::hash_map::DefaultHasher::new();
        ("plot", &self.masks, &self.levels, self.cols, self.rows).hash(&mut h);
        format!("{:?}{:?}", theme.plot, theme.colors.accent).hash(&mut h);
        kitty::upload(h.finish(), self.cols as u16, self.rows as u16, || self.to_png(theme))
    }
}

/// Anti-aliased capsule (segment `a`-`b` thickened by `r`); the strongest
/// coverage per pixel wins, with its color.
fn stamp_capsule(img: &mut image::RgbaImage, a: (f32, f32), b: (f32, f32), r: f32, rgb: [u8; 3]) {
    let (w, h) = (img.width() as f32, img.height() as f32);
    let x0 = (a.0.min(b.0) - r - 1.0).max(0.0) as u32;
    let y0 = (a.1.min(b.1) - r - 1.0).max(0.0) as u32;
    let x1 = (a.0.max(b.0) + r + 1.0).min(w) as u32;
    let y1 = (a.1.max(b.1) + r + 1.0).min(h) as u32;
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let len2 = dx * dx + dy * dy;
    for py in y0..y1 {
        for px in x0..x1 {
            let (qx, qy) = (px as f32 + 0.5 - a.0, py as f32 + 0.5 - a.1);
            let t = if len2 > 0.0 { ((qx * dx + qy * dy) / len2).clamp(0.0, 1.0) } else { 0.0 };
            let d = ((qx - t * dx).powi(2) + (qy - t * dy).powi(2)).sqrt();
            let alpha = ((r - d + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
            let p = img.get_pixel_mut(px, py);
            if alpha > p[3] {
                *p = image::Rgba([rgb[0], rgb[1], rgb[2], alpha]);
            }
        }
    }
}

fn plot_lines(plot: &Plot, theme: &Theme) -> Vec<PanelLine> {
    if graphics() == Graphics::Kitty {
        if let Some(id) = plot.upload(theme) {
            return (0..plot.rows)
                .map(|row| PanelLine {
                    segments: Vec::new(),
                    image: Some(ImageRow { id, cols: plot.cols as u16, row: row as u16 }),
                })
                .collect();
        }
    }
    plot.braille_lines()
}

/// ` low ■■■■■■■■■■ high   z -0.40 … 0.40`
fn legend(plot: &Plot) -> PanelLine {
    let mut segments = vec![(" low ".to_string(), Role::Muted)];
    for i in 0..PLOT_LEVELS {
        segments.push(("■".into(), Role::PlotLevel(i as u8)));
    }
    segments.push((" high".into(), Role::Muted));
    if let Some((lo, hi)) = plot.z_range {
        segments.push((format!("   z {lo:.2} … {hi:.2}"), Role::Dim));
    }
    PanelLine::text(segments)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sugi_doc() -> String {
        serde_json::json!({
            "render": "sugi-v1",
            "title": "flow",
            "width": 40.0,
            "height": 9.0,
            "nodes": [
                {"id": "A", "label": "Intake", "x": 4.0, "y": 0.0, "w": 12.0, "h": 3.0},
                {"id": "B", "label": "Plan", "x": 18.0, "y": 6.0, "w": 8.0, "h": 3.0}
            ],
            "edges": [{"from": "A", "to": "B", "label": "go"}]
        })
        .to_string()
    }

    #[test]
    fn sugi_doc_parses_and_renders_boxes_and_orthogonal_edge() {
        let diagram = SugiDiagram::parse(&sugi_doc()).expect("sugi-v1 doc parses");
        assert_eq!(diagram.title, "flow");
        assert_eq!(diagram.nodes.len(), 2);
        let rows = diagram.render_rows(80);
        let art = rows.join("\n");
        // Rounded box around each label.
        assert!(art.contains('╭') && art.contains('╯'), "boxes:\n{art}");
        assert!(art.contains("Intake") && art.contains("Plan"), "labels:\n{art}");
        // Orthogonal connector: vertical run + arrow into the target.
        assert!(art.contains('│'), "vertical line:\n{art}");
        assert!(art.contains('▼'), "arrowhead:\n{art}");
        // Edge label rides the horizontal segment.
        assert!(art.contains('└') || art.contains('┌'), "elbow:\n{art}");
    }

    #[test]
    fn sugi_renders_within_width_cap() {
        // serde_json compacts: no spaces in the pattern.
        let doc = sugi_doc().replace("\"width\":40.0", "\"width\":40.0,\"max_width\":20.0");
        let diagram = SugiDiagram::parse(&doc).unwrap();
        for row in diagram.render_rows(80) {
            assert!(row.chars().count() <= 21, "row over cap: {row:?}");
        }
    }

    #[test]
    fn sugi_parse_rejects_other_documents() {
        assert!(SugiDiagram::parse("{\"render\": \"braille-v1\"}").is_none());
        assert!(SugiDiagram::parse("not json").is_none());
        // Missing nodes array → not a sugi doc.
        assert!(SugiDiagram::parse("{\"render\": \"sugi-v1\"}").is_none());
    }

    #[test]
    fn tool_block_routes_sugi_output_to_the_diagram_body() {
        let block = ToolBlock::response(Some("id".into()), "sugi".into(), &sugi_doc(), false);
        assert!(matches!(block.body, ToolBody::Sugi(_)), "body: {:?}", block.body);
        assert_eq!(block.subtitle, "flow");
    }


    fn doc(rows: &[&str], levels: &[&str]) -> String {
        serde_json::json!({
            "render": "braille-v1", "title": "sin(x)", "width": rows[0].chars().count(),
            "rows": rows, "levels": levels, "z_range": [-1.0, 1.0]
        })
        .to_string()
    }

    #[test]
    fn parses_render_doc() {
        let p = Plot::parse(&doc(&["⣿⠀⠁", "⠀⠀⠀"], &["9..", "..."])).unwrap();
        assert_eq!((p.cols, p.rows), (3, 2));
        assert_eq!(p.levels[0], Some(9));
        assert_eq!(p.levels[1], None, "empty cell has no level");
        assert_eq!(p.levels[2], None, "missing level digit");
        assert_eq!(p.title, "sin(x)");
        assert_eq!(p.z_range, Some((-1.0, 1.0)));
    }

    #[test]
    fn parses_legacy_braille_but_not_prose() {
        let p = Plot::parse("⠁⠂\n⠃⠇\n").unwrap();
        assert_eq!((p.cols, p.rows), (2, 2));
        assert!(!p.has_levels());
        assert!(Plot::parse("hello ⠁").is_none());
        assert!(Plot::parse("{\"render\": \"other\"}").is_none());
    }

    #[test]
    fn fit_downscales_keeping_dots_and_levels() {
        // A full 8-col row of solid cells at level 3, last one level 7.
        let p = Plot::parse(&doc(&["⣿⣿⣿⣿⣿⣿⣿⣿"], &["33333337"])).unwrap();
        let f = p.fit(4);
        assert_eq!(f.cols, 4);
        assert!(f.rows >= 1);
        assert!(f.masks.iter().take(4).all(|m| *m != 0), "no gaps after pooling");
        assert_eq!(f.levels[3], Some(7), "max level wins");
        // No-op when it already fits.
        assert_eq!(p.fit(80), p);
    }

    #[test]
    fn braille_runs_group_by_level() {
        let p = Plot::parse(&doc(&["⣿⣿⠀⣿"], &["11.9"])).unwrap();
        let lines = p.braille_lines();
        let segs = &lines[0].segments;
        assert_eq!(segs[0], (" ".into(), Role::Plain), "1-col inset");
        assert_eq!(segs[1], ("⣿⣿".into(), Role::PlotLevel(1)));
        assert_eq!(segs[2], (" ".into(), Role::Plain));
        assert_eq!(segs[3], ("⣿".into(), Role::PlotLevel(9)));
    }

    #[test]
    fn png_has_colored_dots_and_transparency() {
        let theme = crate::theme::Theme::default();
        let p = Plot::parse(&doc(&["⠁⠀"], &["9."])).unwrap();
        let png = p.to_png(&theme).unwrap();
        let img = image::load_from_memory(&png).unwrap().to_rgba8();
        assert_eq!((img.width(), img.height()), (16, 16));
        assert!(img.get_pixel(2, 2)[3] > 200, "dot 1 is opaque");
        assert_eq!(img.get_pixel(12, 12)[3], 0, "empty cell is transparent");
    }

    #[test]
    fn block_states_and_layout() {
        let theme = crate::theme::Theme::default();
        let pending = ToolBlock::pending("id1".into(), "landscape".into(), r#"{"expr":"x^2"}"#);
        assert_eq!(pending.state, ToolState::Pending);
        assert_eq!(pending.subtitle, "x^2");
        let lines = pending.layout(40, &theme);
        assert!(lines[0].segments.iter().any(|(s, r)| s == "landscape" && *r == Role::ToolTitle));
        assert!(lines[1].segments[0].0.contains("running"));

        let ok = ToolBlock::response(None, "landscape".into(), &doc(&["⣿"], &["5"]), false);
        assert_eq!(ok.state, ToolState::Success);
        let lines = ok.layout(40, &theme);
        assert!(lines.iter().any(|l| l.segments.iter().any(|(_, r)| *r == Role::PlotLevel(5))));
        assert!(lines.iter().any(|l| l.segments.iter().any(|(s, _)| s == " low ")), "legend");

        let err = ToolBlock::response(None, "landscape".into(), "tool failed: boom", true);
        assert_eq!(err.state, ToolState::Error);
        let lines = err.layout(40, &theme);
        assert!(lines[1].segments[0] == (" tool failed: boom".into(), Role::Error));
    }

    #[test]
    fn args_summary() {
        assert_eq!(summarize_args(r#"{"expr": "sin(x)", "grid": 9}"#), "sin(x)  grid=9");
        assert_eq!(summarize_args(r#"{"grid": 9}"#), "grid=9");
        assert_eq!(summarize_args(r#"{"expr": "cos(x)*sin(y)", "pitch": 0, "distance": 1}"#),
                   "cos(x)*sin(y)  distance=1  pitch=0");
        // A title wins; the bulky 'graph' payload stays out of the subtitle.
        assert_eq!(
            summarize_args(r#"{"graph": "A: Alpha\nB: Beta\nA -> B\n", "title": "agent loop"}"#),
            "agent loop"
        );
        assert_eq!(truncate("abcdef", 4), "abc…");
    }
}
