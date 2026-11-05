//! Thumbnails for avatars and image attachments.
//!
//! A thumbnail occupies a fixed block of cells, so layout code (Taffy,
//! chat wrap) is the same for every backend. What fills the cells depends
//! on the host terminal (see `terminal::graphics`):
//! - Kitty: the real image, via Unicode placeholders;
//! - half-blocks: `▀`/`▄` cells, two pixels per cell.

use crate::terminal::graphics::{Graphics, graphics};
use crate::terminal::kitty;
use image::{DynamicImage, RgbaImage, imageops::FilterType};
use ratatui::{buffer::Buffer, style::Color};
use std::hash::{Hash, Hasher};

#[derive(Clone)]
pub struct AvatarThumbnail {
    width: u16,
    height: u16,
    kind: ThumbKind,
}

#[derive(Clone)]
enum ThumbKind {
    /// (top, bottom) pixel per cell; `None` = transparent (outside the
    /// hexagon mask), showing whatever surface is below.
    HalfBlocks(Vec<(Option<Color>, Option<Color>)>),
    /// Uploaded Kitty image ID.
    Kitty(u32),
}

impl AvatarThumbnail {
    pub fn width(&self) -> u16 {
        self.width
    }

    pub fn height(&self) -> u16 {
        self.height
    }
}

/// Hexagon-masked avatar, `cell_width` x `cell_width / 2` cells (about
/// square, since cells are roughly 1:2).
pub fn decode_avatar_hex(bytes: &[u8], cell_width: u16) -> Option<AvatarThumbnail> {
    let cell_height = (cell_width / 2).max(1);
    match graphics() {
        Graphics::Kitty => {
            let key = source_key(bytes, "avatar-hex", cell_width, cell_height);
            let id = kitty::upload(key, cell_width, cell_height, || {
                let img = image::load_from_memory(bytes).ok()?;
                // Plenty for a few cells even on HiDPI; the terminal scales.
                let mut rgba = img.resize_exact(128, 128, FilterType::Triangle).to_rgba8();
                apply_hex_mask(&mut rgba);
                encode_png(&rgba)
            })?;
            Some(AvatarThumbnail { width: cell_width, height: cell_height, kind: ThumbKind::Kitty(id) })
        }
        Graphics::HalfBlocks => {
            let img = image::load_from_memory(bytes).ok()?;
            let mut rgba = img
                .resize_exact(cell_width as u32, cell_height as u32 * 2, FilterType::Triangle)
                .to_rgba8();
            apply_hex_mask(&mut rgba);
            Some(halfblocks(&rgba, cell_width, cell_height))
        }
    }
}

/// Image attachment thumbnail. With Kitty graphics: the unmasked image,
/// aspect-correct, up to [`ATTACHMENT_MAX_COLS`] x [`ATTACHMENT_MAX_ROWS`]
/// cells. With half-blocks: the small hexagon thumbnail, as before
/// (finer detail doesn't survive two pixels per cell).
pub fn decode_attachment(bytes: &[u8]) -> Option<AvatarThumbnail> {
    match graphics() {
        Graphics::HalfBlocks => decode_avatar_hex(bytes, 6),
        Graphics::Kitty => {
            let img = image::load_from_memory(bytes).ok()?;
            let (cols, rows) = attachment_cells(img.width(), img.height());
            let key = source_key(bytes, "attachment", cols, rows);
            let id = kitty::upload(key, cols, rows, || {
                // Cap the upload size; the cell box is at most ~400px wide.
                encode_png(&img.thumbnail(512, 512).to_rgba8())
            })?;
            Some(AvatarThumbnail { width: cols, height: rows, kind: ThumbKind::Kitty(id) })
        }
    }
}

pub const ATTACHMENT_MAX_COLS: u16 = 32;
pub const ATTACHMENT_MAX_ROWS: u16 = 12;

/// Cell box for an image of `w` x `h` pixels, assuming cells are twice as
/// tall as wide. The terminal fits the image inside, so rounding only
/// affects padding.
fn attachment_cells(w: u32, h: u32) -> (u16, u16) {
    let aspect = w.max(1) as f64 / h.max(1) as f64; // width / height
    let max_c = ATTACHMENT_MAX_COLS as f64;
    let max_r = ATTACHMENT_MAX_ROWS as f64;
    // Rows at full height; cols follow from aspect (2 cols per row-height).
    let (cols, rows) = if max_r * 2.0 * aspect <= max_c {
        (max_r * 2.0 * aspect, max_r)
    } else {
        (max_c, max_c / (2.0 * aspect))
    };
    let clamp = |v: f64, max: u16| (v.round() as u16).clamp(1, max.min(kitty::MAX_CELLS));
    (clamp(cols, ATTACHMENT_MAX_COLS), clamp(rows, ATTACHMENT_MAX_ROWS))
}

fn source_key(bytes: &[u8], kind: &str, cols: u16, rows: u16) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut h);
    (kind, cols, rows).hash(&mut h);
    h.finish()
}

fn encode_png(img: &RgbaImage) -> Option<Vec<u8>> {
    let mut out = std::io::Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(img.clone())
        .write_to(&mut out, image::ImageFormat::Png)
        .ok()?;
    Some(out.into_inner())
}

/// Make everything outside a pointy-top hexagon transparent.
fn apply_hex_mask(img: &mut RgbaImage) {
    let (w, h) = (img.width() as f32, img.height() as f32);
    let (cx, cy) = (w / 2.0, h / 2.0);
    let radius = cx.min(cy);
    for (x, y, px) in img.enumerate_pixels_mut() {
        let dx = (x as f32 + 0.5 - cx).abs();
        let dy = (y as f32 + 0.5 - cy).abs();
        if dx * 0.866025 + dy * 0.5 > radius {
            px[3] = 0;
        }
    }
}

/// Pack an image of `cols` x `rows * 2` pixels into half-block cells.
fn halfblocks(rgba: &RgbaImage, cols: u16, rows: u16) -> AvatarThumbnail {
    let px = |x: u32, y: u32| {
        let p = rgba.get_pixel(x.min(rgba.width() - 1), y.min(rgba.height() - 1));
        (p[3] >= 128).then(|| Color::Rgb(p[0], p[1], p[2]))
    };
    let mut pixels = Vec::with_capacity(cols as usize * rows as usize);
    for ty in 0..rows as u32 {
        for tx in 0..cols as u32 {
            pixels.push((px(tx, ty * 2), px(tx, ty * 2 + 1)));
        }
    }
    AvatarThumbnail { width: cols, height: rows, kind: ThumbKind::HalfBlocks(pixels) }
}

pub fn render_avatar(buf: &mut Buffer, x: u16, y: u16, thumb: &AvatarThumbnail) {
    match &thumb.kind {
        ThumbKind::Kitty(id) => kitty::render(buf, x, y, *id, thumb.width, thumb.height),
        ThumbKind::HalfBlocks(pixels) => {
            for (i, (top, bottom)) in pixels.iter().enumerate() {
                let cx = (i as u16) % thumb.width;
                let cy = (i as u16) / thumb.width;
                let Some(cell) = buf.cell_mut((x + cx, y + cy)) else { continue };
                // Transparent halves keep the surface's background.
                match (top, bottom) {
                    (Some(t), Some(b)) => {
                        cell.set_symbol("▀").set_fg(*t).set_bg(*b);
                    }
                    (Some(t), None) => {
                        cell.set_symbol("▀").set_fg(*t);
                    }
                    (None, Some(b)) => {
                        cell.set_symbol("▄").set_fg(*b);
                    }
                    (None, None) => {
                        cell.set_symbol(" ");
                    }
                }
            }
        }
    }
}

pub fn fallback_avatar(buf: &mut Buffer, x: u16, y: u16, initial: char, bg: Color, fg: Color) {
    let initial_str = initial.to_string();
    for dy in 0..2 {
        for dx in 0..2 {
            if let Some(cell) = buf.cell_mut((x + dx, y + dy)) {
                cell.set_bg(bg);
                cell.set_fg(fg);
                cell.set_symbol(if dx == 0 && dy == 0 {
                    &initial_str
                } else {
                    " "
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png(w: u32, h: u32) -> Vec<u8> {
        encode_png(&RgbaImage::from_pixel(w, h, image::Rgba([10, 200, 30, 255]))).unwrap()
    }

    #[test]
    fn halfblock_hex_is_masked_and_sized() {
        // Tests run with the default backend (half-blocks).
        let t = decode_avatar_hex(&png(16, 16), 4).unwrap();
        assert_eq!((t.width(), t.height()), (4, 2));
        let ThumbKind::HalfBlocks(px) = &t.kind else { panic!("half-blocks expected") };
        assert!(px.iter().any(|(a, b)| a.is_none() || b.is_none()), "corners masked");
        assert!(px.iter().any(|(a, _)| *a == Some(Color::Rgb(10, 200, 30))), "center opaque");
    }

    #[test]
    fn transparent_halves_keep_surface() {
        let t = decode_avatar_hex(&png(16, 16), 4).unwrap();
        let mut buf = Buffer::empty(ratatui::layout::Rect::new(0, 0, 4, 2));
        for cell in buf.content.iter_mut() {
            cell.set_bg(Color::Rgb(1, 2, 3));
        }
        render_avatar(&mut buf, 0, 0, &t);
        // Top-left corner is outside the hexagon in both halves.
        assert_eq!(buf[(0, 0)].bg, Color::Rgb(1, 2, 3));
    }

    #[test]
    fn attachment_boxes() {
        // Square image: rows maxed, cols = 2x rows (cells are 1:2).
        assert_eq!(attachment_cells(100, 100), (24, 12));
        // Wide image: cols maxed.
        assert_eq!(attachment_cells(1600, 400), (32, 4));
        // Tall image: rows maxed, narrow.
        assert_eq!(attachment_cells(100, 1000), (2, 12));
        // Degenerate.
        assert_eq!(attachment_cells(0, 0), (24, 12));
        assert_eq!(attachment_cells(10000, 1), (32, 1));
    }

    #[test]
    fn hex_mask_center_and_corner() {
        let mut img = RgbaImage::from_pixel(64, 64, image::Rgba([0, 0, 0, 255]));
        apply_hex_mask(&mut img);
        assert_eq!(img.get_pixel(32, 32)[3], 255);
        assert_eq!(img.get_pixel(0, 0)[3], 0);
    }
}
