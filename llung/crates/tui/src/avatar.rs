//! Thumbnails for avatars and image attachments.
//!
//! A thumbnail occupies a fixed block of cells, so layout code (Taffy,
//! chat wrap) is the same for every backend. What fills the cells depends
//! on the host terminal (see `terminal::graphics`):
//! - Kitty: the real image, via Unicode placeholders;
//! - braille: a pixelized hexagonal translation of the source image —
//!   each cell is a 2x4-dot braille character whose dot pattern carries
//!   the image's ink (inside a pointy-top hexagon mask) and whose color
//!   is sampled from the source pixels under those dots.

use crate::terminal::graphics::{Graphics, graphics};
use crate::terminal::kitty;
use image::{DynamicImage, ImageBuffer, Luma, RgbaImage, imageops::FilterType};
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
    /// One (braille char, color) per cell; `' '` = transparent (outside
    /// the hexagon mask or no ink), showing the surface below.
    Braille(Vec<(char, Color)>),
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
            Some(braille(&img, cell_width, cell_height))
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

/// Braille dot bit for a dot position within a 2x4 cell. Layout:
/// ```text
/// (0,0)=0x01 (0,1)=0x02 (0,2)=0x04   dots 1 2 3
/// (1,0)=0x08 (1,1)=0x10 (1,2)=0x20   dots 4 5 6
/// (0,3)=0x40 (1,3)=0x80              dots 7 8
/// ```
fn braille_bit(dx: u32, dy: u32) -> u32 {
    match (dx, dy) {
        (0, 0) => 0x01,
        (0, 1) => 0x02,
        (0, 2) => 0x04,
        (1, 0) => 0x08,
        (1, 1) => 0x10,
        (1, 2) => 0x20,
        (0, 3) => 0x40,
        (1, 3) => 0x80,
        _ => 0,
    }
}

/// Relative luminance 0..=255.
fn luminance(p: &image::Rgba<u8>) -> f32 {
    0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32
}

/// Estimate the background luminance from the border ring (avatars sit
/// on a surface; the figure/ink is inside). Median keeps stray dark
/// edge pixels from dragging it down.
fn background_luminance(rgba: &RgbaImage) -> f32 {
    let (w, h) = (rgba.width(), rgba.height());
    let mut ring: Vec<f32> = Vec::new();
    for x in 0..w {
        for y in [0, h - 1] {
            ring.push(luminance(rgba.get_pixel(x, y)));
        }
    }
    for y in 0..h {
        for x in [0, w - 1] {
            ring.push(luminance(rgba.get_pixel(x, y)));
        }
    }
    ring.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ring[ring.len() / 2]
}

/// Ink coverage 0..=1 of one pixel: how far it is into "clearly darker
/// than the background". Thin lines get partial coverage, so averaging
/// this map over a dot's source region keeps them alive where a
/// resize-then-threshold would wash them out to paper.
fn ink_of(p: &image::Rgba<u8>, ink_threshold: f32) -> f32 {
    if p[3] < 128 {
        return 0.0;
    }
    ((ink_threshold - luminance(p)) / 96.0).clamp(0.0, 1.0)
}

/// 1.0 inside the pointy-top hexagon, 0.0 outside — same geometry as
/// [`apply_hex_mask`], as a multiplier at dot-grid resolution.
fn hex_mask_at(x: u32, y: u32, w: u32, h: u32) -> f32 {
    let (wf, hf) = (w as f32, h as f32);
    let (cx, cy) = (wf / 2.0, hf / 2.0);
    let radius = cx.min(cy);
    let dx = (x as f32 + 0.5 - cx).abs();
    let dy = (y as f32 + 0.5 - cy).abs();
    f32::from(dx * 0.866025 + dy * 0.5 <= radius)
}

/// Translate `img` into `cols` x `rows` braille cells (2x4 dots each).
///
/// The dot pattern is the pixelized ink of the source: an ink-coverage
/// map (darkness vs the border-estimated background) averaged down to
/// dot resolution and cut by the hexagon mask — thin line art survives
/// where resize-then-threshold would average it away. Each cell's color
/// is the ink-weighted average of the source colors under its on-dots:
/// the translation draws its colors from the image itself. A cell with
/// no on-dots is transparent, showing the surface below.
fn braille(img: &DynamicImage, cols: u16, rows: u16) -> AvatarThumbnail {
    // Cap the source working size: the dot grid is tiny, and a bounded
    // map keeps the per-pixel ink computation cheap. Shrink only —
    // upscaling adds no information and blurs thin lines.
    let src = if img.width() > 256 || img.height() > 256 {
        img.resize(256, 256, FilterType::Triangle).to_rgba8()
    } else {
        img.to_rgba8()
    };
    let (w, h) = (cols as u32 * 2, rows as u32 * 4);

    let bg = background_luminance(&src);
    let ink_threshold = bg - 48.0;
    let ink_full = ImageBuffer::<Luma<f32>, Vec<f32>>::from_fn(src.width(), src.height(), |x, y| {
        image::Luma([ink_of(src.get_pixel(x, y), ink_threshold)])
    });
    // Degenerate input (uniform patch): nothing reads as ink, but the
    // caller still wants the hexagon shape — fall back to "every opaque
    // (masked) pixel is a dot".
    let inkless = !ink_full.pixels().any(|p| p[0] > 0.0);
    let mut ink = image::imageops::resize(&ink_full, w, h, FilterType::Triangle);
    if inkless {
        ink = ImageBuffer::<Luma<f32>, Vec<f32>>::from_fn(w, h, |x, y| {
            image::Luma([hex_mask_at(x, y, w, h)])
        });
    } else {
        for y in 0..h {
            for x in 0..w {
                ink.get_pixel_mut(x, y)[0] *= hex_mask_at(x, y, w, h);
            }
        }
    }

    // Ink colors at dot resolution: channel maps weighted by the
    // per-pixel ink, so a boundary dot's color comes from the DARK
    // pixels in its source area, not the raw paper-and-ink blend (which
    // would wash edge dots to the background color).
    // Inkless input: weight colors by the hex mask instead, so a
    // uniform patch still comes out in its own color (shape fallback).
    let color_weight_src = if inkless {
        ImageBuffer::<Luma<f32>, Vec<f32>>::from_fn(src.width(), src.height(), |x, y| {
            image::Luma([hex_mask_at(x, y, src.width(), src.height())])
        })
    } else {
        ink_full.clone()
    };
    let mut r_ink = ImageBuffer::<Luma<f32>, Vec<f32>>::from_pixel(src.width(), src.height(), image::Luma([0.0]));
    let mut g_ink = ImageBuffer::<Luma<f32>, Vec<f32>>::from_pixel(src.width(), src.height(), image::Luma([0.0]));
    let mut b_ink = ImageBuffer::<Luma<f32>, Vec<f32>>::from_pixel(src.width(), src.height(), image::Luma([0.0]));
    for y in 0..src.height() {
        for x in 0..src.width() {
            let ink = color_weight_src.get_pixel(x, y)[0];
            if ink > 0.0 {
                let p = src.get_pixel(x, y);
                // 0..=1 normalized: imageops::resize clamps f32 pixels
                // to [0, 1], so channel maps live in that range and
                // rescale to 0..=255 at the end.
                r_ink.get_pixel_mut(x, y)[0] = p[0] as f32 / 255.0 * ink;
                g_ink.get_pixel_mut(x, y)[0] = p[1] as f32 / 255.0 * ink;
                b_ink.get_pixel_mut(x, y)[0] = p[2] as f32 / 255.0 * ink;
            }
        }
    }
    let r_ink = image::imageops::resize(&r_ink, w, h, FilterType::Triangle);
    let g_ink = image::imageops::resize(&g_ink, w, h, FilterType::Triangle);
    let b_ink = image::imageops::resize(&b_ink, w, h, FilterType::Triangle);

    const DOT_ON: f32 = 0.25; // ink coverage that counts as a dot
    let mut cells = Vec::with_capacity(cols as usize * rows as usize);
    for ty in 0..rows as u32 {
        for tx in 0..cols as u32 {
            let mut bits = 0u32;
            // Ink-weighted color average over the on-dots: dots count
            // in proportion to their coverage.
            let mut r = 0.0f32;
            let mut g = 0.0f32;
            let mut b = 0.0f32;
            let mut weight = 0.0f32;
            for dy in 0..4u32 {
                for dx in 0..2u32 {
                    let (px, py) = (tx * 2 + dx, ty * 4 + dy);
                    let coverage = ink.get_pixel(px, py)[0];
                    if coverage > DOT_ON {
                        bits |= braille_bit(dx, dy);
                        // Divide by coverage: the channel maps are
                        // ink-weighted, so this recovers the average
                        // color of the ink itself.
                        r += r_ink.get_pixel(px, py)[0];
                        g += g_ink.get_pixel(px, py)[0];
                        b += b_ink.get_pixel(px, py)[0];
                        weight += coverage;
                    }
                }
            }
            if bits == 0 || weight == 0.0 {
                cells.push((' ', Color::Reset));
            } else {
                let scale = 255.0 / weight;
                cells.push((
                    char::from_u32(0x2800 + bits).unwrap_or(' '),
                    Color::Rgb(
                        (r * scale).round().clamp(0.0, 255.0) as u8,
                        (g * scale).round().clamp(0.0, 255.0) as u8,
                        (b * scale).round().clamp(0.0, 255.0) as u8,
                    ),
                ));
            }
        }
    }
    AvatarThumbnail { width: cols, height: rows, kind: ThumbKind::Braille(cells) }
}

pub fn render_avatar(buf: &mut Buffer, x: u16, y: u16, thumb: &AvatarThumbnail) {
    match &thumb.kind {
        ThumbKind::Kitty(id) => kitty::render(buf, x, y, *id, thumb.width, thumb.height),
        ThumbKind::Braille(cells) => {
            for (i, (ch, color)) in cells.iter().enumerate() {
                let cx = (i as u16) % thumb.width;
                let cy = (i as u16) / thumb.width;
                let Some(cell) = buf.cell_mut((x + cx, y + cy)) else { continue };
                if *ch == ' ' {
                    // Transparent cell: keep the surface as-is.
                    cell.set_symbol(" ");
                } else {
                    let s = ch.to_string();
                    cell.set_symbol(&s).set_fg(*color);
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
    fn braille_hex_is_masked_and_sized() {
        // Tests run with the default backend (braille translation).
        let t = decode_avatar_hex(&png(32, 32), 4).unwrap();
        assert_eq!((t.width(), t.height()), (4, 2));
        let ThumbKind::Braille(cells) = &t.kind else { panic!("braille expected") };
        // Every written cell is a real braille character carrying the
        // source color; the uniform image (no ink contrast) fills the
        // hexagon via the inkless fallback.
        let bits = |ch: char| ch as u32 - 0x2800;
        assert!(
            cells.iter().all(|(ch, c)| *ch == ' ' || (1..=0xFF).contains(&bits(*ch))),
            "only blank or dotted-braille cells: {cells:?}"
        );
        assert!(
            cells.iter().any(|(ch, c)| *ch != ' ' && *c == Color::Rgb(10, 200, 30)),
            "center opaque with the source color: {cells:?}"
        );
        // The hexagon mask must carve SOMETHING from the dot grid (the
        // corners) without gutting the image.
        let on: u32 = cells.iter().map(|(ch, _)| bits(*ch).count_ones()).sum();
        let total = cells.len() as u32 * 8;
        assert!(total / 2 < on && on < total, "mask carves the corners: {on}/{total} dots");
    }

    #[test]
    fn braille_cells_carry_source_image_colors() {
        // Dark ink square on a light ground: on-dots are the ink pixels,
        // so cell colors must come from the INK, not the paper. Edge
        // cells blend ink with paper under Triangle resize, hence the
        // "much closer to ink than paper" assertion, not exact equality.
        let mut img = RgbaImage::from_pixel(16, 16, image::Rgba([240, 240, 235, 255]));
        for y in 4..12 {
            for x in 4..12 {
                img.put_pixel(x, y, image::Rgba([30, 40, 160, 255]));
            }
        }
        let t = decode_avatar_hex(&encode_png(&img).unwrap(), 4).unwrap();
        let ThumbKind::Braille(cells) = &t.kind else { panic!("braille expected") };
        let inked: Vec<_> = cells.iter().filter(|(ch, _)| *ch != ' ').collect();
        assert!(!inked.is_empty(), "ink square must produce dots");
        let closer_to_ink = |c: &Color| match c {
            Color::Rgb(r, g, b) => {
                // Ink (30,40,160) vs paper (240,240,235): blue-dominant
                // and dark — no blended pixel drifts near the paper.
                *b as u32 > *r as u32 + 40 && *b as u32 > *g as u32 + 40 && *b < 200
            }
            _ => false,
        };
        assert!(
            inked.iter().all(|(_, c)| closer_to_ink(c)),
            "colors sampled from the source ink: {inked:?}"
        );
    }

    #[test]
    fn transparent_cells_keep_surface() {
        // A small dark square at the center: only the middle cells
        // carry dots, so the corner cells must stay blank and untouched.
        // (Kept big enough to survive the 4x downscale as solid ink —
        // a 1px dot would smear under the resize filter and trip the
        // inkless-shape fallback, filling the whole hexagon.)
        let mut img = RgbaImage::from_pixel(32, 32, image::Rgba([245, 245, 240, 255]));
        for y in 12..20 {
            for x in 12..20 {
                img.put_pixel(x, y, image::Rgba([10, 10, 10, 255]));
            }
        }
        let t = decode_avatar_hex(&encode_png(&img).unwrap(), 4).unwrap();
        let ThumbKind::Braille(cells) = &t.kind else { panic!("braille expected") };
        assert!(
            cells.iter().any(|(ch, _)| *ch != ' '),
            "ink dot must produce dots"
        );
        let mut buf = Buffer::empty(ratatui::layout::Rect::new(0, 0, 4, 2));
        for cell in buf.content.iter_mut() {
            cell.set_bg(Color::Rgb(1, 2, 3));
        }
        render_avatar(&mut buf, 0, 0, &t);
        assert_eq!(buf[(0, 0)].bg, Color::Rgb(1, 2, 3));
        assert_eq!(buf[(0, 0)].symbol(), " ");
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


