use image::{RgbaImage, imageops::FilterType};
use ratatui::{buffer::Buffer, style::Color};

#[derive(Clone)]
pub struct AvatarThumbnail {
    width: u16,
    height: u16,
    pixels: Vec<(Color, Color)>, // (top_fg, bottom_bg) per cell
}

pub fn decode_avatar_hex(bytes: &[u8], cell_width: u16) -> Option<AvatarThumbnail> {
    let img = image::load_from_memory(bytes).ok()?;
    let cell_height = cell_width / 2; // terminal cells are ~2:1, so 4x2 cells = 4x4 pixels
    let pixel_w = cell_width as u32;
    let pixel_h = cell_height * 2;

    let thumb = img.resize_exact(cell_width as u32, pixel_h as u32, FilterType::Triangle);
    let rgba = thumb.to_rgba8();

    let mut pixels = Vec::with_capacity((cell_width * cell_height) as usize);
    let cx = pixel_w as f32 / 2.0;
    let cy = pixel_h as f32 / 2.0;
    let radius = cx.min(cy);

    for ty in 0..cell_height {
        for tx in 0..cell_width {
            let px = tx as u32;
            let py_top = ty as u32 * 2;
            let py_bot = py_top + 1;

            let top = sample_hex(&rgba, px, py_top, cx, cy, radius);
            let bot = sample_hex(&rgba, px, py_bot, cx, cy, radius);

            pixels.push((top, bot));
        }
    }
    Some(AvatarThumbnail {
        width: cell_width,
        height: cell_height,
        pixels,
    })
}

pub fn sample_hex(img: &RgbaImage, x: u32, y: u32, cx: f32, cy: f32, radius: f32) -> Color {
    let dx = (x as f32 + 0.5 - cx).abs();
    let dy = (y as f32 + 0.5 - cy).abs();

    let hex_dist = dx * 0.866025 + dy * 0.5;
    if hex_dist > radius {
        return Color::Black;
    }

    let pixel = img.get_pixel(x.min(img.width() - 1), y.min(img.height() - 1));
    Color::Rgb(pixel[0], pixel[1], pixel[2])
}

pub fn render_avatar(buf: &mut Buffer, x: u16, y: u16, thumb: &AvatarThumbnail) {
    for (i, (fg, bg)) in thumb.pixels.iter().enumerate() {
        let cx = (i as u16) % thumb.width;
        let cy = (i as u16) / thumb.width;
        if let Some(cell) = buf.cell_mut((x + cx, y + cy)) {
            cell.set_symbol("▀");
            cell.set_fg(*fg);
            cell.set_bg(*bg);
        }
    }
}

pub fn fallback_avatar(buf: &mut Buffer, x: u16, y: u16, initial: char, bg: Color) {
    let initial_str = initial.to_string();
    for dy in 0..2 {
        for dx in 0..2 {
            if let Some(cell) = buf.cell_mut((x + dx, y + dy)) {
                cell.set_bg(bg);
                cell.set_fg(Color::White);
                cell.set_symbol(if dx == 0 && dy == 0 {
                    &initial_str
                } else {
                    " "
                });
            }
        }
    }
}
