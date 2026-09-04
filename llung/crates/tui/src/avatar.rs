use image::{RgbaImage, imageops::FilterType};
use ratatui::{buffer::Buffer, style::Color};

#[derive(Clone)]
pub struct AvatarThumbnail {
    width: u16,
    pixels: Vec<(Color, Color)>, // (top_fg, bottom_bg) per cell
}

pub fn decode_avatar(bytes: &[u8], cell_width: u16) -> Option<AvatarThumbnail> {
    let img = image::load_from_memory(bytes).ok()?;
    let cell_height = cell_width / 2; // terminal cells are ~2:1, so 4x2 cells = 4x4 pixels
    let pixel_h = cell_height * 2;

    let thumb = img.resize_exact(cell_width as u32, pixel_h as u32, FilterType::Triangle);
    let rgba = thumb.to_rgba8();

    let mut pixels = Vec::with_capacity((cell_width * cell_height) as usize);
    for cy in 0..cell_height {
        for cx in 0..cell_width {
            let top = rgba.get_pixel(cx as u32, cy as u32 * 2);
            let bot = rgba.get_pixel(cx as u32, cy as u32 * 2 + 1);
            pixels.push((
                Color::Rgb(top[0], top[1], top[2]),
                Color::Rgb(bot[0], bot[1], bot[2]),
            ));
        }
    }
    Some(AvatarThumbnail {
        width: cell_width,
        pixels,
    })
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
