//! Braille rasterizer: turns projected 2D line segments into the
//! high-density unicode braille grid that terminals render so well.
//!
//! Each character cell holds a 2x4 dot matrix (dots 1-8). Pixel space is
//! uniform, and since a braille cell is 2 dots wide and 4 tall it matches
//! the ~1:2 aspect of terminal characters — no stretching needed.
//!
//! Every dot also remembers the highest surface value `z` drawn onto it,
//! so each cell carries a height *level* (0-9) the viewer can color by.

/// A projected segment endpoint: screen `x`, `y` and surface height `z`.
pub type Point = (f32, f32, f32);

/// A drawing surface of `width` x `height` character cells.
pub struct BrailleCanvas {
    width: usize,
    height: usize,
    /// `width * 2` by `height * 4` subpixels: highest `z` drawn, or NaN.
    pixels: Vec<f32>,
}

/// Rasterized chart: braille rows plus one height level per cell.
#[derive(Debug, Clone, PartialEq)]
pub struct Raster {
    pub width: usize,
    /// Braille rows (trailing empty rows trimmed).
    pub rows: Vec<String>,
    /// Per cell: `'0'..='9'` height level, `'.'` for empty cells.
    pub levels: Vec<String>,
}

impl BrailleCanvas {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            pixels: vec![f32::NAN; width * 2 * height * 4],
        }
    }

    fn set_pixel(&mut self, px: usize, py: usize, z: f32) {
        if px < self.width * 2 && py < self.height * 4 {
            let p = &mut self.pixels[py * self.width * 2 + px];
            // NaN.max(z) == z, so the first write always lands.
            *p = p.max(z);
        }
    }

    /// Draw a line between two pixel-space points, interpolating `z`.
    pub fn draw_segment(&mut self, a: Point, b: Point) {
        let dx = b.0 - a.0;
        let dy = b.1 - a.1;
        let steps = ((dx * dx + dy * dy).sqrt() * 2.0).ceil().max(1.0) as usize;
        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            self.set_pixel(
                (a.0 + dx * t).round() as usize,
                (a.1 + dy * t).round() as usize,
                a.2 + (b.2 - a.2) * t,
            );
        }
    }

    /// Braille dot bitmask and highest `z` for character cell (col, row).
    fn cell(&self, col: usize, row: usize) -> (u8, f32) {
        let base_x = col * 2;
        let base_y = row * 4;
        let mut mask = 0u8;
        let mut z = f32::NAN;
        for (dx, dy, bit) in [
            (0, 0, 0x01), // dot 1
            (0, 1, 0x02), // dot 2
            (0, 2, 0x04), // dot 3
            (1, 0, 0x08), // dot 4
            (1, 1, 0x10), // dot 5
            (1, 2, 0x20), // dot 6
            (1, 3, 0x40), // dot 7
            (0, 3, 0x80), // dot 8
        ] {
            let p = self.pixels[(base_y + dy) * self.width * 2 + (base_x + dx)];
            if !p.is_nan() {
                mask |= bit;
                z = z.max(p);
            }
        }
        (mask, z)
    }

    /// Render rows and levels; `z_range` maps heights onto levels 0-9.
    pub fn render(&self, z_range: (f32, f32)) -> Raster {
        let (z_min, z_max) = z_range;
        let span = (z_max - z_min).max(1e-6);
        let mut rows = Vec::with_capacity(self.height);
        let mut levels = Vec::with_capacity(self.height);
        let mut last_nonempty = None;
        for row in 0..self.height {
            let mut braille = String::with_capacity(self.width * 3);
            let mut level = String::with_capacity(self.width);
            for col in 0..self.width {
                let (mask, z) = self.cell(col, row);
                braille.push(char::from_u32(0x2800 + mask as u32).expect("braille is valid unicode"));
                level.push(if mask == 0 {
                    '.'
                } else {
                    let l = (((z - z_min) / span) * 9.0).round().clamp(0.0, 9.0) as u8;
                    (b'0' + l) as char
                });
                if mask != 0 {
                    last_nonempty = Some(row);
                }
            }
            rows.push(braille);
            levels.push(level);
        }
        // Trim fully-empty trailing rows so a small chart doesn't ship a
        // slab of blank lines.
        let keep = last_nonempty.map_or(0, |r| r + 1);
        rows.truncate(keep);
        levels.truncate(keep);
        Raster { width: self.width, rows, levels }
    }
}

/// Fit `segments` into a canvas of `width` x `height` character cells and
/// rasterize. Points outside the accumulated bounding box are clipped.
pub fn rasterize(segments: &[[Point; 2]], width: usize, height: usize) -> Raster {
    let empty = Raster { width, rows: Vec::new(), levels: Vec::new() };
    let (mut min_x, mut max_x) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut min_y, mut max_y) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut min_z, mut max_z) = (f32::INFINITY, f32::NEG_INFINITY);
    for seg in segments {
        for (x, y, z) in seg {
            min_x = min_x.min(*x);
            max_x = max_x.max(*x);
            min_y = min_y.min(*y);
            max_y = max_y.max(*y);
            min_z = min_z.min(*z);
            max_z = max_z.max(*z);
        }
    }
    if ![min_x, max_x, min_y, max_y, min_z, max_z].iter().all(|v| v.is_finite()) {
        return empty;
    }

    // 4% margin so the wireframe doesn't kiss the edges.
    let span_x = (max_x - min_x).max(1e-6);
    let span_y = (max_y - min_y).max(1e-6);
    let px_w = (width * 2).saturating_sub(1) as f32;
    let px_h = (height * 4).saturating_sub(1) as f32;
    let to_px = |(x, y, z): Point| {
        let nx = ((x - min_x) / span_x - 0.5) * 1.08 + 0.5;
        let ny = ((y - min_y) / span_y - 0.5) * 1.08 + 0.5;
        // Flip y: math up is terminal down.
        (nx.clamp(0.0, 1.0) * px_w, (1.0 - ny.clamp(0.0, 1.0)) * px_h, z)
    };

    let mut canvas = BrailleCanvas::new(width, height);
    for seg in segments {
        canvas.draw_segment(to_px(seg[0]), to_px(seg[1]));
    }
    canvas.render((min_z, max_z))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_braille(c: char) -> bool {
        ('\u{2800}'..='\u{28FF}').contains(&c)
    }

    #[test]
    fn single_segment_produces_braille() {
        let r = rasterize(&[[(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]], 20, 8);
        assert!(r.rows.iter().any(|l| l.chars().any(|c| is_braille(c) && c != '\u{2800}')));
        assert_eq!(r.rows.len(), r.levels.len());
    }

    #[test]
    fn levels_follow_height() {
        // Bottom-left (z=0) to top-right (z=1): levels rise along the line.
        let r = rasterize(&[[(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]], 20, 8);
        let lv: Vec<u8> = r.levels.iter().flat_map(|l| l.bytes()).filter(|b| *b != b'.').collect();
        assert!(lv.contains(&b'0') && lv.contains(&b'9'), "{lv:?}");
        // Top row holds the high end.
        assert!(r.levels[0].contains('9'));
    }

    #[test]
    fn empty_cells_are_dots_and_match_braille() {
        let r = rasterize(&[[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]], 10, 4);
        for (row, lv) in r.rows.iter().zip(&r.levels) {
            for (c, l) in row.chars().zip(lv.chars()) {
                assert_eq!(c == '\u{2800}', l == '.');
            }
        }
    }

    #[test]
    fn degenerate_bbox_does_not_panic() {
        let r = rasterize(&[[(1.0, 1.0, 0.0), (1.0, 1.0, 0.0)]], 10, 5);
        assert!(r.rows.len() <= 5);
    }

    #[test]
    fn empty_segments_render_empty() {
        assert!(rasterize(&[], 10, 5).rows.is_empty());
    }
}
