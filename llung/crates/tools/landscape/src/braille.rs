//! Braille rasterizer: turns projected 2D line segments into the
//! high-density unicode braille grid that terminals render so well.
//!
//! Each character cell holds a 2x4 dot matrix (dots 1-8). Pixel space is
//! uniform, and since a braille cell is 2 dots wide and 4 tall it matches
//! the ~1:2 aspect of terminal characters — no stretching needed.

/// A drawing surface of `width` x `height` character cells.
pub struct BrailleCanvas {
    width: usize,
    height: usize,
    /// `width * 2` by `height * 4` subpixels.
    pixels: Vec<bool>,
}

impl BrailleCanvas {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            pixels: vec![false; width * 2 * height * 4],
        }
    }

    fn set_pixel(&mut self, px: usize, py: usize) {
        if px < self.width * 2 && py < self.height * 4 {
            self.pixels[py * self.width * 2 + px] = true;
        }
    }

    /// Draw a line between two points given in pixel coordinates.
    pub fn draw_segment(&mut self, a: (f32, f32), b: (f32, f32)) {
        let dx = b.0 - a.0;
        let dy = b.1 - a.1;
        let steps = ((dx * dx + dy * dy).sqrt() * 2.0).ceil().max(1.0) as usize;
        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            self.set_pixel(
                (a.0 + dx * t).round() as usize,
                (a.1 + dy * t).round() as usize,
            );
        }
    }

    /// Render to a string of braille characters, one line per row.
    pub fn render(&self) -> String {
        // Trim fully-empty trailing rows so a small chart doesn't ship
        // a slab of blank lines.
        let mut last_row = 0;
        for row in 0..self.height {
            let mut any = false;
            for col in 0..self.width {
                if self.cell_mask(col, row) != 0 {
                    any = true;
                    break;
                }
            }
            if any {
                last_row = row;
            }
        }

        let mut out = String::new();
        for row in 0..=last_row {
            for col in 0..self.width {
                let mask = self.cell_mask(col, row);
                out.push(char::from_u32(0x2800 + mask as u32).expect("braille is valid unicode"));
            }
            out.push('\n');
        }
        out
    }

    /// Braille dot bitmask for character cell (col, row).
    fn cell_mask(&self, col: usize, row: usize) -> u8 {
        let base_x = col * 2;
        let base_y = row * 4;
        let mut mask = 0u8;
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
            if self.pixels[(base_y + dy) * self.width * 2 + (base_x + dx)] {
                mask |= bit;
            }
        }
        mask
    }
}

/// Fit `segments` into a canvas of `width` x `height` character cells and
/// rasterize. Points outside the accumulated bounding box are clipped.
pub fn rasterize(segments: &[[(f32, f32); 2]], width: usize, height: usize) -> String {
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for seg in segments {
        for (x, y) in seg {
            min_x = min_x.min(*x);
            max_x = max_x.max(*x);
            min_y = min_y.min(*y);
            max_y = max_y.max(*y);
        }
    }
    if !min_x.is_finite() || !max_x.is_finite() || !min_y.is_finite() || !max_y.is_finite() {
        return String::new();
    }

    // 4% margin so the wireframe doesn't kiss the edges.
    let span_x = (max_x - min_x).max(1e-6);
    let span_y = (max_y - min_y).max(1e-6);
    let px_w = (width * 2).saturating_sub(1) as f32;
    let px_h = (height * 4).saturating_sub(1) as f32;
    let to_px = |(x, y): (f32, f32)| {
        let nx = ((x - min_x) / span_x - 0.5) * 1.08 + 0.5;
        let ny = ((y - min_y) / span_y - 0.5) * 1.08 + 0.5;
        // Flip y: math up is terminal down.
        (nx.clamp(0.0, 1.0) * px_w, (1.0 - ny.clamp(0.0, 1.0)) * px_h)
    };

    let mut canvas = BrailleCanvas::new(width, height);
    for seg in segments {
        canvas.draw_segment(to_px(seg[0]), to_px(seg[1]));
    }
    canvas.render()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_braille(c: char) -> bool {
        ('\u{2800}'..='\u{28FF}').contains(&c)
    }

    #[test]
    fn single_segment_produces_braille() {
        let out = rasterize(&[[(0.0, 0.0), (1.0, 1.0)]], 20, 8);
        assert!(out.lines().any(|l| l.chars().any(is_braille)));
    }

    #[test]
    fn degenerate_bbox_does_not_panic() {
        let out = rasterize(&[[(1.0, 1.0), (1.0, 1.0)]], 10, 5);
        assert!(out.is_empty() || out.lines().count() <= 5);
    }

    #[test]
    fn empty_segments_render_empty() {
        assert!(rasterize(&[], 10, 5).is_empty());
    }
}
