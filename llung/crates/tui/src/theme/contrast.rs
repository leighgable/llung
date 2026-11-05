//! WCAG 2 contrast and lightness adjustment, shared by the `system` theme
//! generator and derived palettes (e.g. the plot gradient).

use super::oklab;

pub type Rgb = [u8; 3];

/// WCAG 2 relative luminance.
pub fn relative_luminance(c: Rgb) -> f64 {
    let lin = |v: u8| {
        let v = v as f64 / 255.0;
        if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
    };
    0.2126 * lin(c[0]) + 0.7152 * lin(c[1]) + 0.0722 * lin(c[2])
}

/// WCAG 2 contrast ratio, 1-21.
pub fn wcag_contrast(a: Rgb, b: Rgb) -> f64 {
    let (la, lb) = (relative_luminance(a), relative_luminance(b));
    (la.max(lb) + 0.05) / (la.min(lb) + 0.05)
}

/// Move `color` toward white (`lighter`) or black, keeping its OKHSL hue
/// and saturation, until it reaches `min` contrast on every surface.
/// Returns the extreme if even that falls short.
pub fn ensure_contrast(color: Rgb, surfaces: &[Rgb], min: f64, lighter: bool) -> Rgb {
    let meets = |c: Rgb| surfaces.iter().all(|s| wcag_contrast(c, *s) >= min);
    if meets(color) {
        return color;
    }
    let (h, s, l) = oklab::rgb_to_okhsl(color);
    let at = |x: f64| oklab::okhsl_to_rgb(h, s.clamp(0.0, 1.0), x.clamp(0.0, 1.0));
    let extreme = if lighter { 1.0 } else { 0.0 };
    if !meets(at(extreme)) {
        return at(extreme);
    }
    let (mut lo, mut hi) = (l, extreme);
    for _ in 0..20 {
        let mid = (lo + hi) / 2.0;
        if meets(at(mid)) { hi = mid } else { lo = mid }
    }
    at(hi)
}

/// Interpolate two colors in OKLCH (shortest hue path), gamut-mapped.
pub fn mix_oklch(a: Rgb, b: Rgb, t: f64) -> Rgb {
    let lch = |c: Rgb| {
        let [l, x, y] = oklab::rgb_to_oklab(c);
        (l, x.hypot(y), y.atan2(x).to_degrees().rem_euclid(360.0))
    };
    let (l1, c1, h1) = lch(a);
    let (l2, c2, h2) = lch(b);
    // Achromatic ends take the other end's hue.
    let h1 = if c1 < 1e-4 { h2 } else { h1 };
    let h2 = if c2 < 1e-4 { h1 } else { h2 };
    let dh = ((h2 - h1 + 540.0) % 360.0) - 180.0;
    oklab::oklch_to_rgb(l1 + (l2 - l1) * t, c1 + (c2 - c1) * t, h1 + dh * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contrast_extremes() {
        assert!((wcag_contrast([0, 0, 0], [255, 255, 255]) - 21.0).abs() < 0.01);
        assert!((wcag_contrast([9, 9, 9], [9, 9, 9]) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn ensure_contrast_lightens_on_dark() {
        let bg = [20, 20, 30];
        let c = ensure_contrast([30, 30, 60], &[bg], 3.0, true);
        assert!(wcag_contrast(c, bg) >= 3.0);
        let already = [240, 240, 240];
        assert_eq!(ensure_contrast(already, &[bg], 3.0, true), already);
    }

    #[test]
    fn mix_endpoints() {
        let (a, b) = ([200, 40, 40], [40, 40, 200]);
        let close = |x: Rgb, y: Rgb| x.iter().zip(y).all(|(p, q)| (*p as i32 - q as i32).abs() <= 1);
        assert!(close(mix_oklch(a, b, 0.0), a));
        assert!(close(mix_oklch(a, b, 1.0), b));
    }
}
