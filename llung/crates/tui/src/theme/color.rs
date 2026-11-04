//! Theme color literals and terminal color-mode handling.
//!
//! Accepted literal forms match pi's theme format:
//! `"#rgb"`, `"#rrggbb"`, `"oklch(L C H)"`, `"okhsl(H S L)"`, a 256-color
//! index (JSON integer 0-255), and `""` for the terminal default.

use super::oklab;
use ratatui::style::Color;

/// How resolved colors are emitted to the terminal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorMode {
    TrueColor,
    Ansi256,
}

impl ColorMode {
    /// Detect like pi-tui: truecolor when `COLORTERM` is `truecolor`/`24bit`
    /// or `TERM` ends in `-direct`, else 256 colors. `LLUNG_COLOR_MODE`
    /// (`truecolor` | `256`) overrides detection.
    pub fn detect() -> Self {
        let var = |k: &str| std::env::var(k).unwrap_or_default().to_lowercase();
        match var("LLUNG_COLOR_MODE").as_str() {
            "truecolor" | "24bit" => return Self::TrueColor,
            "256" | "256color" => return Self::Ansi256,
            _ => {}
        }
        let colorterm = var("COLORTERM");
        if colorterm == "truecolor" || colorterm == "24bit" || var("TERM").ends_with("-direct") {
            Self::TrueColor
        } else {
            Self::Ansi256
        }
    }

    /// Adapt a resolved color to this mode.
    pub fn adapt(self, color: Color) -> Color {
        match (self, color) {
            (Self::Ansi256, Color::Rgb(r, g, b)) => Color::Indexed(rgb_to_ansi256([r, g, b])),
            _ => color,
        }
    }
}

/// Parse a literal color string (not a variable reference). Returns
/// `Ok(None)` when the string isn't a literal and should be treated as a
/// variable reference.
pub fn parse_literal(value: &str) -> Result<Option<Color>, String> {
    let v = value.trim();
    if v.is_empty() {
        return Ok(Some(Color::Reset));
    }
    if let Some(hex) = v.strip_prefix('#') {
        return parse_hex(hex)
            .map(Some)
            .ok_or_else(|| format!("invalid hex color: {value}"));
    }
    let lower = v.to_ascii_lowercase();
    if lower.starts_with("okhsl") && lower.contains('(') {
        return func_args(&lower, "okhsl")
            .and_then(|a| parse_okhsl(&a))
            .map(Some)
            .ok_or_else(|| format!("invalid okhsl color: {value} (expected okhsl(H S% L%))"));
    }
    if lower.starts_with("oklch") && lower.contains('(') {
        return func_args(&lower, "oklch")
            .and_then(|a| parse_oklch(&a))
            .map(Some)
            .ok_or_else(|| format!("invalid oklch color: {value} (expected oklch(L% C H))"));
    }
    Ok(None)
}

/// A 256-color palette index.
pub fn from_index(n: u64) -> Result<Color, String> {
    u8::try_from(n)
        .map(Color::Indexed)
        .map_err(|_| format!("ANSI color index must be 0-255: {n}"))
}

/// Best-effort sRGB value of a resolved color, for color math (contrast,
/// avatar derivation). `None` for the terminal default.
pub fn to_rgb(color: Color) -> Option<[u8; 3]> {
    match color {
        Color::Rgb(r, g, b) => Some([r, g, b]),
        Color::Indexed(i) => Some(indexed_to_rgb(i)),
        _ => None,
    }
}

pub fn rgb(c: [u8; 3]) -> Color {
    Color::Rgb(c[0], c[1], c[2])
}

fn parse_hex(hex: &str) -> Option<Color> {
    if !hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    let digits: String = match hex.len() {
        3 => hex.chars().flat_map(|c| [c, c]).collect(),
        6 => hex.to_string(),
        _ => return None,
    };
    let n = u32::from_str_radix(&digits, 16).ok()?;
    Some(Color::Rgb((n >> 16) as u8, (n >> 8) as u8, n as u8))
}

/// `name(a b c)` -> `["a", "b", "c"]`; exactly three space-separated args.
fn func_args(s: &str, name: &str) -> Option<Vec<String>> {
    let inner = s
        .strip_prefix(name)?
        .trim_start()
        .strip_prefix('(')?
        .strip_suffix(')')?;
    let args: Vec<String> = inner.split_whitespace().map(str::to_string).collect();
    (args.len() == 3).then_some(args)
}

/// A number with an optional `%` (divided by 100).
fn num_pct(s: &str) -> Option<f64> {
    match s.strip_suffix('%') {
        Some(n) => n.parse::<f64>().ok().map(|v| v / 100.0),
        None => s.parse().ok(),
    }
    .filter(|v| v.is_finite())
}

fn hue(s: &str) -> Option<f64> {
    s.strip_suffix("deg")
        .unwrap_or(s)
        .parse()
        .ok()
        .filter(|v: &f64| v.is_finite())
}

fn parse_okhsl(args: &[String]) -> Option<Color> {
    let h = hue(&args[0])?;
    let s = num_pct(&args[1]).filter(|v| (0.0..=1.0).contains(v))?;
    let l = num_pct(&args[2]).filter(|v| (0.0..=1.0).contains(v))?;
    Some(rgb(oklab::okhsl_to_rgb(h, s, l)))
}

fn parse_oklch(args: &[String]) -> Option<Color> {
    let l = num_pct(&args[0]).filter(|v| (0.0..=1.0).contains(v))?;
    let c = args[1]
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite() && *v >= 0.0)?;
    let h = hue(&args[2])?;
    Some(rgb(oklab::oklch_to_rgb(l, c, h)))
}

// ── 256-color approximation (port of pi-tui's `rgbToAnsi256`) ──

const BASIC: [[u8; 3]; 16] = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [192, 192, 192],
    [128, 128, 128],
    [255, 0, 0],
    [0, 255, 0],
    [255, 255, 0],
    [0, 0, 255],
    [255, 0, 255],
    [0, 255, 255],
    [255, 255, 255],
];
const CUBE: [u8; 6] = [0, 95, 135, 175, 215, 255];

fn gray_value(i: usize) -> u8 {
    (8 + i * 10) as u8
}

fn indexed_to_rgb(i: u8) -> [u8; 3] {
    match i {
        0..=15 => BASIC[i as usize],
        16..=231 => {
            let c = (i - 16) as usize;
            [CUBE[c / 36], CUBE[(c % 36) / 6], CUBE[c % 6]]
        }
        _ => [gray_value((i - 232) as usize); 3],
    }
}

fn closest(values: impl Iterator<Item = u8>, target: f64) -> usize {
    let mut best = (0, f64::INFINITY);
    for (i, v) in values.enumerate() {
        let d = (target - v as f64).abs();
        if d < best.1 {
            best = (i, d);
        }
    }
    best.0
}

fn distance(a: [u8; 3], b: [u8; 3]) -> f64 {
    let d = |i: usize| a[i] as f64 - b[i] as f64;
    d(0).powi(2) * 0.299 + d(1).powi(2) * 0.587 + d(2).powi(2) * 0.114
}

pub fn rgb_to_ansi256(c: [u8; 3]) -> u8 {
    let ri = closest(CUBE.into_iter(), c[0] as f64);
    let gi = closest(CUBE.into_iter(), c[1] as f64);
    let bi = closest(CUBE.into_iter(), c[2] as f64);
    let cube = [CUBE[ri], CUBE[gi], CUBE[bi]];
    let cube_index = 16 + 36 * ri + 6 * gi + bi;

    let gray = (0.299 * c[0] as f64 + 0.587 * c[1] as f64 + 0.114 * c[2] as f64).round();
    let go = closest((0..24).map(gray_value), gray);
    let gv = gray_value(go);
    let spread = c.iter().max().unwrap() - c.iter().min().unwrap();
    if spread < 10 && distance(c, [gv; 3]) < distance(c, cube) {
        return (232 + go) as u8;
    }
    cube_index as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literals() {
        assert_eq!(parse_literal("").unwrap(), Some(Color::Reset));
        assert_eq!(
            parse_literal("#0af").unwrap(),
            Some(Color::Rgb(0, 0xaa, 0xff))
        );
        assert_eq!(
            parse_literal("#00AAFF").unwrap(),
            Some(Color::Rgb(0, 0xaa, 0xff))
        );
        assert_eq!(
            parse_literal("okhsl(234 3% 89%)").unwrap(),
            Some(Color::Rgb(0xde, 0xe0, 0xe1))
        );
        assert_eq!(
            parse_literal("OKLCH(62% 0.1 200deg)").unwrap(),
            Some(Color::Rgb(0x1c, 0x98, 0x9e))
        );
        assert_eq!(parse_literal("accent").unwrap(), None);
    }

    #[test]
    fn bad_literals() {
        assert!(parse_literal("#12").is_err());
        assert!(parse_literal("#gggggg").is_err());
        assert!(parse_literal("okhsl(1 2)").is_err());
        assert!(parse_literal("okhsl(1 200% 50%)").is_err());
        assert!(parse_literal("oklch(50% -1 20)").is_err());
        assert!(from_index(256).is_err());
        assert_eq!(from_index(39).unwrap(), Color::Indexed(39));
    }

    #[test]
    fn ansi256() {
        assert_eq!(rgb_to_ansi256([0, 0, 0]), 16);
        assert_eq!(rgb_to_ansi256([255, 0, 0]), 196);
        assert_eq!(rgb_to_ansi256([255, 255, 255]), 231);
        assert_eq!(rgb_to_ansi256([128, 128, 128]), 244);
        assert_eq!(
            ColorMode::Ansi256.adapt(Color::Rgb(255, 0, 0)),
            Color::Indexed(196)
        );
        assert_eq!(ColorMode::Ansi256.adapt(Color::Reset), Color::Reset);
        assert_eq!(
            ColorMode::TrueColor.adapt(Color::Rgb(1, 2, 3)),
            Color::Rgb(1, 2, 3)
        );
    }
}
