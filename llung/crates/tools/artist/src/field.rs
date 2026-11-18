//! The procedural styles: (point, seed) → [0, 1]. Each is a small
//! recipe over the noise module. `mix` resolves deterministically per
//! seed to one of the concrete styles.

use crate::noise::{derive, fbm};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Style {
    /// Domain-warped fBm — soft billowing clouds (iq's classic warp).
    Clouds,
    /// Sharp ridged multifractal — mountain-range crests.
    Ridged,
    /// Concentric rings warped by noise — ripples in water.
    Rings,
    /// Trigonometric interference — woven fabric / moiré.
    Weave,
}

impl Style {
    pub fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "clouds" | "cloud" => Some(Self::Clouds),
            "ridged" | "ridge" | "mountains" => Some(Self::Ridged),
            "rings" | "ring" | "ripples" => Some(Self::Rings),
            "weave" | "woven" | "plaid" | "moire" => Some(Self::Weave),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Clouds => "clouds",
            Self::Ridged => "ridged",
            Self::Rings => "rings",
            Self::Weave => "weave",
        }
    }

    /// "mix" (or an unrecognized style) picks one deterministically per
    /// seed — the same seed always yields the same family.
    pub fn for_seed(seed: u64) -> Self {
        match derive(seed, 0x5EED) % 4 {
            0 => Self::Clouds,
            1 => Self::Ridged,
            2 => Self::Rings,
            _ => Self::Weave,
        }
    }

    /// Evaluate the field at `p` (already scaled into pattern space).
    pub fn sample(self, x: f64, y: f64, seed: u64) -> f64 {
        let s2 = seed ^ 0xA076_1D64_78BD_642F;
        match self {
            Self::Clouds => {
                // Two-stage domain warp.
                let qx = fbm(x, y, seed, 4);
                let qy = fbm(x + 5.2, y + 1.3, s2, 4);
                let rx = fbm(x + 1.6 * qx, y + 1.6 * qy, seed ^ 0x0F0F, 4);
                let ry = fbm(x + 1.6 * qx + 8.3, y + 1.6 * qy + 2.8, s2 ^ 0x0F0F, 4);
                fbm(x + 1.2 * rx, y + 1.2 * ry, seed ^ 0x7777, 5)
            }
            Self::Ridged => {
                let n = fbm(x, y, seed, 5);
                // Invert around the mean and square: bright sharp crests.
                let r = 1.0 - (2.0 * n - 1.0).abs();
                r * r
            }
            Self::Rings => {
                let d = (x * x + y * y).sqrt();
                let warp = 2.5 * fbm(x, y, seed, 4);
                let k = 6.0 + (derive(seed, 0x4B1D) % 6) as f64; // ring density per seed
                0.5 + 0.5 * (d * k + warp).sin()
            }
            Self::Weave => {
                let jitter = 0.8 * fbm(x, y, seed, 3);
                let a = (x * 3.0 + jitter).sin();
                let b = (y * 3.0 - jitter).sin();
                // Weave the two axes' interference with a noise wash.
                let v = a * b * 0.5 + 0.5;
                (v * 0.75 + fbm(x * 0.5, y * 0.5, s2, 3) * 0.25).clamp(0.0, 1.0)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn styles_parse_and_are_stable() {
        assert_eq!(Style::parse("Clouds"), Some(Style::Clouds));
        assert_eq!(Style::parse("RIPPLES"), Some(Style::Rings));
        assert_eq!(Style::parse("plaid"), Some(Style::Weave));
        assert_eq!(Style::parse("nonsense"), None);
        assert_eq!(Style::for_seed(42), Style::for_seed(42));
    }

    #[test]
    fn all_styles_sample_in_range() {
        for style in [Style::Clouds, Style::Ridged, Style::Rings, Style::Weave] {
            for i in 0..50 {
                let v = style.sample(i as f64 * 0.13, i as f64 * 0.29, 99);
                assert!((0.0..=1.0).contains(&v), "{style:?} gave {v}");
            }
        }
    }
}
