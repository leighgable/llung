//! Oklab and OKHSL <-> sRGB conversion.
//!
//! Rust port of pi-tui's `oklab.js`, itself a port of Björn Ottosson's
//! reference implementation (https://bottosson.github.io/posts/colorpicker/).
//! Kept numerically identical so pi theme files render the same colors here.
//!
//! Copyright (c) 2021 Björn Ottosson, used under the MIT license:
//!
//! Permission is hereby granted, free of charge, to any person obtaining a copy
//! of this software and associated documentation files (the "Software"), to
//! deal in the Software without restriction, including without limitation the
//! rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
//! sell copies of the Software, and to permit persons to whom the Software is
//! furnished to do so, subject to the following conditions: The above
//! copyright notice and this permission notice shall be included in all copies
//! or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS",
//! WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.

type V3 = [f64; 3];
type M3 = [V3; 3];

fn mul(m: &M3, v: V3) -> V3 {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn dot(a: &V3, b: &V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

const LINEAR_SRGB_TO_LMS: M3 = [
    [0.4122214694707629, 0.5363325372617349, 0.0514459932675022],
    [0.2119034958178251, 0.6806995506452344, 0.1073969535369405],
    [0.0883024591900564, 0.2817188391361215, 0.6299787016738222],
];
const LMS_TO_LAB: M3 = [
    [0.210454268309314, 0.793617774702305, -0.0040720430116193],
    [1.9779985324311684, -2.42859224204858, 0.450593709617411],
    [0.0259040424655478, 0.7827717124575296, -0.8086757549230774],
];
const LAB_TO_LMS: M3 = [
    [1.0, 0.3963377773761749, 0.2158037573099136],
    [1.0, -0.1055613458156586, -0.0638541728258133],
    [1.0, -0.0894841775298119, -1.2914855480194092],
];
const LMS_TO_LINEAR_SRGB: M3 = [
    [4.0767416360759583, -3.3077115392580629, 0.2309699031821043],
    [-1.2684379732850315, 2.6097573492876882, -0.341319376002657],
    [-0.0041960761386756, -0.7034186179359362, 1.7076146940746117],
];

/// Per sRGB channel: the (a, b) half-plane where that channel clips first,
/// and the polynomial approximating the maximum saturation there.
const SATURATION_FIT: [([f64; 2], [f64; 5]); 3] = [
    (
        [-1.8817031, -0.80936501],
        [1.19086277, 1.76576728, 0.59662641, 0.75515197, 0.56771245],
    ),
    (
        [1.8144408, -1.19445267],
        [0.73956515, -0.45954404, 0.08285427, 0.12541073, -0.14503204],
    ),
    (
        [0.13110758, 1.81333971],
        [1.35733652, -0.00915799, -1.1513021, -0.50559606, 0.00692167],
    ),
];

const K1: f64 = 0.206;
const K2: f64 = 0.03;
const K3: f64 = (1.0 + K1) / (1.0 + K2);

/// Oklab lightness to OKHSL lightness.
pub fn oklab_to_okhsl_lightness(x: f64) -> f64 {
    0.5 * (K3 * x - K1 + ((K3 * x - K1).powi(2) + 4.0 * K2 * K3 * x).sqrt())
}

/// OKHSL lightness to Oklab lightness.
fn okhsl_to_oklab_lightness(x: f64) -> f64 {
    (x * x + K1 * x) / (K3 * (x + K2))
}

fn linear_to_srgb(v: f64) -> f64 {
    if v > 0.0031308 {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    } else {
        12.92 * v
    }
}

fn srgb_to_linear(v: f64) -> f64 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Oklab [L, a, b] to linear sRGB (0-1, may leave the gamut).
pub fn oklab_to_linear_srgb(lab: V3) -> V3 {
    let lms = mul(&LAB_TO_LMS, lab);
    mul(&LMS_TO_LINEAR_SRGB, lms.map(|v| v * v * v))
}

/// sRGB channels (0-255) to Oklab [L, a, b].
pub fn rgb_to_oklab(rgb: [u8; 3]) -> V3 {
    let lin = rgb.map(|c| srgb_to_linear(c as f64 / 255.0));
    mul(&LMS_TO_LAB, mul(&LINEAR_SRGB_TO_LMS, lin).map(f64::cbrt))
}

/// Linear sRGB to sRGB channels (0-255, rounded), clipping out-of-gamut.
pub fn linear_srgb_to_rgb(lin: V3) -> [u8; 3] {
    lin.map(|v| (linear_to_srgb(v).clamp(0.0, 1.0) * 255.0).round() as u8)
}

/// Rate of change of each cube-root LMS component along chroma direction (a, b).
fn lms_slopes(a: f64, b: f64) -> V3 {
    LAB_TO_LMS.map(|row| row[1] * a + row[2] * b)
}

/// Largest saturation (C/L) inside sRGB for hue (a, b): fit plus one Halley step.
fn max_saturation(a: f64, b: f64) -> f64 {
    let channel = SATURATION_FIT
        .iter()
        .enumerate()
        .position(|(i, ([x, y], _))| i == 2 || x * a + y * b > 1.0)
        .unwrap_or(2);
    let [k0, k1, k2, k3, k4] = SATURATION_FIT[channel].1;
    let weights = LMS_TO_LINEAR_SRGB[channel];
    let s = k0 + k1 * a + k2 * b + k3 * a * a + k4 * a * b;
    let slopes = lms_slopes(a, b);
    let base = slopes.map(|k| 1.0 + s * k);
    let f = dot(&weights, &base.map(|v| v.powi(3)));
    let f1 = dot(
        &weights,
        &[0, 1, 2].map(|i| 3.0 * slopes[i] * base[i].powi(2)),
    );
    let f2 = dot(
        &weights,
        &[0, 1, 2].map(|i| 6.0 * slopes[i].powi(2) * base[i]),
    );
    s - (f * f1) / (f1 * f1 - 0.5 * f * f2)
}

/// Oklab lightness and chroma of the most saturated sRGB color of hue (a, b).
fn cusp(a: f64, b: f64) -> (f64, f64) {
    let s = max_saturation(a, b);
    let rgb = oklab_to_linear_srgb([1.0, s * a, s * b]);
    let l = (1.0 / rgb[0].max(rgb[1]).max(rgb[2])).cbrt();
    (l, l * s)
}

/// Chroma where the constant-lightness line at `l` leaves the sRGB gamut.
fn max_chroma(a: f64, b: f64, l: f64, (cusp_l, cusp_c): (f64, f64)) -> f64 {
    if l <= cusp_l {
        return cusp_c * l / cusp_l;
    }
    let t = cusp_c * (l - 1.0) / (cusp_l - 1.0);
    let slopes = lms_slopes(a, b);
    let lms = slopes.map(|k| l + t * k);
    let cubes = lms.map(|v| v.powi(3));
    let first = [0, 1, 2].map(|i| 3.0 * slopes[i] * lms[i].powi(2));
    let second = [0, 1, 2].map(|i| 6.0 * slopes[i].powi(2) * lms[i]);
    let step = LMS_TO_LINEAR_SRGB
        .iter()
        .map(|row| {
            let f = dot(row, &cubes) - 1.0;
            let f1 = dot(row, &first);
            let f2 = dot(row, &second);
            let u = f1 / (f1 * f1 - 0.5 * f * f2);
            if u >= 0.0 { -f * u } else { f64::MAX }
        })
        .fold(f64::MAX, f64::min);
    t + step
}

/// OKHSL's chroma reference points at lightness `l` and hue (a, b).
fn chroma_stops(l: f64, a: f64, b: f64) -> (f64, f64, f64) {
    let peak = cusp(a, b);
    let c_max = max_chroma(a, b, l, peak);
    let k = c_max / (l * (peak.1 / peak.0)).min((1.0 - l) * (peak.1 / (1.0 - peak.0)));
    let mid_s = 0.11516993
        + 1.0
            / (7.4477897
                + 4.1590124 * b
                + a * (-2.19557347
                    + 1.75198401 * b
                    + a * (-2.13704948 - 10.02301043 * b
                        + a * (-4.24894561 + 5.38770819 * b + 4.69891013 * a))));
    let mid_t = 0.11239642
        + 1.0
            / (1.6132032 - 0.68124379 * b
                + a * (0.40370612
                    + 0.90148123 * b
                    + a * (-0.27087943
                        + 0.6122399 * b
                        + a * (0.00299215 - 0.45399568 * b - 0.14661872 * a))));
    let c_mid = 0.9
        * k
        * (1.0 / (1.0 / (l * mid_s).powi(4) + 1.0 / ((1.0 - l) * mid_t).powi(4)))
            .sqrt()
            .sqrt();
    let c0 = (1.0 / (1.0 / (l * 0.4).powi(2) + 1.0 / ((1.0 - l) * 0.8).powi(2))).sqrt();
    (c0, c_mid, c_max)
}

/// OKHSL (hue degrees, saturation 0-1, lightness 0-1) to sRGB channels.
pub fn okhsl_to_rgb(hue: f64, saturation: f64, lightness: f64) -> [u8; 3] {
    let l = okhsl_to_oklab_lightness(lightness);
    let mut lab = [l, 0.0, 0.0];
    if l > 0.0 && l < 1.0 && saturation > 0.0 {
        let angle = hue.rem_euclid(360.0).to_radians();
        let (a, b) = (angle.cos(), angle.sin());
        let (c0, c_mid, c_max) = chroma_stops(l, a, b);
        let chroma = if saturation < 0.8 {
            let t = 1.25 * saturation;
            let k1 = 0.8 * c0;
            t * k1 / (1.0 - (1.0 - k1 / c_mid) * t)
        } else {
            let t = 5.0 * (saturation - 0.8);
            let k1 = 0.2 * c_mid.powi(2) * 1.25f64.powi(2) / c0;
            c_mid + t * k1 / (1.0 - (1.0 - k1 / (c_max - c_mid)) * t)
        };
        lab = [l, chroma * a, chroma * b];
    }
    linear_srgb_to_rgb(oklab_to_linear_srgb(lab))
}

/// sRGB to OKHSL: (hue degrees, saturation 0-1, lightness 0-1).
pub fn rgb_to_okhsl(rgb: [u8; 3]) -> (f64, f64, f64) {
    let [l, la, lb] = rgb_to_oklab(rgb);
    let chroma = la.hypot(lb);
    let lightness = oklab_to_okhsl_lightness(l);
    if chroma < 1e-9 || lightness <= 0.0 || lightness >= 1.0 {
        return (0.0, 0.0, lightness);
    }
    let hue = lb.atan2(la).to_degrees().rem_euclid(360.0);
    let (c0, c_mid, c_max) = chroma_stops(l, la / chroma, lb / chroma);
    let s = if chroma < c_mid {
        let k1 = 0.8 * c0;
        0.8 * (chroma / (k1 + (1.0 - k1 / c_mid) * chroma))
    } else {
        let k1 = 0.2 * c_mid.powi(2) * 1.25f64.powi(2) / c0;
        let off = chroma - c_mid;
        0.8 + 0.2 * (off / (k1 + (1.0 - k1 / (c_max - c_mid)) * off))
    };
    (hue, s.clamp(0.0, 1.0), lightness)
}

fn in_gamut(lin: &V3) -> bool {
    const EPS: f64 = 1e-7;
    lin.iter().all(|&c| c >= -EPS && c <= 1.0 + EPS)
}

/// OKLCH to sRGB, reducing chroma (hue and lightness fixed) until in gamut.
pub fn oklch_to_rgb(l: f64, c: f64, h: f64) -> [u8; 3] {
    let rad = h.rem_euclid(360.0).to_radians();
    let (cos, sin) = (rad.cos(), rad.sin());
    let at = |chroma: f64| oklab_to_linear_srgb([l, chroma * cos, chroma * sin]);
    let direct = at(c);
    if in_gamut(&direct) {
        return linear_srgb_to_rgb(direct);
    }
    let mut lin = at(0.0);
    let (mut lo, mut hi) = (0.0, c);
    for _ in 0..20 {
        let mid = (lo + hi) / 2.0;
        let cand = at(mid);
        if in_gamut(&cand) {
            lo = mid;
            lin = cand;
        } else {
            hi = mid;
        }
    }
    linear_srgb_to_rgb(lin)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(rgb: [u8; 3]) -> String {
        format!("#{:02x}{:02x}{:02x}", rgb[0], rgb[1], rgb[2])
    }

    /// Reference values produced by pi-tui's `parseColor` + `colorToHex`.
    #[test]
    fn matches_pi_okhsl() {
        assert_eq!(hex(okhsl_to_rgb(234.0, 0.03, 0.89)), "#dee0e1");
        assert_eq!(hex(okhsl_to_rgb(295.0, 0.50, 0.67)), "#a798d7");
        assert_eq!(hex(okhsl_to_rgb(233.0, 0.41, 0.24)), "#213b49");
        assert_eq!(hex(okhsl_to_rgb(20.0, 0.72, 0.67)), "#ea7f81");
        assert_eq!(hex(okhsl_to_rgb(83.0, 0.99, 0.47)), "#8f6802");
    }

    #[test]
    fn matches_pi_oklch() {
        assert_eq!(hex(oklch_to_rgb(0.62, 0.1, 200.0)), "#1c989e");
        // Out of gamut: chroma reduced.
        assert_eq!(hex(oklch_to_rgb(0.70, 0.4, 30.0)), "#ff6551");
        // Fully out of gamut at L=1 maps to white.
        assert_eq!(hex(oklch_to_rgb(1.0, 0.3, 150.0)), "#ffffff");
    }

    #[test]
    fn okhsl_extremes() {
        assert_eq!(okhsl_to_rgb(0.0, 0.0, 0.0), [0, 0, 0]);
        assert_eq!(okhsl_to_rgb(0.0, 0.0, 1.0), [255, 255, 255]);
    }

    #[test]
    fn okhsl_round_trip() {
        let rgb = okhsl_to_rgb(295.0, 0.5, 0.67);
        let (h, s, l) = rgb_to_okhsl(rgb);
        assert!((h - 295.0).abs() < 1.0, "{h}");
        assert!((s - 0.5).abs() < 0.02, "{s}");
        assert!((l - 0.67).abs() < 0.01, "{l}");
    }
}
