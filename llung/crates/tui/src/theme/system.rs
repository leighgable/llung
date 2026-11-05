//! The `system` theme: pi's colors derived from the terminal's own theme.
//!
//! Port of pi's `system-theme.js`. Every token belongs to a color family
//! (its hue) and has contrast rules: it must reach a contrast *level* on
//! the background and on the panels it is drawn on. Hue and saturation
//! come from the terminal's palette color for the family's ANSI slot, or
//! from the family's own hue when the terminal reports no palette.
//! Lightness comes from the rules alone. Colors are built in OKHSL and
//! fade toward gray near black and white. A palette color never gains
//! OKLCH chroma when moved to another lightness, so pastel palettes stay
//! pastel.
//!
//! A level is a target-lightness curve: the OKLab lightness a token needs,
//! given the lightness of the surface below it (polynomials fitted by pi).
//!
//! Tiers, by what the terminal reports:
//! - background + palette: hues from the palette, lightness from the background;
//! - background only: the families' own hues, lightness from the background;
//! - nothing: ANSI indices and default colors, rendered by the terminal itself.

use super::file::Appearance;
use super::contrast::{ensure_contrast, wcag_contrast};
use super::oklab;
use crate::terminal::query::TerminalColors;
use serde_json::{Map, Value, json};
use std::collections::HashMap;
use std::sync::OnceLock;

pub const SYSTEM_THEME_NAME: &str = "system";

// ── Terminal colors, queried once at startup ──

static TERMINAL: OnceLock<TerminalColors> = OnceLock::new();

/// Record the startup query result. Later calls are ignored.
pub fn set_terminal_colors(colors: TerminalColors) {
    let _ = TERMINAL.set(colors);
}

/// What the terminal reported (empty if never queried).
/// See [`crate::terminal::init`].
pub fn terminal_colors() -> TerminalColors {
    TERMINAL.get().cloned().unwrap_or_default()
}

// ── Families and rules (pi's tables) ──

#[derive(Clone, Copy)]
struct Family {
    hue: f64,
    s_min: f64,
    s_max: f64,
    slot: usize,
}

const fn fam(hue: f64, s_min: f64, s_max: f64, slot: usize) -> Family {
    Family { hue, s_min, s_max, slot }
}

const NEUTRAL: Family = fam(231.49, 0.02, 0.08, 8);
const BLUE: Family = fam(231.49, 0.1, 0.68, 4);
const GREEN: Family = fam(158.68, 0.1, 0.76, 2);
const RED: Family = fam(20.0, 0.1, 0.92, 1);
const YELLOW: Family = fam(82.36, 0.5, 1.0, 3);
const ORANGE: Family = fam(52.0, 0.12, 0.85, 3);
const VIOLET: Family = fam(295.0, 0.2, 0.6, 5);
const CALAMINE: Family = fam(202.43, 0.1, 0.74, 6);
const THINKING_SLATE: Family = fam(231.49, 0.08, 0.2, 4);
const THINKING_BLUE: Family = fam(231.49, 0.2, 0.45, 4);
const THINKING_PERIWINKLE: Family = fam(263.25, 0.3, 0.6, 6);
const THINKING_VIOLET: Family = fam(295.0, 0.4, 0.75, 5);
const THINKING_MAGENTA: Family = fam(337.5, 0.5, 0.85, 13);
const THINKING_RED: Family = fam(20.0, 0.95, 1.0, 1);

/// Every pi token with its family, in pi's order.
const TOKENS: &[(&str, Family)] = &[
    ("selectedBg", BLUE),
    ("searchMatchBg", ORANGE),
    ("userMessageBg", BLUE),
    ("customMessageBg", VIOLET),
    ("toolPendingBg", NEUTRAL),
    ("toolSuccessBg", GREEN),
    ("toolErrorBg", RED),
    ("text", NEUTRAL),
    ("userMessageText", NEUTRAL),
    ("customMessageText", NEUTRAL),
    ("toolTitle", NEUTRAL),
    ("syntaxOperator", NEUTRAL),
    ("syntaxPunctuation", NEUTRAL),
    ("muted", NEUTRAL),
    ("dim", NEUTRAL),
    ("thinkingText", NEUTRAL),
    ("toolOutput", NEUTRAL),
    ("mdLinkUrl", NEUTRAL),
    ("mdQuote", NEUTRAL),
    ("mdQuoteBorder", NEUTRAL),
    ("mdHr", NEUTRAL),
    ("mdCodeBlockBorder", NEUTRAL),
    ("toolDiffContext", NEUTRAL),
    ("syntaxComment", NEUTRAL),
    ("scrollbarTrack", NEUTRAL),
    ("scrollbarThumb", NEUTRAL),
    ("searchMatchText", NEUTRAL),
    ("borderMuted", NEUTRAL),
    ("accent", VIOLET),
    ("borderAccent", VIOLET),
    ("customMessageLabel", VIOLET),
    ("mdCode", VIOLET),
    ("mdListBullet", VIOLET),
    ("syntaxType", VIOLET),
    ("border", BLUE),
    ("mdLink", BLUE),
    ("syntaxKeyword", BLUE),
    ("syntaxVariable", CALAMINE),
    ("success", GREEN),
    ("mdCodeBlock", GREEN),
    ("toolDiffAdded", GREEN),
    ("bashMode", GREEN),
    ("syntaxNumber", GREEN),
    ("error", RED),
    ("toolDiffRemoved", RED),
    ("warning", YELLOW),
    ("mdHeading", YELLOW),
    ("syntaxFunction", YELLOW),
    ("syntaxString", ORANGE),
    ("thinkingOff", NEUTRAL),
    ("thinkingMinimal", THINKING_SLATE),
    ("thinkingLow", THINKING_BLUE),
    ("thinkingMedium", THINKING_PERIWINKLE),
    ("thinkingHigh", THINKING_VIOLET),
    ("thinkingXhigh", THINKING_MAGENTA),
    ("thinkingMax", THINKING_RED),
];

fn family_of(token: &str) -> Family {
    TOKENS.iter().find(|(t, _)| *t == token).map(|(_, f)| *f).unwrap_or(NEUTRAL)
}

/// Palette slots for tokens that would otherwise share a hue with a similar token.
fn slot_of(token: &str) -> usize {
    match token {
        "syntaxString" => 2,
        "syntaxNumber" => 5,
        "searchMatchBg" => 3,
        _ => family_of(token).slot,
    }
}

struct Curve {
    c: [f64; 6],
    reach: (f64, f64),
}

struct Level {
    name: &'static str,
    dark: Curve,
    light: Curve,
}

macro_rules! level {
    ($name:literal, [$($d:expr),*], $dr:expr, [$($l:expr),*], $lr:expr) => {
        Level {
            name: $name,
            dark: Curve { c: [$($d),*], reach: $dr },
            light: Curve { c: [$($l),*], reach: $lr },
        }
    };
}

#[rustfmt::skip]
const LEVELS: &[Level] = &[
    level!("panel", [0.29131, -0.39746, 2.33185, -0.85524, -1.2076, 0.86276], (0.0, 0.979),
        [-3.74073, 27.94549, -78.44258, 112.6798, -79.60015, 22.11277], (0.348, 1.0)),
    level!("track", [0.39028, -0.23015, 0.83573, 2.43829, -4.38292, 2.01582], (0.0, 0.946),
        [-5.24921, 38.37322, -107.28833, 152.10005, -106.17127, 29.18061], (0.368, 1.0)),
    level!("thinking0", [0.52988, -0.05809, -0.30924, 4.63567, -6.52933, 2.89108], (0.0, 0.873),
        [-28.27749, 182.85284, -469.62416, 603.15916, -384.59976, 97.35147], (0.51, 1.0)),
    level!("thinking1", [0.55278, -0.03667, -0.45659, 4.95347, -6.90265, 3.0706], (0.0, 0.858),
        [-37.10484, 235.86282, -596.62344, 754.3633, -474.00763, 118.3551], (0.535, 1.0)),
    level!("thinking2", [0.57486, -0.01765, -0.58987, 5.25227, -7.27175, 3.25532], (0.0, 0.842),
        [-59.89653, 377.05024, -945.07843, 1182.03145, -734.96375, 181.68658], (0.556, 1.0)),
    level!("thinking3", [0.59621, -0.00062, -0.71148, 5.53588, -7.6392, 3.44606], (0.0, 0.827),
        [-72.07122, 445.84082, -1099.57352, 1353.88793, -829.53392, 202.26164], (0.58, 1.0)),
    level!("thinking4", [0.61691, 0.01462, -0.82288, 5.80651, -8.00641, 3.64333], (0.0, 0.811),
        [-110.14338, 674.21488, -1645.75941, 2004.32367, -1215.15899, 293.3183], (0.6, 1.0)),
    level!("thinking5", [0.63702, 0.02826, -0.92498, 6.06465, -8.37246, 3.84651], (0.0, 0.795),
        [-175.47701, 1063.54495, -2570.70594, 3098.80776, -1860.15527, 444.76392], (0.62, 1.0)),
    level!("thinking6", [0.65658, 0.04044, -1.01835, 6.30989, -8.73529, 4.05439], (0.0, 0.779),
        [-183.81712, 1094.70055, -2602.68539, 3088.71276, -1826.91131, 430.75931], (0.643, 1.0)),
    level!("subtle", [0.56762, -0.02475, -0.5383, 5.12628, -7.10931, 3.17324], (0.0, 0.848),
        [-232.85459, 1376.54473, -3249.11801, 3827.91186, -2248.29472, 526.55751], (0.657, 1.0)),
    level!("thumb", [0.60323, 0.00278, -0.73328, 5.57157, -7.68067, 3.46933], (0.0, 0.823),
        [-82.89897, 511.01355, -1255.98095, 1540.76821, -940.68087, 228.58523], (0.586, 1.0)),
    level!("readable", [0.66937, 0.04704, -1.06871, 6.43941, -8.9332, 4.17229], (0.0, 0.77),
        [-1554.52576, 8733.56817, -19604.93507, 21977.72696, -12300.99599, 2749.81288], (0.751, 1.0)),
    level!("emphasis", [0.7303, 0.07695, -1.31626, 7.1681, -10.14436, 4.92846], (0.0, 0.712),
        [-4948.31942, 26870.91986, -58334.48399, 63280.17197, -34298.01053, 7430.30146], (0.811, 1.0)),
    level!("textOnPanel", [0.86713, 0.05232, -0.89428, 4.79014, -5.5432, 1.75023], (0.0, 0.542),
        [-8570.89457, 43954.60805, -90084.00702, 92220.6791, -47152.15802, 9632.27113], (0.867, 1.0)),
    level!("text", [0.89242, 0.02311, -0.44862, 2.34417, -0.06084, -2.63844], (0.0, 0.5),
        [-2004.67048, 6664.47299, -6060.70202, -1792.61209, 5133.82359, -1939.85583], (0.894, 1.0)),
];

const TOOL_PANELS: &[&str] = &["toolPendingBg", "toolSuccessBg", "toolErrorBg"];
const MESSAGE_PANELS: &[&str] = &["userMessageBg", "customMessageBg"];
const PANELS: &[&str] = &[
    "userMessageBg",
    "toolPendingBg",
    "toolSuccessBg",
    "toolErrorBg",
    "selectedBg",
    "searchMatchBg",
    "customMessageBg",
];
const THINKING: &[&str] = &[
    "thinkingOff",
    "thinkingMinimal",
    "thinkingLow",
    "thinkingMedium",
    "thinkingHigh",
    "thinkingXhigh",
    "thinkingMax",
];
const THINKING_LEVELS: &[&str] = &[
    "thinking0", "thinking1", "thinking2", "thinking3", "thinking4", "thinking5", "thinking6",
];
/// Text-level tokens that take the terminal's foreground.
const FOREGROUND_TOKENS: &[&str] = &["text", "userMessageText", "toolTitle"];
/// Body text uses the terminal's foreground when it reaches this level.
const FOREGROUND_LEVEL: &str = "emphasis";
/// WCAG 2 contrast ratio body text must reach on its surfaces.
const TEXT_MIN_CONTRAST: f64 = 4.5;

struct Rule {
    token: &'static str,
    on: Vec<&'static str>,
    level: &'static str,
}

fn cat(parts: &[&[&'static str]]) -> Vec<&'static str> {
    parts.iter().flat_map(|p| p.iter().copied()).collect()
}

fn rules() -> Vec<Rule> {
    let mut r = Vec::new();
    let mut each = |tokens: &[&'static str], on: Vec<&'static str>, level: &'static str| {
        for t in tokens {
            r.push(Rule { token: t, on: on.clone(), level });
        }
    };
    let bg = "background";
    each(PANELS, vec![bg], "panel");
    each(&["text"], vec![bg], "text");
    each(&["text"], vec!["selectedBg"], "textOnPanel");
    each(&["userMessageText"], vec!["userMessageBg"], "textOnPanel");
    each(&["toolTitle"], TOOL_PANELS.to_vec(), "textOnPanel");
    each(
        &["accent", "success", "error", "warning"],
        cat(&[&[bg, "selectedBg"], TOOL_PANELS]),
        "readable",
    );
    each(&["muted"], cat(&[&[bg, "selectedBg", "customMessageBg"], TOOL_PANELS]), "readable");
    each(&["dim"], cat(&[&[bg, "selectedBg", "customMessageBg"], TOOL_PANELS]), "subtle");
    each(&["thinkingText"], vec![bg], "readable");
    each(&["customMessageText"], cat(&[&["customMessageBg"], TOOL_PANELS]), "readable");
    each(
        &["customMessageLabel"],
        cat(&[&[bg, "customMessageBg", "selectedBg"], TOOL_PANELS]),
        "readable",
    );
    each(&["toolOutput"], cat(&[&[bg], TOOL_PANELS]), "readable");
    each(
        &["mdHeading", "mdLink", "mdLinkUrl", "mdCode", "mdQuote", "mdCodeBlockBorder", "mdListBullet"],
        cat(&[&[bg], MESSAGE_PANELS]),
        "readable",
    );
    each(&["mdCodeBlock"], cat(&[&[bg], MESSAGE_PANELS, TOOL_PANELS]), "readable");
    each(
        &["toolDiffAdded", "toolDiffRemoved", "toolDiffContext"],
        cat(&[&[bg], TOOL_PANELS]),
        "readable",
    );
    each(
        &[
            "syntaxComment", "syntaxKeyword", "syntaxFunction", "syntaxVariable", "syntaxString",
            "syntaxNumber", "syntaxType", "syntaxOperator", "syntaxPunctuation",
        ],
        cat(&[&[bg], MESSAGE_PANELS, TOOL_PANELS]),
        "readable",
    );
    each(&["searchMatchText"], vec!["searchMatchBg"], "readable");
    each(&["bashMode", "border", "borderAccent"], vec![bg], "readable");
    each(&["borderMuted"], vec![bg], "subtle");
    each(&["mdQuoteBorder", "mdHr"], cat(&[&[bg], MESSAGE_PANELS, TOOL_PANELS]), "readable");
    each(&["scrollbarTrack"], vec![bg], "track");
    each(&["scrollbarThumb"], vec!["scrollbarTrack"], "thumb");
    for (t, l) in THINKING.iter().zip(THINKING_LEVELS) {
        each(&[t], vec![bg], l);
    }
    r
}

/// Tokens in dependency order: every surface before the tokens drawn on it.
fn solve_order(rules: &[Rule]) -> Vec<&'static str> {
    fn visit(token: &'static str, rules: &[Rule], order: &mut Vec<&'static str>) {
        if order.contains(&token) {
            return;
        }
        for rule in rules.iter().filter(|r| r.token == token) {
            for s in &rule.on {
                if *s != "background" {
                    visit(s, rules, order);
                }
            }
        }
        order.push(token);
    }
    let mut order = Vec::new();
    for rule in rules {
        visit(rule.token, rules, &mut order);
    }
    order
}

// ── Color helpers ──

type Rgb = [u8; 3];

fn oklab_l(c: Rgb) -> f64 {
    oklab::rgb_to_oklab(c)[0]
}

fn oklch_of(c: Rgb) -> (f64, f64, f64) {
    let [l, a, b] = oklab::rgb_to_oklab(c);
    (l, a.hypot(b), b.atan2(a).to_degrees().rem_euclid(360.0))
}

fn okhsl(h: f64, s: f64, l: f64) -> Rgb {
    oklab::okhsl_to_rgb(h, s.clamp(0.0, 1.0), l.clamp(0.0, 1.0))
}

const WHITE: Rgb = [255, 255, 255];
const BLACK: Rgb = [0, 0, 0];

/// Dark or light, from reported colors: the direction of the terminal's
/// own foreground when text can be readable that way, otherwise dark
/// when white text has more contrast on the background than black.
pub fn terminal_appearance(background: Rgb, foreground: Option<Rgb>) -> Appearance {
    let white = wcag_contrast(WHITE, background);
    let black = wcag_contrast(BLACK, background);
    if let Some(fg) = foreground {
        let (fl, bl) = (oklab_l(fg), oklab_l(background));
        if (fl - bl).abs() > 0.05 {
            let (app, best) = if fl > bl {
                (Appearance::Dark, white)
            } else {
                (Appearance::Light, black)
            };
            if best >= TEXT_MIN_CONTRAST {
                return app;
            }
        }
    }
    if white >= black { Appearance::Dark } else { Appearance::Light }
}

/// Saturation weight at a lightness: a Gaussian (center 0.5, sigma 0.25),
/// 0 at black and white, 1 in the middle.
fn bell(l: f64) -> f64 {
    let g = |x: f64| (-(x - 0.5).powi(2) / (2.0 * 0.25f64.powi(2))).exp();
    (g(l) - g(0.0)) / (1.0 - g(0.0))
}

/// A family's saturation relative to its maximum: 1 at mid lightness,
/// `min / max` at black and white.
fn saturation_curve(f: &Family, l: f64) -> f64 {
    let floor = if f.s_max > 0.0 { f.s_min / f.s_max } else { 1.0 };
    floor + (1.0 - floor) * bell(l)
}

fn level_target(level: &str, app: Appearance, surface_l: f64) -> Option<f64> {
    let lv = LEVELS.iter().find(|l| l.name == level)?;
    let curve = match app {
        Appearance::Dark => &lv.dark,
        Appearance::Light => &lv.light,
    };
    if surface_l < curve.reach.0 || surface_l > curve.reach.1 {
        return None;
    }
    Some(curve.c.iter().enumerate().map(|(p, k)| k * surface_l.powi(p as i32)).sum())
}

/// A palette color, as OKHSL plus its OKLCH chroma.
#[derive(Clone, Copy)]
struct Source {
    h: f64,
    s: f64,
    l: f64,
    chroma: f64,
}

fn source_of(c: Rgb) -> Source {
    let (h, s, l) = oklab::rgb_to_okhsl(c);
    Source { h, s, l, chroma: oklch_of(c).1 }
}

/// A source color's hue at another OKHSL lightness. Saturation falls off
/// toward black and white along the family curve, and chroma is capped at
/// the source's, so pastels stay pastel.
fn anchored(src: &Source, f: &Family, lightness: f64) -> Rgb {
    let anchor = saturation_curve(f, src.l);
    let falloff = if anchor > 0.0 {
        (saturation_curve(f, lightness) / anchor).min(1.0)
    } else {
        1.0
    };
    let color = okhsl(src.h, src.s * falloff, lightness);
    let cap = src.chroma * falloff;
    let (l, c, _) = oklch_of(color);
    if c <= cap { color } else { oklab::oklch_to_rgb(l, cap, src.h) }
}

/// Move a text color toward white/black until it reaches the WCAG minimum
/// on every surface.
fn with_text_contrast(color: Rgb, surfaces: &[Rgb], lighter: bool) -> Rgb {
    ensure_contrast(color, surfaces, TEXT_MIN_CONTRAST, lighter)
}

// ── Generation ──

/// Generated `system` theme, as a pi theme document plus render hints.
pub struct Generated {
    /// A complete pi theme JSON document.
    pub theme: Value,
    #[cfg_attr(not(test), allow(dead_code))] // also in `theme["appearance"]`
    pub appearance: Appearance,
    /// Nothing-reported tier: neutral secondary tokens are "" and should
    /// render faint (SGR 2) instead.
    pub faint_neutrals: bool,
}

/// Generate the system theme. `appearance_hint` is used only when the
/// terminal reported no background.
pub fn generate(input: &TerminalColors, appearance_hint: Appearance) -> Generated {
    let colors = match input.background {
        Some(bg) => generate_from_background(bg, input.foreground, input.palette.as_ref()),
        None => None,
    };
    let (colors, appearance, faint) = match colors {
        Some((c, app)) => (c, app, false),
        None => (indexed_colors(), appearance_hint, true),
    };
    let app_str = match appearance {
        Appearance::Dark => "dark",
        Appearance::Light => "light",
    };
    Generated {
        theme: json!({ "name": SYSTEM_THEME_NAME, "appearance": app_str, "colors": colors }),
        appearance,
        faint_neutrals: faint,
    }
}

fn hex(c: Rgb) -> Value {
    Value::String(format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2]))
}

/// Terminal reported nothing: ANSI indices 0-15 and default colors, which
/// the terminal renders with its own theme. Panels get no background.
fn indexed_colors() -> Map<String, Value> {
    TOKENS
        .iter()
        .map(|(token, f)| {
            // NEUTRAL is the only family on slot 8 (bright black).
            let v = if PANELS.contains(token) || f.slot == NEUTRAL.slot {
                Value::String(String::new())
            } else {
                json!(slot_of(token))
            };
            (token.to_string(), v)
        })
        .collect()
}

fn generate_from_background(
    background: Rgb,
    foreground: Option<Rgb>,
    palette: Option<&[Rgb; 16]>,
) -> Option<(Map<String, Value>, Appearance)> {
    let palette: Option<Vec<Source>> = palette.map(|p| p.iter().map(|c| source_of(*c)).collect());
    let appearance = terminal_appearance(background, foreground);
    let lighter = appearance == Appearance::Dark;
    let extreme = if lighter { 1.0 } else { 0.0 };
    let background_l = oklab_l(background);
    let readable_floor = if lighter { "readable" } else { "subtle" };
    let rules = rules();
    let order = solve_order(&rules);

    let paint = |token: &str, oklab_lightness: f64| -> Rgb {
        let lightness = oklab::oklab_to_okhsl_lightness(oklab_lightness);
        let f = family_of(token);
        match &palette {
            None => okhsl(f.hue, f.s_min + (f.s_max - f.s_min) * bell(lightness), lightness),
            Some(p) => anchored(&p[slot_of(token)], &f, lightness),
        }
    };

    // Lightness a rule needs on a surface, relaxed by `t`: 0→1 compresses
    // levels stronger than the readable floor toward it; 1→2 moves all
    // levels toward the surface itself.
    let target = |level: &str, surface_l: f64, t: f64| -> Option<f64> {
        let reached = level_target(level, appearance, surface_l);
        if reached.is_none() && t == 0.0 {
            return None;
        }
        let distance = reached.unwrap_or(extreme) - surface_l;
        let floor = level_target(readable_floor, appearance, surface_l).unwrap_or(extreme) - surface_l;
        let compressed = if distance.abs() > floor.abs() {
            distance - (distance - floor) * t.min(1.0)
        } else {
            distance
        };
        Some(surface_l + compressed * (1.0 - (t - 1.0).max(0.0)))
    };

    // Keep panels such that white (black) text still reaches the body-text
    // minimum on them; matters only near mid-gray backgrounds.
    let extreme_text = if lighter { WHITE } else { BLACK };
    let readable = |c: Rgb| wcag_contrast(extreme_text, c) >= TEXT_MIN_CONTRAST;
    let limit_panel = |token: &str, l: f64| -> Rgb {
        let color = paint(token, l);
        if readable(color) {
            return color;
        }
        let (mut lo, mut hi) = (background_l, l);
        for _ in 0..20 {
            let mid = (lo + hi) / 2.0;
            if readable(paint(token, mid)) { lo = mid } else { hi = mid }
        }
        paint(token, lo)
    };

    let solve = |t: f64| -> Option<HashMap<&'static str, Rgb>> {
        let mut colors: HashMap<&'static str, Rgb> = HashMap::from([("background", background)]);
        for &token in &order {
            let mut targets = Vec::new();
            for rule in rules.iter().filter(|r| r.token == token) {
                for s in &rule.on {
                    let surface_l = oklab_l(*colors.get(s).unwrap_or(&background));
                    let v = target(rule.level, surface_l, t)?;
                    if !(0.0..=1.0).contains(&v) {
                        return None;
                    }
                    targets.push(v);
                }
            }
            let l = if lighter {
                targets.iter().copied().fold(f64::MIN, f64::max)
            } else {
                targets.iter().copied().fold(f64::MAX, f64::min)
            };
            let c = if PANELS.contains(&token) { limit_panel(token, l) } else { paint(token, l) };
            colors.insert(token, c);
        }
        Some(colors)
    };

    let mut relaxation = 0.0;
    let solved = match solve(0.0) {
        Some(c) => c,
        None => {
            // Mid-gray backgrounds can't fit every level: relax as little
            // as possible. Full relaxation always fits.
            let (mut lo, mut hi) = (0.0, 2.0);
            let mut best = solve(hi);
            for _ in 0..20 {
                let mid = (lo + hi) / 2.0;
                match solve(mid) {
                    Some(a) => {
                        hi = mid;
                        best = Some(a);
                    }
                    None => lo = mid,
                }
            }
            relaxation = hi;
            best?
        }
    };

    let surfaces_of = |token: &str| -> Vec<Rgb> {
        rules
            .iter()
            .filter(|r| r.token == token)
            .flat_map(|r| r.on.iter().map(|s| *solved.get(s).unwrap_or(&background)))
            .collect()
    };

    let mut result = Map::new();
    for (token, _) in TOKENS {
        result.insert(token.to_string(), solved.get(token).map(|c| hex(*c)).unwrap_or(json!("")));
    }

    for &token in FOREGROUND_TOKENS {
        let surfaces = surfaces_of(token);
        let mut text = solved.get(token).copied();
        if let Some(fg) = foreground {
            let targets: Option<Vec<f64>> = surfaces
                .iter()
                .map(|s| target(FOREGROUND_LEVEL, oklab_l(*s), relaxation).filter(|v| (0.0..=1.0).contains(v)))
                .collect();
            if let Some(targets) = targets {
                let needed = if lighter {
                    targets.iter().copied().fold(f64::MIN, f64::max)
                } else {
                    targets.iter().copied().fold(f64::MAX, f64::min)
                };
                let fg_l = oklab_l(fg);
                // The terminal's own foreground is strong enough: use it.
                if (lighter && fg_l >= needed) || (!lighter && fg_l <= needed) {
                    result.insert(token.to_string(), json!(""));
                    continue;
                }
                text = Some(anchored(&source_of(fg), &NEUTRAL, oklab::oklab_to_okhsl_lightness(needed)));
            }
        }
        if let Some(t) = text {
            result.insert(token.to_string(), hex(with_text_contrast(t, &surfaces, lighter)));
        }
    }
    Some((result, appearance))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb_of(v: &Value) -> Option<Rgb> {
        let s = v.as_str()?.strip_prefix('#')?;
        let n = u32::from_str_radix(s, 16).ok()?;
        Some([(n >> 16) as u8, (n >> 8) as u8, n as u8])
    }

    fn colors(g: &Generated) -> &Map<String, Value> {
        g.theme["colors"].as_object().unwrap()
    }

    fn h(s: &str) -> Rgb {
        let n = u32::from_str_radix(&s[1..], 16).unwrap();
        [(n >> 16) as u8, (n >> 8) as u8, n as u8]
    }

    /// Byte-for-byte parity with pi's `generateSystemThemeColors`, on
    /// fixtures produced by running pi's own JS (Catppuccin Mocha with
    /// palette, white background only, mid-gray needing relaxation).
    #[test]
    fn matches_pi_generator() {
        let fixtures: Value =
            serde_json::from_str(include_str!("testdata/pi_system.json")).unwrap();
        let mocha = [
            "#45475a", "#f38ba8", "#a6e3a1", "#f9e2af", "#89b4fa", "#f5c2e7", "#94e2d5", "#bac2de",
            "#585b70", "#f38ba8", "#a6e3a1", "#f9e2af", "#89b4fa", "#f5c2e7", "#94e2d5", "#a6adc8",
        ]
        .map(h);
        let cases = [
            ("mocha", TerminalColors {
                background: Some(h("#1e1e2e")),
                foreground: Some(h("#cdd6f4")),
                palette: Some(mocha),
            }),
            ("white_bg_only", TerminalColors {
                background: Some(h("#ffffff")),
                ..Default::default()
            }),
            ("midgray", TerminalColors {
                background: Some(h("#808080")),
                foreground: Some(h("#000000")),
                palette: None,
            }),
        ];
        for (name, input) in cases {
            let g = generate(&input, Appearance::Dark);
            let ours = colors(&g);
            let theirs = fixtures[name].as_object().unwrap();
            assert_eq!(ours.len(), theirs.len(), "{name}: token count");
            let diffs: Vec<String> = theirs
                .iter()
                .filter(|(k, v)| ours.get(*k) != Some(v))
                .map(|(k, v)| format!("{k}: pi {v} ours {}", ours.get(k).unwrap_or(&Value::Null)))
                .collect();
            assert!(diffs.is_empty(), "{name}: {} mismatches:\n{}", diffs.len(), diffs.join("\n"));
        }
    }

    #[test]
    fn every_token_has_rules_and_family() {
        let rules = rules();
        for (t, _) in TOKENS {
            assert!(rules.iter().any(|r| r.token == *t), "no rule for {t}");
        }
        assert_eq!(solve_order(&rules).len(), TOKENS.len());
    }

    #[test]
    fn appearance_detection() {
        assert_eq!(terminal_appearance([0x1e, 0x1e, 0x2e], None), Appearance::Dark);
        assert_eq!(terminal_appearance([0xff, 0xff, 0xff], None), Appearance::Light);
        // Foreground direction wins when readable.
        assert_eq!(
            terminal_appearance([0x80, 0x80, 0x80], Some([0, 0, 0])),
            Appearance::Light
        );
    }

    #[test]
    fn nothing_reported_uses_ansi_indices() {
        let g = generate(&TerminalColors::default(), Appearance::Light);
        assert!(g.faint_neutrals);
        assert_eq!(g.appearance, Appearance::Light);
        let c = colors(&g);
        assert_eq!(c["accent"], json!(5));
        assert_eq!(c["error"], json!(1));
        assert_eq!(c["text"], json!(""));
        assert_eq!(c["userMessageBg"], json!(""));
        assert_eq!(c["thinkingXhigh"], json!(13));
    }

    /// Body text must reach WCAG 4.5:1 on the background, across a sweep
    /// of backgrounds including the hard mid-gray range.
    #[test]
    fn text_contrast_holds_across_backgrounds() {
        for v in (0..=255).step_by(15) {
            let bg = [v as u8, v as u8, (v as u8).saturating_add(8)];
            let g = generate(
                &TerminalColors { background: Some(bg), foreground: None, palette: None },
                Appearance::Dark,
            );
            assert!(!g.faint_neutrals);
            let c = colors(&g);
            let text = rgb_of(&c["text"]).expect("text is concrete without a foreground");
            let ratio = wcag_contrast(text, bg);
            assert!(ratio >= 4.5 - 1e-6, "bg {bg:?}: text {text:?} ratio {ratio}");
            for (k, v) in c {
                assert!(v.is_string(), "{k} should be a hex color");
            }
        }
    }

    #[test]
    fn strong_foreground_is_kept() {
        let g = generate(
            &TerminalColors {
                background: Some([0x1e, 0x1e, 0x2e]),
                foreground: Some([0xff, 0xff, 0xff]),
                palette: None,
            },
            Appearance::Dark,
        );
        assert_eq!(colors(&g)["text"], json!(""));
    }

    #[test]
    fn palette_hues_are_used() {
        // A palette whose "red" slot is pure blue: error should come out bluish.
        let mut palette = [[128u8, 128, 128]; 16];
        palette[1] = [0x40, 0x60, 0xff];
        let g = generate(
            &TerminalColors { background: Some([0, 0, 0]), foreground: None, palette: Some(palette) },
            Appearance::Dark,
        );
        let err = rgb_of(&colors(&g)["error"]).unwrap();
        assert!(err[2] > err[0], "error {err:?} should follow the palette's blue");
    }
}
