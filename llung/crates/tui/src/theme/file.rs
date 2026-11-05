//! Theme JSON files, compatible with pi's theme schema
//! (https://pi.dev/schemas/theme.schema.json).
//!
//! Any valid pi theme loads unchanged. llung-specific colors are set
//! through reserved `llung.*` entries in `vars` (see [`LLUNG_VARS`]). pi
//! treats `vars` as a free-form map, so a llung theme stays a valid pi
//! theme.

use super::color::{self, ColorMode};
use super::oklab;
use ratatui::style::Color;
use serde::Deserialize;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct ThemeError(pub String);

impl fmt::Display for ThemeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ThemeError {}

fn err<T>(msg: impl Into<String>) -> Result<T, ThemeError> {
    Err(ThemeError(msg.into()))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Appearance {
    Dark,
    Light,
}

/// Defines the pi color tokens once and generates:
/// - `ColorsFile`: strict serde struct for the `colors` object,
/// - `Palette`: the resolved colors, one `Color` field per token,
/// - required-token checking and optional-token fallbacks.
macro_rules! pi_tokens {
    (
        required { $($r:ident = $rk:literal),* $(,)? }
        optional { $($o:ident = $ok:literal => $fb:ident),* $(,)? }
    ) => {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct ColorsFile {
            $( #[serde(rename = $rk, default)] $r: Option<Value>, )*
            $( #[serde(rename = $ok, default)] $o: Option<Value>, )*
        }

        /// Resolved pi palette (all tokens, in theme-file order).
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct Palette {
            $( pub $r: Color, )*
            $( pub $o: Color, )*
        }

        impl ColorsFile {
            fn resolve(&self, res: &mut Resolver) -> Result<Palette, ThemeError> {
                let missing: Vec<&str> = [$( ($rk, self.$r.is_none()), )*]
                    .into_iter()
                    .filter_map(|(k, m)| m.then_some(k))
                    .collect();
                if !missing.is_empty() {
                    return err(format!("missing required colors: {}", missing.join(", ")));
                }
                $( let $r = res.resolve(self.$r.as_ref().unwrap(), concat!("colors.", $rk))?; )*
                $( let $o = match &self.$o {
                    Some(v) => res.resolve(v, concat!("colors.", $ok))?,
                    None => $fb,
                }; )*
                Ok(Palette { $( $r, )* $( $o, )* })
            }
        }
    };
}

pi_tokens! {
    required {
        accent = "accent",
        border = "border",
        border_accent = "borderAccent",
        border_muted = "borderMuted",
        success = "success",
        error = "error",
        warning = "warning",
        muted = "muted",
        dim = "dim",
        text = "text",
        thinking_text = "thinkingText",
        selected_bg = "selectedBg",
        user_message_bg = "userMessageBg",
        user_message_text = "userMessageText",
        custom_message_bg = "customMessageBg",
        custom_message_text = "customMessageText",
        custom_message_label = "customMessageLabel",
        tool_pending_bg = "toolPendingBg",
        tool_success_bg = "toolSuccessBg",
        tool_error_bg = "toolErrorBg",
        tool_title = "toolTitle",
        tool_output = "toolOutput",
        md_heading = "mdHeading",
        md_link = "mdLink",
        md_link_url = "mdLinkUrl",
        md_code = "mdCode",
        md_code_block = "mdCodeBlock",
        md_code_block_border = "mdCodeBlockBorder",
        md_quote = "mdQuote",
        md_quote_border = "mdQuoteBorder",
        md_hr = "mdHr",
        md_list_bullet = "mdListBullet",
        tool_diff_added = "toolDiffAdded",
        tool_diff_removed = "toolDiffRemoved",
        tool_diff_context = "toolDiffContext",
        syntax_comment = "syntaxComment",
        syntax_keyword = "syntaxKeyword",
        syntax_function = "syntaxFunction",
        syntax_variable = "syntaxVariable",
        syntax_string = "syntaxString",
        syntax_number = "syntaxNumber",
        syntax_type = "syntaxType",
        syntax_operator = "syntaxOperator",
        syntax_punctuation = "syntaxPunctuation",
        thinking_off = "thinkingOff",
        thinking_minimal = "thinkingMinimal",
        thinking_low = "thinkingLow",
        thinking_medium = "thinkingMedium",
        thinking_high = "thinkingHigh",
        thinking_xhigh = "thinkingXhigh",
        bash_mode = "bashMode",
    }
    optional {
        scrollbar_track = "scrollbarTrack" => muted,
        scrollbar_thumb = "scrollbarThumb" => text,
        search_match_bg = "searchMatchBg" => selected_bg,
        search_match_text = "searchMatchText" => text,
        thinking_max = "thinkingMax" => thinking_xhigh,
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExportFile {
    #[serde(rename = "pageBg", default)]
    page_bg: Option<Value>,
    #[serde(rename = "cardBg", default)]
    card_bg: Option<Value>,
    #[serde(rename = "infoBg", default)]
    info_bg: Option<Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ThemeFile {
    #[serde(rename = "$schema", default)]
    _schema: Option<String>,
    name: String,
    #[serde(default)]
    appearance: Option<Appearance>,
    #[serde(default)]
    vars: HashMap<String, Value>,
    colors: ColorsFile,
    #[serde(default)]
    export: Option<ExportFile>,
}

/// llung-specific colors, set via reserved `llung.*` vars.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LlungColors {
    /// `llung.bg`: main canvas. Default: terminal default (`""`).
    pub bg: Color,
    /// `llung.panelBg`: sidebar. Default: `toolPendingBg`.
    pub panel_bg: Color,
    /// `llung.peerName`: peer display names. Default: `accent`.
    pub peer_name: Color,
    /// `llung.peerMessageText`: others' messages. Default: `text`.
    pub peer_message_text: Color,
    /// OKHSL (saturation, lightness) of `llung.avatar`, applied with a
    /// per-peer hue to generated fallback avatars.
    pub avatar_sl: (f64, f64),
    /// `llung.plotLow` / `llung.plotHigh`: ends of the plot elevation
    /// gradient. Default: `syntaxKeyword` / `error` (via success, warning).
    pub plot_low: Option<Color>,
    pub plot_high: Option<Color>,
}

/// Reserved var names. Unknown `llung.*` names are rejected to catch typos.
pub const LLUNG_VARS: &[&str] = &[
    "llung.bg",
    "llung.panelBg",
    "llung.peerName",
    "llung.peerMessageText",
    "llung.avatar",
    "llung.plotLow",
    "llung.plotHigh",
];

/// A fully resolved theme file.
#[derive(Clone, Debug)]
pub struct ResolvedTheme {
    pub name: String,
    pub appearance: Appearance,
    pub colors: Palette,
    pub llung: LlungColors,
}

/// Resolves color values: literals, 256-color indexes and (chained)
/// variable references, adapting the result to the terminal color mode.
struct Resolver<'a> {
    vars: &'a HashMap<String, Value>,
    mode: ColorMode,
}

impl Resolver<'_> {
    fn resolve(&mut self, value: &Value, at: &str) -> Result<Color, ThemeError> {
        Ok(self.mode.adapt(self.resolve_raw(value, at)?))
    }

    /// Resolve without mode adaptation (for color math on exact values).
    fn resolve_raw(&self, value: &Value, at: &str) -> Result<Color, ThemeError> {
        let mut seen = HashSet::new();
        let mut current = value;
        loop {
            match current {
                Value::Number(n) => {
                    let idx = n.as_u64().ok_or_else(|| {
                        ThemeError(format!(
                            "{at}: ANSI color index must be an integer 0-255: {n}"
                        ))
                    })?;
                    return color::from_index(idx).map_err(|e| ThemeError(format!("{at}: {e}")));
                }
                Value::String(s) => {
                    match color::parse_literal(s).map_err(|e| ThemeError(format!("{at}: {e}")))? {
                        Some(c) => return Ok(c),
                        None => {
                            if !seen.insert(s.as_str()) {
                                return err(format!(
                                    "{at}: circular variable reference detected: {s}"
                                ));
                            }
                            current = self.vars.get(s).ok_or_else(|| {
                                ThemeError(format!("{at}: variable reference not found: {s}"))
                            })?;
                        }
                    }
                }
                other => {
                    return err(format!(
                        "{at}: expected a color string or 0-255 integer, got {other}"
                    ));
                }
            }
        }
    }
}

/// Lightness (OKHSL, 0-1) of a color, if it has a concrete value.
fn lightness(c: Color) -> Option<f64> {
    color::to_rgb(c).map(|rgb| oklab::rgb_to_okhsl(rgb).2)
}

/// Parse and resolve a theme file.
pub fn parse(json: &str, mode: ColorMode) -> Result<ResolvedTheme, ThemeError> {
    let json = json.strip_prefix('\u{feff}').unwrap_or(json);
    let file: ThemeFile =
        serde_json::from_str(json).map_err(|e| ThemeError(format!("invalid theme JSON: {e}")))?;
    resolve(file, mode, false)
}

/// Resolve the generated `system` theme document (the one place the
/// reserved name is allowed). It goes through the same validation as
/// files, so a generator bug can't produce a partial palette.
pub fn parse_system(doc: Value, mode: ColorMode) -> Result<ResolvedTheme, ThemeError> {
    let file: ThemeFile = serde_json::from_value(doc)
        .map_err(|e| ThemeError(format!("generated system theme invalid: {e}")))?;
    resolve(file, mode, true)
}

fn resolve(file: ThemeFile, mode: ColorMode, system: bool) -> Result<ResolvedTheme, ThemeError> {
    let name = file.name.trim();
    if name.is_empty() {
        return err("theme name must not be empty");
    }
    if name.contains('/') {
        return err(format!(
            "invalid theme name \"{name}\": '/' is reserved for light/dark theme pairs"
        ));
    }
    if name == "system" && !system {
        return err("\"system\" is a reserved theme name");
    }
    if let Some(bad) = file
        .vars
        .keys()
        .find(|k| k.starts_with("llung.") && !LLUNG_VARS.contains(&k.as_str()))
    {
        return err(format!(
            "unknown llung var \"{bad}\" (known: {})",
            LLUNG_VARS.join(", ")
        ));
    }

    let mut res = Resolver {
        vars: &file.vars,
        mode,
    };
    let colors = file.colors.resolve(&mut res)?;

    // Validate export colors (unused by the tui, but keep pi's strictness).
    if let Some(export) = &file.export {
        for (k, v) in [
            ("export.pageBg", &export.page_bg),
            ("export.cardBg", &export.card_bg),
            ("export.infoBg", &export.info_bg),
        ] {
            if let Some(v) = v {
                res.resolve(v, k)?;
            }
        }
    }

    // Appearance: explicit, else inferred from text vs. background lightness.
    let raw_text = res.resolve_raw(file.colors.text.as_ref().unwrap(), "colors.text")?;
    let appearance = file.appearance.unwrap_or_else(|| {
        let light_text = lightness(raw_text)
            .map(|l| l > 0.5)
            .or_else(|| lightness(colors.user_message_bg).map(|l| l < 0.5))
            .unwrap_or(true);
        if light_text {
            Appearance::Dark
        } else {
            Appearance::Light
        }
    });

    let var = |res: &mut Resolver, key: &str, default: Color| -> Result<Color, ThemeError> {
        match file.vars.get(key) {
            Some(v) => res.resolve(v, &format!("vars.{key}")),
            None => Ok(default),
        }
    };
    let avatar_sl = match file.vars.get("llung.avatar") {
        Some(v) => {
            let c = res.resolve_raw(v, "vars.llung.avatar")?;
            let rgb = color::to_rgb(c).ok_or_else(|| {
                ThemeError("vars.llung.avatar: needs a concrete color, not \"\"".into())
            })?;
            let (_, s, l) = oklab::rgb_to_okhsl(rgb);
            (s, l)
        }
        None => match appearance {
            Appearance::Dark => (0.45, 0.40),
            Appearance::Light => (0.50, 0.72),
        },
    };
    let llung = LlungColors {
        bg: var(&mut res, "llung.bg", Color::Reset)?,
        panel_bg: var(&mut res, "llung.panelBg", colors.tool_pending_bg)?,
        peer_name: var(&mut res, "llung.peerName", colors.accent)?,
        peer_message_text: var(&mut res, "llung.peerMessageText", colors.text)?,
        avatar_sl,
        plot_low: match file.vars.get("llung.plotLow") {
            Some(v) => Some(res.resolve(v, "vars.llung.plotLow")?),
            None => None,
        },
        plot_high: match file.vars.get("llung.plotHigh") {
            Some(v) => Some(res.resolve(v, "vars.llung.plotHigh")?),
            None => None,
        },
    };

    Ok(ResolvedTheme {
        name: name.to_string(),
        appearance,
        colors,
        llung,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const DARK: &str = include_str!("../../themes/dark.json");
    const LIGHT: &str = include_str!("../../themes/light.json");

    /// Minimal valid theme: every required token points at one var.
    fn minimal(extra_vars: &str, extra_colors: &str) -> String {
        let required = [
            "accent",
            "border",
            "borderAccent",
            "borderMuted",
            "success",
            "error",
            "warning",
            "muted",
            "dim",
            "text",
            "thinkingText",
            "selectedBg",
            "userMessageBg",
            "userMessageText",
            "customMessageBg",
            "customMessageText",
            "customMessageLabel",
            "toolPendingBg",
            "toolSuccessBg",
            "toolErrorBg",
            "toolTitle",
            "toolOutput",
            "mdHeading",
            "mdLink",
            "mdLinkUrl",
            "mdCode",
            "mdCodeBlock",
            "mdCodeBlockBorder",
            "mdQuote",
            "mdQuoteBorder",
            "mdHr",
            "mdListBullet",
            "toolDiffAdded",
            "toolDiffRemoved",
            "toolDiffContext",
            "syntaxComment",
            "syntaxKeyword",
            "syntaxFunction",
            "syntaxVariable",
            "syntaxString",
            "syntaxNumber",
            "syntaxType",
            "syntaxOperator",
            "syntaxPunctuation",
            "thinkingOff",
            "thinkingMinimal",
            "thinkingLow",
            "thinkingMedium",
            "thinkingHigh",
            "thinkingXhigh",
            "bashMode",
        ];
        let colors: Vec<String> = required.iter().map(|k| format!("\"{k}\": \"c\"")).collect();
        format!(
            r##"{{"name": "t", "vars": {{"c": "#808080" {extra_vars}}},
                 "colors": {{ {} {extra_colors} }} }}"##,
            colors.join(",")
        )
    }

    #[test]
    fn loads_pi_builtins() {
        let d = parse(DARK, ColorMode::TrueColor).unwrap();
        assert_eq!(d.name, "dark");
        assert_eq!(d.appearance, Appearance::Dark);
        assert_eq!(d.colors.text, Color::Rgb(0xde, 0xe0, 0xe1));
        assert_eq!(d.colors.accent, Color::Rgb(0xa7, 0x98, 0xd7));
        assert_eq!(d.llung.bg, Color::Reset);
        assert_eq!(d.llung.panel_bg, d.colors.tool_pending_bg);
        assert_eq!(d.llung.peer_name, d.colors.accent);

        let l = parse(LIGHT, ColorMode::TrueColor).unwrap();
        assert_eq!(l.appearance, Appearance::Light);
    }

    #[test]
    fn ansi256_mode_indexes_everything() {
        let d = parse(DARK, ColorMode::Ansi256).unwrap();
        assert!(matches!(d.colors.accent, Color::Indexed(_)));
        assert!(matches!(d.llung.panel_bg, Color::Indexed(_)));
    }

    #[test]
    fn minimal_theme_and_fallbacks() {
        let t = parse(&minimal("", ""), ColorMode::TrueColor).unwrap();
        let gray = Color::Rgb(128, 128, 128);
        assert_eq!(t.colors.thinking_max, gray);
        assert_eq!(t.colors.search_match_bg, gray);
        // Inferred appearance: mid-gray text (OKHSL L ≈ 0.6) reads as dark.
        assert_eq!(t.appearance, Appearance::Dark);
    }

    #[test]
    fn optional_tokens_override_fallbacks() {
        let t = parse(
            &minimal("", r##","thinkingMax": "#ff0000""##),
            ColorMode::TrueColor,
        )
        .unwrap();
        assert_eq!(t.colors.thinking_max, Color::Rgb(255, 0, 0));
    }

    #[test]
    fn llung_vars() {
        let t = parse(
            &minimal(
                r##","llung.bg": "#000000", "llung.peerName": 39, "llung.avatar": "okhsl(0 60% 70%)""##,
                "",
            ),
            ColorMode::TrueColor,
        )
        .unwrap();
        assert_eq!(t.llung.bg, Color::Rgb(0, 0, 0));
        assert_eq!(t.llung.peer_name, Color::Indexed(39));
        assert!((t.llung.avatar_sl.0 - 0.6).abs() < 0.02);
        assert!((t.llung.avatar_sl.1 - 0.7).abs() < 0.01);
    }

    #[test]
    fn rejects_unknown_llung_var() {
        let e = parse(
            &minimal(r#","llung.panelbg": "c""#, ""),
            ColorMode::TrueColor,
        )
        .unwrap_err();
        assert!(e.0.contains("llung.panelbg"), "{e}");
    }

    #[test]
    fn missing_colors_are_listed() {
        let json = r##"{"name": "t", "colors": {"accent": "#fff"}}"##;
        let e = parse(json, ColorMode::TrueColor).unwrap_err();
        assert!(e.0.contains("missing required colors"), "{e}");
        assert!(
            e.0.contains("borderAccent") && e.0.contains("bashMode"),
            "{e}"
        );
        assert!(!e.0.contains("thinkingMax"), "optional token listed: {e}");
    }

    #[test]
    fn rejects_unknown_keys() {
        let e = parse(&minimal("", r#","notAToken": "c""#), ColorMode::TrueColor).unwrap_err();
        assert!(e.0.contains("notAToken"), "{e}");
        let top = DARK.replacen("\"name\"", "\"bogus\": 1, \"name\"", 1);
        assert!(parse(&top, ColorMode::TrueColor).is_err());
    }

    #[test]
    fn var_errors() {
        let e = parse(
            &minimal(r#","a": "b", "b": "a""#, r#","thinkingMax": "a""#),
            ColorMode::TrueColor,
        )
        .unwrap_err();
        assert!(e.0.contains("circular"), "{e}");
        let e = parse(
            &minimal("", r#","thinkingMax": "nope""#),
            ColorMode::TrueColor,
        )
        .unwrap_err();
        assert!(
            e.0.contains("colors.thinkingMax") && e.0.contains("nope"),
            "{e}"
        );
        let e = parse(&minimal("", r#","thinkingMax": 300"#), ColorMode::TrueColor).unwrap_err();
        assert!(e.0.contains("0-255"), "{e}");
    }

    #[test]
    fn name_rules() {
        for (name, needle) in [("a/b", "reserved"), ("system", "reserved"), (" ", "empty")] {
            let json =
                minimal("", "").replacen("\"name\": \"t\"", &format!("\"name\": \"{name}\""), 1);
            let e = parse(&json, ColorMode::TrueColor).unwrap_err();
            assert!(e.0.contains(needle), "{name}: {e}");
        }
    }

    #[test]
    fn strips_bom() {
        assert!(parse(&format!("\u{feff}{DARK}"), ColorMode::TrueColor).is_ok());
    }
}
