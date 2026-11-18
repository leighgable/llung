//! `artist` — a llung wasm tool.
//!
//! Procedurally generated art from a seed: deterministic value-noise
//! fields in a handful of styles (domain-warped clouds, ridged crests,
//! warped rings, woven interference), rasterized to braille with an
//! ordered Bayer dither so gradients become stipple. The output is a
//! `braille-v1` render document — the exact format `landscape` emits —
//! so viewers color it by level with no new code.
//!
//! The model controls the artwork via JSON arguments; everything is
//! derived from the integer seed by avalanche hashing: no RNG state, no
//! I/O, no side effects, bounded cost.

wit_bindgen::generate!({
    path: "wit",
    world: "universal-tool",
});

mod field;
mod noise;

// `Guest` and `ToolMetadata` come from the `generate!` macro above — it
// emits them at the invocation site (crate root).
use field::Style;

/// KEEP IN SYNC with `ARTIST_TOOL_SCHEMA` in crates/agent/src/tools.rs
/// (the host-side copy that goes into the LLM's system prompt).
/// (Only referenced by the component export glue, hence wasm-only.)
#[cfg(target_arch = "wasm32")]
const INPUT_SCHEMA: &str = r#"{
  "type": "object",
  "properties": {
    "seed": {"type": "integer", "default": 42, "description": "Art seed. The same seed always produces the same artwork; change it for a new piece."},
    "style": {"type": "string", "default": "mix", "description": "Art style: clouds (soft billowing), ridged (sharp crests), rings (warped ripples), weave (woven interference), or mix (pick per seed)."},
    "zoom": {"type": "number", "default": 3, "description": "Pattern scale (must be > 0); larger zooms into the noise."},
    "width": {"type": "integer", "default": 120, "description": "Output width in characters, 8..=400 (width*height is capped at 10000 cells)"},
    "height": {"type": "integer", "default": 60, "description": "Output height in characters, 4..=200 (width*height is capped at 10000 cells)"}
  }
}"#;

const DEFAULT_SEED: u64 = 42;

/// Output cap, same reasoning as landscape: results travel as gossipsub
/// chat messages (64 KiB max); a cell costs ~4 bytes (3-byte braille char
/// + 1 level byte), so 10k cells is ~40 KiB.
pub const MAX_CELLS: usize = 10_000;

/// Ordered Bayer 8x8 dither matrix (normalized by /64 at use).
#[rustfmt::skip]
const BAYER8: [[u8; 8]; 8] = [
    [ 0, 32,  8, 40,  2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44,  4, 36, 14, 46,  6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [ 3, 35, 11, 43,  1, 33,  9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47,  7, 39, 13, 45,  5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21],
];

/// Braille dot bit for dot (dx, dy) within a cell (dots 1-8).
const DOT_BITS: [[u8; 4]; 2] = [[0x01, 0x02, 0x04, 0x80], [0x08, 0x10, 0x20, 0x40]];

#[cfg(target_arch = "wasm32")]
struct Artist;

#[cfg(target_arch = "wasm32")]
impl Guest for Artist {
    fn get_metadata() -> ToolMetadata {
        ToolMetadata {
            name: "artist".into(),
            description: "Generate procedural art from a seed as colored terminal \
                          braille. Use when asked to draw, paint, or make generative art, \
                          a pattern, or a picture."
                .into(),
            input_schema: INPUT_SCHEMA.into(),
        }
    }

    fn execute(arguments_json: String) -> Result<String, String> {
        run(&arguments_json)
    }
}

#[cfg(target_arch = "wasm32")]
export!(Artist);

struct Args {
    seed: u64,
    style: Style,
    zoom: f64,
    width: usize,
    height: usize,
}

fn parse_args(arguments_json: &str) -> Result<Args, String> {
    let v: serde_json::Value = serde_json::from_str(arguments_json)
        .map_err(|e| format!("arguments must be a JSON object: {e}"))?;
    if !v.is_object() {
        return Err("arguments must be a JSON object".into());
    }

    let seed = v
        .get("seed")
        .and_then(|n| n.as_u64())
        .unwrap_or(DEFAULT_SEED);

    let style_source = v
        .get("style")
        .and_then(|s| s.as_str())
        .unwrap_or("mix")
        .to_string();
    let style = if style_source.eq_ignore_ascii_case("mix") {
        Style::for_seed(seed)
    } else {
        Style::parse(&style_source).ok_or_else(|| {
            format!(
                "unknown style '{style_source}' (try: clouds, ridged, rings, weave, mix)"
            )
        })?
    };

    let zoom = v
        .get("zoom")
        .and_then(|n| n.as_f64())
        .unwrap_or(3.0)
        .clamp(0.5, 20.0);

    // Keep the aspect ratio while shrinking into the cell budget
    // (same policy as landscape).
    let (mut width, mut height) = (
        v.get("width").and_then(|n| n.as_i64()).unwrap_or(120).clamp(8, 400) as usize,
        v.get("height").and_then(|n| n.as_i64()).unwrap_or(60).clamp(4, 200) as usize,
    );
    if width * height > MAX_CELLS {
        let scale = (MAX_CELLS as f64 / (width * height) as f64).sqrt();
        width = ((width as f64 * scale).floor() as usize).max(8);
        height = ((height as f64 * scale).floor() as usize).max(4);
    }

    Ok(Args {
        seed,
        style,
        zoom,
        width,
        height,
    })
}

/// Generate the artwork and return a `braille-v1` document:
///
/// ```json
/// {"render": "braille-v1", "title": "artist:clouds:42", "width": 120,
///  "rows":   ["⠀⣠⠔…", …],   // braille, one string per text row
///  "levels": ["..12…", …],   // per cell: '0'-'9' brightness, '.' empty
///  "z_range": [0.0, 1.0]}
/// ```
///
/// The field is sampled per braille DOT (2×4 per cell) at a uniform
/// scale, then each dot is thresholded against its position in an 8×8
/// Bayer matrix — ordered dithering: smooth gradients become stipple,
/// and each cell's brightness level feeds the viewer's color gradient.
/// KEEP IN SYNC with the parser in crates/tui/src/tool_panel.rs.
pub fn run(arguments_json: &str) -> Result<String, String> {
    let args = parse_args(arguments_json)?;

    // Uniform pattern scale across the dot grid (features keep their
    // shape regardless of canvas size).
    let step = args.zoom / (args.width * 2) as f64;

    let mut rows = Vec::with_capacity(args.height);
    let mut levels = Vec::with_capacity(args.height);
    for cy in 0..args.height {
        let mut row = String::with_capacity(args.width * 3);
        let mut level_row = String::with_capacity(args.width);
        for cx in 0..args.width {
            let mut mask: u8 = 0;
            let mut sum = 0.0f64;
            for dy in 0..4 {
                for dx in 0..2 {
                    let px = cx * 2 + dx;
                    let py = cy * 4 + dy;
                    let v = args.style.sample(px as f64 * step, py as f64 * step, args.seed);
                    sum += v;
                    let threshold = (BAYER8[py % 8][px % 8] as f64 + 0.5) / 64.0;
                    if v > threshold {
                        mask |= DOT_BITS[dx][dy];
                    }
                }
            }
            row.push(char::from_u32(0x2800 + mask as u32).expect("braille is valid unicode"));
            level_row.push(if mask == 0 {
                '.'
            } else {
                (b'0' + (sum / 8.0 * 9.0).round().clamp(0.0, 9.0) as u8) as char
            });
        }
        rows.push(row);
        levels.push(level_row);
    }

    let title = format!("artist:{}:{}", args.style.name(), args.seed);
    let doc = serde_json::json!({
        "render": "braille-v1",
        "title": title,
        "width": args.width,
        "rows": rows,
        "levels": levels,
        "z_range": [0.0, 1.0],
    });
    Ok(doc.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(args: &str) -> serde_json::Value {
        serde_json::from_str(&run(args).expect("args should work")).expect("valid JSON")
    }

    #[test]
    fn same_seed_same_art_different_seed_different_art() {
        let a1 = run(r#"{"seed": 7, "style": "clouds", "width": 24, "height": 10}"#).unwrap();
        let a2 = run(r#"{"seed": 7, "style": "clouds", "width": 24, "height": 10}"#).unwrap();
        assert_eq!(a1, a2, "same seed must regenerate identical art");
        let b = run(r#"{"seed": 8, "style": "clouds", "width": 24, "height": 10}"#).unwrap();
        assert_ne!(a1, b);
    }

    #[test]
    fn mix_resolves_deterministically_to_a_named_style() {
        let d = doc(r#"{"seed": 99, "style": "mix", "width": 16, "height": 8}"#);
        let title = d["title"].as_str().unwrap();
        assert!(title.starts_with("artist:"));
        assert!(title.ends_with(":99"));
        assert_ne!(title, "artist:mix:99", "mix resolves to a concrete style");
    }

    #[test]
    fn doc_shape_matches_the_braille_v1_contract() {
        let d = doc(r#"{"seed": 1, "style": "rings", "width": 30, "height": 12}"#);
        assert_eq!(d["render"], "braille-v1");
        assert_eq!(d["width"], 30);
        let rows = d["rows"].as_array().unwrap();
        let levels = d["levels"].as_array().unwrap();
        assert_eq!(rows.len(), 12);
        assert_eq!(levels.len(), 12);
        let mut dots = 0;
        for (r, l) in rows.iter().zip(levels) {
            assert_eq!(r.as_str().unwrap().chars().count(), 30);
            assert_eq!(l.as_str().unwrap().chars().count(), 30);
            for (c, lv) in r.as_str().unwrap().chars().zip(l.as_str().unwrap().chars()) {
                assert!(('\u{2800}'..='\u{28FF}').contains(&c));
                if c != '\u{2800}' {
                    dots += 1;
                    assert!(lv.is_ascii_digit(), "drawn cell carries a level");
                } else {
                    assert_eq!(lv, '.');
                }
            }
        }
        assert!(dots > 30, "a real piece of art ({dots} drawn cells)");
    }

    #[test]
    fn every_style_produces_drawn_cells() {
        for style in ["clouds", "ridged", "rings", "weave"] {
            let d = doc(&format!(
                r#"{{"seed": 5, "style": "{style}", "width": 20, "height": 10}}"#
            ));
            let drawn: usize = d["rows"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|r| r.as_str().unwrap().chars())
                .filter(|c| *c != '\u{2800}')
                .count();
            assert!(drawn > 10, "{style} drew only {drawn} cells");
        }
    }

    #[test]
    fn unknown_style_is_a_clean_error() {
        let err = run(r#"{"style": "cubist"}"#).unwrap_err();
        assert!(err.contains("unknown style"), "{err}");
    }

    /// The largest request must still fit a gossipsub message (64 KiB),
    /// including JSON-in-JSON escaping in the chat message envelope.
    #[test]
    fn output_fits_a_chat_message() {
        let out = run(r#"{"width": 400, "height": 200, "seed": 3}"#).unwrap();
        let d: serde_json::Value = serde_json::from_str(&out).unwrap();
        let w = d["width"].as_u64().unwrap() as usize;
        assert!(w * d["rows"].as_array().unwrap().len() <= MAX_CELLS);
        let envelope = serde_json::to_string(&serde_json::json!({ "content": out })).unwrap();
        assert!(envelope.len() < 48 * 1024, "{} bytes", envelope.len());
    }
}
